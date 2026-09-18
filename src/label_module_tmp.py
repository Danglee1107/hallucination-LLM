"""
Bản tối ưu của label_module.py: gom batch + tự lùi batch size khi OOM.

Gán nhãn hallucination cho response (sinh bởi generate_responses_tmp.py) bằng
2 phương pháp kết hợp: Entailment (NLI so với context) + Self-consistency
(sample nhiều lần, đo độ nhất quán ngữ nghĩa).

Khác bản gốc:
  - NLI chạy theo lô nhiều cặp (premise, hypothesis) thay vì 1 cặp/forward,
    gom cả chunk trong 1 record lẫn nhiều record liền nhau.
  - Self-consistency sinh N sample cho nhiều prompt trong 1 lần generate
    (B prompt × N sequence) và dồn cặp NLI của cả nhóm record vào 1 lần gọi.
  - Chỉ đọc id/domain/response_text từ file responses (bỏ mảng hidden states),
    mỗi id giữ bản cuối nên file responses bị trùng id không sinh nhãn trùng.
  - --no-resume ghi đè file nhãn thay vì cộng dồn.
  - Gặp torch.OutOfMemoryError thì chia đôi lô và chạy tiếp, không chết phiên.

CHẠY (3 stage, chạy tuần tự):

    # Stage 1 — Entailment (nhẹ, CPU hay GPU đều được, không cần load LLM)
    uv run src/label_module_tmp.py --stage entailment --model qwen2.5-0.5b \
        --responses data/processed/responses/qwen2.5-0.5b.jsonl \
        --unified data/processed/unified_prompts.jsonl

    # Stage 2 — Self-consistency (NẶNG — load LLM để sample N lần/prompt)
    uv run src/label_module_tmp.py --stage self_consistency --model llama3.1-8b \
        --responses data/processed/responses/llama3.1-8b.jsonl \
        --unified data/processed/unified_prompts.jsonl --n_samples 10 \
        --sample-batch-size 4 --nli-device cuda

    # Stage 3 — gộp 2 kết quả thành 1 nhãn
    uv run src/label_module_tmp.py --stage combine --model llama3.1-8b

    uv run src/label_module_tmp.py --self-check   # kiểm tra nhanh, không cần GPU

Thêm --tag <nhãn> (lặp lại được) để đánh dấu run trên wandb khi so sánh cấu hình.
"""

import argparse
import gc
import json
import sys
import tempfile
import time
from itertools import combinations
from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from config import settings, tracking
from config.logs import get_logger
from generate_responses_tmp import MODEL_REGISTRY, build_chat_prompt
from utils.jsonl import iter_heads

log = get_logger("label_module")

LABEL_DIR = settings.label_dir

# --- NLI config (cả 2 stage đều dùng model này) ---
NLI_MODEL_ID = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
NLI_MAX_TOKENS = 400  # chunk context theo cửa sổ này (NLI model giới hạn ~512 token)
NLI_CHUNK_STRIDE = 350  # overlap giữa các chunk để không cắt đứt câu quan trọng

# Số record self-consistency xử lý mỗi vòng: sample hết nhóm (chia lô theo
# --sample-batch-size) rồi dồn toàn bộ cặp NLI của nhóm vào nli_probs 1 lần.
SC_RECORD_GROUP = 16

# --- Ngưỡng quyết định nhãn ---
ENTAILMENT_SUPPORT_THRESHOLD = 0.5  # entailment_prob >= ngưỡng -> "được context ủng hộ"
ENTAILMENT_CONTRADICT_THRESHOLD = (
    0.3  # contradiction_prob >= ngưỡng -> "mâu thuẫn với context"
)
CONSISTENCY_THRESHOLD = 0.6  # tỉ lệ cặp sample nhất quán >= ngưỡng -> "nhất quán"
# Threshold riêng cho self-consistency, tách khỏi ENTAILMENT_SUPPORT_THRESHOLD:
# 2 câu hỏi khác nhau ("context có ủng hộ response" vs "2 sample có nhất quán").
SELF_CONSISTENCY_PAIR_ENTAIL_THRESHOLD = 0.5


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_responses(path, limit=None):
    """Chỉ id/domain/response_text (bỏ qua mảng hidden states nặng hàng GB), mỗi id
    giữ bản CUỐI — file responses cũ từng bị ghi trùng id, không lọc thì ra nhãn trùng."""
    rows = list({r["id"]: r for r in iter_heads(Path(path))}.values())
    return rows[:limit] if limit else rows


def load_done_ids(path):
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def free_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def map_batched(fn, items, state, name):
    """Chạy fn theo lô, fn(list) trả list kết quả cùng độ dài.

    OOM thì xả cache, chia đôi lô rồi chạy lại. Kích thước mới ghi vào state nên
    các lô sau không lặp lại lỗi. Lô còn 1 phần tử mà vẫn OOM thì bỏ phần tử đó
    (kết quả None) và chạy tiếp — phiên chạy không chết.
    Lỗi khác OOM vẫn ném lên để không nuốt bug thật.
    """
    results = []
    i = 0
    while i < len(items):
        size = max(1, state["size"])
        sub = items[i : i + size]
        try:
            results.extend(fn(sub))
            i += len(sub)
        except torch.OutOfMemoryError:
            free_cuda()
            if size == 1:
                log.warning(f"{name}: OOM ở lô 1 phần tử, bỏ phần tử thứ {i}")
                results.append(None)
                i += 1
            else:
                state["size"] = size // 2
                log.warning(f"{name}: OOM, giảm batch {size} -> {state['size']}")
    return results


def chunk_text(text, tokenizer, max_tokens=NLI_MAX_TOKENS, stride=NLI_CHUNK_STRIDE):
    """Chia context dài thành chunk theo token, có overlap, để không vượt giới hạn NLI."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return [text]
    chunks = []
    start = 0
    while start < len(ids):
        chunks.append(tokenizer.decode(ids[start : start + max_tokens]))
        if start + max_tokens >= len(ids):
            break
        start += stride
    return chunks


# STAGE 1: ENTAILMENT
def load_nli(device=None, fp32=False):
    tok = AutoTokenizer.from_pretrained(NLI_MODEL_ID)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    if device == "cuda" and not fp32 and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16  # nửa VRAM, gấp đôi throughput trên Ampere trở lên
    model = AutoModelForSequenceClassification.from_pretrained(
        NLI_MODEL_ID, dtype=dtype
    )
    model = model.to(device)
    model.eval()

    # Thứ tự label khác nhau giữa các checkpoint NLI (bart-large-mnli đảo ngược),
    # nên verify lúc chạy thay vì tin vào model card.
    log.info(
        f"NLI {NLI_MODEL_ID} trên {device} ({dtype}), id2label = {model.config.id2label}"
    )
    assert (
        model.config.id2label[0].lower() == "entailment"
    ), "Thứ tự label không như giả định — code đang giả định index 0 = entailment!"
    assert (
        model.config.id2label[2].lower() == "contradiction"
    ), "Thứ tự label không như giả định — code đang giả định index 2 = contradiction!"
    return tok, model


@torch.inference_mode()
def nli_probs(pairs, nli_tok, nli_model, state):
    """pairs: list (premise, hypothesis) -> list [entail, neutral, contra].
    Phần tử None = lô 1 cặp vẫn OOM, bỏ qua cặp đó."""
    if not pairs:
        return []

    def run(sub):
        premises, hypotheses = zip(*sub)
        inputs = nli_tok(
            list(premises),
            list(hypotheses),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(nli_model.device)
        logits = nli_model(**inputs).logits.float()
        return torch.softmax(logits, dim=-1).cpu().tolist()  # 1 lần sync cho cả lô

    return map_batched(run, pairs, state, "nli")


def best_chunk_scores(probs):
    """Chọn 1 chunk có net support (entail - contra) cao nhất, lấy CẢ HAI giá trị từ
    ĐÚNG chunk đó. Lấy max entail và max contra độc lập từ 2 chunk khác nhau là sai."""
    best = None
    for p in probs:
        if p is None:
            continue
        entail_p, _, contra_p = p
        if best is None or entail_p - contra_p > best[0] - best[1]:
            best = (entail_p, contra_p)
    return best


def run_entailment_stage(
    model_key,
    responses_path,
    unified_path,
    limit=None,
    resume=True,
    nli_batch_size=16,
    nli_device=None,
    fp32=False,
):
    responses = load_responses(responses_path, limit)
    unified_by_id = {r["id"]: r for r in load_jsonl(unified_path)}

    out_path = LABEL_DIR / f"entailment_{model_key}.jsonl"
    done_ids = load_done_ids(out_path) if resume else set()
    todo = [r for r in responses if r["id"] not in done_ids]
    slog = log.bind(stage="entailment")
    slog.info(f"entailment: {len(done_ids)} record đã xong, {len(todo)} record cần chạy")
    if not todo:
        return {"ok": 0, "failed": 0, "sec": 0.0}

    nli_tok, nli_model = load_nli(nli_device, fp32)
    state = {"size": nli_batch_size}

    failed = 0
    t0 = time.perf_counter()
    bar = tqdm(total=len(todo), desc="entailment", unit="rec", dynamic_ncols=True)
    # Mỗi record có >= 1 chunk nên gom nli_batch_size record là đủ lấp đầy 1 lô NLI
    # (QA gần như luôn 1 chunk; gom ít hơn thì lô NLI thực tế nhỏ hơn tham số).
    group_size = nli_batch_size
    # --no-resume phải GHI ĐÈ, "a" sẽ cộng dồn nhãn trùng id.
    with open(out_path, "a" if resume else "w", encoding="utf-8") as fout:
        for start in range(0, len(todo), group_size):
            group = todo[start : start + group_size]

            # Trải phẳng (chunk, response) của cả nhóm record thành 1 danh sách cặp,
            # nhớ owner để gom lại sau. Nhờ vậy record 1 chunk vẫn chạy full lô.
            pairs, owners, usable = [], [], []
            for resp in group:
                record = unified_by_id.get(resp["id"])
                if record is None:
                    slog.warning(
                        f"entailment: không thấy context cho id={resp['id']}, bỏ qua"
                    )
                    failed += 1
                    bar.update(1)
                    continue
                chunks = chunk_text(record["context"], nli_tok)
                owners.extend([len(usable)] * len(chunks))
                pairs.extend((c, resp["response_text"]) for c in chunks)
                usable.append(resp)

            probs = nli_probs(pairs, nli_tok, nli_model, state)

            by_owner = [[] for _ in usable]
            for owner, prob in zip(owners, probs):
                by_owner[owner].append(prob)

            for resp, chunk_probs in zip(usable, by_owner):
                best = best_chunk_scores(chunk_probs)
                if best is None:
                    slog.warning(
                        f"entailment: id={resp['id']} không chunk nào chạy được, bỏ qua"
                    )
                    failed += 1
                    bar.update(1)
                    continue
                entail_p, contra_p = best
                supported = (
                    entail_p >= ENTAILMENT_SUPPORT_THRESHOLD
                    and contra_p < ENTAILMENT_CONTRADICT_THRESHOLD
                )
                fout.write(
                    json.dumps(
                        {
                            "id": resp["id"],
                            "domain": resp["domain"],
                            "model": model_key,
                            "entailment_prob": entail_p,
                            "contradiction_prob": contra_p,
                            "entailment_label": (
                                "supported" if supported else "hallucinated"
                            ),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                bar.update(1)
            fout.flush()
            bar.set_postfix(batch=state["size"], failed=failed)
            # nli_batch theo dõi luôn kích thước lô THỰC TẾ sau khi map_batched tự
            # giảm vì OOM — số này quyết định throughput, đừng chỉ nhìn tham số CLI.
            tracking.log_metrics(
                {
                    "entailment/records_done": bar.n,
                    "entailment/failed": failed,
                    "entailment/rec_per_s": bar.n / (time.perf_counter() - t0),
                    "entailment/nli_batch": state["size"],
                }
            )

    bar.close()
    sec = time.perf_counter() - t0
    slog.info(
        f"entailment xong: {len(todo) - failed} ok, {failed} lỗi, {sec:.0f}s -> {out_path}"
    )
    return {"ok": len(todo) - failed, "failed": failed, "sec": round(sec, 1)}


# STAGE 2: SELF-CONSISTENCY
@torch.inference_mode()
def sample_batch(
    model, tokenizer, records, n_samples, max_new_tokens=256, temperature=0.7, top_p=0.9
):
    """N sample cho MỖI record trong 1 lần generate: B prompt × N sequence.
    Trước đây 1 prompt/lần nên lô chỉ có N sequence và GPU rảnh phần lớn thời gian.
    Đây là mẫu để đo độ nhất quán, KHÔNG phải response chính thức."""
    texts = [
        tokenizer.apply_chat_template(
            build_chat_prompt(r), add_generation_prompt=True, tokenize=False
        )
        for r in records
    ]
    # add_special_tokens=False: chat template đã chèn BOS rồi, thêm nữa là lặp.
    encoded = tokenizer(
        texts, return_tensors="pt", padding=True, add_special_tokens=False
    ).to(model.device)
    generated = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=n_samples,
        pad_token_id=tokenizer.pad_token_id,
    )
    # generate xếp output theo prompt: [p0s0 .. p0s(N-1), p1s0, ...]. Pad/EOS ở
    # đuôi là special token nên skip_special_tokens bỏ luôn.
    out = tokenizer.batch_decode(
        generated[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return [out[k * n_samples : (k + 1) * n_samples] for k in range(len(records))]


def consistency_scores(sample_groups, nli_fn):
    """Mỗi nhóm sample -> tỉ lệ cặp 'nhất quán' (None nếu không cặp nào chạy được NLI).
    Nhất quán = 2 sample entail LẪN NHAU (cả 2 chiều), giống ý tưởng gom cụm theo
    semantic entailment trong semantic entropy. Cặp của MỌI nhóm dồn vào 1 lần
    nli_fn để lô NLI đầy. nli_fn: list (premise, hypothesis) -> list probs | None."""
    pairs, owners = [], []
    for k, samples in enumerate(sample_groups):
        for a, b in combinations(samples, 2):
            pairs += [(a, b), (b, a)]  # 2 chiều đứng liền nhau
            owners.append(k)
    probs = nli_fn(pairs)

    agree = [[] for _ in sample_groups]
    for k, fwd, bwd in zip(owners, probs[0::2], probs[1::2]):
        if fwd is not None and bwd is not None:  # None = cặp OOM ở lô 1 cặp
            agree[k].append(min(fwd[0], bwd[0]) >= SELF_CONSISTENCY_PAIR_ENTAIL_THRESHOLD)
    return [
        1.0 if len(samples) < 2 else (sum(a) / len(a) if a else None)
        for samples, a in zip(sample_groups, agree)
    ]


def run_self_consistency_stage(
    model_key,
    responses_path,
    unified_path,
    n_samples=5,
    limit=None,
    resume=True,
    nli_batch_size=16,
    sample_batch_size=4,
    nli_device=None,
    fp32=False,
):
    responses = load_responses(responses_path, limit)
    unified_by_id = {r["id"]: r for r in load_jsonl(unified_path)}

    out_path = LABEL_DIR / f"self_consistency_{model_key}.jsonl"
    done_ids = load_done_ids(out_path) if resume else set()
    todo = [r for r in responses if r["id"] not in done_ids]
    slog = log.bind(stage="self_consistency")
    slog.info(
        f"self_consistency: {len(done_ids)} record đã xong, {len(todo)} record cần chạy"
    )
    if not todo:
        return {"ok": 0, "failed": 0, "sec": 0.0}

    cfg = MODEL_REGISTRY[model_key]
    slog.info(f"loading {model_key} ({cfg['repo_id']})")
    tokenizer = AutoTokenizer.from_pretrained(cfg["repo_id"])
    tokenizer.padding_side = "left"  # bắt buộc cho generate theo batch
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # llama không có pad token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["repo_id"],
        device_map="auto",
        attn_implementation="sdpa",  # stage này không đọc attentions, sdpa nhanh hơn eager
        **cfg["load_kwargs"],
    )
    model.eval()

    nli_tok, nli_model = load_nli(nli_device, fp32)
    nli_state = {"size": nli_batch_size}
    sample_state = {"size": sample_batch_size}  # số PROMPT mỗi lần generate

    def nli_fn(pairs):
        return nli_probs(pairs, nli_tok, nli_model, nli_state)

    def sample_fn(items):
        return sample_batch(model, tokenizer, [rec for _, rec in items], n_samples)

    failed = 0
    t0 = time.perf_counter()
    bar = tqdm(total=len(todo), desc="self_consistency", unit="rec", dynamic_ncols=True)
    items = []
    for resp in todo:
        record = unified_by_id.get(resp["id"])
        if record is None:
            slog.warning(f"self_consistency: không thấy prompt cho id={resp['id']}, bỏ qua")
            failed += 1
            bar.update(1)
        else:
            items.append((resp, record))
    # Prompt dài đi với prompt dài: đỡ pad trái. Batch size chỉ giảm khi OOM, mà
    # prompt dài dồn về cuối, nên các lô đầu vẫn chạy ở batch size lớn nhất.
    items.sort(key=lambda it: len(it[1]["context"]) + len(it[1]["prompt"]))

    with open(out_path, "a" if resume else "w", encoding="utf-8") as fout:
        for start in range(0, len(items), SC_RECORD_GROUP):
            group = items[start : start + SC_RECORD_GROUP]
            try:
                # None = prompt vẫn OOM ở lô 1 prompt
                sample_groups = [
                    g or [] for g in map_batched(sample_fn, group, sample_state, "sample")
                ]
                scores = consistency_scores(sample_groups, nli_fn)
            except Exception:
                failed += len(group)
                slog.exception(f"self_consistency: lỗi ở nhóm bắt đầu từ id={group[0][0]['id']}")
                free_cuda()
                bar.update(len(group))
                continue

            for (resp, _), samples, score in zip(group, sample_groups, scores):
                if not samples or score is None:
                    failed += 1
                    slog.warning(f"self_consistency: id={resp['id']} không sample/NLI được, bỏ qua")
                    continue
                fout.write(
                    json.dumps(
                        {
                            "id": resp["id"],
                            "domain": resp["domain"],
                            "model": model_key,
                            "n_samples": len(samples),
                            "consistency_score": score,
                            "consistency_label": (
                                "consistent"
                                if score >= CONSISTENCY_THRESHOLD
                                else "inconsistent"
                            ),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            fout.flush()
            bar.update(len(group))
            bar.set_postfix(
                sample=sample_state["size"], nli=nli_state["size"], failed=failed
            )
            tracking.log_metrics(
                {
                    "self_consistency/records_done": bar.n,
                    "self_consistency/failed": failed,
                    "self_consistency/rec_per_s": bar.n / (time.perf_counter() - t0),
                    "self_consistency/sample_batch": sample_state["size"],
                    "self_consistency/nli_batch": nli_state["size"],
                    "self_consistency/vram_peak_gb": tracking.vram_peak_gb(),
                }
            )

    bar.close()
    # Gán None thay vì del: nli_fn/sample_fn giữ closure tới các biến này.
    model = tokenizer = nli_model = None
    gc.collect()
    free_cuda()
    sec = time.perf_counter() - t0
    slog.info(
        f"self_consistency xong: {len(todo) - failed} ok, {failed} lỗi, {sec:.0f}s "
        f"-> {out_path}"
    )
    return {"ok": len(todo) - failed, "failed": failed, "sec": round(sec, 1)}


# STAGE 3: COMBINE
def run_combine_stage(model_key):
    ent_path = LABEL_DIR / f"entailment_{model_key}.jsonl"
    sc_path = LABEL_DIR / f"self_consistency_{model_key}.jsonl"
    out_path = LABEL_DIR / f"combined_{model_key}.jsonl"

    if not ent_path.exists() or not sc_path.exists():
        raise FileNotFoundError(
            "Cần chạy xong cả stage 'entailment' và 'self_consistency' cho model này trước khi combine."
        )

    ent_by_id = {r["id"]: r for r in load_jsonl(ent_path)}
    sc_by_id = {r["id"]: r for r in load_jsonl(sc_path)}
    common_ids = sorted(set(ent_by_id) & set(sc_by_id))
    slog = log.bind(stage="combine")
    slog.info(
        f"combine: {len(common_ids)} id có đủ 2 nhãn (entailment ∩ self_consistency)"
    )

    counts = {"hallucination": 0, "not_hallucination": 0, "disagreement": 0}
    samples = []  # vài chục dòng đầu để xem định tính trên wandb
    with open(out_path, "w", encoding="utf-8") as f:
        for rid in common_ids:
            ent, sc = ent_by_id[rid], sc_by_id[rid]
            ent_bad = ent["entailment_label"] == "hallucinated"
            sc_bad = sc["consistency_label"] == "inconsistent"

            if ent_bad and sc_bad:
                final_label, confidence = "hallucination", "high"
            elif not ent_bad and not sc_bad:
                final_label, confidence = "not_hallucination", "high"
            else:
                final_label, confidence = (
                    "disagreement",
                    "low",
                )  # cần review tay trên subset nhỏ
            counts[final_label] += 1
            if len(samples) < 50:
                samples.append(
                    [
                        rid,
                        ent["domain"],
                        round(ent["entailment_prob"], 3),
                        round(ent["contradiction_prob"], 3),
                        round(sc["consistency_score"], 3),
                        final_label,
                    ]
                )

            f.write(
                json.dumps(
                    {
                        "id": rid,
                        "domain": ent["domain"],
                        "model": model_key,
                        "entailment_label": ent["entailment_label"],
                        "entailment_prob": ent["entailment_prob"],
                        "contradiction_prob": ent["contradiction_prob"],
                        "consistency_label": sc["consistency_label"],
                        "consistency_score": sc["consistency_score"],
                        "final_label": final_label,
                        "confidence": confidence,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    total = len(common_ids) or 1
    for label, n in counts.items():
        slog.info(f"combine: {label}: {n} ({n / total:.1%})")
    slog.info(f"combine xong -> {out_path}")

    tracking.log_metrics(
        {f"combine/{label}": n for label, n in counts.items()}
        | {f"combine/{label}_ratio": n / total for label, n in counts.items()}
    )
    tracking.log_bar("combine/label_distribution", counts)
    tracking.log_table(
        "combine/samples",
        ["id", "domain", "entail_p", "contra_p", "consistency", "final_label"],
        samples,
    )
    return {"total": len(common_ids), **counts}


def self_check():
    """Check nhỏ chạy được không cần GPU/model."""
    # map_batched: OOM thì chia đôi lô, kết quả vẫn đủ và đúng thứ tự
    state = {"size": 8}
    seen = []

    def halves_on_oom(sub):
        seen.append(len(sub))
        if len(sub) > 2:
            raise torch.OutOfMemoryError("fake OOM")
        return [x * 2 for x in sub]

    assert map_batched(halves_on_oom, list(range(6)), state, "t") == [0, 2, 4, 6, 8, 10]
    assert state["size"] == 2, state
    assert seen[:3] == [6, 4, 2], seen  # lô 8 chỉ có 6 item -> 4 -> 2 rồi mới chạy được

    # OOM cả ở lô 1 phần tử: trả None, không lặp vô hạn
    always = {"size": 2}

    def always_oom(sub):
        raise torch.OutOfMemoryError("fake OOM")

    assert map_batched(always_oom, [1, 2], always, "t") == [None, None]

    # lỗi khác OOM phải ném lên, không nuốt
    def boom(sub):
        raise ValueError("bug thật")

    try:
        map_batched(boom, [1], {"size": 1}, "t")
        raise AssertionError("lỗi thường phải ném lên")
    except ValueError:
        pass

    # best_chunk_scores: lấy cả 2 giá trị từ ĐÚNG chunk có net support cao nhất
    probs = [[0.9, 0.05, 0.8], [0.6, 0.2, 0.05], None]
    assert best_chunk_scores(probs) == (0.6, 0.05)  # không phải (0.9, 0.05)
    assert best_chunk_scores([None]) is None

    # consistency_scores: fake NLI entail cặp khi 2 câu cùng chữ cái đầu
    def fake_nli(pairs):
        return [[0.9, 0.05, 0.05] if a[0] == b[0] else [0.1, 0.1, 0.8] for a, b in pairs]

    got = consistency_scores([["a1", "a2", "b1"], ["x"], ["c1", "c2"]], fake_nli)
    assert got == [1 / 3, 1.0, 1.0], got  # nhóm 1: chỉ cặp (a1, a2) nhất quán
    assert consistency_scores([["a", "b"]], lambda pairs: [None] * len(pairs)) == [None]

    # load_responses: giữ bản cuối mỗi id, không đọc mảng hidden states
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "r.jsonl"
        path.write_text(
            "".join(
                json.dumps({"id": i, "response_text": t, "hidden_states_per_layer": [[1.0]]}) + "\n"
                for i, t in [("q1", "old"), ("q2", "x"), ("q1", "new")]
            )
        )
        rows = load_responses(path)
        assert [(r["id"], r["response_text"]) for r in rows] == [("q1", "new"), ("q2", "x")]
        assert "hidden_states_per_layer" not in rows[0]

    print("self-check OK")


# logger.catch: lỗi ở main vẫn vào file log kèm traceback dù thư viện khác có
# ghi đè sys.excepthook. default=1 để `sys.exit(main())` trả mã lỗi khác 0.
@logger.catch(default=1)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=["entailment", "self_consistency", "combine"]
    )
    parser.add_argument("--model", help="model_key trong MODEL_REGISTRY")
    parser.add_argument(
        "--responses", help="Path tới output của generate_responses_tmp.py"
    )
    parser.add_argument("--unified", help="Path tới unified_prompts.jsonl")
    parser.add_argument(
        "--n_samples", type=int, default=5, help="Số lần sample cho self_consistency"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--nli-batch-size", type=int, default=16, help="Số cặp NLI mỗi forward"
    )
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=4,
        # ponytail: mỗi prompt nhân N sequence và HF không chia sẻ KV prefix giữa
        # chúng, nên VRAM ~ B × N × (prompt + max_new_tokens). OOM thì tự chia đôi.
        help="Số PROMPT mỗi lần generate, mỗi prompt sinh --n_samples sequence",
    )
    parser.add_argument("--nli-device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--fp32", action="store_true", help="Tắt bf16 cho NLI")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tag", action="append", default=[], help="Tag cho wandb run, lặp lại được"
    )
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return 0
    if not args.stage or not args.model:
        parser.error("--stage và --model là bắt buộc")

    tracking.set_seed(args.seed)
    info = tracking.runtime_info(args.seed)
    log.info(f"runtime: {info}")
    tracking.init_run(f"label_{args.stage}", {**vars(args), **info}, tags=args.tag)

    if args.stage == "combine":
        tracking.finish(run_combine_stage(args.model))
        return 0

    if not args.responses or not args.unified:
        parser.error(f"stage {args.stage} cần --responses và --unified")

    if args.stage == "entailment":
        stats = run_entailment_stage(
            args.model,
            Path(args.responses),
            Path(args.unified),
            limit=args.limit,
            resume=not args.no_resume,
            nli_batch_size=args.nli_batch_size,
            nli_device=args.nli_device,
            fp32=args.fp32,
        )
    else:
        stats = run_self_consistency_stage(
            args.model,
            Path(args.responses),
            Path(args.unified),
            n_samples=args.n_samples,
            limit=args.limit,
            resume=not args.no_resume,
            nli_batch_size=args.nli_batch_size,
            sample_batch_size=args.sample_batch_size,
            nli_device=args.nli_device,
            fp32=args.fp32,
        )
    tracking.finish(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())

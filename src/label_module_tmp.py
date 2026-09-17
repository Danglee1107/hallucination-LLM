"""
Bản tối ưu của label_module.py: gom batch + tự lùi batch size khi OOM.

Gán nhãn hallucination cho response (sinh bởi generate_responses_tmp.py) bằng
2 phương pháp kết hợp: Entailment (NLI so với context) + Self-consistency
(sample nhiều lần, đo độ nhất quán ngữ nghĩa).

Khác bản gốc:
  - NLI chạy theo lô nhiều cặp (premise, hypothesis) thay vì 1 cặp/forward,
    gom cả chunk trong 1 record lẫn nhiều record liền nhau.
  - sample_n_responses dùng num_return_sequences thay cho N lần generate().
  - Gặp torch.OutOfMemoryError thì chia đôi lô và chạy tiếp, không chết phiên.

CHẠY (3 stage, chạy tuần tự):

    # Stage 1 — Entailment (nhẹ, CPU hay GPU đều được, không cần load LLM)
    uv run label_module_tmp.py --stage entailment --model qwen2.5-0.5b \
        --responses ../data/processed/responses/qwen2.5-0.5b.jsonl \
        --unified ../data/processed/unified_prompts.jsonl

    # Stage 2 — Self-consistency (NẶNG — load LLM để sample N lần/prompt)
    uv run label_module_tmp.py --stage self_consistency --model llama3.1-8b \
        --responses ../data/processed/responses/llama3.1-8b.jsonl \
        --unified ../data/processed/unified_prompts.jsonl --n_samples 10 \
        --nli-device cuda 

    # Stage 3 — gộp 2 kết quả thành 1 nhãn
    uv run label_module_tmp.py --stage combine --model llama3.1-8b

    uv run label_module_tmp.py --self-check   # kiểm tra nhanh, không cần GPU
"""

import argparse
import gc
import json
import sys
from itertools import combinations
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from config import settings
from config.logs import get_logger
from generate_responses_tmp import MODEL_REGISTRY, build_chat_prompt

log = get_logger("label_module")

LABEL_DIR = settings.label_dir

# --- NLI config (cả 2 stage đều dùng model này) ---
NLI_MODEL_ID = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
NLI_MAX_TOKENS = 400  # chunk context theo cửa sổ này (NLI model giới hạn ~512 token)
NLI_CHUNK_STRIDE = 350  # overlap giữa các chunk để không cắt đứt câu quan trọng

# Số record gom lại trước khi gọi NLI. Context QA thường chỉ 1 chunk nên nếu
# chỉ gom trong 1 record thì lô vẫn bằng 1 và GPU rảnh.
ENTAILMENT_RECORD_GROUP = 32

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
    responses = load_jsonl(responses_path)
    if limit:
        responses = responses[:limit]
    unified_by_id = {r["id"]: r for r in load_jsonl(unified_path)}

    out_path = LABEL_DIR / f"entailment_{model_key}.jsonl"
    done_ids = load_done_ids(out_path) if resume else set()
    todo = [r for r in responses if r["id"] not in done_ids]
    log.info(f"entailment: {len(done_ids)} record đã xong, {len(todo)} record cần chạy")
    if not todo:
        return

    nli_tok, nli_model = load_nli(nli_device, fp32)
    state = {"size": nli_batch_size}

    failed = 0
    bar = tqdm(total=len(todo), desc="entailment", unit="rec", dynamic_ncols=True)
    with open(out_path, "a", encoding="utf-8") as fout:
        for start in range(0, len(todo), ENTAILMENT_RECORD_GROUP):
            group = todo[start : start + ENTAILMENT_RECORD_GROUP]

            # Trải phẳng (chunk, response) của cả nhóm record thành 1 danh sách cặp,
            # nhớ owner để gom lại sau. Nhờ vậy record 1 chunk vẫn chạy full lô.
            pairs, owners, usable = [], [], []
            for resp in group:
                record = unified_by_id.get(resp["id"])
                if record is None:
                    log.warning(
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
                    log.warning(
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

    bar.close()
    log.info(f"entailment xong: {len(todo) - failed} ok, {failed} lỗi -> {out_path}")


# STAGE 2: SELF-CONSISTENCY
@torch.inference_mode()
def sample_n_responses(
    model,
    tokenizer,
    messages,
    n_samples,
    state,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
):
    """Sample N response khác nhau cho cùng 1 prompt bằng num_return_sequences
    (1 lần generate thay vì N lần). Đây là mẫu để đo độ nhất quán, KHÔNG phải
    response chính thức (đã có ở generate_responses_tmp.py)."""
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    input_ids = encoded["input_ids"]
    prompt_len = input_ids.shape[1]

    def run(sub):
        generated = model.generate(
            input_ids,
            attention_mask=encoded.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=len(sub),
            pad_token_id=tokenizer.eos_token_id,
        )
        return [
            tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            for seq in generated
        ]

    texts = map_batched(run, list(range(n_samples)), state, "sample")
    return [t for t in texts if t is not None]


def consistency_score(samples, nli_tok, nli_model, state):
    """Tỉ lệ cặp sample 'nhất quán'. Nhất quán = 2 sample entail LẪN NHAU (cả 2 chiều),
    giống ý tưởng gom cụm theo semantic entailment trong semantic entropy."""
    if len(samples) < 2:
        return 1.0
    pairs = list(combinations(samples, 2))
    # Cả 2 chiều đưa vào 1 lô: nửa đầu a->b, nửa sau b->a.
    probs = nli_probs(
        [(a, b) for a, b in pairs] + [(b, a) for a, b in pairs],
        nli_tok,
        nli_model,
        state,
    )

    scores = []
    for k in range(len(pairs)):
        forward, backward = probs[k], probs[len(pairs) + k]
        if forward is None or backward is None:
            continue
        scores.append(min(forward[0], backward[0]))  # entail prob 2 chiều
    if not scores:
        raise RuntimeError("không cặp nào chạy được NLI")
    consistent = sum(1 for s in scores if s >= SELF_CONSISTENCY_PAIR_ENTAIL_THRESHOLD)
    return consistent / len(scores)


def run_self_consistency_stage(
    model_key,
    responses_path,
    unified_path,
    n_samples=5,
    limit=None,
    resume=True,
    nli_batch_size=16,
    sample_batch_size=None,
    nli_device=None,
    fp32=False,
):
    responses = load_jsonl(responses_path)
    if limit:
        responses = responses[:limit]
    unified_by_id = {r["id"]: r for r in load_jsonl(unified_path)}

    out_path = LABEL_DIR / f"self_consistency_{model_key}.jsonl"
    done_ids = load_done_ids(out_path) if resume else set()
    todo = [r for r in responses if r["id"] not in done_ids]
    log.info(
        f"self_consistency: {len(done_ids)} record đã xong, {len(todo)} record cần chạy"
    )
    if not todo:
        return

    cfg = MODEL_REGISTRY[model_key]
    log.info(f"loading {model_key} ({cfg['repo_id']})")
    tokenizer = AutoTokenizer.from_pretrained(cfg["repo_id"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["repo_id"],
        device_map="auto",
        attn_implementation="sdpa",  # stage này không đọc attentions, sdpa nhanh hơn eager
        **cfg["load_kwargs"],
    )
    model.eval()

    nli_tok, nli_model = load_nli(nli_device, fp32)
    nli_state = {"size": nli_batch_size}
    sample_state = {"size": sample_batch_size or n_samples}

    failed = 0
    bar = tqdm(total=len(todo), desc="self_consistency", unit="rec", dynamic_ncols=True)
    with open(out_path, "a", encoding="utf-8") as fout:
        for resp in todo:
            record = unified_by_id.get(resp["id"])
            if record is None:
                log.warning(
                    f"self_consistency: không thấy prompt cho id={resp['id']}, bỏ qua"
                )
                failed += 1
                bar.update(1)
                continue
            try:
                samples = sample_n_responses(
                    model, tokenizer, build_chat_prompt(record), n_samples, sample_state
                )
                if not samples:
                    raise RuntimeError("không sample được response nào")
                score = consistency_score(samples, nli_tok, nli_model, nli_state)
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
            except Exception:
                failed += 1
                log.exception(f"self_consistency: lỗi ở id={resp['id']}")
                free_cuda()
            bar.update(1)
            bar.set_postfix(
                sample=sample_state["size"], nli=nli_state["size"], failed=failed
            )

    bar.close()
    del model, tokenizer, nli_model
    gc.collect()
    free_cuda()
    log.info(
        f"self_consistency xong: {len(todo) - failed} ok, {failed} lỗi -> {out_path}"
    )


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
    log.info(
        f"combine: {len(common_ids)} id có đủ 2 nhãn (entailment ∩ self_consistency)"
    )

    counts = {"hallucination": 0, "not_hallucination": 0, "disagreement": 0}
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
        log.info(f"combine: {label}: {n} ({n / total:.1%})")
    log.info(f"combine xong -> {out_path}")


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

    print("self-check OK")


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
        default=None,
        help="Số sample mỗi lần generate (mặc định = --n_samples)",
    )
    parser.add_argument("--nli-device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--fp32", action="store_true", help="Tắt bf16 cho NLI")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return
    if not args.stage or not args.model:
        parser.error("--stage và --model là bắt buộc")

    if args.stage == "combine":
        run_combine_stage(args.model)
        return

    if not args.responses or not args.unified:
        parser.error(f"stage {args.stage} cần --responses và --unified")

    if args.stage == "entailment":
        run_entailment_stage(
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
        run_self_consistency_stage(
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


if __name__ == "__main__":
    sys.exit(main())

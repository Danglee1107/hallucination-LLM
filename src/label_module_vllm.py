"""
Stage 2 (self-consistency) bằng vLLM — bản TỐI ƯU + TỰ CHỈNH THAM SỐ.

Thay thế run_self_consistency_stage() trong label_module_tmp.py. Stage entailment
và combine giữ nguyên, vẫn chạy bằng label_module_tmp.py.

VÌ SAO BẢN vLLM CŨ CHẬM (và bản này sửa gì)
-------------------------------------------
1. NLI mới là nút cổ chai, không phải generate.
   n_samples=5 -> C(5,2)=10 cặp × 2 chiều = 20 forward NLI / record. Bản
   transformers chậm ở generate nên không ai thấy; vLLM làm generate nhanh gấp
   chục lần, NLI lộ ra thành phần chiếm ~80-90% thời gian. Sửa:
     - bỏ qua NLI khi 2 sample GIỐNG HỆT chuỗi (rất hay gặp ở QA ngắn,
       temperature 0.7) -> entail = 1.0 hiển nhiên, khỏi chạy model;
     - khử trùng lặp cặp trong cùng 1 lô;
     - SẮP XẾP cặp theo độ dài trước khi batch. `padding=True` pad theo phần tử
       DÀI NHẤT của lô, trộn câu 10 token với câu 400 token nghĩa là trả tiền
       cho 400 token trên mọi phần tử. Riêng cái này thường 2-3x.
     - batch NLI TỰ TĂNG (bản cũ chỉ biết giảm khi OOM, kẹt ở 16 mãi mãi).
2. load_nli() gọi SAU LLM(). vLLM chiếm sẵn `gpu_memory_utilization` × VRAM ngay
   lúc khởi tạo, NLI nạp sau phải chen vào phần thừa -> hoặc OOM (rồi tụt batch
   xuống 1) hoặc phân mảnh. Bản này nạp NLI TRƯỚC, rồi tính util theo VRAM còn lại.
3. max_model_len=4096 cứng. KV cache = (max_model_len × số seq đồng thời); đặt dư
   thì vLLM giảm số sequence chạy song song -> throughput thấp. Bản này ĐO độ dài
   prompt thật rồi đặt max_model_len vừa đủ.
4. max_new_tokens=256 cứng, trong khi response thật thường ngắn hơn nhiều. Bản
   này lấy p95 độ dài response_text đã sinh ở bước trước làm ngưỡng.
5. generate và NLI chạy nối tiếp. Bản này chồng lấn: NLI của nhóm k chạy trên
   thread nền trong khi vLLM generate nhóm k+1.
6. apply_chat_template(tokenize=False) rồi để vLLM tự tokenize -> BOS bị chèn 2
   lần (bản transformers có add_special_tokens=False, bản vLLM cũ quên). Bản này
   truyền thẳng token ids nên không lệch prompt giữa 2 đường chạy.
7. Bật prefix caching + xếp record theo context: nhiều record dùng CHUNG context
   (HaluEval QA) -> prefill được tái dùng.

ĐIỂM QUAN TRỌNG: định nghĩa nhãn KHÔNG đổi. consistency_scores() và
CONSISTENCY_THRESHOLD import y nguyên từ label_module_tmp.py, nên nhãn sinh ra
từ file này so sánh được với nhãn sinh từ bản transformers.

TỰ CHỈNH THAM SỐ
----------------
Mọi flag về hiệu năng mặc định = auto. Chạy trần:

    uv run label_module_self_consistency_vllm.py --model llama3.1-8b \
        --responses data/processed/responses/llama3.1-8b.jsonl \
        --unified data/processed/unified_prompts.jsonl --n_samples 10

là đã ở điểm tối ưu cho GPU đang thuê + dữ liệu đang chạy: không cần chạy thử
rồi chạy lại với tham số khác. Muốn ghim tay thì vẫn truyền được, giá trị truyền
tay luôn thắng auto. Kế hoạch đã chốt được log ra đầu run (mục "plan").

Sau đó gộp nhãn như cũ:
    uv run label_module_tmp.py --stage combine --model llama3.1-8b
"""

import argparse
import gc
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams

from config import settings, tracking
from config.logs import get_logger
from generate_responses_tmp import MODEL_REGISTRY, build_chat_prompt
from label_module_tmp import (  # tái dùng nguyên xi, KHÔNG viết lại logic nhãn
    CONSISTENCY_THRESHOLD,
    LABEL_DIR,
    consistency_scores,
    load_done_ids,
    load_jsonl,
    load_nli,
    load_responses,
)

log = get_logger("label_module_vllm")

# entail=1, neutral=0, contra=0 — dùng cho cặp 2 sample giống hệt nhau.
IDENTICAL_PROBS = [1.0, 0.0, 0.0]

NLI_MAX_LENGTH = 512  # giới hạn cứng của DeBERTa-v3-base
NLI_BATCH_START = 32  # điểm xuất phát, tự tăng gấp đôi nếu không OOM
NLI_BATCH_MAX = 512
NLI_GROW_AFTER = 6  # số lô chạy trót lọt trước khi thử tăng batch

# Chừa VRAM ngoài pool của vLLM cho weight + activation của NLI.
NLI_RESERVE_GB = 2.5


def free_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def encode_chat_prompt(tokenizer, record):
    """Chat template -> list[int] token ids.

    KHÔNG dùng apply_chat_template(tokenize=True): tuỳ phiên bản transformers nó
    trả về list[int], BatchEncoding, hoặc dict {'input_ids': ..., 'attention_mask':
    ...} — đưa thẳng dict cho vLLM thì nó vỡ ở _validate_model_input ('<' not
    supported between 'str' and 'int'), và len(dict) = 2 làm hỏng luôn phép đo
    độ dài prompt. Render ra text rồi tự encode với add_special_tokens=False
    (template đã chèn special token rồi, thêm nữa là lặp BOS).
    """
    text = tokenizer.apply_chat_template(
        build_chat_prompt(record), add_generation_prompt=True, tokenize=False
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(ids, list) or (ids and not isinstance(ids[0], int)):
        raise TypeError(f"token ids phải là list[int], nhận được {type(ids)}")
    return ids


def gpu_mem_gb():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    free, total = torch.cuda.mem_get_info()
    return free / 1024**3, total / 1024**3


# ---------------------------------------------------------------- NLI nhanh


def dedupe_pairs(pairs, run_unique):
    """pairs -> list probs cùng thứ tự, nhưng chỉ chạy model trên cặp THỰC SỰ cần.

    Bỏ qua 2 trường hợp: (a) premise == hypothesis sau strip -> entail hiển nhiên;
    (b) cặp trùng cặp đã có trong lô. run_unique(list[(p, h)]) -> list probs|None.
    Trả thêm thống kê để biết mình tiết kiệm được bao nhiêu.
    """
    out = [None] * len(pairs)
    uniq = {}
    n_identical = 0
    for i, (premise, hypothesis) in enumerate(pairs):
        premise, hypothesis = premise.strip(), hypothesis.strip()
        if premise == hypothesis:
            out[i] = list(IDENTICAL_PROBS)
            n_identical += 1
            continue
        uniq.setdefault((premise, hypothesis), []).append(i)

    # Cặp ngắn đi với cặp ngắn: padding=True pad theo phần tử dài nhất của lô.
    keys = sorted(uniq, key=lambda k: len(k[0]) + len(k[1]))
    for key, prob in zip(keys, run_unique(keys)):
        for i in uniq[key]:
            out[i] = prob
    return out, {"identical": n_identical, "unique": len(keys), "total": len(pairs)}


class NLIRunner:
    """nli_fn cho consistency_scores(): khử trùng lặp + sort theo độ dài + batch
    tự co giãn. Giảm khi OOM (như map_batched cũ) và TĂNG khi chạy trót lọt."""

    def __init__(self, tok, model, batch_size=None):
        self.tok = tok
        self.model = model
        self.batch = batch_size or NLI_BATCH_START
        self.can_grow = batch_size is None  # ghim tay thì tôn trọng, không tự tăng
        self.streak = 0
        self.sec = 0.0
        self.n_forward = 0
        self.n_pairs = 0
        self.n_skipped = 0

    @torch.inference_mode()
    def _forward(self, sub):
        premises, hypotheses = zip(*sub)
        inputs = self.tok(
            list(premises),
            list(hypotheses),
            padding=True,
            truncation=True,
            max_length=NLI_MAX_LENGTH,
            return_tensors="pt",
        ).to(self.model.device)
        logits = self.model(**inputs).logits.float()
        return torch.softmax(logits, dim=-1).cpu().tolist()

    def _run_unique(self, keys):
        results = []
        i = 0
        while i < len(keys):
            size = max(1, self.batch)
            sub = keys[i : i + size]
            try:
                t0 = time.perf_counter()
                results.extend(self._forward(sub))
                self.sec += time.perf_counter() - t0
                self.n_forward += 1
                i += len(sub)
                self.streak += 1
                if (
                    self.can_grow
                    and self.streak >= NLI_GROW_AFTER
                    and self.batch < NLI_BATCH_MAX
                ):
                    self.batch = min(NLI_BATCH_MAX, self.batch * 2)
                    self.streak = 0
                    log.debug(f"nli: tăng batch -> {self.batch}")
            except torch.OutOfMemoryError:
                free_cuda()
                self.can_grow = False  # chạm trần rồi thì thôi, đừng thử lại
                self.streak = 0
                if size == 1:
                    log.warning("nli: OOM ở lô 1 cặp, bỏ cặp này")
                    results.append(None)
                    i += 1
                else:
                    self.batch = size // 2
                    log.warning(f"nli: OOM, giảm batch {size} -> {self.batch}")
        return results

    def __call__(self, pairs):
        if not pairs:
            return []
        probs, stats = dedupe_pairs(pairs, self._run_unique)
        self.n_pairs += stats["total"]
        self.n_skipped += stats["total"] - stats["unique"]
        return probs


# ------------------------------------------------------- tự chốt tham số


def percentile(values, q):
    if not values:
        return 0
    ordered = sorted(values)
    k = min(len(ordered) - 1, int(math.ceil(q / 100 * len(ordered))) - 1)
    return ordered[max(0, k)]


def auto_max_new_tokens(tokenizer, responses, sample_cap=2000):
    """Sample self-consistency chỉ cần dài bằng response thật ở bước trước.
    Lấy p95 độ dài response_text đã sinh, cộng biên 20%."""
    texts = [r.get("response_text", "") for r in responses[:sample_cap] if r.get("response_text")]
    if not texts:
        return 256
    lens = [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts]
    return int(min(512, max(96, percentile(lens, 95) * 1.2)))


def auto_max_model_len(prompt_lens, max_new_tokens, model_max):
    """Vừa đủ prompt dài nhất + phần sinh thêm, làm tròn lên bội 256.
    Đặt dư = KV cache phình = ít sequence song song = chậm."""
    need = max(prompt_lens) + max_new_tokens + 16
    need = int(math.ceil(need / 256) * 256)
    return max(1024, min(model_max, need))


def auto_gpu_memory_utilization():
    """Chừa NLI_RESERVE_GB cho NLI, phần còn trống lấy 95% cho vLLM."""
    free, total = gpu_mem_gb()
    if total <= 0:
        return 0.85
    used = total - free
    budget = used + max(0.0, free - NLI_RESERVE_GB) * 0.95
    return round(min(0.92, max(0.50, budget / total)), 3)


def auto_group_size(n_items):
    """Nhóm càng to càng ít overhead, nhưng cần đủ nhiều nhóm để chồng lấn
    generate/NLI và để flush file thường xuyên."""
    return int(min(512, max(64, n_items // 8)))


def model_max_len(repo_id):
    try:
        cfg = AutoConfig.from_pretrained(repo_id)
        return int(getattr(cfg, "max_position_embeddings", 4096) or 4096)
    except Exception:
        return 4096


def vllm_quantization(cfg):
    """load_in_8bit/4bit của transformers -> quantization của vLLM."""
    kwargs = cfg.get("load_kwargs", {}) or {}
    if kwargs.get("load_in_8bit") or kwargs.get("load_in_4bit"):
        return "bitsandbytes"
    return None


# ------------------------------------------------------------- stage chính


def run_self_consistency_stage_vllm(
    model_key,
    responses_path,
    unified_path,
    n_samples=5,
    limit=None,
    resume=True,
    nli_batch_size=None,
    nli_device=None,
    fp32=False,
    max_new_tokens=None,
    gpu_memory_utilization=None,
    max_model_len=None,
    group_size=None,
    overlap=True,
    sampling_seed=None,
):
    responses = load_responses(responses_path, limit)
    unified_by_id = {r["id"]: r for r in load_jsonl(unified_path)}

    out_path = LABEL_DIR / f"self_consistency_{model_key}.jsonl"
    done_ids = load_done_ids(out_path) if resume else set()
    todo = [r for r in responses if r["id"] not in done_ids]
    slog = log.bind(stage="self_consistency_vllm")
    slog.info(f"self_consistency(vllm): {len(done_ids)} đã xong, {len(todo)} cần chạy")
    if not todo:
        return {"ok": 0, "failed": 0, "sec": 0.0}

    cfg = MODEL_REGISTRY[model_key]
    tokenizer = AutoTokenizer.from_pretrained(cfg["repo_id"])

    items, failed = [], 0
    for resp in todo:
        record = unified_by_id.get(resp["id"])
        if record is None:
            slog.warning(f"self_consistency(vllm): thiếu prompt cho id={resp['id']}")
            failed += 1
        else:
            items.append((resp, record))

    # Record dùng chung context đứng cạnh nhau -> prefix cache của vLLM tái dùng
    # được phần prefill; trong cùng context thì prompt ngắn đi trước.
    items.sort(key=lambda it: (hash(it[1]["context"]), len(it[1]["prompt"])))

    # Tokenize 1 lần: vừa để đo độ dài (chốt max_model_len) vừa để truyền thẳng
    # token ids cho vLLM — tránh BOS bị chèn 2 lần như bản cũ.
    t_tok = time.perf_counter()
    prompt_ids = [encode_chat_prompt(tokenizer, rec) for _, rec in items]
    prompt_lens = [len(ids) for ids in prompt_ids]
    slog.info(
        f"tokenize {len(prompt_ids)} prompt trong {time.perf_counter() - t_tok:.1f}s "
        f"(median {percentile(prompt_lens, 50)}, p99 {percentile(prompt_lens, 99)}, "
        f"max {max(prompt_lens)} token)"
    )
    # Prompt có context + câu hỏi thì không thể chỉ vài token. Con số vô lý ở đây
    # nghĩa là encode sai, và max_model_len auto sẽ sai theo -> dừng sớm còn hơn
    # chạy hết rồi mới biết.
    if max(prompt_lens) < 16:
        raise ValueError(
            f"prompt dài nhất chỉ {max(prompt_lens)} token — chat template hoặc "
            "tokenizer đang trả về sai định dạng, kiểm tra encode_chat_prompt()"
        )

    # --- NẠP NLI TRƯỚC vLLM: vLLM profile VRAM lúc khởi tạo, phải thấy NLI đã ở đó.
    nli_tok, nli_model = load_nli(nli_device, fp32)
    nli = NLIRunner(nli_tok, nli_model, nli_batch_size)

    # --- chốt kế hoạch
    if max_new_tokens is None:
        max_new_tokens = auto_max_new_tokens(tokenizer, responses)
    hard_max = model_max_len(cfg["repo_id"])
    if max_model_len is None:
        max_model_len = auto_max_model_len(prompt_lens, max_new_tokens, hard_max)
    if gpu_memory_utilization is None:
        gpu_memory_utilization = auto_gpu_memory_utilization()
    if group_size is None:
        group_size = auto_group_size(len(items))

    # Prompt dài hơn cửa sổ thì cắt từ TRÁI (giữ câu hỏi ở cuối), hiếm nhưng
    # không được để vLLM ném lỗi giữa chừng.
    budget = max_model_len - max_new_tokens
    n_trunc = 0
    for i, ids in enumerate(prompt_ids):
        if len(ids) > budget:
            prompt_ids[i] = ids[-budget:]
            n_trunc += 1
    if n_trunc:
        slog.warning(f"{n_trunc} prompt dài hơn {budget} token, đã cắt từ trái")

    free_gb, total_gb = gpu_mem_gb()
    plan = {
        "max_new_tokens": max_new_tokens,
        "max_model_len": max_model_len,
        "model_max_len": hard_max,
        "gpu_memory_utilization": gpu_memory_utilization,
        "group_size": group_size,
        "nli_batch_start": nli.batch,
        "nli_auto_grow": nli.can_grow,
        "overlap": overlap,
        "n_samples": n_samples,
        "vram_free_gb": round(free_gb, 1),
        "vram_total_gb": round(total_gb, 1),
        "prompt_truncated": n_trunc,
    }
    slog.info(f"plan: {plan}")
    tracking.log_metrics({f"plan/{k}": v for k, v in plan.items() if isinstance(v, (int, float))})

    slog.info(f"loading {model_key} ({cfg['repo_id']}) qua vLLM")
    llm = LLM(
        model=cfg["repo_id"],
        dtype="bfloat16",
        quantization=vllm_quantization(cfg),
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_prefix_caching=True,  # nhiều record chung context -> tái dùng prefill
        disable_log_stats=True,
    )
    sampling_params = SamplingParams(
        n=n_samples,
        temperature=0.7,
        top_p=0.9,
        max_tokens=max_new_tokens,
        seed=sampling_seed,
    )

    t0 = time.perf_counter()
    gen_sec = 0.0
    bar = tqdm(total=len(todo), desc=f"{model_key}-sc-vllm", unit="rec", dynamic_ncols=True)
    bar.update(failed)

    def score_and_write(group, sample_groups, fout):
        """Chạy NLI + ghi file cho 1 nhóm. Chạy trên thread nền để chồng lấn với
        lần generate kế tiếp. Chỉ 1 worker nên thứ tự ghi vẫn tuần tự."""
        nonlocal failed
        scores = consistency_scores(sample_groups, nli)
        for (resp, _), samples, score in zip(group, sample_groups, scores):
            if not samples or score is None:
                failed += 1
                slog.warning(f"self_consistency(vllm): id={resp['id']} không NLI được, bỏ qua")
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
                            "consistent" if score >= CONSISTENCY_THRESHOLD else "inconsistent"
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        fout.flush()
        bar.update(len(group))

    pool = ThreadPoolExecutor(max_workers=1) if overlap else None
    pending = None
    with open(out_path, "a" if resume else "w", encoding="utf-8") as fout:
        for start in range(0, len(items), group_size):
            group = items[start : start + group_size]
            prompts = [{"prompt_token_ids": ids} for ids in prompt_ids[start : start + group_size]]

            try:
                t_gen = time.perf_counter()
                outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
                gen_sec += time.perf_counter() - t_gen
            except Exception:
                slog.exception(
                    f"self_consistency(vllm): lỗi generate ở nhóm từ id={group[0][0]['id']}"
                )
                failed += len(group)
                bar.update(len(group))
                continue

            sample_groups = [[o.text for o in out.outputs] for out in outputs]

            if pool is None:
                score_and_write(group, sample_groups, fout)
            else:
                # Chờ nhóm trước ghi xong RỒI mới giao nhóm này: giữ thứ tự, và
                # phần chờ đó đã được lần generate vừa rồi che lấp.
                if pending is not None:
                    pending.result()
                pending = pool.submit(score_and_write, group, sample_groups, fout)

            elapsed = time.perf_counter() - t0
            bar.set_postfix(nli=nli.batch, failed=failed)
            tracking.log_metrics(
                {
                    "self_consistency_vllm/records_done": bar.n,
                    "self_consistency_vllm/failed": failed,
                    "self_consistency_vllm/rec_per_s": bar.n / max(elapsed, 1e-9),
                    "self_consistency_vllm/nli_batch": nli.batch,
                    "self_consistency_vllm/gen_share": gen_sec / max(elapsed, 1e-9),
                    "self_consistency_vllm/nli_share": nli.sec / max(elapsed, 1e-9),
                }
            )

        if pending is not None:
            pending.result()
    if pool is not None:
        pool.shutdown()

    bar.close()
    del llm
    nli_model = None
    nli.model = None
    gc.collect()
    free_cuda()

    sec = time.perf_counter() - t0
    ok = len(todo) - failed
    slog.info(
        f"self_consistency(vllm) xong: {ok} ok, {failed} lỗi, {sec:.0f}s -> {out_path}"
    )
    # Đọc 3 số này để biết còn chỉnh được gì: gen_sec cao -> tăng group_size /
    # gpu_memory_utilization; nli_sec cao -> giảm n_samples hoặc max_new_tokens.
    slog.info(
        f"phân bổ thời gian: generate {gen_sec:.0f}s, nli {nli.sec:.0f}s "
        f"({nli.n_forward} forward, batch cuối {nli.batch}), "
        f"bỏ qua {nli.n_skipped}/{nli.n_pairs} cặp NLI nhờ khử trùng lặp"
    )
    return {
        "ok": ok,
        "failed": failed,
        "sec": round(sec, 1),
        "gen_sec": round(gen_sec, 1),
        "nli_sec": round(nli.sec, 1),
        "nli_pairs_skipped": nli.n_skipped,
        "nli_pairs_total": nli.n_pairs,
        "nli_batch_final": nli.batch,
        **{f"plan_{k}": v for k, v in plan.items()},
    }


# ------------------------------------------------------------- self-check


def self_check():
    """Check chạy được không cần GPU/model."""
    calls = []

    def fake_run(keys):
        calls.append(list(keys))
        return [[0.9, 0.05, 0.05]] * len(keys)

    # cặp giống hệt không gọi model; cặp trùng chỉ chạy 1 lần; thứ tự giữ nguyên
    pairs = [("a", "a"), ("x", "y"), ("x", "y"), ("b ", "b")]
    probs, stats = dedupe_pairs(pairs, fake_run)
    assert stats == {"identical": 3, "unique": 1, "total": 4}, stats
    assert calls == [[("x", "y")]], calls
    assert probs[0] == IDENTICAL_PROBS and probs[3] == IDENTICAL_PROBS
    assert probs[1] == probs[2] == [0.9, 0.05, 0.05]

    # sort theo tổng độ dài
    calls.clear()
    dedupe_pairs([("xxxx", "yyyy"), ("a", "b")], fake_run)
    assert calls[0] == [("a", "b"), ("xxxx", "yyyy")], calls

    # sample giống hệt nhau -> consistency = 1.0, không đụng tới model
    calls.clear()
    got = consistency_scores([["same", "same", "same"]], lambda p: dedupe_pairs(p, fake_run)[0])
    assert got == [1.0], got
    assert calls == [], calls

    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5
    assert auto_max_model_len([600], 128, 8192) == 768
    assert auto_max_model_len([600], 128, 512) == 1024  # không xuống dưới sàn
    assert auto_max_model_len([9000], 256, 4096) == 4096  # cắt theo cửa sổ model
    assert auto_group_size(100) == 64 and auto_group_size(100000) == 512

    print("self-check OK")


@logger.catch(default=1)
def main():
    parser = argparse.ArgumentParser(
        description="Stage self_consistency bằng vLLM. Mọi flag hiệu năng mặc "
        "định là auto — chạy trần là đã tối ưu, truyền tay chỉ khi muốn ghim."
    )
    parser.add_argument("--model", help="model_key trong MODEL_REGISTRY")
    parser.add_argument("--responses", help="Path tới output của generate_responses_tmp.py")
    parser.add_argument("--unified", help="Path tới unified_prompts.jsonl")
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--nli-device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--fp32", action="store_true", help="Tắt bf16 cho NLI")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=None,
        help="Seed riêng cho vLLM sampling. Bỏ trống = nhanh hơn (vLLM gộp "
        "request tự do); đặt vào = tái lập được đúng sample.",
    )
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--self-check", action="store_true")

    auto = parser.add_argument_group(
        "auto", "để trống = tự đo và tự chọn; truyền vào = ghim, auto không đụng tới"
    )
    auto.add_argument("--nli-batch-size", type=int, default=None,
                      help="auto: bắt đầu 32 rồi tự tăng tới khi OOM")
    auto.add_argument("--max-new-tokens", type=int, default=None,
                      help="auto: p95 độ dài response đã sinh + 20%%")
    auto.add_argument("--gpu-memory-utilization", type=float, default=None,
                      help="auto: theo VRAM còn trống, chừa ~2.5GB cho NLI")
    auto.add_argument("--max-model-len", type=int, default=None,
                      help="auto: prompt dài nhất + max_new_tokens")
    auto.add_argument("--group-size", type=int, default=None,
                      help="auto: số record mỗi lần llm.generate()")
    auto.add_argument("--no-overlap", action="store_true",
                      help="tắt chồng lấn NLI với generate (chỉ để debug)")
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return 0
    if not args.model or not args.responses or not args.unified:
        parser.error("--model, --responses và --unified là bắt buộc")

    tracking.set_seed(args.seed)
    info = tracking.runtime_info(args.seed)
    log.info(f"runtime: {info}")
    tracking.init_run("label_self_consistency_vllm", {**vars(args), **info}, tags=args.tag)

    stats = run_self_consistency_stage_vllm(
        args.model,
        Path(args.responses),
        Path(args.unified),
        n_samples=args.n_samples,
        limit=args.limit,
        resume=not args.no_resume,
        nli_batch_size=args.nli_batch_size,
        nli_device=args.nli_device,
        fp32=args.fp32,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        group_size=args.group_size,
        overlap=not args.no_overlap,
        sampling_seed=args.sampling_seed,
    )
    tracking.finish(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())

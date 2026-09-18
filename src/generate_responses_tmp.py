"""
Bản tối ưu của generate_responses.py: batch inference.

Sinh response cho unified_prompts.jsonl bằng các model open-source, trích
internal state (hidden states / attention entropy / logit confidence) làm
feature cho detector.

Vì sao batch: decode greedy 1 sequence bị chặn bởi memory bandwidth, mỗi bước
đọc toàn bộ weight từ VRAM trong khi SM gần như rảnh. Batch B chia chi phí đọc
weight cho B sequence nên throughput tăng gần tuyến tính.

File output: sinh theo thứ tự độ dài prompt (đỡ pad) nhưng cuối mỗi run được dọn
lại theo thứ tự input, mỗi id đúng 1 dòng (bản mới nhất). --no-resume ghi đè file.

Chạy:
    uv run src/generate_responses_tmp.py --model qwen2.5-0.5b --input data/processed/unified_prompts.jsonl --batch-size 16
    uv run src/generate_responses_tmp.py --model all --input data/processed/unified_prompts.jsonl --batch-size 16
    uv run src/generate_responses_tmp.py --self-check   # kiểm tra nhanh phần toán, không cần GPU
    uv run src/generate_responses_tmp.py --model qwen2.5-0.5b --input data/processed/unified_prompts.jsonl --limit 8 --no-resume --batch-size 8
    uv run src/generate_responses_tmp.py --model qwen2.5-0.5b --input ... --tag baseline --tag bs16   # gắn nhãn run trên wandb
"""

import argparse
import gc
import json
import sys
import tempfile
import time
from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import settings, tracking
from config.logs import get_logger
from utils.jsonl import compact_jsonl, head

log = get_logger("generate_responses")

MODEL_REGISTRY = {
    "llama3.1-8b": {
        "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
        "load_kwargs": {"dtype": torch.bfloat16},
    },
    "mistral-7b": {
        "repo_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "load_kwargs": {"dtype": torch.bfloat16},
    },
    "qwen2.5-7b": {
        "repo_id": "Qwen/Qwen2.5-7B-Instruct",
        "load_kwargs": {"dtype": torch.bfloat16},
    },
    "qwen3-8b": {
        "repo_id": "Qwen/Qwen3-8B",
        "load_kwargs": {"dtype": torch.bfloat16},
    },
    "gemma2-9b": {
        "repo_id": "google/gemma-2-9b-it",
        "load_kwargs": {"dtype": torch.bfloat16},
    },
    "llama2-13b": {
        "repo_id": "meta-llama/Llama-2-13b-chat-hf",
        "load_kwargs": {"load_in_8bit": True},
    },
    # --- DEMO ---
    "qwen2.5-0.5b": {
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "load_kwargs": {"dtype": torch.bfloat16},
    },
}

PROMPT_TEMPLATES = {
    "qa": "Context: {context}\n\nQuestion: {prompt}\nAnswer concisely based on the context.",
    "dialogue": "Relevant knowledge: {context}\n\nConversation so far:\n{prompt}\nRespond as the assistant.",
    "summarization": "Document:\n{context}\n\n{prompt}",
}


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_chat_prompt(record):
    """Ghép prompt theo domain. Model tự áp chat template riêng của nó."""
    try:
        template = PROMPT_TEMPLATES[record["domain"]]
    except KeyError:
        raise ValueError(f"Unknown domain: {record['domain']}") from None
    user_msg = template.format(context=record["context"], prompt=record["prompt"])
    return [{"role": "user", "content": user_msg}]


def token_stats(logits):
    """logits [T, V] của response tokens -> (mean top-1 prob, mean entropy).
    Dùng log_softmax: ổn định số học hơn softmax + log, không cần clamp."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    probs = logp.exp()
    mean_confidence = probs.max(dim=-1).values.mean().item()
    mean_entropy = (-(probs * logp).sum(dim=-1)).mean().item()
    return mean_confidence, mean_entropy


def done_ids(out_path: Path) -> set:
    """Id đã sinh xong. Chỉ tính là done khi attn_entropy_per_layer có dữ liệu —
    tránh resume bỏ qua record cũ bị lỗi/thiếu. Kiểm tra trên bytes + head() để
    không parse hàng GB mảng float chỉ để lấy id."""
    if not out_path.exists():
        return set()
    with open(out_path, "rb") as f:
        return {
            head(line)["id"]
            for line in f
            if b'"attn_entropy_per_layer": [' in line
            and b'"attn_entropy_per_layer": []' not in line
        }


def cut_response(gen_ids, eos):
    """Phần model sinh ra còn pad ở đuôi sau EOS (do sequence khác trong batch
    còn chạy tiếp). Cắt ngay sau EOS đầu tiên, giữ lại EOS như khi chạy batch 1."""
    for i, token in enumerate(gen_ids.tolist()):
        if token in eos:
            return gen_ids[: i + 1]
    return gen_ids


def eos_id_set(model, tokenizer) -> set:
    """eos_token_id có thể là int hoặc list (llama3 có thêm <|eot_id|>)."""
    ids = model.generation_config.eos_token_id
    if ids is None:
        ids = tokenizer.eos_token_id
    if ids is None:
        return set()
    return {ids} if isinstance(ids, int) else set(ids)


def pad_batch(seqs, pad_id, device):
    """Pad PHẢI: mọi sequence bắt đầu ở vị trí 0 nên prompt_len của từng sequence
    dùng thẳng làm chỉ số tuyệt đối khi cắt hidden states / logits."""
    lens = [s.numel() for s in seqs]
    input_ids = torch.full(
        (len(seqs), max(lens)), pad_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros(
        (len(seqs), max(lens)), dtype=torch.long, device=device
    )
    for i, seq in enumerate(seqs):
        input_ids[i, : lens[i]] = seq.to(device)
        attention_mask[i, : lens[i]] = 1
    return input_ids, attention_mask, lens


def set_attn(model, impl, enabled=True):
    if enabled:
        model.set_attn_implementation(impl)


def probe_attn_switch(model) -> bool:
    """sdpa cho generate (nhanh, không dựng ma trận T²), eager cho pass phân tích
    (chỉ eager mới trả attentions). Model nào không đổi được thì chạy eager hết."""
    try:
        model.set_attn_implementation("sdpa")
        model.set_attn_implementation("eager")
        return True
    except Exception as e:
        log.warning(
            f"không đổi được attn implementation ({e!r}), dùng eager cho cả generate"
        )
        return False


@torch.inference_mode()
def generate_batch(model, tokenizer, records, max_new_tokens):
    """Trả list (prompt_ids, response_ids) đã bỏ pad hai đầu."""
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
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    prompt_width = encoded["input_ids"].shape[1]
    eos = eos_id_set(model, tokenizer)
    pairs = []
    for i in range(len(records)):
        keep = encoded["attention_mask"][i].bool()
        prompt_ids = encoded["input_ids"][i][keep]  # bỏ pad TRÁI
        response_ids = cut_response(generated[i, prompt_width:], eos)  # bỏ pad PHẢI
        pairs.append((prompt_ids, response_ids))
    return pairs


@torch.inference_mode()
def analyze_chunk(model, pad_id, pairs):
    """1 forward pass trên (prompt + response) để lấy internal state ổn định.
    use_cache=False: pass này không decode nên không cần KV cache."""
    input_ids, attention_mask, lens = pad_batch(
        [torch.cat([p, r]) for p, r in pairs], pad_id, model.device
    )
    outputs = model(
        input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        output_attentions=True,
        use_cache=False,
    )

    results = []
    for i, (prompt_ids, _) in enumerate(pairs):
        start, end = prompt_ids.numel(), lens[i]

        # hidden_states: tuple(num_layers+1) of [batch, seq_len, hidden_dim]
        # mean-pool riêng phần RESPONSE cho từng layer. Stack rồi chép về CPU 1 lần
        # thay vì 1 lần/layer (mỗi lần là 1 lần đồng bộ GPU).
        hidden = (
            torch.stack([hs[i, start:end, :].mean(dim=0) for hs in outputs.hidden_states])
            .float()
            .cpu()
            .tolist()
        )

        # logit tại t dự đoán token t+1 -> lùi 1 vị trí
        confidence, entropy = token_stats(outputs.logits[i, start - 1 : end - 1, :])

        # attention entropy trung bình qua layer/head cho phần response.
        # Cắt cột tới lens[i] để cột pad không lọt vào tổng.
        attn = []
        for layer_attn in outputs.attentions:  # [batch, num_heads, seq_len, seq_len]
            a = layer_attn[i, :, start:end, : lens[i]].float().clamp_min(1e-12)
            attn.append((-(a * a.log()).sum(dim=-1)).mean())
        attn = torch.stack(attn).tolist()

        results.append(
            {
                "hidden_states_per_layer": hidden,
                "mean_token_confidence": confidence,
                "mean_token_entropy": entropy,
                "attn_entropy_per_layer": attn,
            }
        )
    return results


def compact(out_path, records, slog):
    """Sinh theo thứ tự độ dài (đỡ pad) nhưng file cuối phải theo thứ tự input
    và mỗi id 1 dòng: người đọc file / ghép theo dòng không bị lệch."""
    if out_path.exists():
        dropped = compact_jsonl(out_path, [r["id"] for r in records])
        slog.info(f"dọn {out_path.name}: theo thứ tự input, bỏ {dropped} dòng trùng id")


def run_model(
    model_key,
    records,
    max_new_tokens=256,
    batch_size=8,
    analysis_batch_size=1,
    resume=True,
):
    cfg = MODEL_REGISTRY[model_key]
    out_path = settings.responses_dir / f"{model_key}.jsonl"
    slog = log.bind(stage=model_key)  # mọi dòng log dưới đây có nhãn model

    # Lọc trước khi load model: nếu xong hết thì khỏi tốn thời gian load weights.
    done = done_ids(out_path) if resume else set()
    todo = [r for r in records if r["id"] not in done]
    slog.info(f"{model_key}: {len(done)} record đã xong, {len(todo)} record cần chạy")
    if not todo:
        # Vẫn dọn file: sửa được file cũ bị trùng id / sai thứ tự mà không load model.
        compact(out_path, records, slog)
        return {"ok": 0, "failed": 0, "sec": 0.0}

    # Gom record dài với record dài: prompt summarization dài hơn QA nhiều lần,
    # trộn chung thì phần lớn batch là pad. Đếm ký tự đủ dùng, khỏi tokenize 2 lần.
    todo.sort(key=lambda r: len(r["context"]) + len(r["prompt"]))

    slog.info(f"loading {model_key} ({cfg['repo_id']})")
    tokenizer = AutoTokenizer.from_pretrained(cfg["repo_id"])
    tokenizer.padding_side = "left"  # bắt buộc cho decode theo batch
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # llama2 không có pad token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["repo_id"],
        device_map="auto",
        attn_implementation="eager",  # sdpa không trả attentions
        **cfg["load_kwargs"],
    )
    model.eval()
    switchable = probe_attn_switch(model)

    failed = 0
    n_tokens = 0
    t0 = time.perf_counter()
    bar = tqdm(total=len(todo), desc=model_key, unit="rec", dynamic_ncols=True)

    def log_progress():
        """Throughput + VRAM đỉnh lên wandb để so sánh giữa các lần chạy/cấu hình."""
        elapsed = time.perf_counter() - t0
        tracking.log_metrics(
            {
                f"{model_key}/records_done": bar.n,
                f"{model_key}/failed": failed,
                f"{model_key}/rec_per_s": bar.n / elapsed,
                f"{model_key}/tokens_per_s": n_tokens / elapsed,
                f"{model_key}/vram_peak_gb": tracking.vram_peak_gb(),
            }
        )
    # --no-resume phải GHI ĐÈ. Trước đây luôn "a" nên chạy lại cộng dồn bản trùng id.
    with open(out_path, "a" if resume else "w", encoding="utf-8") as fout:
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            try:
                set_attn(model, "sdpa", switchable)
                pairs = generate_batch(model, tokenizer, batch, max_new_tokens)
                set_attn(model, "eager", switchable)
            except torch.cuda.OutOfMemoryError:
                failed += len(batch)
                slog.exception(
                    f"OOM lúc generate, giảm --batch-size (hiện {batch_size})"
                )
                torch.cuda.empty_cache()
                bar.update(len(batch))
                log_progress()
                continue
            except Exception:
                failed += len(batch)
                slog.exception(f"lỗi generate ở batch bắt đầu từ {batch[0]['id']}")
                bar.update(len(batch))
                log_progress()
                continue

            for j in range(0, len(batch), analysis_batch_size):
                sub_records = batch[j : j + analysis_batch_size]
                sub_pairs = pairs[j : j + analysis_batch_size]
                n_sub = len(sub_pairs)  # đếm cho thanh tiến trình, không đổi khi lọc
                # Response rỗng thì mean() trả NaN và ghi thẳng vào jsonl -> loại trước.
                keep = [k for k, (_, r) in enumerate(sub_pairs) if r.numel() > 0]
                failed += n_sub - len(keep)
                for k in set(range(n_sub)) - set(keep):
                    slog.warning(
                        f"record {sub_records[k]['id']}: response rỗng, bỏ qua"
                    )
                if not keep:
                    bar.update(n_sub)
                    continue
                sub_records = [sub_records[k] for k in keep]
                sub_pairs = [sub_pairs[k] for k in keep]

                try:
                    rows = analyze_chunk(model, tokenizer.pad_token_id, sub_pairs)
                except torch.cuda.OutOfMemoryError:
                    failed += len(sub_pairs)
                    slog.exception("OOM lúc phân tích, giảm --analysis-batch-size")
                    torch.cuda.empty_cache()
                    bar.update(n_sub)
                    continue
                except Exception:
                    failed += len(sub_pairs)
                    slog.exception(f"lỗi phân tích ở record {sub_records[0]['id']}")
                    bar.update(n_sub)
                    continue

                n_tokens += sum(r.numel() for _, r in sub_pairs)
                for record, (_, response_ids), row in zip(sub_records, sub_pairs, rows):
                    fout.write(
                        json.dumps(
                            {
                                "id": record["id"],
                                "domain": record["domain"],
                                "model": model_key,
                                "response_text": tokenizer.decode(
                                    response_ids, skip_special_tokens=True
                                ),
                                **row,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                fout.flush()
                bar.update(n_sub)
                bar.set_postfix(failed=failed)

            log_progress()

    bar.close()
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    compact(out_path, records, slog)
    sec = time.perf_counter() - t0
    slog.info(
        f"xong {model_key}: {len(todo) - failed} ok, {failed} lỗi, {sec:.0f}s, "
        f"VRAM đỉnh {tracking.vram_peak_gb()} GB -> {out_path}"
    )
    return {"ok": len(todo) - failed, "failed": failed, "sec": round(sec, 1)}


def self_check():
    """Check nhỏ chạy được không cần GPU/model."""
    msgs = build_chat_prompt({"domain": "qa", "context": "C", "prompt": "P", "id": "x"})
    assert msgs[0]["role"] == "user" and "C" in msgs[0]["content"]
    try:
        build_chat_prompt({"domain": "nope", "context": "", "prompt": ""})
        raise AssertionError("domain lạ phải raise ValueError")
    except ValueError:
        pass

    # logits đều nhau trên vocab 4 -> conf = 0.25, entropy = ln(4)
    conf, ent = token_stats(torch.zeros(3, 4))
    assert abs(conf - 0.25) < 1e-6, conf
    assert abs(ent - torch.log(torch.tensor(4.0)).item()) < 1e-6, ent

    # logits nhọn -> conf ~ 1, entropy ~ 0
    sharp = torch.full((2, 4), -50.0)
    sharp[:, 0] = 50.0
    conf, ent = token_stats(sharp)
    assert conf > 0.999 and ent < 1e-4, (conf, ent)

    # cắt pad phải: giữ EOS đầu tiên, bỏ phần sau
    eos = {2, 5}
    assert cut_response(torch.tensor([7, 8, 2, 9, 9]), eos).tolist() == [7, 8, 2]
    assert cut_response(torch.tensor([7, 8, 9]), eos).tolist() == [7, 8, 9]
    assert cut_response(torch.tensor([5, 9]), eos).tolist() == [5]

    # pad phải: sequence bắt đầu ở vị trí 0, mask khớp độ dài thật
    ids, mask, lens = pad_batch(
        [torch.tensor([1, 2, 3]), torch.tensor([4, 5])], pad_id=0, device="cpu"
    )
    assert lens == [3, 2]
    assert ids.tolist() == [[1, 2, 3], [4, 5, 0]]
    assert mask.tolist() == [[1, 1, 1], [1, 1, 0]]
    # chỉ số cắt response dùng cho analyze_chunk: prompt_len=1 -> response = [2, 3]
    assert ids[0, 1 : lens[0]].tolist() == [2, 3]

    # head(): chuỗi giống key nặng nằm TRONG response_text (đã escape) không được cắt nhầm
    tricky = 'x", "hidden_states_per_layer": [1]'
    row = {"id": "a", "response_text": tricky, "hidden_states_per_layer": [[0.5]]}
    assert head(json.dumps(row).encode())["response_text"] == tricky

    # compact_jsonl: giữ bản cuối mỗi id, theo thứ tự input, id lạ xếp sau,
    # dòng cuối file thiếu \n vẫn tách dòng đúng; resume đọc được file sau khi dọn
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "r.jsonl"
        rows = [("b", 1, [1.0]), ("a", 1, [1.0]), ("b", 2, [1.0]), ("z", 1, []), ("c", 1, [1.0])]
        path.write_text(
            "\n".join(
                json.dumps({"id": i, "v": v, "hidden_states_per_layer": [], "attn_entropy_per_layer": a})
                for i, v, a in rows
            )
        )
        assert compact_jsonl(path, ["a", "b", "c"]) == 1
        got = [(r["id"], r["v"]) for r in map(json.loads, path.read_text().splitlines())]
        assert got == [("a", 1), ("b", 2), ("c", 1), ("z", 1)], got
        assert done_ids(path) == {"a", "b", "c"}  # z có attn rỗng -> chưa xong
    print("self-check OK")


# logger.catch: lỗi ở main vẫn vào file log kèm traceback dù thư viện khác có
# ghi đè sys.excepthook. default=1 để `sys.exit(main())` trả mã lỗi khác 0.
@logger.catch(default=1)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_REGISTRY) + ["all"])
    parser.add_argument("--input", help="Path to unified_prompts.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ chạy N record đầu")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Số record mỗi lần generate"
    )
    parser.add_argument(
        "--analysis-batch-size",
        type=int,
        default=1,
        # ponytail: attention memory = layers × heads × B × T² nên đây là chỗ OOM
        # trước tiên. Tăng dần khi VRAM còn dư, đừng đặt bằng --batch-size vô tội vạ.
        help="Số record mỗi forward pass phân tích (tốn VRAM theo T²)",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tag", action="append", default=[], help="Tag cho wandb run, lặp lại được"
    )
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return 0
    if not args.model or not args.input:
        parser.error("--model và --input là bắt buộc")

    tracking.set_seed(args.seed)
    info = tracking.runtime_info(args.seed)
    log.info(f"runtime: {info}")
    tracking.init_run("generate_responses", {**vars(args), **info}, tags=args.tag)

    records = load_jsonl(args.input)
    if args.limit:
        records = records[: args.limit]
    log.info(f"nạp {len(records)} record từ {args.input}")

    targets = list(MODEL_REGISTRY) if args.model == "all" else [args.model]
    summary = {}
    for model_key in targets:
        stats = run_model(
            model_key,
            records,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            analysis_batch_size=args.analysis_batch_size,
            resume=not args.no_resume,
        )
        summary.update({f"{model_key}/{k}": v for k, v in stats.items()})
    tracking.finish(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())

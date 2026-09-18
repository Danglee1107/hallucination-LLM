"""
Generate responses for all unified_prompts.jsonl from 5 model open-source,
extract internal state (hidden states / attentions / logit
confidence) and use as feature for detector.

REQUIREMENT:

    pip install transformers torch accelerate bitsandbytes --break-system-packages
    OR
    uv add transformers torch accelerate bitsandbytes --break-system-packages

    huggingface-cli login   # cần cho Llama-3.1, Llama-2, Gemma-2 (gated models)

how to run:
    uv run generate_responses.py --model llama3.1-8b --input ../data/processed/unified_prompts.jsonl
    uv run generate_responses.py --model all --input ../data/processed/unified_prompts.jsonl
"""

import json
import argparse
import gc
import traceback
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import settings

OUT_DIR = settings.responses_dir

# HuggingFace repo id cho từng model.
# Llama/Gemma là gated -> cần HF_TOKEN đã login trước.
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


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_chat_prompt(record):
    """Ghép prompt theo domain: QA dùng context+question, dialogue dùng lịch sử hội thoại,
    summarization dùng document. Model tự áp dụng chat template riêng của nó."""
    domain = record["domain"]
    context = record["context"]
    prompt = record["prompt"]

    if domain == "qa":
        user_msg = f"Context: {context}\n\nQuestion: {prompt}\nAnswer concisely based on the context."
    elif domain == "dialogue":
        user_msg = f"Relevant knowledge: {context}\n\nConversation so far:\n{prompt}\nRespond as the assistant."
    elif domain == "summarization":
        user_msg = f"Document:\n{context}\n\n{prompt}"
    else:
        raise ValueError(f"Unknown domain: {domain}")

    return [{"role": "user", "content": user_msg}]


@torch.no_grad()
def generate_with_internals(model, tokenizer, messages, max_new_tokens=256):
    """Generate response, sau đó chạy 1 forward pass riêng trên (input+response)
    để lấy hidden_states/attentions/logits ổn định (tránh phải giữ cache qua từng bước generate).
    """
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")

    gen_out = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    response_ids = gen_out[0][input_ids.shape[1] :]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    # Forward pass riêng lấy internal states cho full sequence (prompt + response)
    full_ids = gen_out
    outputs = model(
        full_ids,
        output_hidden_states=True,
        output_attentions=True,
    )

    # hidden_states: tuple(num_layers+1) of [batch, seq_len, hidden_dim]
    # Lấy mean-pooled hidden state của phần RESPONSE (không tính phần prompt) cho mỗi layer
    prompt_len = input_ids.shape[1]
    hidden_states_response = [
        layer_hs[0, prompt_len:, :].mean(dim=0).float().cpu().numpy().tolist()
        for layer_hs in outputs.hidden_states
    ]

    # logit-based confidence: mean top-1 probability của response tokens
    logits = outputs.logits[
        0, prompt_len - 1 : -1, :
    ]  # align: logit tại t dự đoán token t+1
    probs = torch.softmax(logits.float(), dim=-1)
    top1_probs = probs.max(dim=-1).values
    mean_confidence = top1_probs.mean().item()
    token_entropy = (-probs * torch.log(probs + 1e-12)).sum(dim=-1).mean().item()

    # attention: trung bình attention entropy qua các layer/head cho phần response
    # (giữ nhẹ ở đây; phân tích attention head-level chi tiết làm ở bước feature engineering riêng)
    attn_entropies = []
    for layer_attn in outputs.attentions:  # [batch, num_heads, seq_len, seq_len]
        a = layer_attn[0, :, prompt_len:, :].float()
        a = a.clamp_min(1e-12)
        ent = (-a * torch.log(a)).sum(dim=-1).mean().item()
        attn_entropies.append(ent)

    return {
        "response_text": response_text,
        "hidden_states_per_layer": hidden_states_response,  # list[num_layers+1][hidden_dim]
        "mean_token_confidence": mean_confidence,
        "mean_token_entropy": token_entropy,
        "attn_entropy_per_layer": attn_entropies,
    }


def run_model(model_key, records, resume=True):
    cfg = MODEL_REGISTRY[model_key]
    print(f"\n=== Loading {model_key} ({cfg['repo_id']}) ===")

    tokenizer = AutoTokenizer.from_pretrained(cfg["repo_id"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["repo_id"],
        device_map="auto",
        attn_implementation="eager",
        **cfg["load_kwargs"],
    )
    model.eval()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{model_key}.jsonl"

    done_ids = set()
    if resume and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                # Chỉ tính là "done" nếu attn_entropy_per_layer thực sự có dữ liệu —
                # tránh resume bỏ qua record cũ bị lỗi/thiếu do code từng có bug (ví dụ sdpa không trả attentions).
                if row.get("attn_entropy_per_layer"):
                    done_ids.add(row["id"])
        print(f"Resuming: {len(done_ids)} records already done.")

    with open(out_path, "a", encoding="utf-8") as fout:
        for i, record in enumerate(records):
            if record["id"] in done_ids:
                continue
            try:
                messages = build_chat_prompt(record)
                result = generate_with_internals(model, tokenizer, messages)
                row = {
                    "id": record["id"],
                    "domain": record["domain"],
                    "model": model_key,
                    **result,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
            except Exception as e:
                print(f"[WARN] Failed on {record['id']}: {e!r}")
                traceback.print_exc()

            if (i + 1) % 50 == 0:
                print(f"  {model_key}: {i + 1}/{len(records)} done")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"=== Finished {model_key}, saved to {out_path} ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, choices=list(MODEL_REGISTRY.keys()) + ["all"]
    )
    parser.add_argument("--input", required=True, help="Path to unified_prompts.jsonl")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: only process first N records (quick test)",
    )
    args = parser.parse_args()

    records = load_jsonl(args.input)
    if args.limit:
        records = records[: args.limit]

    targets = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    for model_key in targets:
        run_model(model_key, records)


if __name__ == "__main__":
    main()

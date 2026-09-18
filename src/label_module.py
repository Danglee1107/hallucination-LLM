"""
label hallucination to response were generated (from generate_responses.py) by
2 methods combine: Entailment (NLI fit with context) + Self-consistency
(multiple sample, measure consistent meaning).

REQUIREMENT:
    pip install transformers torch sentencepiece --break-system-packages

HOW TO RUN (3 stage, run sequencially):

    # Stage 1 — Entailment (light, run in CPU or GPU, no need to load LLM)
    python label_module.py --stage entailment --model qwen2.5-0.5b \
        --responses ../data/processed/responses/qwen2.5-0.5b.jsonl \
        --unified ../data/processed/unified_prompts.jsonl

    # Stage 2 — Self-consistency (HEAVY — need to load that LLM to sample N lần/prompt)
    python label_module.py --stage self_consistency --model qwen2.5-0.5b \
        --responses ../data/processed/responses/qwen2.5-0.5b.jsonl \
        --unified ../data/processed/unified_prompts.jsonl \
        --n_samples 5

    # Stage 3 — combine 2 results into 1 label
    python label_module.py --stage combine --model qwen2.5-0.5b

Lưu ý chi phí: self_consistency phải generate thêm N lần cho MỖI prompt/model —
tốn compute hơn hẳn generate_responses.py gốc. Nên test kỹ trên subset nhỏ
(--limit) trước khi chạy full, và đây là bước có khả năng cần thêm GPU thứ 2
(RTX 4090) nếu chạy quá chậm trên 1 RTX 5090.
"""

import json
import argparse
import gc
import traceback
from pathlib import Path
from itertools import combinations

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from config import settings
from generate_responses import MODEL_REGISTRY, build_chat_prompt

LABEL_DIR = settings.label_dir

# --- NLI config for model use Entailment (2 stage use this model) ---
NLI_MODEL_ID = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
NLI_MAX_TOKENS = 400          # chunk context theo cửa sổ này (NLI model giới hạn ~512 token)
NLI_CHUNK_STRIDE = 350        # overlap giữa các chunk để không cắt đứt câu quan trọng

# --- Ngưỡng quyết định nhãn ---
ENTAILMENT_SUPPORT_THRESHOLD = 0.5     # entailment_prob >= ngưỡng này -> coi là "được context ủng hộ"
ENTAILMENT_CONTRADICT_THRESHOLD = 0.3  # contradiction_prob >= ngưỡng này -> coi là "mâu thuẫn với context"
CONSISTENCY_THRESHOLD = 0.6            # tỉ lệ cặp sample nhất quán >= ngưỡng này -> coi là "nhất quán"
# FIX: threshold riêng cho self-consistency (tách khỏi ENTAILMENT_SUPPORT_THRESHOLD ở trên
# để có thể tinh chỉnh độc lập — 2 câu hỏi khác nhau: "context có ủng hộ response" vs
# "2 sample có nhất quán với nhau").
SELF_CONSISTENCY_PAIR_ENTAIL_THRESHOLD = 0.5


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done_ids(path):
    if not path.exists():
        return set()
    return {json.loads(l)["id"] for l in open(path, encoding="utf-8") if l.strip()}


def chunk_text(text, tokenizer, max_tokens=NLI_MAX_TOKENS, stride=NLI_CHUNK_STRIDE):
    """Chia context dài thành các chunk theo token, có overlap, để không vượt giới hạn NLI model."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return [text]
    chunks = []
    start = 0
    while start < len(ids):
        chunk_ids = ids[start: start + max_tokens]
        chunks.append(tokenizer.decode(chunk_ids))
        if start + max_tokens >= len(ids):
            break
        start += stride
    return chunks


# STAGE 1: ENTAILMENT
def load_nli():
    tok = AutoTokenizer.from_pretrained(NLI_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_ID)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    # Thứ tự label của model này: 0=entailment, 1=neutral, 2=contradiction (theo model card)
    # FIX: verify runtime thay vì chỉ tin vào comment trên — thứ tự label khác nhau giữa
    # các checkpoint NLI (ví dụ bart-large-mnli đảo ngược thứ tự), nên assert lại cho chắc.
    print(f"[NLI] id2label = {model.config.id2label}")
    assert model.config.id2label[0].lower() == "entailment", \
        "Thứ tự label không như giả định — code đang giả định index 0 = entailment!"
    assert model.config.id2label[2].lower() == "contradiction", \
        "Thứ tự label không như giả định — code đang giả định index 2 = contradiction!"
    return tok, model


@torch.no_grad()
def entailment_scores(premise_chunks, hypothesis, nli_tok, nli_model):
    """So khớp hypothesis (response) với từng chunk của premise (context),
    trả về (max_entailment_prob, max_contradiction_prob) qua toàn bộ chunk —
    lấy MAX vì chỉ cần 1 chunk ủng hộ là đủ coi response được grounding."""
    # FIX: không lấy max entailment và max contradiction độc lập từ 2 chunk khác nhau nữa
    # (bug cũ: có thể ghép nhầm entail cao của chunk A với contra cao của chunk B không liên quan).
    # Giờ chọn 1 chunk duy nhất có net support (entail - contra) cao nhất, lấy CẢ HAI giá trị
    # từ ĐÚNG chunk đó để đảm bảo nhất quán — quan trọng với context dài (domain summarization).
    best_entail, best_contra, best_score = 0.0, 0.0, float("-inf")
    device = nli_model.device
    for chunk in premise_chunks:
        inputs = nli_tok(chunk, hypothesis, truncation=True, max_length=512, return_tensors="pt").to(device)
        logits = nli_model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
        entail_p, _, contra_p = probs
        score = entail_p - contra_p
        if score > best_score:
            best_score = score
            best_entail, best_contra = entail_p, contra_p
    return best_entail, best_contra


def run_entailment_stage(model_key, responses_path, unified_path, limit=None, resume=True):
    responses = load_jsonl(responses_path)
    if limit:
        responses = responses[:limit]
    unified_by_id = {r["id"]: r for r in load_jsonl(unified_path)}

    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LABEL_DIR / f"entailment_{model_key}.jsonl"
    done_ids = load_done_ids(out_path) if resume else set()
    print(f"[entailment] {len(done_ids)} record was processed, resume right here.")

    nli_tok, nli_model = load_nli()

    for i, resp in enumerate(responses):
        rid = resp["id"]
        if rid in done_ids:
            continue
        record = unified_by_id.get(rid)
        if record is None:
            print(f"[entailment][WARN] cannot see the context for id={rid}, skip.")
            continue
        try:
            chunks = chunk_text(record["context"], nli_tok)
            entail_p, contra_p = entailment_scores(chunks, resp["response_text"], nli_tok, nli_model)
            supported = entail_p >= ENTAILMENT_SUPPORT_THRESHOLD and contra_p < ENTAILMENT_CONTRADICT_THRESHOLD
            append_jsonl(out_path, {
                "id": rid,
                "domain": resp["domain"],
                "model": model_key,
                "entailment_prob": entail_p,
                "contradiction_prob": contra_p,
                "entailment_label": "supported" if supported else "hallucinated",
            })
        except Exception as e:
            print(f"[entailment][WARN] ERROR at id={rid}: {e!r}")
            traceback.print_exc()

        if (i + 1) % 50 == 0:
            print(f"[entailment] {i + 1}/{len(responses)} done")

    print(f"[entailment] done, save at {out_path}")


# STAGE 2: SELF-CONSISTENCY
@torch.no_grad()
def sample_n_responses(model, tokenizer, messages, n_samples=5, max_new_tokens=256,
                        temperature=0.7, top_p=0.9):
    """Sample N response khác nhau cho cùng 1 prompt (do_sample=True) —
    dùng để đo độ nhất quán, KHÔNG phải response chính thức (đã có ở generate_responses.py)."""
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")

    outputs_text = []
    for _ in range(n_samples):
        gen_out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(gen_out[0][input_ids.shape[1]:], skip_special_tokens=True)
        outputs_text.append(text)
    return outputs_text


@torch.no_grad()
def mutual_entailment(text_a, text_b, nli_tok, nli_model):
    """Coi 2 sample là 'nhất quán' nếu chúng entail LẪN NHAU (cả 2 chiều) —
    cách này tương tự ý tưởng gom cụm theo semantic entailment trong semantic entropy."""
    device = nli_model.device

    def entail_prob(premise, hypothesis):
        inputs = nli_tok(premise, hypothesis, truncation=True, max_length=512, return_tensors="pt").to(device)
        probs = torch.softmax(nli_model(**inputs).logits[0], dim=-1).cpu().tolist()
        return probs[0]  # entailment prob

    p_ab = entail_prob(text_a, text_b)
    p_ba = entail_prob(text_b, text_a)
    return min(p_ab, p_ba)  # cả 2 chiều đều phải cao mới coi là nhất quán thật


def consistency_score(samples, nli_tok, nli_model):
    """Tính tỉ lệ cặp sample 'nhất quán' (mutual entailment cao) trên tổng số cặp."""
    if len(samples) < 2:
        return 1.0
    pair_scores = [
        mutual_entailment(a, b, nli_tok, nli_model)
        for a, b in combinations(samples, 2)
    ]
    # FIX: dùng threshold riêng SELF_CONSISTENCY_PAIR_ENTAIL_THRESHOLD thay vì
    # ENTAILMENT_SUPPORT_THRESHOLD (trước đây dùng chung 1 threshold cho 2 mục đích khác nhau).
    consistent_pairs = sum(1 for s in pair_scores if s >= SELF_CONSISTENCY_PAIR_ENTAIL_THRESHOLD)
    return consistent_pairs / len(pair_scores)


def run_self_consistency_stage(model_key, responses_path, unified_path, n_samples=5,
                                limit=None, resume=True):
    responses = load_jsonl(responses_path)
    if limit:
        responses = responses[:limit]
    unified_by_id = {r["id"]: r for r in load_jsonl(unified_path)}

    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LABEL_DIR / f"self_consistency_{model_key}.jsonl"
    done_ids = load_done_ids(out_path) if resume else set()
    print(f"[self_consistency] {len(done_ids)} record was processed, resume right here.")

    cfg = MODEL_REGISTRY[model_key]
    print(f"[self_consistency] Loading generation model {cfg['repo_id']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["repo_id"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["repo_id"], device_map="auto", attn_implementation="eager", **cfg["load_kwargs"]
    )
    model.eval()

    nli_tok, nli_model = load_nli()

    for i, resp in enumerate(responses):
        rid = resp["id"]
        if rid in done_ids:
            continue
        record = unified_by_id.get(rid)
        if record is None:
            print(f"[self_consistency][WARN] cannot see prompt for id={rid}, skip.")
            continue
        try:
            messages = build_chat_prompt(record)
            samples = sample_n_responses(model, tokenizer, messages, n_samples=n_samples)
            score = consistency_score(samples, nli_tok, nli_model)
            append_jsonl(out_path, {
                "id": rid,
                "domain": resp["domain"],
                "model": model_key,
                "n_samples": n_samples,
                "consistency_score": score,
                "consistency_label": "consistent" if score >= CONSISTENCY_THRESHOLD else "inconsistent",
            })
        except Exception as e:
            print(f"[self_consistency][WARN] ERROR at id={rid}: {e!r}")
            traceback.print_exc()

        if (i + 1) % 20 == 0:
            print(f"[self_consistency] {i + 1}/{len(responses)} done")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[self_consistency] done, save at {out_path}")


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
    common_ids = set(ent_by_id) & set(sc_by_id)
    print(f"[combine] {len(common_ids)} id has 2 labels (entailment ∩ self_consistency).")

    rows = []
    n_agree_halluc, n_agree_ok, n_disagree = 0, 0, 0
    for rid in common_ids:
        ent = ent_by_id[rid]
        sc = sc_by_id[rid]
        ent_bad = ent["entailment_label"] == "hallucinated"
        sc_bad = sc["consistency_label"] == "inconsistent"

        if ent_bad and sc_bad:
            final_label, confidence = "hallucination", "high"
            n_agree_halluc += 1
        elif not ent_bad and not sc_bad:
            final_label, confidence = "not_hallucination", "high"
            n_agree_ok += 1
        else:
            final_label, confidence = "disagreement", "low"  # cần review tay trên subset nhỏ
            n_disagree += 1

        rows.append({
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
        })

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(rows) or 1
    print(f"[combine] hallucination: {n_agree_halluc} ({n_agree_halluc/total:.1%})")
    print(f"[combine] not_hallucination: {n_agree_ok} ({n_agree_ok/total:.1%})")
    print(f"[combine] disagreement: {n_disagree} ({n_disagree/total:.1%})")
    print(f"[combine] done, save at {out_path}")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["entailment", "self_consistency", "combine"])
    parser.add_argument("--model", required=True, help="model_key trong MODEL_REGISTRY (generate_responses.py)")
    parser.add_argument("--responses", help="Path tới file output của generate_responses.py")
    parser.add_argument("--unified", help="Path tới unified_prompts.jsonl (để lấy context/prompt)")
    parser.add_argument("--n_samples", type=int, default=5, help="Số lần sample cho self_consistency")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.stage == "entailment":
        run_entailment_stage(args.model, Path(args.responses), Path(args.unified), limit=args.limit)
    elif args.stage == "self_consistency":
        run_self_consistency_stage(args.model, Path(args.responses), Path(args.unified),
                                    n_samples=args.n_samples, limit=args.limit)
    elif args.stage == "combine":
        run_combine_stage(args.model)


if __name__ == "__main__":
    main()

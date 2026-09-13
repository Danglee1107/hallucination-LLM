"""
Chuẩn hoá 3 domain của HaluEval (QA / Dialogue / Summarization) về schema chung:

    {
        "id": str,
        "domain": "qa" | "dialogue" | "summarization",
        "prompt": str,          # câu hỏi / lượt hội thoại cuối / yêu cầu tóm tắt
        "context": str,         # knowledge / dialogue_history / document — dùng để label bằng entailment
        "source_right": str,    # right_answer/response/summary gốc của HaluEval (KHÔNG dùng làm label cho model mới,
                                 #   chỉ giữ lại để tham khảo / sanity check)
        "source_hallucinated": str,  # tương tự, chỉ để tham khảo
    }

Lưu ý quan trọng: right_answer/hallucinated_answer trong HaluEval là do ChatGPT sinh,
KHÔNG phải là label cho response mà 5 model local của bạn sẽ sinh ra sau này.
File output này chỉ chứa (prompt, context) để đưa vào 5 model generate response mới;
bước gán label hallucination cho response mới sẽ làm ở script riêng (entailment-based),
dùng "context" ở đây làm nguồn đối chiếu.
"""

import json
import argparse
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

RAW_DIR = BASE / "data" / "raw" / "halueval" / "data"
OUT_DIR = BASE / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def convert_qa(records):
    out = []
    for i, r in enumerate(records):
        out.append({
            "id": f"qa_{i}",
            "domain": "qa",
            "prompt": r["question"],
            "context": r["knowledge"],
            "source_right": r["right_answer"],
            "source_hallucinated": r["hallucinated_answer"],
        })
    return out


def convert_dialogue(records):
    out = []
    for i, r in enumerate(records):
        out.append({
            "id": f"dialogue_{i}",
            "domain": "dialogue",
            "prompt": r["dialogue_history"],
            "context": r["knowledge"],
            "source_right": r["right_response"],
            "source_hallucinated": r["hallucinated_response"],
        })
    return out


def convert_summarization(records):
    out = []
    for i, r in enumerate(records):
        out.append({
            "id": f"summarization_{i}",
            "domain": "summarization",
            "prompt": "Summarize the following document.",
            "context": r["document"],
            "source_right": r["right_summary"],
            "source_hallucinated": r["hallucinated_summary"],
        })
    return out


def main(sample_size=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    qa = convert_qa(load_jsonl(RAW_DIR / "qa_data.json"))
    dialogue = convert_dialogue(load_jsonl(RAW_DIR / "dialogue_data.json"))
    summarization = convert_summarization(load_jsonl(RAW_DIR / "summarization_data.json"))

    if sample_size:
        qa = qa[:sample_size]
        dialogue = dialogue[:sample_size]
        summarization = summarization[:sample_size]

    all_records = qa + dialogue + summarization

    out_path = OUT_DIR / "unified_prompts.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"QA: {len(qa)} | Dialogue: {len(dialogue)} | Summarization: {len(summarization)}")
    print(f"Total: {len(all_records)} -> saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=None,
                         help="Optional: limit records per domain (for quick testing)")
    args = parser.parse_args()
    main(sample_size=args.sample_size)

"""
Normalize 3 domain của HaluEval (QA / Dialogue / Summarization) về cùng một schema:

    {
        "id": str,
        "domain": "qa" | "dialogue" | "summarization",
        "prompt": str,          # question / dialogue_history / summary prompt
        "context": str,         # knowledge / knowledge / document -> dùng để label bằng entailment
        "source_right": str,    # right_answer/response/summary gốc của HaluEval (chỉ để sanity check)
        "source_hallucinated": str,
    }

Khác với bản cũ: thay vì ghi hết QA -> hết Dialogue -> hết Summarization,
file output được ghi xen kẽ theo từng "đợt" (block) 2500 records:
    2500 QA -> 2500 Dialogue -> 2500 Summarization -> 2500 QA -> ...
Domain nào hết dữ liệu trước thì bị bỏ qua ở các vòng sau.

Lưu ý: right_answer/hallucinated_answer trong HaluEval do ChatGPT sinh,
KHÔNG phải label cho response mà 5 model local sẽ sinh sau này.
"""

import json
import argparse
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

RAW_DIR = BASE / "data/raw/halueval/data/"
OUT_DIR = BASE / "data/processed"


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def convert_qa(records):
    return [
        {
            "id": f"qa_{i}",
            "domain": "qa",
            "prompt": r["question"],
            "context": r["knowledge"],
            "source_right": r["right_answer"],
            "source_hallucinated": r["hallucinated_answer"],
        }
        for i, r in enumerate(records)
    ]


def convert_dialogue(records):
    return [
        {
            "id": f"dialogue_{i}",
            "domain": "dialogue",
            "prompt": r["dialogue_history"],
            "context": r["knowledge"],
            "source_right": r["right_response"],
            "source_hallucinated": r["hallucinated_response"],
        }
        for i, r in enumerate(records)
    ]


def convert_summarization(records):
    return [
        {
            "id": f"summarization_{i}",
            "domain": "summarization",
            "prompt": "Summarize the following document.",
            "context": r["document"],
            "source_right": r["right_summary"],
            "source_hallucinated": r["hallucinated_summary"],
        }
        for i, r in enumerate(records)
    ]


def interleave_blocks(domain_lists, block_size):
    """Ghép các list theo từng đợt block_size: A[0:n] + B[0:n] + C[0:n] + A[n:2n] + ..."""
    out = []
    offsets = [0] * len(domain_lists)
    while any(offsets[i] < len(lst) for i, lst in enumerate(domain_lists)):
        for i, lst in enumerate(domain_lists):
            start = offsets[i]
            if start >= len(lst):
                continue
            out.extend(lst[start:start + block_size])
            offsets[i] = start + block_size
    return out


def main(sample_size=None, block_size=2500):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    qa = convert_qa(load_jsonl(RAW_DIR / "qa_data.json"))
    dialogue = convert_dialogue(load_jsonl(RAW_DIR / "dialogue_data.json"))
    summarization = convert_summarization(load_jsonl(RAW_DIR / "summarization_data.json"))

    if sample_size:
        qa = qa[:sample_size]
        dialogue = dialogue[:sample_size]
        summarization = summarization[:sample_size]

    all_records = interleave_blocks([qa, dialogue, summarization], block_size)

    out_path = OUT_DIR / "unified_prompts.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"QA: {len(qa)} | Dialogue: {len(dialogue)} | Summarization: {len(summarization)}")
    print(f"Block size: {block_size} (xen kẽ theo từng đợt)")
    print(f"Total: {len(all_records)} -> saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Optional: giới hạn số record mỗi domain (test nhanh)")
    parser.add_argument("--block_size", type=int, default=2500,
                        help="Số record mỗi đợt của một domain trước khi chuyển sang domain kế tiếp")
    args = parser.parse_args()
    main(sample_size=args.sample_size, block_size=args.block_size)

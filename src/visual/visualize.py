"""
Vẽ hình cho báo cáo / LaTeX: sơ đồ pipeline + phân tích nhãn hallucination.

Xuất mỗi hình 2 bản vào settings.assets_dir (mặc định <repo>/assets/):
  - .pdf: vector, dùng thẳng \\includegraphics trong LaTeX (phóng to không vỡ).
  - .png: xem nhanh / chèn slide.

Hình:
  pipeline                    sơ đồ 3 tầng: sinh response -> gán nhãn -> detector
  label_distribution          tỉ lệ final_label theo model × domain
  hallucination_rate          heatmap tỉ lệ hallucination, model × domain
  label_scatter_<model>       entailment_prob vs consistency_score, 2 phương pháp đồng ý tới đâu
  label_agreement_<model>     ma trận 2×2 entailment_label × consistency_label

Đọc mọi file labels/combined_*.jsonl nên chạy combine cho model mới là hình tự có.

Chạy dạng module từ src/ để import được config và label_module_tmp:
    cd src && uv run python -m visual.visualize
    cd src && uv run python -m visual.visualize --only pipeline
    cd src && uv run python -m visual.visualize --self-check   # vẽ trên dữ liệu giả vào thư mục tạm
"""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # không cần màn hình, chạy được qua ssh
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import PercentFormatter

from config import settings

# Lấy ngưỡng từ chính module gán nhãn để đường ngưỡng trên hình không lệch khi
# ai đó chỉnh ngưỡng. Đổi lại: import kéo theo torch/transformers, chậm vài giây.
from label_module_tmp import CONSISTENCY_THRESHOLD, ENTAILMENT_SUPPORT_THRESHOLD

# fontTools log từng bước subset font khi ghi PDF, bị loguru gom ra console -> tắt.
logging.getLogger("fontTools").setLevel(logging.WARNING)

DOMAINS = ["qa", "dialogue", "summarization"]
LABELS = ["not_hallucination", "hallucination", "disagreement"]

# 3 slot categorical đầu (blue, orange, aqua): đã kiểm tra CVD cho mọi cặp nên
# dùng được cho scatter. Màu gắn theo nhãn, không theo thứ hạng.
LABEL_COLOR = dict(zip(LABELS, ["#2a78d6", "#eb6834", "#1baf7a"]))
DOMAIN_MARKER = {"qa": "o", "dialogue": "s", "summarization": "^"}
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
# Ramp xanh 1 hue sáng -> đậm cho độ lớn (heatmap).
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

plt.rcParams.update(
    {
        "pdf.fonttype": 42,  # nhúng TrueType: LaTeX/Overleaf hiển thị và tìm chữ được
        "font.size": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": SURFACE,
    }
)


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"-> {out_dir / name}.pdf / .png")


def load_labels(label_dir: Path) -> dict:
    """{model_key: [row, ...]} từ mọi file combined_<model>.jsonl."""
    out = {}
    for path in sorted(label_dir.glob("combined_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if rows:
            out[path.stem.removeprefix("combined_")] = rows
    return out


def label_shares(rows) -> dict:
    """{label: (count, share)} theo đúng thứ tự LABELS."""
    n = len(rows)
    counts = {lab: sum(r["final_label"] == lab for r in rows) for lab in LABELS}
    return {lab: (c, c / n if n else 0.0) for lab, c in counts.items()}


def agreement_matrix(rows):
    """Hàng: entailment supported/hallucinated. Cột: consistent/inconsistent."""
    m = np.zeros((2, 2), dtype=int)
    for r in rows:
        i = int(r["entailment_label"] == "hallucinated")
        j = int(r["consistency_label"] == "inconsistent")
        m[i, j] += 1
    return m


def hide_spines(ax):
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


# ---------------------------------------------------------------- nhóm 1


def plot_pipeline(out_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 5.9)
    ax.axis("off")

    gen = {"face": "#f0efec", "edge": INK_2, "ls": "-"}
    lab = {"face": "#cde2fb", "edge": "#1c5cab", "ls": "-"}
    todo = {"face": SURFACE, "edge": INK_2, "ls": "--"}
    # key: (x tâm, y tâm, rộng, cao, tiêu đề, mô tả, style)
    boxes = {
        "data": (1.15, 3.0, 2.0, 1.2, "HaluEval", "QA · Dialogue\n· Summarization", gen),
        "prep": (3.65, 3.0, 2.2, 1.2, "prepare_data", "unified_prompts.jsonl\n(prompt, context)", gen),
        "gen": (6.35, 3.0, 2.6, 1.45, "generate_responses", "6 open LLMs\ngreedy decoding\n+ 1 analysis forward pass", gen),
        "state": (6.35, 4.85, 2.6, 1.3, "Internal states", "hidden states / layer\nattention entropy / layer\ntoken confidence, entropy", gen),
        "ent": (9.4, 2.6, 2.6, 1.1, "Entailment", "NLI: context ⇒ response\nDeBERTa-v3 (MNLI/FEVER/ANLI)", lab),
        "sc": (9.4, 1.0, 2.6, 1.1, "Self-consistency", "N = 10 sampled responses\npairwise NLI agreement", lab),
        "comb": (12.15, 1.8, 2.4, 1.5, "Combine", "both flag → hallucination\nboth pass → not_hallucination\nsplit → disagreement", lab),
        "det": (12.15, 4.85, 2.4, 1.3, "Detector", "internal states → label\n(planned)", todo),
    }
    for x, y, w, h, title, desc, st in boxes.values():
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2), w, h,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                facecolor=st["face"], edgecolor=st["edge"], linestyle=st["ls"], linewidth=1.2,
            )
        )
        ax.text(x, y + h / 2 - 0.18, title, ha="center", va="top", fontsize=10.5, fontweight="bold", color=INK)
        ax.text(x, y + h / 2 - 0.48, desc, ha="center", va="top", fontsize=8, color=INK_2, linespacing=1.35)

    def edge(key, side):
        x, y, w, h = boxes[key][:4]
        return {"l": (x - w / 2, y), "r": (x + w / 2, y), "t": (x, y + h / 2), "b": (x, y - h / 2)}[side]

    def arrow(a, b, text=None):
        ax.annotate(
            "", xy=b, xytext=a,
            arrowprops={"arrowstyle": "-|>", "color": INK_2, "lw": 1.2, "shrinkA": 2, "shrinkB": 2},
        )
        if text:  # mũi tên ngang: chữ nằm trên đường; dọc: chữ bên phải
            flat = a[1] == b[1]
            ax.text(
                (a[0] + b[0]) / 2 + (0 if flat else 0.08), (a[1] + b[1]) / 2 + (0.15 if flat else 0), text,
                ha="center" if flat else "left", va="center", fontsize=7.5, color=INK_2, style="italic",
            )

    arrow(edge("data", "r"), edge("prep", "l"))
    arrow(edge("prep", "r"), edge("gen", "l"))
    arrow(edge("gen", "t"), edge("state", "b"))
    for dst in ("ent", "sc"):
        arrow(edge("gen", "r"), edge(dst, "l"))
    arrow(edge("ent", "r"), edge("comb", "l"))
    arrow(edge("sc", "r"), edge("comb", "l"))
    arrow(edge("state", "r"), edge("det", "l"), "features")
    arrow(edge("comb", "t"), edge("det", "b"), "labels")

    for x, text in ((3.75, "① Response generation"), (9.4, "② Labeling"), (12.15, "③ Detection")):
        ax.text(x, 5.75, text, ha="center", fontsize=9, color=INK_2, fontweight="bold")
    save(fig, out_dir, "pipeline")


# ---------------------------------------------------------------- nhóm 2


def plot_label_distribution(data: dict, out_dir: Path):
    """Bar ngang 100% stacked, mỗi hàng 1 cặp model × domain."""
    rows = [
        (f"{model} · {dom}", recs)
        for model, all_recs in data.items()
        for dom in DOMAINS
        if (recs := [r for r in all_recs if r["domain"] == dom])
    ]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.4))
    for i, (_, recs) in enumerate(rows):
        left = 0.0
        for lab, (_, share) in label_shares(recs).items():
            # viền trắng 2px tách các đoạn liền nhau
            ax.barh(i, share, left=left, height=0.6, color=LABEL_COLOR[lab], edgecolor=SURFACE, linewidth=2)
            if share >= 0.07:
                ax.text(left + share / 2, i, f"{share:.0%}", ha="center", va="center", fontsize=8, color=SURFACE, fontweight="bold")
            left += share
        ax.text(1.01, i, f"n={len(recs)}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(rows)), [name for name, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("share of responses")
    ax.legend(
        [plt.Rectangle((0, 0), 1, 1, color=LABEL_COLOR[lab]) for lab in LABELS], LABELS,
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
    )
    save(fig, out_dir, "label_distribution")


def plot_hallucination_rate(data: dict, out_dir: Path):
    models = list(data)
    rate = np.full((len(models), len(DOMAINS)), np.nan)
    n = np.zeros(rate.shape, dtype=int)
    for i, model in enumerate(models):
        for j, dom in enumerate(DOMAINS):
            recs = [r for r in data[model] if r["domain"] == dom]
            if recs:
                n[i, j] = len(recs)
                rate[i, j] = label_shares(recs)["hallucination"][1]

    fig, ax = plt.subplots(figsize=(1.6 * len(DOMAINS) + 2, 0.6 * len(models) + 1.3))
    cmap = BLUES.copy()
    cmap.set_bad("#f0efec")  # chưa có dữ liệu: xám, không lẫn với 0%
    im = ax.imshow(np.ma.masked_invalid(rate), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(models)):
        for j in range(len(DOMAINS)):
            if np.isnan(rate[i, j]):
                ax.text(j, i, "no data", ha="center", va="center", fontsize=8, color=INK_2)
            else:
                ink = SURFACE if rate[i, j] > 0.5 else INK
                ax.text(j, i, f"{rate[i, j]:.0%}\nn={n[i, j]}", ha="center", va="center", fontsize=8, color=ink)
    ax.set_xticks(range(len(DOMAINS)), DOMAINS)
    ax.set_yticks(range(len(models)), models)
    hide_spines(ax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    cbar.outline.set_visible(False)
    ax.set_title("hallucination rate (final_label)", fontsize=10, color=INK, loc="left")
    save(fig, out_dir, "hallucination_rate")


# ---------------------------------------------------------------- nhóm 3


def plot_label_scatter(model: str, rows, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.axvline(ENTAILMENT_SUPPORT_THRESHOLD, color=INK_2, ls="--", lw=1)
    ax.axhline(CONSISTENCY_THRESHOLD, color=INK_2, ls="--", lw=1)
    for lab in LABELS:
        for dom in DOMAINS:
            pts = [r for r in rows if r["final_label"] == lab and r["domain"] == dom]
            if pts:
                ax.scatter(
                    [r["entailment_prob"] for r in pts], [r["consistency_score"] for r in pts],
                    s=36, marker=DOMAIN_MARKER[dom], color=LABEL_COLOR[lab],
                    edgecolor=SURFACE, linewidth=0.8, alpha=0.85,
                )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("entailment_prob  (NLI: context ⇒ response)")
    ax.set_ylabel("consistency_score  (share of agreeing sample pairs)")
    ax.set_title(f"{model}: do the two labelers agree?", fontsize=10, color=INK, loc="left")
    ax.grid(color=GRID, lw=0.6)

    domains = [d for d in DOMAINS if any(r["domain"] == d for r in rows)]
    handles = [plt.Line2D([], [], ls="", marker="o", ms=7, color=LABEL_COLOR[lab]) for lab in LABELS]
    handles += [plt.Line2D([], [], ls="", marker=DOMAIN_MARKER[d], ms=7, color=INK_2) for d in domains]
    ax.legend(handles, LABELS + domains, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    ax.text(
        1.02, 0.02,
        f"dashed: thresholds\nentailment ≥ {ENTAILMENT_SUPPORT_THRESHOLD}\n"
        f"consistency ≥ {CONSISTENCY_THRESHOLD}\n(entailment also needs\nlow contradiction_prob)",
        transform=ax.transAxes, fontsize=7.5, color=INK_2, va="bottom",
    )
    save(fig, out_dir, f"label_scatter_{model}")


def plot_label_agreement(model: str, rows, out_dir: Path):
    m = agreement_matrix(rows)
    total = m.sum()
    verdict = [["not_hallucination", "disagreement"], ["disagreement", "hallucination"]]
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.imshow(m / total, cmap=BLUES, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ink = SURFACE if m[i, j] / total > 0.5 else INK
            ax.text(j, i - 0.12, f"{m[i, j]}  ({m[i, j] / total:.0%})", ha="center", va="center", fontsize=11, fontweight="bold", color=ink)
            ax.text(j, i + 0.18, f"→ {verdict[i][j]}", ha="center", va="center", fontsize=7.5, color=ink)
    ax.set_xticks([0, 1], ["consistent", "inconsistent"])
    ax.set_yticks([0, 1], ["supported", "hallucinated"])
    ax.set_xlabel("self-consistency")
    ax.set_ylabel("entailment")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    hide_spines(ax)
    ax.set_title(f"{model} (n={total})", fontsize=10, color=INK, loc="left", pad=28)
    save(fig, out_dir, f"label_agreement_{model}")


# ----------------------------------------------------------------


def render(out_dir: Path, label_dir: Path, only=None):
    plot_pipeline(out_dir)
    if only == "pipeline":
        return
    data = load_labels(label_dir)
    if not data:
        print(f"không có file combined_*.jsonl trong {label_dir}, bỏ qua hình nhãn")
        return
    plot_label_distribution(data, out_dir)
    plot_hallucination_rate(data, out_dir)
    for model, rows in data.items():
        plot_label_scatter(model, rows, out_dir)
        plot_label_agreement(model, rows, out_dir)


def self_check():
    rows = [
        {"domain": "qa", "final_label": "hallucination", "entailment_label": "hallucinated",
         "consistency_label": "inconsistent", "entailment_prob": 0.1, "consistency_score": 0.2},
        {"domain": "qa", "final_label": "disagreement", "entailment_label": "supported",
         "consistency_label": "inconsistent", "entailment_prob": 0.9, "consistency_score": 0.3},
        {"domain": "dialogue", "final_label": "not_hallucination", "entailment_label": "supported",
         "consistency_label": "consistent", "entailment_prob": 0.95, "consistency_score": 0.9},
    ]
    shares = label_shares(rows)
    assert shares["hallucination"] == (1, 1 / 3), shares
    assert sum(c for c, _ in shares.values()) == len(rows)
    assert label_shares([])["hallucination"] == (0, 0.0)
    assert agreement_matrix(rows).tolist() == [[1, 1], [0, 1]]

    # smoke test: mọi hình vẽ được, kể cả ô heatmap không có dữ liệu
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "labels").mkdir()
        with open(tmp / "labels" / "combined_fake.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        render(tmp / "fig", tmp / "labels")
        assert len(list((tmp / "fig").glob("*.pdf"))) == 5
    print("self-check OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=settings.assets_dir)
    parser.add_argument("--only", choices=["pipeline"], help="chỉ vẽ sơ đồ pipeline")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        render(args.out, settings.label_dir, args.only)
    return 0


if __name__ == "__main__":
    sys.exit(main())

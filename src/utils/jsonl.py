"""
Đọc / dọn file responses/<model>.jsonl mà không parse mảng float.

Mỗi dòng nặng vài MB, gần hết là hidden_states_per_layer. Các bước chỉ cần
id/domain/response_text (resume, label, dedupe) nên cắt dòng trước key đó rồi
mới json.loads: nhanh hơn nhiều lần và không tốn hàng chục GB RAM cho list float.

Vì sao cắt được an toàn: trong chuỗi JSON mọi dấu " đều bị escape thành \\",
nên chuỗi `, "hidden_states_per_layer"` (dấu " không escape) không thể nằm
trong response_text. Dòng không có key đó thì parse cả dòng như thường.
"""

import json
import os
from pathlib import Path

HEAVY_KEY = b', "hidden_states_per_layer"'


def head(line: bytes) -> dict:
    """Các field đứng trước hidden_states_per_layer (id, domain, model, response_text)."""
    cut = line.find(HEAVY_KEY)
    return json.loads(line if cut == -1 else line[:cut] + b"}")


def iter_heads(path: Path):
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                yield head(line)


def compact_jsonl(path: Path, order) -> int:
    """Giữ dòng CUỐI của mỗi id, ghi lại theo thứ tự `order` (id không có trong
    order xếp sau, giữ thứ tự trong file). Trả số dòng bị bỏ.

    Dòng cuối = kết quả mới nhất, khớp cách combine/dict đọc (bản sau ghi đè).
    Ghi ra file tạm rồi os.replace: chết giữa chừng thì file gốc còn nguyên.
    Chỉ giữ offset trong RAM, không giữ nội dung dòng; cần disk trống ~ 1 lần file.
    """
    last, total = {}, 0
    with open(path, "rb") as f:
        while line := f.readline():
            if line.strip():
                last[head(line)["id"]] = f.tell() - len(line)
                total += 1

    rank = {rid: i for i, rid in enumerate(order)}
    ids = sorted(last, key=lambda rid: (rank.get(rid, len(rank)), last[rid]))
    tmp = path.with_name(path.name + ".tmp")
    with open(path, "rb") as src, open(tmp, "wb") as dst:
        for rid in ids:
            src.seek(last[rid])
            row = src.readline()
            dst.write(row if row.endswith(b"\n") else row + b"\n")
    os.replace(tmp, path)
    return total - len(ids)

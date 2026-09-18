"""wandb: so sánh metric, cấu hình và thông số máy giữa các lần chạy.

Không có WANDB_API_KEY (hoặc đặt WANDB_MODE=disabled) thì mọi hàm ở đây thành
no-op, nên script gọi thẳng, không cần bọc if. Nhờ vậy chạy trên máy chưa login
wandb cũng không bị treo ở prompt nhập key.

wandb tự ghi sẵn, không cần log tay:
  - system metrics: GPU util, VRAM, nhiệt độ, CPU, RAM, disk (mỗi vài giây)
  - git commit + diff của lần chạy

Máy không có mạng (vast.ai): WANDB_MODE=offline khi chạy, xong đồng bộ sau:
    wandb sync logs/wandb/offline-run-*

KHÔNG upload file trong logs/ lên wandb: traceback bật diagnose=True có thể
chứa HF_TOKEN.
"""

import os
import random
import socket
import time

import torch
from loguru import logger

from .config import settings

_run = None


def set_seed(seed: int) -> int:
    """Seed cho random/torch/cuda. Chỉ ảnh hưởng stage có do_sample=True."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op khi không có GPU
    return seed


def runtime_info(seed: int | None = None) -> dict:
    """Thông số máy + flag ảnh hưởng tới kết quả. Đưa vào cả log lẫn wandb config
    để biết run chậm/khác kết quả là do máy hay do cấu hình."""
    info = {
        "host": socket.gethostname(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "seed": seed,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["gpu_capability"] = f"{props.major}.{props.minor}"
        info["gpu_total_gb"] = round(props.total_memory / 1e9, 1)
        info["bf16"] = torch.cuda.is_bf16_supported()
    return info


def vram_peak_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.max_memory_allocated() / 1e9, 2)


def init_run(session: str, config: dict, tags: list[str] | None = None):
    """Mở wandb run. Trả None khi tắt tracking."""
    global _run
    mode = os.getenv("WANDB_MODE") or ("online" if settings.wandb_api_key else "disabled")
    if mode == "disabled":
        logger.info("wandb tắt (thiếu WANDB_API_KEY), chỉ ghi log file")
        return None
    try:
        import wandb
    except ImportError:
        logger.warning("chưa cài wandb (uv add wandb), bỏ qua tracking")
        return None

    if settings.wandb_api_key:
        os.environ.setdefault("WANDB_API_KEY", settings.wandb_api_key)
    _run = wandb.init(
        project=settings.wandb_project,
        name=f"{session}_{time.strftime('%Y%m%d_%H%M%S')}",
        config=config,
        tags=tags or None,
        dir=str(settings.log_dir),  # run offline nằm cạnh log, dễ wandb sync
        mode=mode,
    )
    logger.info(f"wandb {mode}: {getattr(_run, 'url', None)}")
    return _run


def log_metrics(data: dict, step: int | None = None) -> None:
    if _run is None:
        return
    _run.log(data, step=step)


def log_table(name: str, columns: list[str], rows: list[list]) -> None:
    """Bảng sample để so sánh định tính giữa các run."""
    if _run is None:
        return
    import wandb

    _run.log({name: wandb.Table(columns=columns, data=rows)})


def log_bar(name: str, mapping: dict) -> None:
    """Bar chart từ {nhãn: số lượng}."""
    if _run is None:
        return
    import wandb

    table = wandb.Table(
        columns=["label", "count"], data=[[k, v] for k, v in mapping.items()]
    )
    _run.log({name: wandb.plot.bar(table, "label", "count", title=name)})


def finish(summary: dict | None = None) -> None:
    """Ghi summary (số cuối cùng hiện ở bảng so sánh run) rồi đóng run."""
    global _run
    if _run is None:
        return
    if summary:
        _run.summary.update(summary)
    _run.finish()
    _run = None

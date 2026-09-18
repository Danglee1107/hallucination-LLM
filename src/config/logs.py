"""Cấu hình loguru dùng chung cho cả project.

3 sink:
  - console: ghi qua tqdm.write để không làm vỡ thanh tiến trình.
  - <session>_<time>.log: cho người đọc, có backtrace + giá trị biến.
  - <session>_<time>.jsonl: cùng nội dung dạng JSON, để grep/thống kê lỗi bằng script.

Mọi dòng log đều có {name}:{function}:{line} nên biết ngay bug ở file nào dòng nào.
Log của logging stdlib (transformers, torch, urllib3) và exception không ai bắt
cũng được gom vào đúng các sink này.

CẢNH BÁO: file sink bật diagnose=True nên traceback in cả giá trị biến cục bộ.
Nếu frame lỗi giữ object settings thì HF_TOKEN có thể lọt vào file log. Vì vậy
logs/ nằm trong .gitignore và không bao giờ upload file log lên wandb.
Muốn tắt: sửa diagnose=False bên dưới (đổi lại mất giá trị biến khi debug).

Self-check:  cd src && uv run python -m config.logs
"""

import logging
import sys

from loguru import logger
from tqdm import tqdm

from .config import settings

FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[session]}/{extra[stage]} | "
    "{name}:{function}:{line} | {message}"
)

_configured = False


class _InterceptHandler(logging.Handler):
    """Chuyển record của logging stdlib sang loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno  # level lạ (vd TRACE của lib khác)

        # Lấy vị trí thẳng từ LogRecord: chính xác hơn lùi frame và không phụ
        # thuộc đường dẫn của module logging. Không có bước này thì mọi log của
        # thư viện đều hiện là logging:callHandlers.
        patched = logger.patch(
            lambda r: r.update(
                name=record.name, function=record.funcName, line=record.lineno
            )
        )
        patched.opt(exception=record.exc_info).log(level, record.getMessage())


def _route_transformers() -> None:
    """transformers gắn handler riêng và tắt propagate nên _InterceptHandler ở
    root logger không thấy gì. Trả nó về đường propagate để warning của nó
    ("Setting pad_token_id...", "sliding window attention...") vào file log."""
    try:
        from transformers.utils import logging as hf_logging
    except ImportError:
        return
    hf_logging.disable_default_handler()
    hf_logging.enable_propagation()


def get_logger(session_name: str, stage: str = "-"):
    """Logger đã cấu hình. Gọi lại nhiều lần không nhân đôi handler.

    stage: nhãn phụ in ở mỗi dòng. Đổi theo từng giai đoạn bằng
    log.bind(stage="entailment") thay vì gọi lại get_logger.
    """
    global _configured
    logger.configure(extra={"session": session_name, "stage": stage})
    if _configured:
        return logger

    logger.add(
        lambda msg: tqdm.write(msg, end="", file=sys.stderr),
        format=FORMAT,
        level=settings.log_level,
        colorize=True,
        backtrace=True,
    )

    stem = session_name + "_{time:YYYYMMDD_HHmmss}"
    for suffix, serialize in ((".log", False), (".jsonl", True)):
        logger.add(
            settings.log_dir / (stem + suffix),
            format=FORMAT,
            level="DEBUG",
            encoding="utf-8",
            backtrace=True,
            diagnose=True,
            serialize=serialize,
            retention=settings.log_retention,
            # Ghi ở thread riêng: log từ dataloader worker / subprocess không kẹt.
            # Đổi lại phải logger.complete() trước khi đọc file log.
            enqueue=True,
        )

    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    _route_transformers()

    # Crash không ai bắt cũng phải vào file log kèm traceback, không chỉ in terminal.
    sys.excepthook = lambda t, v, tb: logger.opt(exception=(t, v, tb)).critical(
        "uncaught exception"
    )
    _configured = True
    return logger


# logger mặc định của loguru ghi thẳng stderr; bỏ để get_logger tự dựng sink.
logger.remove()


def _self_check():
    """Check chạy được không cần GPU: sink ghi đủ vị trí gọi, traceback, log stdlib."""
    import json
    import logging as std_logging

    from . import tracking

    log = get_logger("selfcheck", stage="test")
    log.info("dòng info")
    try:
        1 / 0
    except ZeroDivisionError:
        log.exception("lỗi cố ý")
    std_logging.getLogger("third_party").warning("log stdlib")

    logger.complete()  # sink enqueue=True ghi async, phải đợi trước khi đọc

    path = max(
        settings.log_dir.glob("selfcheck_*.log"), key=lambda p: p.stat().st_mtime
    )
    text = path.read_text(encoding="utf-8")
    assert ":_self_check:" in text, "thiếu file:function:line trong log"
    assert "ZeroDivisionError" in text, "traceback không vào file log"
    assert "1 / 0" in text, "backtrace không có dòng code gây lỗi"
    assert "log stdlib" in text, "không bắt được log của logging stdlib"
    assert "third_party:_self_check:" in text, "log stdlib bị ghi sai vị trí gọi"

    rows = [
        json.loads(line)
        for line in path.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["record"]["line"] > 0, "sink jsonl thiếu số dòng"
    assert rows[0]["record"]["extra"]["stage"] == "test", "bind stage không vào jsonl"

    # tracking không có key -> mọi hàm là no-op, script gọi thẳng không cần if
    tracking.init_run("selfcheck", {"x": 1}, tags=["t"])
    tracking.log_metrics({"a": 1})
    tracking.log_bar("b", {"x": 1})
    tracking.log_table("c", ["k"], [[1]])
    tracking.finish({"done": True})

    print(f"self-check OK -> {path}")


if __name__ == "__main__":
    import os

    # Không tạo run thật lúc self-check. Ghi đè để thử đường wandb:
    #     WANDB_MODE=offline uv run python -m config.logs
    os.environ.setdefault("WANDB_MODE", "disabled")
    _self_check()

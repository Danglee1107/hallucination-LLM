import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class Settings(BaseSettings):
    # env_file dùng đường dẫn tuyệt đối để chạy từ thư mục nào cũng đọc được .env
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    hf_token: str = Field(..., alias="HF_TOKEN")

    data_dir: Path = BASE_DIR / "data"
    log_dir: Path = BASE_DIR / "logs"
    # Hình xuất bởi visual/visualize.py (pdf cho LaTeX + png), tạo lúc lưu hình.
    assets_dir: Path = BASE_DIR / "assets"

    # Level của sink console. File sink luôn DEBUG. Bật chi tiết khi debug:
    # LOG_LEVEL=DEBUG uv run src/generate_responses_tmp.py ...
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    log_retention: int = Field(20, alias="LOG_RETENTION")  # giữ N file log gần nhất

    # Không có key -> tracking tự tắt, script vẫn chạy bình thường.
    wandb_api_key: str | None = Field(None, alias="WANDB_API_KEY")
    wandb_project: str = Field("hallucination-LLM", alias="WANDB_PROJECT")

    # Các thư mục con suy ra từ data_dir, tạo lazy khi được dùng tới
    # (trước đây tạo hết ở model_post_init dù script chỉ cần 1 thư mục).
    @property
    def raw_dir(self) -> Path:
        return _ensure(self.data_dir / "raw" / "halueval" / "data")

    @property
    def processed_dir(self) -> Path:
        return _ensure(self.data_dir / "processed")

    @property
    def responses_dir(self) -> Path:
        return _ensure(self.processed_dir / "responses")

    @property
    def label_dir(self) -> Path:
        return _ensure(self.processed_dir / "labels")


settings = Settings()

# huggingface_hub / transformers tự đọc HF_TOKEN từ env cho gated repo.
# Dùng env thay cho login() lúc import: không gọi mạng, không ghi token ra ~/.cache.
os.environ.setdefault("HF_TOKEN", settings.hf_token)

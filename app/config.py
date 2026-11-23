from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "ParaSight Backend"
    app_version: str = "0.1.0"

    model_path: Path = BASE_DIR / "models" / "malaria_model.keras"
    last_conv_layer_name: str = "out_relu"
    gradcam_alpha: float = 0.4

    threshold: float = 0.5
    max_upload_mb: int = 5
    input_size: int = 224

    allowed_origins: str = "http://localhost:3000,http://localhost:5173"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

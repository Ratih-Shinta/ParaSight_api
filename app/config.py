from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Malaria Detection API"
    app_version: str = "0.2.0"

    model_path: Path = BASE_DIR / "models" / "malaria_model.keras"
    last_conv_layer_name: str = "out_relu"
    gradcam_alpha: float = 0.4

    threshold: float = 0.5
    max_upload_mb: int = 5
    input_size: int = 224

    # smear mode
    max_smear_cells: int = Field(64, ge=1, le=500)
    smear_max_side: int = Field(1280, ge=320)
    smear_batch_size: int = Field(16, ge=1, le=128)
    smear_gradcam_max_cells: int = Field(12, ge=0, le=100)
    smear_min_area_frac: float = 0.0003
    smear_max_area_frac: float = 0.03
    smear_max_aspect_ratio: float = 2.0
    smear_crop_padding: float = 0.10
    smear_min_contrast: float = 6.0
    smear_crop_background: Literal["black", "bg", "keep"] = "black"

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
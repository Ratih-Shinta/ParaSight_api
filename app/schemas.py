from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool


class PredictionResponse(BaseModel):
    prediction: Literal["Parasitized", "Uninfected"]
    raw_sigmoid: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=100.0, description="Persentase 0-100")
    threshold: float
    gradcam_base64: str = Field(..., description="data:image/png;base64,...")
    disclaimer: str


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail

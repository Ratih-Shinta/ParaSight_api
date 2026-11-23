from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool

class PredictionResponse(BaseModel):
    mode: Literal["single"] = "single"
    prediction: Literal["Parasitized", "Uninfected"]
    raw_sigmoid: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=100.0, description="Persentase 0-100")
    threshold: float
    gradcam_base64: str = Field(..., description="data:image/png;base64,...")
    disclaimer: str

class CellResult(BaseModel):
    index: int = Field(..., ge=1, description="Nomor sel (mulai dari 1), sama dengan label pada gambar anotasi")
    bbox: list[int] = Field(..., min_length=4, max_length=4, description="[x1, y1, x2, y2] pada citra kerja")
    prediction: Literal["Parasitized", "Uninfected"]
    confidence: float = Field(..., ge=0.0, le=100.0)
    raw_sigmoid: float = Field(..., ge=0.0, le=1.0)
    threshold: float
    gradcam_base64: Optional[str] = Field(None, description="data:image/png;base64,... (hanya sel terpilih)")

class SmearPredictionResponse(BaseModel):
    mode: Literal["smear"] = "smear"
    image_width: int = Field(..., description="Lebar citra kerja (acuan koordinat bbox)")
    image_height: int = Field(..., description="Tinggi citra kerja (acuan koordinat bbox)")
    total_cells: int = Field(..., ge=0)
    parasitized_count: int = Field(..., ge=0)
    uninfected_count: int = Field(..., ge=0)
    parasitemia_rate: float = Field(..., ge=0.0, le=100.0, description="% sel terinfeksi dari total sel")
    annotated_image_base64: str = Field(..., description="data:image/jpeg;base64,...")
    cells: list[CellResult]
    cells_truncated: bool = Field(False, description="True bila kandidat melebihi max_smear_cells")
    message: Optional[str] = None
    disclaimer: str

PredictResponse = Union[PredictionResponse, SmearPredictionResponse]

class ErrorDetail(BaseModel):
    code: str
    message: str

class ErrorResponse(BaseModel):
    error: ErrorDetail
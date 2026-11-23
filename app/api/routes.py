from dataclasses import asdict

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.config import get_settings
from app.errors import AppError
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
    PredictResponse,
    SmearPredictionResponse,
)
from app.services.image_utils import (
    decode_image,
    decode_smear_image,
    read_upload_limited,
    validate_upload_metadata,
)

router = APIRouter()
VALID_MODES = ("single", "smear")

@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    loaded = getattr(request.app.state, "inference_service", None) is not None
    return HealthResponse(status="ok", model_loaded=loaded)

@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Form("single"),
):
    settings = get_settings()

    mode_norm = (mode or "single").strip().lower()
    if mode_norm not in VALID_MODES:
        await file.close()
        raise AppError(400, "INVALID_MODE", "Nilai 'mode' tidak valid. Gunakan 'single' atau 'smear'.")

    try:
        service = getattr(request.app.state, "inference_service", None)
        if service is None:
            raise AppError(503, "MODEL_NOT_READY", "Model belum siap. Coba lagi nanti.")
        validate_upload_metadata(file)
        data = await read_upload_limited(file, settings.max_upload_bytes)
    finally:
        await file.close()

    if mode_norm == "single":
        rgb = await run_in_threadpool(decode_image, data, settings.input_size)
        del data
        result = await run_in_threadpool(service.predict, rgb)
        return PredictionResponse(**asdict(result))

    rgb = await run_in_threadpool(decode_smear_image, data, settings.smear_max_side)
    del data
    result = await run_in_threadpool(service.analyze_smear, rgb)
    return SmearPredictionResponse(**asdict(result))
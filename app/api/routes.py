from fastapi import APIRouter, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.config import get_settings
from app.errors import AppError
from app.schemas import ErrorResponse, HealthResponse, PredictionResponse
from app.services.image_utils import decode_image, read_upload_limited, validate_upload_metadata

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    loaded = getattr(request.app.state, "inference_service", None) is not None
    return HealthResponse(status="ok", model_loaded=loaded)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def predict(request: Request, file: UploadFile = File(...)):
    settings = get_settings()
    service = getattr(request.app.state, "inference_service", None)
    if service is None:
        raise AppError(503, "MODEL_NOT_READY", "Model belum siap. Coba lagi nanti.")

    validate_upload_metadata(file)
    data = await read_upload_limited(file, settings.max_upload_bytes)
    rgb = await run_in_threadpool(decode_image, data, settings.input_size)
    result = await run_in_threadpool(service.predict, rgb)

    return PredictionResponse(**result.__dict__)

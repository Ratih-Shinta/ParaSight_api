import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import get_settings
from app.errors import AppError
from app.services.inference import MalariaInferenceService, load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.inference_service = None
    try:
        logger.info("Memuat model dari %s ...", settings.model_path)
        model = load_model(settings)
        service = MalariaInferenceService(model, settings)
        try:
            service.warm_up()
        except Exception:
            logger.exception("Warm-up gagal (server tetap berjalan)")
        app.state.inference_service = service
        logger.info("Model siap.")
    except Exception:
        logger.exception("Gagal memuat model; /health akan melaporkan model_loaded=false")
    yield
    app.state.inference_service = None

app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)

origins = settings.allowed_origins_list
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials="*" not in origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})

@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError):
    return _error(exc.status_code, exc.code, exc.message)

@app.exception_handler(RequestValidationError)
async def validation_handler(_: Request, exc: RequestValidationError):
    return _error(422, "VALIDATION_ERROR", "Request tidak valid. Kirim file pada field 'file' (multipart/form-data).")

@app.exception_handler(Exception)
async def unhandled_handler(_: Request, exc: Exception):
    logger.exception("Unhandled error")
    return _error(500, "INTERNAL_ERROR", "Terjadi kesalahan internal pada server.")

app.include_router(router)
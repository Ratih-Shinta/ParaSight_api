import base64
import io

import numpy as np
from fastapi import UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from app.errors import AppError

ALLOWED_MIME = {"image/png", "image/jpeg", "image/webp"}
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_PIL_FORMATS = {"PNG", "JPEG", "WEBP"}

Image.MAX_IMAGE_PIXELS = 50_000_000

def validate_upload_metadata(file: UploadFile) -> None:
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type not in ALLOWED_MIME:
        raise AppError(
            415,
            "UNSUPPORTED_MEDIA_TYPE",
            "Tipe file tidak didukung. Gunakan image/png, image/jpeg, atau image/webp.",
        )
    name = (file.filename or "").lower()
    if not any(name.endswith(ext) for ext in ALLOWED_EXT):
        raise AppError(
            415,
            "UNSUPPORTED_EXTENSION",
            "Ekstensi file tidak didukung. Gunakan .png, .jpg, .jpeg, atau .webp.",
        )

async def read_upload_limited(file: UploadFile, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > max_bytes:
            raise AppError(
                413,
                "FILE_TOO_LARGE",
                f"Ukuran file melebihi batas {max_bytes // (1024 * 1024)} MB.",
            )
        chunks.append(chunk)
    if total == 0:
        raise AppError(400, "EMPTY_FILE", "File kosong.")
    return b"".join(chunks)

def _load_rgb_pil(data: bytes, apply_exif: bool = False) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in ALLOWED_PIL_FORMATS:
                raise AppError(
                    415,
                    "UNSUPPORTED_MEDIA_TYPE",
                    "Isi file bukan PNG, JPEG, atau WEBP yang valid.",
                )
            if apply_exif:
                im = ImageOps.exif_transpose(im)
            return im.convert("RGB")
    except AppError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError, SyntaxError):
        raise AppError(400, "INVALID_IMAGE", "File rusak atau bukan gambar yang valid.")

def decode_image(data: bytes, size: int) -> np.ndarray:
    img = _load_rgb_pil(data).resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)

def decode_smear_image(data: bytes, max_side: int) -> np.ndarray:
    img = _load_rgb_pil(data, apply_exif=True)
    w, h = img.size
    if min(w, h) < 64:
        raise AppError(400, "IMAGE_TOO_SMALL", "Gambar apusan terlalu kecil (minimal 64x64 piksel).")
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)

def encode_data_uri(rgb: np.ndarray, fmt: str = "PNG", quality: int = 90) -> str:
    buf = io.BytesIO()
    im = Image.fromarray(np.ascontiguousarray(rgb))
    if fmt.upper() == "JPEG":
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        mime = "image/jpeg"
    else:
        im.save(buf, format="PNG")
        mime = "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("ascii")
import base64
import io
import logging
import threading
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from PIL import Image

from app.config import Settings
from app.errors import AppError

from app.gradcam import make_gradcam_heatmap, overlay_gradcam

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "Sistem ini adalah prototipe/alat bantu riset dan BUKAN pengganti "
    "diagnosis medis profesional. Konsultasikan hasil dengan tenaga medis."
)


@dataclass
class PredictionResult:
    prediction: str
    raw_sigmoid: float
    confidence: float
    threshold: float
    gradcam_base64: str
    disclaimer: str = DISCLAIMER


def load_model(settings: Settings) -> tf.keras.Model:
    path = settings.model_path
    if not path.exists():
        raise FileNotFoundError(f"File model tidak ditemukan: {path}")
    return tf.keras.models.load_model(path)


class MalariaInferenceService:
    def __init__(self, model: tf.keras.Model, settings: Settings):
        self.model = model
        self.settings = settings
        self._lock = threading.Lock()

    def warm_up(self) -> None:
        size = self.settings.input_size
        self.predict(np.zeros((size, size, 3), dtype=np.uint8))

    def predict(self, rgb: np.ndarray) -> PredictionResult:
        s = self.settings
        # Skala 0-255; normalisasi layer `rescaling` didlm model
        img_array = np.expand_dims(rgb.astype("float32"), axis=0)

        with self._lock:
            try:
                out = self.model(img_array, training=False)
                raw = float(np.asarray(out).ravel()[0])
            except Exception:
                logger.exception("Inferensi model gagal")
                raise AppError(500, "INFERENCE_FAILED", "Terjadi kesalahan saat menjalankan prediksi.")

            try:
                heatmap = make_gradcam_heatmap(
                    tf.convert_to_tensor(img_array),
                    self.model,
                    last_conv_layer_name=s.last_conv_layer_name,
                )
                overlay = overlay_gradcam(rgb, heatmap, alpha=s.gradcam_alpha)
                gradcam_b64 = self._to_data_uri(overlay)
            except Exception:
                logger.exception("Grad-CAM gagal")
                raise AppError(500, "GRADCAM_FAILED", "Gagal membuat visualisasi Grad-CAM.")

        if raw < s.threshold:
            label, conf = "Parasitized", (1.0 - raw) * 100.0
        else:
            label, conf = "Uninfected", raw * 100.0

        return PredictionResult(
            prediction=label,
            raw_sigmoid=raw,
            confidence=round(conf, 2),
            threshold=s.threshold,
            gradcam_base64=gradcam_b64,
        )

    @staticmethod
    def _to_data_uri(rgb_uint8: np.ndarray) -> str:
        buf = io.BytesIO()
        Image.fromarray(rgb_uint8).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

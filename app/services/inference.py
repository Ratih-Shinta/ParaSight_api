import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import tensorflow as tf

from app.config import Settings
from app.errors import AppError

from app.gradcam import make_gradcam_heatmap, overlay_gradcam
from app.services.annotation import draw_annotations
from app.services.image_utils import encode_data_uri
from app.services.segmentation import segment_cells

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
    mode: str = "single"

@dataclass
class CellPrediction:
    index: int
    bbox: list[int]
    prediction: str
    confidence: float
    raw_sigmoid: float
    threshold: float
    gradcam_base64: Optional[str] = None

@dataclass
class SmearResult:
    image_width: int
    image_height: int
    total_cells: int
    parasitized_count: int
    uninfected_count: int
    parasitemia_rate: float
    annotated_image_base64: str
    cells: list[CellPrediction] = field(default_factory=list)
    cells_truncated: bool = False
    message: Optional[str] = None
    disclaimer: str = DISCLAIMER
    mode: str = "smear"

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

    # util
    def warm_up(self) -> None:
        size = self.settings.input_size
        self.predict(np.zeros((size, size, 3), dtype=np.uint8))

    def _classify(self, raw: float) -> tuple[str, float]:
        if raw < self.settings.threshold:
            return "Parasitized", (1.0 - raw) * 100.0
        return "Uninfected", raw * 100.0

    def _forward(self, batch_uint8: np.ndarray) -> np.ndarray:
        x = tf.convert_to_tensor(batch_uint8.astype("float32"))
        with self._lock:
            out = self.model(x, training=False)
        return np.asarray(out).reshape(-1)

    def _gradcam_data_uri(self, rgb224: np.ndarray) -> str:
        img_array = np.expand_dims(rgb224.astype("float32"), axis=0)
        with self._lock:
            heatmap = make_gradcam_heatmap(
                tf.convert_to_tensor(img_array),
                self.model,
                last_conv_layer_name=self.settings.last_conv_layer_name,
            )
        overlay = overlay_gradcam(rgb224, heatmap, alpha=self.settings.gradcam_alpha)
        return encode_data_uri(overlay, "PNG")

    # single mode
    def predict(self, rgb: np.ndarray) -> PredictionResult:
        try:
            raw = float(self._forward(np.expand_dims(rgb, 0))[0])
        except Exception:
            logger.exception("Inferensi model gagal")
            raise AppError(500, "INFERENCE_FAILED", "Terjadi kesalahan saat menjalankan prediksi.")

        try:
            gradcam_b64 = self._gradcam_data_uri(rgb)
        except Exception:
            logger.exception("Grad-CAM gagal")
            raise AppError(500, "GRADCAM_FAILED", "Gagal membuat visualisasi Grad-CAM.")

        label, conf = self._classify(raw)
        return PredictionResult(
            prediction=label,
            raw_sigmoid=raw,
            confidence=round(conf, 2),
            threshold=self.settings.threshold,
            gradcam_base64=gradcam_b64,
        )

    # smear mode
    def analyze_smear(self, rgb: np.ndarray) -> SmearResult:
        s = self.settings
        h, w = rgb.shape[:2]

        # 1. segment
        try:
            seg = segment_cells(rgb, s)
        except Exception:
            logger.exception("Segmentasi gagal")
            raise AppError(500, "SEGMENTATION_FAILED", "Gagal memproses segmentasi sel pada citra apusan.")

        if not seg.cells:
            return SmearResult(
                image_width=w, image_height=h,
                total_cells=0, parasitized_count=0, uninfected_count=0, parasitemia_rate=0.0,
                annotated_image_base64=encode_data_uri(rgb, "JPEG"),
                message=(
                    "Tidak ada sel terdeteksi. Pastikan gambar berupa apusan darah dengan latar terang, "
                    "fokus, dan sel terlihat jelas."
                ),
            )

        # 2. klasifikasi batch
        crops = np.stack([c.crop for c in seg.cells])
        bs = s.smear_batch_size
        try:
            raws = np.concatenate([self._forward(crops[i:i + bs]) for i in range(0, len(crops), bs)])
        except Exception:
            logger.exception("Inferensi batch gagal")
            raise AppError(500, "INFERENCE_FAILED", "Terjadi kesalahan saat menjalankan prediksi.")

        preds: list[CellPrediction] = []
        for i, (cell, raw) in enumerate(zip(seg.cells, raws), start=1):
            raw_f = float(min(max(raw, 0.0), 1.0))
            label, conf = self._classify(raw_f)
            preds.append(CellPrediction(
                index=i, bbox=list(cell.bbox), prediction=label,
                confidence=round(conf, 2), raw_sigmoid=raw_f, threshold=s.threshold,
            ))

        # 3. Grad-CAM terbatas: parasitized (confidence tertinggi dulu), sisa -> uninfected paling ga yakin
        para = sorted((i for i, p in enumerate(preds) if p.prediction == "Parasitized"),
                      key=lambda i: -preds[i].confidence)
        unin = sorted((i for i, p in enumerate(preds) if p.prediction == "Uninfected"),
                      key=lambda i: preds[i].confidence)
        for i in (para + unin)[: s.smear_gradcam_max_cells]:
            try:
                preds[i].gradcam_base64 = self._gradcam_data_uri(seg.cells[i].crop)
            except Exception:
                logger.exception("Grad-CAM gagal untuk sel #%d (dilewati)", preds[i].index)

        # 4. statistik + gambar anotasi
        total = len(preds)
        n_para = len(para)
        annotated = draw_annotations(rgb, [(p.index, tuple(p.bbox), p.prediction) for p in preds])

        return SmearResult(
            image_width=w, image_height=h,
            total_cells=total,
            parasitized_count=n_para,
            uninfected_count=total - n_para,
            parasitemia_rate=round(n_para / total * 100.0, 2) if total else 0.0,
            annotated_image_base64=encode_data_uri(annotated, "JPEG"),
            cells=preds,
            cells_truncated=seg.truncated,
            message=(
                f"Terdeteksi {seg.candidates_found} kandidat sel; hanya {total} sel yang dianalisis "
                f"(batas max_smear_cells)."
                if seg.truncated else None
            ),
        )
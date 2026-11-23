import cv2
import numpy as np

COLOR_PARASITIZED = (220, 38, 38) # merah
COLOR_UNINFECTED = (22, 163, 74) # hijau
_FONT = cv2.FONT_HERSHEY_SIMPLEX

def draw_annotations(rgb: np.ndarray, items: list[tuple[int, tuple[int, int, int, int], str]]) -> np.ndarray:
    out = rgb.copy()
    h, w = out.shape[:2]
    scale = max(h, w) / 1000.0
    thick = max(1, int(round(2 * scale)))
    fscale = max(0.35, 0.5 * scale)

    for idx, (x1, y1, x2, y2), label in items:
        color = COLOR_PARASITIZED if label == "Parasitized" else COLOR_UNINFECTED
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)

        text = str(idx)
        (tw, th), base = cv2.getTextSize(text, _FONT, fscale, 1)
        box_h = th + base + 2
        ty = y1 - box_h if y1 - box_h >= 0 else y1
        cv2.rectangle(out, (x1, ty), (x1 + tw + 4, ty + box_h), color, cv2.FILLED)
        cv2.putText(out, text, (x1 + 2, ty + th + 1), _FONT, fscale, (255, 255, 255), 1, cv2.LINE_AA)
    return out
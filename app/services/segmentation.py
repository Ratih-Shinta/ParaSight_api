from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)

CORE_RATIO = 0.7
MIN_FILL_RATIO = 0.45
REL_AREA_MIN, REL_AREA_MAX = 0.35, 2.5

@dataclass
class CellCandidate:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 (x2/y2 eksklusif)
    area: int
    crop: np.ndarray # (out, out, 3) RGB uint8

@dataclass
class SegmentationResult:
    cells: list[CellCandidate]
    candidates_found: int
    truncated: bool

def _empty() -> SegmentationResult:
    return SegmentationResult(cells=[], candidates_found=0, truncated=False)

def _label_boxes(markers: np.ndarray):
    w = markers.shape[1]
    flat = markers.ravel()
    idx = np.flatnonzero(flat >= 2)
    if idx.size == 0:
        return None
    lab = flat[idx]
    order = np.argsort(lab, kind="stable")
    lab_s, idx_s = lab[order], idx[order]
    ys, xs = np.divmod(idx_s, w)
    starts = np.flatnonzero(np.r_[True, lab_s[1:] != lab_s[:-1]])
    labels = lab_s[starts]
    areas = np.diff(np.r_[starts, lab_s.size])
    x0 = np.minimum.reduceat(xs, starts)
    x1 = np.maximum.reduceat(xs, starts)
    y0 = np.minimum.reduceat(ys, starts)
    y1 = np.maximum.reduceat(ys, starts)
    return labels, areas, x0, y0, x1, y1

def _make_crop(rgb, markers, label, box, pad_frac, mode, bg_color, out_size) -> np.ndarray:
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    side = max(8, int(round(max(bw, bh) * (1.0 + 2.0 * pad_frac))))
    cx, cy = (x0 + x1 + 1) / 2.0, (y0 + y1 + 1) / 2.0
    sx0, sy0 = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))

    cx0, cy0 = max(sx0, 0), max(sy0, 0)
    cx1, cy1 = min(sx0 + side, w), min(sy0 + side, h)
    patch = rgb[cy0:cy1, cx0:cx1].copy()

    fill = np.zeros(3, np.uint8) if mode == "black" else bg_color
    if mode != "keep":
        region = (markers[cy0:cy1, cx0:cx1] == label).astype(np.uint8)
        region = cv2.dilate(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        patch[region == 0] = fill

    canvas = np.empty((side, side, 3), np.uint8)
    canvas[:] = fill
    canvas[cy0 - sy0: cy0 - sy0 + patch.shape[0], cx0 - sx0: cx0 - sx0 + patch.shape[1]] = patch
    interp = cv2.INTER_AREA if side > out_size else cv2.INTER_LINEAR
    return cv2.resize(canvas, (out_size, out_size), interpolation=interp)

def segment_cells(rgb: np.ndarray, s: "Settings") -> SegmentationResult:
    h, w = rgb.shape[:2]
    img_area = h * w

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if float(gray.std()) < s.smear_min_contrast:
        return _empty()

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.count_nonzero(mask) / img_area > 0.75:
        mask = cv2.bitwise_not(mask)

    k = max(3, int(round(min(h, w) / 200))) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _empty()
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    # inti sel via distance transform
    dist = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    win = max(9, int(round(min(h, w) / 30))) | 1
    local_max = cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (win, win)))
    peaks = dist[(dist == local_max) & (dist > 2.0)]
    if peaks.size:
        ref = float(np.percentile(peaks, 90))
        r_typ = float(np.median(peaks[peaks >= 0.4 * ref]))
    else:
        r_typ = float(dist.max())
    if r_typ <= 1.0:
        return _empty()

    core = (dist >= CORE_RATIO * r_typ).astype(np.uint8) * 255
    n_core, core_lbl = cv2.connectedComponents(core)
    if n_core <= 1:
        return _empty()

    # watershed
    try:
        markers = core_lbl.astype(np.int32) + 1
        sure_bg = cv2.dilate(filled, kernel)
        markers[(sure_bg > 0) & (core == 0)] = 0
        markers = cv2.watershed(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), markers)
    except cv2.error:
        logger.warning("Watershed gagal; fallback ke connected components", exc_info=True)
        n, lbl = cv2.connectedComponents(filled)
        markers = np.where(lbl > 0, lbl + 1, 1).astype(np.int32)

    boxes = _label_boxes(markers)
    if boxes is None:
        return _empty()
    labels, areas, x0, y0, x1, y1 = boxes
    bw, bh = x1 - x0 + 1, y1 - y0 + 1

    # filter
    keep = (areas >= s.smear_min_area_frac * img_area) & (areas <= s.smear_max_area_frac * img_area)
    ar = s.smear_max_aspect_ratio
    keep &= (bw / bh <= ar) & (bh / bw <= ar)
    keep &= (areas / (bw * bh)) >= MIN_FILL_RATIO
    if keep.sum() >= 5:
        med = float(np.median(areas[keep]))
        keep &= (areas >= REL_AREA_MIN * med) & (areas <= REL_AREA_MAX * med)
    touches = (x0 <= 1) | (y0 <= 1) | (x1 >= w - 2) | (y1 >= h - 2)
    if (keep & ~touches).any():
        keep &= ~touches

    cand = np.flatnonzero(keep)
    found = int(cand.size)
    if found == 0:
        return _empty()

    truncated = found > s.max_smear_cells
    if truncated:
        med = float(np.median(areas[cand]))
        cand = cand[np.argsort(np.abs(areas[cand] - med))[: s.max_smear_cells]]

    band = max(1.0, float(np.median(bh[cand])))
    cyc = (y0[cand] + y1[cand] + 1) / 2.0
    cxc = (x0[cand] + x1[cand] + 1) / 2.0
    cand = cand[np.lexsort((cxc, np.floor(cyc / band)))]

    bg_px = rgb[filled == 0]
    bg_color = (np.median(bg_px, axis=0).astype(np.uint8) if bg_px.size else np.full(3, 255, np.uint8))

    cells: list[CellCandidate] = []
    for i in cand:
        box = (int(x0[i]), int(y0[i]), int(x1[i]), int(y1[i]))
        crop = _make_crop(
            rgb, markers, int(labels[i]), box,
            s.smear_crop_padding, s.smear_crop_background, bg_color, s.input_size,
        )
        cells.append(CellCandidate(bbox=(box[0], box[1], box[2] + 1, box[3] + 1), area=int(areas[i]), crop=crop))

    return SegmentationResult(cells=cells, candidates_found=found, truncated=truncated)
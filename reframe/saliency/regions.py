from __future__ import annotations

import math
import numpy as np
from typing import Optional
from reframe.geometry import clamp

def extract_saliency_region(
    saliency_map: np.ndarray,
    bounds: tuple[int, int, int, int],
    frame_shape: Optional[tuple[int, int]] = None,
) -> Optional[tuple[float, float, float, float, float, float, float]]:
    """Returns center_x, center_y, left, top, right, bottom, confidence."""
    h, w = saliency_map.shape[:2]
    sx = (frame_shape[1] / w) if frame_shape else 1.0
    sy = (frame_shape[0] / h) if frame_shape else 1.0
    x1, y1, x2, y2 = bounds
    x1, y1, x2, y2 = x1 / sx, y1 / sy, math.ceil(x2 / sx), math.ceil(y2 / sy)
    x1 = int(clamp(x1, 0, w - 1))
    y1 = int(clamp(y1, 0, h - 1))
    x2 = int(clamp(x2, x1 + 1, w))
    y2 = int(clamp(y2, y1 + 1, h))

    roi = saliency_map[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    roi_sum = float(roi.sum())
    if roi_sum <= 1e-6:
        return None

    roi_max = float(roi.max())
    roi_mean = float(roi.mean())
    roi_std = float(roi.std())
    threshold = max(roi_mean + roi_std * 0.75, roi_max * 0.6)
    salient_mask = roi >= threshold
    if not np.any(salient_mask):
        salient_mask = roi >= max(roi_mean + roi_std * 0.25, roi_max * 0.4)
    if not np.any(salient_mask):
        return None

    ys, xs = np.where(salient_mask)
    weights = roi[salient_mask].astype(np.float64)
    weight_sum = float(weights.sum())
    if weight_sum <= 1e-6:
        return None

    center_x = x1 + float(np.dot(xs, weights) / weight_sum)
    center_y = y1 + float(np.dot(ys, weights) / weight_sum)
    left = x1 + float(xs.min())
    top = y1 + float(ys.min())
    right = x1 + float(xs.max() + 1)
    bottom = y1 + float(ys.max() + 1)
    confidence = clamp(weight_sum / max(1.0, roi.size), 0.0, 1.0)
    return (
        center_x * sx,
        center_y * sy,
        left * sx,
        top * sy,
        right * sx,
        bottom * sy,
        confidence,
    )

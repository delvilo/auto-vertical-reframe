"""CPU saliency maps: sparse optical flow, EMA, and inference provenance."""
from __future__ import annotations

from dataclasses import replace
import cv2
import numpy as np
from reframe.contracts import BackendPrediction, FrameContext, SaliencyResult


class SaliencyCache:
    def __init__(self, ema: float = 0.65):
        if not 0 < ema <= 1:
            raise ValueError("Saliency EMA must be in (0, 1]")
        self.ema = ema
        self.reset()

    def reset(self) -> None:
        self.previous_gray: np.ndarray | None = None
        self.result: SaliencyResult | None = None

    def propagate(self, gray: np.ndarray) -> np.ndarray | None:
        propagated = self.result.map if self.result is not None else None
        if propagated is not None and self.previous_gray is not None:
            points = cv2.goodFeaturesToTrack(
                self.previous_gray, maxCorners=40, qualityLevel=0.02, minDistance=12
            )
            if points is not None:
                new, ok, _ = cv2.calcOpticalFlowPyrLK(
                    self.previous_gray, gray, points, None, winSize=(15, 15), maxLevel=2
                )
                if new is not None and ok is not None and np.count_nonzero(ok) >= 4:
                    displacement = (new - points).reshape(-1, 2)[ok.ravel().astype(bool)]
                    dx, dy = np.median(displacement, axis=0)
                    if (np.isfinite([dx, dy]).all()
                            and abs(dx) < gray.shape[1] * 0.2
                            and abs(dy) < gray.shape[0] * 0.2):
                        propagated = cv2.warpAffine(
                            propagated, np.float32([[1, 0, dx], [0, 1, dy]]),
                            (gray.shape[1], gray.shape[0]), borderMode=cv2.BORDER_REPLICATE,
                        )
        return propagated

    def refresh(self, prediction: BackendPrediction, context: FrameContext,
                propagated: np.ndarray | None, shape: tuple[int, int]) -> SaliencyResult:
        fresh = np.asarray(prediction.map, dtype=np.float32)
        if fresh.ndim != 2 or not fresh.size or not np.isfinite(fresh).all():
            raise ValueError("Saliency backend must return a finite, nonempty 2-D map")
        if fresh.min() < -1e-6 or fresh.max() > 1 + 1e-6:
            raise ValueError("Saliency backend map must be in [0, 1]")
        fresh = cv2.resize(fresh, (shape[1], shape[0]))
        history_weight = 0.0 if propagated is None else 1 - self.ema
        blended = fresh if propagated is None else self.ema * fresh + history_weight * propagated
        self.result = SaliencyResult(
            blended, context, context, prediction.backend, prediction.status,
            "refresh" if propagated is None else "ema", prediction.reason, history_weight,
        )
        return self.result

    def reuse(self, propagated: np.ndarray, context: FrameContext) -> SaliencyResult:
        if self.result is None:
            raise RuntimeError("Cannot propagate an empty saliency cache")
        self.result = replace(self.result, map=propagated, frame=context, source="propagated")
        return self.result

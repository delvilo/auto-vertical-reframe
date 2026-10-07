from __future__ import annotations

from typing import Any, Optional
import cv2
import numpy as np
from reframe.contracts import BackendPrediction, FrameContext
from reframe.saliency.base import SaliencyBackend



class HandcraftedSaliencyHelper(SaliencyBackend):
    """Computes spectral residual saliency combined with motion energy."""

    def __init__(self) -> None:
        self.prev_gray_small: Optional[np.ndarray] = None
        self.backend_name = "handcrafted"
        self.active_backend = "handcrafted"
        self.frames_total = 0
        self.frames_backend = 0
        self.frames_fallback = 0

    def compute_map(self, frame_bgr: np.ndarray) -> np.ndarray:
        self.frames_total += 1
        self.frames_backend += 1
        frame_h, frame_w = frame_bgr.shape[:2]
        scale = min(1.0, 320.0 / max(frame_h, frame_w))
        small_w = max(32, int(round(frame_w * scale)))
        small_h = max(32, int(round(frame_h * scale)))

        # Optimization: skip redundant resize if frame_bgr is already at target dimensions
        if frame_w == small_w and frame_h == small_h:
            resized_bgr = frame_bgr
        else:
            resized_bgr = cv2.resize(
                frame_bgr,
                (small_w, small_h),
                interpolation=cv2.INTER_AREA,
            )

        gray_small = cv2.cvtColor(
            resized_bgr,
            cv2.COLOR_BGR2GRAY,
        ).astype(np.float32)

        dft = cv2.dft(gray_small, flags=cv2.DFT_COMPLEX_OUTPUT)
        real = dft[:, :, 0]
        imag = dft[:, :, 1]
        magnitude = cv2.magnitude(real, imag)
        log_amplitude = np.log(magnitude + 1e-6)
        spectral_residual = log_amplitude - cv2.blur(log_amplitude, (3, 3))

        # Optimization: Avoid costly cv2.phase (atan2) and np.cos/np.sin trigonometric calls.
        # Since cos(phase) = real / magnitude and sin(phase) = imag / magnitude,
        # exp_residual * cos(phase) == real * (exp(spectral_residual) / magnitude).
        scale = np.exp(spectral_residual) / (magnitude + 1e-6)
        residual_real = real * scale
        residual_imag = imag * scale
        residual_spectrum = np.dstack([residual_real, residual_imag]).astype(np.float32)
        saliency_small = cv2.idft(
            residual_spectrum,
            flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT,
        )
        saliency_small = cv2.GaussianBlur(
            saliency_small * saliency_small,
            (7, 7),
            0,
        )

        saliency_small = cv2.normalize(saliency_small, None, 0.0, 1.0, cv2.NORM_MINMAX)

        if (
            self.prev_gray_small is not None
            and self.prev_gray_small.shape == gray_small.shape
        ):
            motion = cv2.absdiff(gray_small, self.prev_gray_small)
            motion = cv2.GaussianBlur(motion, (5, 5), 0)
            motion = cv2.normalize(motion, None, 0.0, 1.0, cv2.NORM_MINMAX)
            saliency_small = saliency_small * 0.72 + motion * 0.28

        self.prev_gray_small = gray_small

        # Optimization: skip redundant output resize if saliency_small is already full frame
        if frame_w == small_w and frame_h == small_h:
            return saliency_small

        return cv2.resize(
            saliency_small,
            (frame_w, frame_h),
            interpolation=cv2.INTER_LINEAR,
        )

    def reset_temporal_state(self) -> None:
        self.prev_gray_small = None

    def get_telemetry(self) -> dict[str, Any]:
        return {
            "requested_backend": self.backend_name,
            "active_backend": self.active_backend,
            "frames_total": self.frames_total,
            "frames_backend": self.frames_backend,
            "frames_fallback": self.frames_fallback,
            "model_loaded": False,
        }

    def load(self) -> None:
        pass

    def observe(self, frame: np.ndarray, context: FrameContext) -> None:
        # Motion differences retain the original refresh-to-refresh behavior.
        pass

    def predict(self, frame: np.ndarray, context: FrameContext) -> BackendPrediction:
        return BackendPrediction(self.compute_map(frame).astype(np.float32, copy=False), "handcrafted")

    def reset(self) -> None:
        self.reset_temporal_state()

    def close(self) -> None:
        self.reset()

    def telemetry(self) -> dict[str, Any]:
        return self.get_telemetry()

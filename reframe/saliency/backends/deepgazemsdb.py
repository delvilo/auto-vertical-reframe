"""Spatial MSDB saliency on original frames, with explicit priors and geometry."""
from __future__ import annotations

from contextlib import nullcontext
import logging
from time import perf_counter
from typing import Any
import cv2
import numpy as np
from reframe.runtime import torch
from reframe.contracts import BackendPrediction, FrameContext
from reframe.saliency.base import SaliencyBackend
from reframe.saliency.backends.handcrafted import HandcraftedSaliencyHelper
from reframe.saliency.msdb_assets import (
    DEEPGAZE_REVISION, load_msdb_model, load_mit1003_template, center_bias_log_density,
)
from reframe.saliency.viewing import ViewingGeometry


class DeepGazeMSDBSaliencyHelper(SaliencyBackend):
    def __init__(self, device: str = "auto", center_bias: str = "mit1003",
                 screen_inches: float = 24.0, viewing_distance_cm: float = 60.0,
                 output_size: tuple[int, int] = (1080, 1920),
                 pixel_per_dva: float | None = None, allow_fallback: bool = False,
                 max_failures: int = 3, use_amp: bool = False):
        if center_bias not in {"mit1003", "uniform"}:
            raise ValueError("center_bias must be mit1003 or uniform")
        self.requested_device = device
        self.device_name: str | None = None
        self.center_bias_mode = center_bias
        self.viewing = ViewingGeometry(screen_inches, viewing_distance_cm, *output_size, pixel_per_dva)
        self.allow_fallback = allow_fallback
        self.max_failures = max_failures
        self.use_amp = use_amp
        self.model = self.template = self.bias_tensor = None
        self.bias_shape = None
        self.fallback = HandcraftedSaliencyHelper()
        self._disabled = False
        self.active_backend = "not_run"
        self.last_error: str | None = None
        self.frames_total = self.frames_backend = self.frames_fallback = 0
        self.observed_frames = self.consecutive_failures = 0
        self.last_ppd = self.last_input_size = self.last_crop_height = None
        self.load_seconds = self.inference_seconds = 0.0
        self.last_inference_seconds: float | None = None

    def _resolve_device(self) -> str:
        if torch is None:
            raise RuntimeError("PyTorch is unavailable for DeepGaze MSDB")
        device = self.requested_device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif device == "cuda":
            device = "cuda:0"
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("Requested MSDB CUDA device is unavailable")
        elif device == "mps":
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                raise RuntimeError("Requested MSDB MPS device is unavailable")
        elif device != "cpu":
            raise ValueError(f"Unsupported MSDB device: {device}")
        return device

    def load(self) -> None:
        if self.model is not None or self._disabled:
            return
        started = perf_counter()
        try:
            self.device_name = self._resolve_device()
            logging.info("MSDB full-frame FP32 baseline; requested_device=%s selected_device=%s "
                         "center_bias=%s dataset=None screen=%.1fin distance=%.1fcm "
                         "output_ppd=%.6f input_ppd_override=%s amp=%s license=unconfirmed",
                         self.requested_device, self.device_name, self.center_bias_mode,
                         self.viewing.screen_inches, self.viewing.distance_cm,
                         self.viewing.output_ppd, self.viewing.override_ppd, self.use_amp)
            if self.center_bias_mode == "mit1003":
                self.template = load_mit1003_template()
            self.model = load_msdb_model()
            self.model.to(self.device_name)
            self.model.eval()
            actual = {str(parameter.device) for parameter in self.model.parameters()}
            if actual and actual != {self.device_name} and not (
                self.device_name == "mps" and actual == {"mps:0"}
            ):
                raise RuntimeError(f"MSDB device mismatch: selected={self.device_name} actual={actual}")
            logging.info("Loaded DeepGaze MSDB on %s; upstream revision=%s",
                         self.device_name, DEEPGAZE_REVISION)
        except Exception as exc:
            self.model = None
            self._failure(exc, loading=True)
        finally:
            self.load_seconds += perf_counter() - started

    def _failure(self, exc: Exception, loading: bool = False) -> None:
        self.last_error = f"{type(exc).__name__}: {exc}"
        self.consecutive_failures += 1
        if not self.allow_fallback:
            # The CLI preserves the original exception/traceback and fails. No
            # hidden CPU fallback, downscaling, or precision change on OOM.
            raise exc
        self._disabled = loading or self.consecutive_failures >= self.max_failures
        logging.warning("MSDB auto fallback to handcrafted (disabled=%s): %s",
                        self._disabled, self.last_error, exc_info=True)
        if self._disabled:
            self.model = self.bias_tensor = None

    def observe(self, frame: np.ndarray, context: FrameContext) -> None:
        # MSDB is spatial: no 16-frame warmup or tensor ring is needed.
        self.observed_frames += 1

    def _fallback(self, frame: np.ndarray) -> BackendPrediction:
        self.frames_fallback += 1
        self.active_backend = "handcrafted"
        return BackendPrediction(self.fallback.compute_map(frame).astype(np.float32, copy=False),
                                 "handcrafted", "fallback", self.last_error)

    def predict(self, frame: np.ndarray, context: FrameContext) -> BackendPrediction:
        self.frames_total += 1
        if self.model is None and not self._disabled:
            self.load()
        if self._disabled:
            return self._fallback(frame)
        started = perf_counter()
        try:
            if (frame.shape != (context.height, context.width, 3)
                    or frame.dtype != np.uint8):
                raise ValueError("MSDB requires original-size BGR uint8 frames; downscaling is disabled")
            # DeepGaze's own normalizers expect 0..255, not an external /255.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))
            image_tensor = torch.from_numpy(chw).unsqueeze(0).to(self.device_name, dtype=torch.float32)
            if self.bias_tensor is None or self.bias_shape != frame.shape[:2]:
                prior = center_bias_log_density(frame.shape[:2], self.template)
                self.bias_tensor = torch.from_numpy(prior).unsqueeze(0).to(self.device_name)
                self.bias_shape = frame.shape[:2]
            ppd = self.viewing.input_ppd(context)
            if self.last_ppd is None:
                logging.info("MSDB actual input=%dx%d RGB[0,255] crop_height=%s input_ppd=%.6f",
                             context.width, context.height, context.view_crop_height, ppd)
            self.last_ppd = ppd
            self.last_crop_height = context.view_crop_height
            self.last_input_size = [context.width, context.height]
            amp = (torch.autocast(device_type="cuda", dtype=torch.float16)
                   if self.use_amp and self.device_name.startswith("cuda") else nullcontext())
            with torch.inference_mode(), amp:
                prediction = self.model(image_tensor, self.bias_tensor, pixel_per_dva=ppd, dataset=None)
            log_density = prediction.detach().float().cpu().numpy()
            if log_density.shape != (1, context.height, context.width) or not np.isfinite(log_density).all():
                raise ValueError(f"Invalid MSDB log density: shape={log_density.shape}")
            density = np.exp(log_density[0] - log_density.max())
            # Constant probabilities still contain uniform saliency. Returning
            # all zeros would incorrectly suppress saliency in that case.
            saliency = density.astype(np.float32, copy=False)
            self.frames_backend += 1
            self.consecutive_failures = 0
            self.last_error = None
            self.active_backend = "deepgazemsdb"
            self.last_inference_seconds = perf_counter() - started  # .cpu() synchronizes the output
            self.inference_seconds += self.last_inference_seconds
            logging.debug("MSDB frame=%d input=%s crop_height=%s ppd=%.6f bias=%s seconds=%.6f",
                          context.frame_index, self.last_input_size, self.last_crop_height,
                          ppd, self.center_bias_mode, self.last_inference_seconds)
            return BackendPrediction(saliency, "deepgazemsdb")
        except Exception as exc:
            self._failure(exc)
            return self._fallback(frame)

    def reset(self) -> None:
        self.fallback.reset()

    def close(self) -> None:
        self.model = self.template = self.bias_tensor = None
        self.bias_shape = None
        self.fallback.close()

    def telemetry(self) -> dict[str, Any]:
        result = {
            "requested_backend": "auto" if self.allow_fallback else "deepgazemsdb",
            "active_backend": self.active_backend, "model_loaded": self.model is not None,
            "device": self.device_name, "observed_frames": self.observed_frames,
            "frames_total": self.frames_total, "frames_backend": self.frames_backend,
            "frames_fallback": self.frames_fallback, "fallback_reason": self.last_error,
            "center_bias": self.center_bias_mode, "dataset": None,
            "input_resolution": "original", "input_size": self.last_input_size,
            "screen_inches": self.viewing.screen_inches, "viewing_distance_cm": self.viewing.distance_cm,
            "output_pixel_per_dva": self.viewing.output_ppd, "input_pixel_per_dva": self.last_ppd,
            "view_crop_height": self.last_crop_height, "amp_enabled": self.use_amp,
            "load_seconds": self.load_seconds, "inference_seconds": self.inference_seconds,
            "last_inference_seconds": self.last_inference_seconds,
            "license_status": "unconfirmed", "upstream_revision": DEEPGAZE_REVISION,
        }
        if self.device_name is not None and self.device_name.startswith("cuda"):
            result["process_cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(self.device_name)
            result["process_cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(self.device_name)
        return result

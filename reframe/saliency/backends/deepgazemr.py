from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any
import cv2
import numpy as np
from reframe.runtime import torch
from reframe.contracts import BackendPrediction, FrameContext
from reframe.frame_pixels import TensorFrame
from reframe.saliency.base import SaliencyBackend
from reframe.saliency.backends.handcrafted import HandcraftedSaliencyHelper
from reframe.saliency.backends.deepgaze_loader import load_deepgaze_mr
from reframe.timing import HostTimings, timed_method


class DeepGazeMRSaliencyHelper(SaliencyBackend):
    """Neural video saliency helper with a feature-only DeepGaze MR loader."""

    def __init__(
        self,
        device: str = "auto",
        max_side: int = 384,
        trust_repo: bool = False,
    ) -> None:
        self.backend_name = "deepgazemr"
        self.device_name = self._resolve_device(device)
        self.max_side = max(128, int(max_side))
        self.trust_repo = trust_repo
        self.model = None
        self.tensor_ring = self.host_ring = self.cpu_ring = None
        self.observation_tensor_ring = None
        self.tensor_observed_frames = self.tensor_preprocessed_frames = 0
        self.tensor_preprocess_batches = 0
        self.tensor_cross_device_copies = self.tensor_cross_device_bytes = 0
        self.tensor_cross_device_batches = 0
        self.copy_events = []
        self.ring_pos = self.ring_count = 0
        self._temporal_sequence = self._device_sequence = 0
        self.observed_frames = 0
        self.last_observation_ok = False
        self.use_amp = True
        self.fallback = HandcraftedSaliencyHelper()
        self._disabled = False
        self.active_backend = "handcrafted"
        self.frames_total = 0
        self.frames_backend = 0
        self.frames_fallback = 0
        self.actual_forward_calls = 0
        self.model_loaded = False
        self.model_provenance: dict[str, Any] = {}
        self.consecutive_failures = 0
        self.max_failures = 3
        self._timings = HostTimings(("init", "observe", "preprocess", "transfer",
                                    "forward", "postprocess", "fallback", "bookkeeping"))

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            if device == "cuda" and (torch is None or not torch.cuda.is_available()):
                logging.warning("CUDA unavailable; using CPU for saliency")
                return "cpu"
            if device == "mps" and (
                torch is None
                or not hasattr(torch.backends, "mps")
                or not torch.backends.mps.is_available()
            ):
                logging.warning("MPS unavailable; using CPU for saliency")
                return "cpu"
            return device
        if torch is None:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @timed_method("init")
    def _load_model(self) -> bool:
        if self._disabled:
            return False
        if self.model is not None:
            return True
        if torch is None:
            logging.warning(
                "PyTorch unavailable; falling back to handcrafted saliency."
            )
            self._disabled = True
            return False

        try:
            self.model, self.model_provenance = load_deepgaze_mr()
            self.model.to(self.device_name)
            if hasattr(self.model, "center_bias") and torch.is_tensor(self.model.center_bias):
                self.model.center_bias = self.model.center_bias.to(self.device_name)
            self.model.eval()
            self.model_loaded = True
            logging.info("Loaded feature-only DeepGaze MR on %s.", self.device_name)
            return True
        except Exception as exc:
            logging.warning(
                "DeepGaze MR direct checkpoint load failed: %s", exc, exc_info=True,
            )

        self._disabled = True
        self.model = None
        self.active_backend = "handcrafted"
        return False

    def _resize_frame(
        self,
        frame_bgr: np.ndarray | TensorFrame,
    ) -> np.ndarray | TensorFrame:
        frame_h, frame_w = frame_bgr.shape[:2]
        scale = min(1.0, self.max_side / max(frame_h, frame_w))
        resized_w = max(64, int(round(frame_w * scale)))
        resized_h = max(64, int(round(frame_h * scale)))
        if (resized_w, resized_h) == (frame_w, frame_h):
            return frame_bgr
        if isinstance(frame_bgr, TensorFrame):
            return frame_bgr.resized((resized_w, resized_h))
        return cv2.resize(
            frame_bgr,
            (resized_w, resized_h),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)))

    @staticmethod
    def _preprocess_tensor_frame(rgb, out=None):
        """Normalize RGB CHW/NCHW pixels, optionally into the existing ring.

        Copying uint8 pixels directly into a float32 destination avoids a
        temporary float batch. Division retains the NumPy path's arithmetic.
        """
        if out is not None:
            return out.copy_(rgb).div_(255.0)
        return rgb.to(dtype=torch.float32).div_(255.0)

    def _discard_device_window(self) -> None:
        for event in self.copy_events:
            if event is not None:
                event.synchronize()
        self.copy_events = []
        self.tensor_ring = self.host_ring = None
        self._device_sequence = 0

    @timed_method("observe")
    def observe_frame(self, frame_bgr: np.ndarray | TensorFrame) -> None:
        """Keep a small uint8 window without loading or touching the model.

        Tensor inputs stay on their source device; copying into this bounded
        window releases references to decoder batches. Float conversion and the
        model's device window remain deferred until a neural prediction.
        """
        self.observed_frames += 1
        self.last_observation_ok = False
        try:
            resized = self._resize_frame(frame_bgr)
            if isinstance(resized, TensorFrame):
                rgb = resized.rgb
                if (self.observation_tensor_ring is None
                        or self.observation_tensor_ring.shape[1:] != rgb.shape
                        or self.observation_tensor_ring.device != rgb.device):
                    self._discard_device_window()
                    self.observation_tensor_ring = torch.empty(
                        (16, *rgb.shape), dtype=rgb.dtype, device=rgb.device)
                    self.cpu_ring = None
                    self.ring_pos = self.ring_count = self._temporal_sequence = 0
                self.observation_tensor_ring[self.ring_pos].copy_(rgb)
                self.tensor_observed_frames += 1
            else:
                if self.cpu_ring is None or self.cpu_ring.shape[1:] != resized.shape:
                    self._discard_device_window()
                    self.cpu_ring = np.empty((16, *resized.shape), dtype=np.uint8)
                    self.observation_tensor_ring = None
                    self.ring_pos = self.ring_count = self._temporal_sequence = 0
                self.cpu_ring[self.ring_pos] = resized
            self.ring_pos = (self.ring_pos + 1) % 16
            self.ring_count = min(16, self.ring_count + 1)
            self._temporal_sequence += 1
            self.last_observation_ok = True
        except Exception as exc:
            self._inference_failure(exc)

    @timed_method("transfer")
    def _sync_device_window(self) -> None:
        """Upload only unseen slots; a long pause uploads the latest full window."""
        if self.observation_tensor_ring is not None:
            self._sync_tensor_device_window()
            return
        if self.cpu_ring is None:
            return
        shape = (3, *self.cpu_ring.shape[1:3])
        cuda = self.device_name.startswith("cuda")
        if self.tensor_ring is None or tuple(self.tensor_ring.shape[1:]) != shape:
            for event in self.copy_events:
                if event is not None:
                    event.synchronize()
            self.tensor_ring = torch.empty(
                (32, *shape), device=self.device_name, dtype=torch.float32
            )
            self.host_ring = torch.empty(
                (16, *shape), dtype=torch.float32, pin_memory=cuda
            )
            self.copy_events = [None] * 16
            self._device_sequence = 0
        pending = min(16, self._temporal_sequence - self._device_sequence)
        for offset in range(pending):
            i = (self.ring_pos - pending + offset) % 16
            if self.copy_events[i] is not None:
                self.copy_events[i].synchronize()
            with self._timings.measure("preprocess"):
                chw = self._preprocess_frame(self.cpu_ring[i])
                self.host_ring[i].copy_(torch.from_numpy(chw))
            self.tensor_ring[i].copy_(self.host_ring[i], non_blocking=cuda)
            self.tensor_ring[i + 16].copy_(self.tensor_ring[i])
            if cuda:
                event = torch.cuda.Event()
                event.record()
                self.copy_events[i] = event
        self._device_sequence = self._temporal_sequence

    def _sync_tensor_device_window(self) -> None:
        """Refresh unseen slots with at most two contiguous batch operations.

        Both halves of the 32-slot device ring mirror the 16 observations, so
        the chronological model clip stays a view even when its start wraps.
        Keeping the source as uint8 bounds storage and any cross-device copy.
        """
        source = self.observation_tensor_ring
        shape = tuple(source.shape[1:])
        if self.tensor_ring is None or tuple(self.tensor_ring.shape[1:]) != shape:
            self._discard_device_window()
            self.tensor_ring = torch.empty(
                (32, *shape), device=self.device_name, dtype=torch.float32)
        pending = min(16, self._temporal_sequence - self._device_sequence)
        start = (self.ring_pos - pending) % 16
        first_count = min(pending, 16 - start)
        for index, count in ((start, first_count), (0, pending - first_count)):
            if not count:
                continue
            end = index + count
            pixels = source[index:end]
            # A deliberately different saliency device copies only the small
            # uint8 observation. Same-device CUDA processing never downloads it.
            if pixels.device != self.tensor_ring.device:
                # Retain the historical frame count; batches count actual .to
                # operations now that a transfer can contain multiple frames.
                self.tensor_cross_device_copies += count
                self.tensor_cross_device_bytes += pixels.numel() * pixels.element_size()
                self.tensor_cross_device_batches += 1
                pixels = pixels.to(device=self.tensor_ring.device)
            with self._timings.measure("preprocess"):
                self._preprocess_tensor_frame(pixels, out=self.tensor_ring[index:end])
                self.tensor_preprocessed_frames += count
                self.tensor_preprocess_batches += 1
            self.tensor_ring[index + 16:end + 16].copy_(self.tensor_ring[index:end])
        self._device_sequence = self._temporal_sequence

    def _inference_failure(self, exc: Exception) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.max_failures:
            self._disabled = True
            self.active_backend = "handcrafted"
            logging.warning("DeepGaze disabled after repeated failures: %s", exc, exc_info=True)
        else:
            logging.warning("DeepGaze frame fallback: %s", exc, exc_info=True)

    @timed_method("fallback")
    def _fallback_map(self, frame_bgr: np.ndarray | TensorFrame) -> np.ndarray:
        if isinstance(frame_bgr, TensorFrame):
            frame_bgr = frame_bgr.bgr()
        return self.fallback.compute_map(frame_bgr)

    @timed_method("bookkeeping")
    def compute_map(self, frame_bgr: np.ndarray | TensorFrame, ingest: bool = True) -> np.ndarray:
        self.frames_total += 1
        if ingest:
            self.observe_frame(frame_bgr)
        if (
            self._disabled
            or not self.last_observation_ok
            or self.ring_count < 16
            or not self._load_model()
        ):
            self.active_backend = "handcrafted"
            self.frames_fallback += 1
            return self._fallback_map(frame_bgr)
        try:
            self._sync_device_window()
            with self._timings.measure("forward"):
                clip = self.tensor_ring[self.ring_pos : self.ring_pos + 16]
                amp = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if self.use_amp and self.device_name.startswith("cuda")
                    else nullcontext()
                )
                with torch.inference_mode(), amp:
                    self.actual_forward_calls += 1
                    prediction = self.model(clip)
            with self._timings.measure("postprocess"):
                # The existing .cpu() can wait for CUDA. Without introducing a
                # synchronize, forward is host dispatch and that wait is here.
                saliency = np.squeeze(prediction.detach().float().cpu().numpy())
                if saliency.ndim != 2 or not np.isfinite(saliency).all():
                    raise ValueError("DeepGaze returned invalid saliency values")
                saliency = np.exp(saliency - saliency.max())
                saliency = cv2.normalize(saliency, None, 0.0, 1.0, cv2.NORM_MINMAX)
                result = cv2.resize(saliency, (frame_bgr.shape[1], frame_bgr.shape[0]))
            self.active_backend = "deepgazemr"
            self.frames_backend += 1
            self.consecutive_failures = 0
            return result
        except Exception as exc:
            if self.use_amp:
                self.use_amp = False
                logging.warning("Disabling DeepGaze autocast after inference failure")
            self._inference_failure(exc)
            self.active_backend = "handcrafted"
            self.frames_fallback += 1
            return self._fallback_map(frame_bgr)

    def reset_temporal_state(self) -> None:
        self.ring_pos = self.ring_count = 0
        self._temporal_sequence = self._device_sequence = 0
        self.last_observation_ok = False
        self.fallback.reset_temporal_state()

    def get_telemetry(self) -> dict[str, Any]:
        return {
            "requested_backend": self.backend_name,
            "active_backend": self.active_backend,
            "frames_total": self.frames_total,
            "frames_backend": self.frames_backend,
            "frames_fallback": self.frames_fallback,
            "prediction_requests": self.frames_total,
            "actual_forward_calls": self.actual_forward_calls,
            "model_loaded": self.model_loaded,
            "model_provenance": dict(self.model_provenance),
            "device": self.device_name,
            "observed_frames": self.observed_frames,
            "tensor_observed_frames": self.tensor_observed_frames,
            "tensor_preprocessed_frames": self.tensor_preprocessed_frames,
            "tensor_preprocess_batches": self.tensor_preprocess_batches,
            "tensor_cross_device_copies": self.tensor_cross_device_copies,
            "tensor_cross_device_bytes": self.tensor_cross_device_bytes,
            "tensor_cross_device_batches": self.tensor_cross_device_batches,
            "observation_storage": (
                "rgb_uint8_tensor" if self.observation_tensor_ring is not None else "bgr_uint8_numpy"),
            "observation_device": (
                str(self.observation_tensor_ring.device)
                if self.observation_tensor_ring is not None else "cpu"),
            "temporal_window_frames": self.ring_count,
            "required_window_frames": 16,
            "amp_enabled": self.use_amp,
            "deepgaze_timing_seconds": dict(self._timings.seconds),
            "deepgaze_total_seconds": self._timings.total,
            "deepgaze_timing_kind": "exclusive_host_wall",
        }

    def load(self) -> None:
        self._load_model()

    def observe(self, frame: np.ndarray | TensorFrame, context: FrameContext) -> None:
        self.observe_frame(frame)

    @timed_method("bookkeeping")
    def predict(self, frame: np.ndarray | TensorFrame, context: FrameContext) -> BackendPrediction:
        previous = self.frames_backend
        saliency = self.compute_map(frame, ingest=False).astype(np.float32, copy=False)
        if self.frames_backend > previous:
            return BackendPrediction(saliency, "deepgazemr")
        warming = (not self._disabled
                   and self.last_observation_ok and self.ring_count < 16)
        return BackendPrediction(saliency, "handcrafted", "warmup" if warming else "fallback",
                                 "temporal_window" if warming else "model_unavailable_or_inference_failed")

    def reset(self) -> None:
        self.reset_temporal_state()

    def close(self) -> None:
        try:
            for event in self.copy_events:
                if event is not None:
                    event.synchronize()
        finally:
            self.copy_events = []
            self.tensor_ring = self.host_ring = self.cpu_ring = self.model = None
            self.observation_tensor_ring = None
            self.model_loaded = False
            self.fallback.close()
            self.reset_temporal_state()

    def telemetry(self) -> dict[str, Any]:
        return self.get_telemetry()

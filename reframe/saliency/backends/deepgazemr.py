from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from reframe.runtime import torch
from reframe.contracts import BackendPrediction, FrameContext
from reframe.saliency.base import SaliencyBackend
from reframe.saliency.backends.handcrafted import HandcraftedSaliencyHelper


class DeepGazeMRSaliencyHelper(SaliencyBackend):
    """Neural video saliency helper leveraging DeepGaze MR via PyTorch Hub."""

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
        self.tensor_ring = self.host_ring = None
        self.copy_events = []
        self.ring_pos = self.ring_count = 0
        self.observed_frames = 0
        self.last_observation_ok = False
        self.use_amp = True
        self.fallback = HandcraftedSaliencyHelper()
        self._disabled = False
        self.active_backend = "handcrafted"
        self.frames_total = 0
        self.frames_backend = 0
        self.frames_fallback = 0
        self.model_loaded = False
        self.consecutive_failures = 0
        self.max_failures = 3

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
            self.model = torch.hub.load(
                "mtangemann/deepgazemr",
                "DeepGazeMR",
                pretrained=True,
                trust_repo=self.trust_repo,
            )
            self.model.to(self.device_name)
            if hasattr(self.model, "center_bias") and torch.is_tensor(
                self.model.center_bias
            ):
                self.model.center_bias = self.model.center_bias.to(self.device_name)
            self.model.eval()
            self.model_loaded = True
            logging.info("Loaded DeepGaze MR saliency model on %s.", self.device_name)
            return True
        except Exception as exc:
            logging.warning(
                "DeepGaze MR automated hub load failed (%s); checking cached repo...",
                exc, exc_info=True,
            )

        try:
            self.model = None
            hub_dir = Path(torch.hub.get_dir())
            candidate_dirs = sorted(
                d
                for d in hub_dir.glob("mtangemann_deepgazemr_*")
                if d.is_dir()
                and (d / "data/deepgazemr-ledov.pt").is_file()
                and (d / "data/center-bias-ledov.pt").is_file()
            )
            if candidate_dirs:
                repo_path = candidate_dirs[0]
                self.model = torch.hub.load(
                    str(repo_path),
                    "DeepGazeMR",
                    source="local",
                    pretrained=False,
                    trust_repo=self.trust_repo,
                )
                ckpt_path = repo_path / "data" / "deepgazemr-ledov.pt"
                bias_path = repo_path / "data" / "center-bias-ledov.pt"
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                center_bias = torch.load(bias_path, map_location="cpu", weights_only=True)
                if not torch.is_tensor(center_bias):
                    raise ValueError("Missing or invalid center-bias tensor")
                self.model.center_bias = center_bias.to(self.device_name)
                self.model.to(self.device_name)
                self.model.eval()
                self.model_loaded = True
                logging.info("Loaded local DeepGaze MR on %s.", self.device_name)
                return True
        except Exception as inner_exc:
            logging.warning("Local DeepGaze MR fallback failed: %s", inner_exc, exc_info=True)

        self._disabled = True
        self.model = None
        self.active_backend = "handcrafted"
        return False

    def _preprocess_frame(
        self,
        frame_bgr: np.ndarray,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        frame_h, frame_w = frame_bgr.shape[:2]
        scale = min(1.0, self.max_side / max(frame_h, frame_w))
        resized_w = max(64, int(round(frame_w * scale)))
        resized_h = max(64, int(round(frame_h * scale)))
        resized = cv2.resize(
            frame_bgr,
            (resized_w, resized_h),
            interpolation=cv2.INTER_AREA,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))
        return chw, (frame_w, frame_h)

    def observe_frame(self, frame_bgr: np.ndarray) -> None:
        """Transfers one frame into a mirrored device ring, avoiding 16-frame stack allocations."""
        self.observed_frames += 1
        self.last_observation_ok = False
        if not self._load_model():
            return
        try:
            chw, self.original_size = self._preprocess_frame(frame_bgr)
            shape = tuple(chw.shape)
            cuda = self.device_name.startswith("cuda")
            if self.tensor_ring is None or tuple(self.tensor_ring.shape[1:]) != shape:
                self.tensor_ring = torch.empty(
                    (32, *shape), device=self.device_name, dtype=torch.float32
                )
                self.host_ring = torch.empty(
                    (16, *shape), dtype=torch.float32, pin_memory=cuda
                )
                self.copy_events = [None] * 16
                self.ring_pos = self.ring_count = 0
            i = self.ring_pos
            if self.copy_events[i] is not None:
                self.copy_events[i].synchronize()
            self.host_ring[i].copy_(torch.from_numpy(np.ascontiguousarray(chw)))
            self.tensor_ring[i].copy_(self.host_ring[i], non_blocking=cuda)
            self.tensor_ring[i + 16].copy_(self.tensor_ring[i])
            if cuda:
                event = torch.cuda.Event()
                event.record()
                self.copy_events[i] = event
            self.ring_pos = (i + 1) % 16
            self.ring_count = min(16, self.ring_count + 1)
            self.last_observation_ok = True
        except Exception as exc:
            self._inference_failure(exc)

    def _inference_failure(self, exc: Exception) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.max_failures:
            self._disabled = True
            self.active_backend = "handcrafted"
            logging.warning("DeepGaze disabled after repeated failures: %s", exc, exc_info=True)
        else:
            logging.warning("DeepGaze frame fallback: %s", exc, exc_info=True)

    def compute_map(self, frame_bgr: np.ndarray, ingest: bool = True) -> np.ndarray:
        self.frames_total += 1
        if ingest:
            self.observe_frame(frame_bgr)
        if (
            self._disabled
            or self.model is None
            or not self.last_observation_ok
            or self.ring_count < 16
        ):
            self.active_backend = "handcrafted"
            self.frames_fallback += 1
            return self.fallback.compute_map(frame_bgr)
        try:
            clip = self.tensor_ring[self.ring_pos : self.ring_pos + 16]
            amp = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self.use_amp and self.device_name.startswith("cuda")
                else nullcontext()
            )
            with torch.inference_mode(), amp:
                prediction = self.model(clip)
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
            return self.fallback.compute_map(frame_bgr)

    def reset_temporal_state(self) -> None:
        self.ring_pos = self.ring_count = 0
        self.last_observation_ok = False
        self.fallback.reset_temporal_state()

    def get_telemetry(self) -> dict[str, Any]:
        return {
            "requested_backend": self.backend_name,
            "active_backend": self.active_backend,
            "frames_total": self.frames_total,
            "frames_backend": self.frames_backend,
            "frames_fallback": self.frames_fallback,
            "model_loaded": self.model_loaded,
            "device": self.device_name,
            "observed_frames": self.observed_frames,
            "amp_enabled": self.use_amp,
        }

    def load(self) -> None:
        self._load_model()

    def observe(self, frame: np.ndarray, context: FrameContext) -> None:
        self.observe_frame(frame)

    def predict(self, frame: np.ndarray, context: FrameContext) -> BackendPrediction:
        previous = self.frames_backend
        saliency = self.compute_map(frame, ingest=False).astype(np.float32, copy=False)
        if self.frames_backend > previous:
            return BackendPrediction(saliency, "deepgazemr")
        warming = (not self._disabled and self.model is not None
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
            self.tensor_ring = self.host_ring = self.model = None
            self.model_loaded = False
            self.fallback.close()
            self.reset_temporal_state()

    def telemetry(self) -> dict[str, Any]:
        return self.get_telemetry()

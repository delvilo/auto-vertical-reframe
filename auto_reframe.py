#!/usr/bin/env python3
"""Auto Vertical Reframe — full performance and composition architecture.

Combines low-overhead GPU mask statistics, shared lazy ROI evaluation,
optical-flow saliency sampling, Theil-Sen robust velocity prediction,
two-person boundary fitting, and live encoder preflight into a single-pass pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import urllib.request
from collections import deque
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from scenedetect import AdaptiveDetector, SceneManager, open_video
from ultralytics import YOLO

# Modern MediaPipe Tasks Vision API
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    HAS_MP_TASKS = True
except Exception:
    mp = None
    mp_python = None
    mp_vision = None
    HAS_MP_TASKS = False

# Legacy solutions fallback (for mediapipe < 0.10.31)
try:
    from mediapipe.python.solutions import face_detection as legacy_mp_face
    from mediapipe.python.solutions import pose as legacy_mp_pose
except Exception:
    try:
        from mediapipe import solutions as mp_solutions

        legacy_mp_face = getattr(mp_solutions, "face_detection", None)
        legacy_mp_pose = getattr(mp_solutions, "pose", None)
    except Exception:
        legacy_mp_face = None
        legacy_mp_pose = None

try:
    import torch
except Exception:
    torch = None

CLASS_IDS = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
    "cat": 15,
    "dog": 16,
}

PRESETS = {
    "talking_head": {
        "classes": ["person"],
        "min_zoom": 1.00,
        "max_zoom": 1.85,
        "max_step_x": 6.0,
        "max_step_y": 4.0,
    },
    "sports": {
        "classes": ["person", "car", "bicycle", "motorcycle"],
        "min_zoom": 1.00,
        "max_zoom": 1.35,
        "max_step_x": 12.0,
        "max_step_y": 8.0,
    },
    "pets": {
        "classes": ["dog", "cat", "person"],
        "min_zoom": 1.00,
        "max_zoom": 1.55,
        "max_step_x": 10.0,
        "max_step_y": 7.0,
    },
    "cars": {
        "classes": ["car", "truck", "bus", "motorcycle", "person"],
        "min_zoom": 1.00,
        "max_zoom": 1.30,
        "max_step_x": 11.0,
        "max_step_y": 7.0,
    },
}

DEFAULT_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
DEFAULT_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

def ensure_mediapipe_model(
    model_name_or_path: Optional[str],
    default_filename: str,
    url: str,
) -> Optional[Path]:
    """Ensures that required MediaPipe model weights are present locally or cached."""
    if model_name_or_path:
        custom_path = Path(model_name_or_path)
        if not custom_path.is_file():
            raise FileNotFoundError(
                f"Explicit MediaPipe model not found: {custom_path}"
            )
        return custom_path

    cache_dir = Path.home() / ".cache" / "mediapipe"
    target_path = cache_dir / default_filename

    if target_path.exists():
        return target_path

    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info("Downloading MediaPipe model %s from %s...", default_filename, url)
        with tempfile.NamedTemporaryFile(
            dir=target_path.parent, suffix=".tmp", delete=False
        ) as tmp:
            temp_file = Path(tmp.name)
            with urllib.request.urlopen(url, timeout=30) as response:
                shutil.copyfileobj(response, tmp)
        if temp_file.stat().st_size == 0:
            raise ValueError("Downloaded model is empty")
        os.replace(temp_file, target_path)
        logging.info("Saved MediaPipe model to %s", target_path)
        return target_path
    except Exception as exc:
        if "temp_file" in locals() and temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass
        logging.warning(
            "Could not download MediaPipe model (%s): %s. "
            "Supply a local model via CLI options if offline.",
            default_filename,
            exc,
        )
        return None

@dataclass
class Candidate:
    cls_id: int
    cls_name: str
    track_id: Optional[int]
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    cx: float
    cy: float
    width: float
    height: float
    area: float
    mask_area: float
    mask_cx: float
    mask_cy: float
    mask_top_y: float
    framing_cx: float
    framing_cy: float
    face_box: Optional[tuple[int, int, int, int]]
    score: float = 0.0
    rank_confidence: float = 0.5
    eye_y: Optional[float] = None
    chin_y: Optional[float] = None
    body_top_y: Optional[float] = None
    body_bottom_y: Optional[float] = None
    body_cx: Optional[float] = None
    shoulder_span: Optional[float] = None
    body_bottom_confident: bool = False
    body_min_x: Optional[float] = None
    body_max_x: Optional[float] = None
    salient_x1: Optional[float] = None
    salient_y1: Optional[float] = None
    salient_x2: Optional[float] = None
    salient_y2: Optional[float] = None
    saliency_confidence: float = 0.0


@dataclass
class CameraObservation:
    center_x: float
    center_y: float
    zoom: float
    confidence: float


@dataclass
class CameraState:
    crop_center_x: float
    crop_center_y: float
    zoom: float
    target_center_x: float
    target_center_y: float
    target_zoom: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    zoom_velocity: float = 0.0
    tracked_id: Optional[int] = None
    tracked_cls_id: Optional[int] = None
    missed_frames: int = 0
    lock_track_id: Optional[int] = None
    lock_cls_id: Optional[int] = None
    current_scene_index: int = 0
    frames_since_subject_switch: int = 0
    last_framing_cx: Optional[float] = None
    last_framing_cy: Optional[float] = None
    last_subject_key: Optional[tuple[Optional[int], int]] = None
    framing_vx: float = 0.0
    framing_vy: float = 0.0
    is_scene_cut: bool = True
    two_person_active_frames: int = 0
    zoom_pending_frames: int = 0
    zoom_pending_seconds: float = 0.0
    motion_history: deque = field(default_factory=lambda: deque(maxlen=32))

class HandcraftedSaliencyHelper:
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

class DeepGazeMRSaliencyHelper:
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
            self.active_backend = "deepgazemr"
            self.model_loaded = True
            logging.info("Loaded DeepGaze MR saliency model on %s.", self.device_name)
            return True
        except Exception as exc:
            logging.warning(
                "DeepGaze MR automated hub load failed (%s); checking cached repo...",
                exc,
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
                self.active_backend = "deepgazemr"
                self.model_loaded = True
                logging.info("Loaded local DeepGaze MR on %s.", self.device_name)
                return True
        except Exception as inner_exc:
            logging.warning("Local DeepGaze MR fallback failed: %s", inner_exc)

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
            logging.warning("DeepGaze disabled after repeated failures: %s", exc)
        else:
            logging.warning("DeepGaze frame fallback: %s", exc)

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
            self.frames_backend += 1
            self.consecutive_failures = 0
            return result
        except Exception as exc:
            if self.use_amp:
                self.use_amp = False
                logging.warning("Disabling DeepGaze autocast after inference failure")
            self._inference_failure(exc)
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

def build_saliency_helper(args: argparse.Namespace) -> Any:
    if args.saliency_model == "handcrafted":
        backend = HandcraftedSaliencyHelper()
    else:
        backend = DeepGazeMRSaliencyHelper(
            device=args.saliency_device,
            max_side=args.saliency_max_side,
            trust_repo=args.saliency_trust_repo,
        )
        backend.max_failures = args.saliency_max_failures
        backend.use_amp = args.saliency_amp
    return SampledSaliency(
        backend, args.saliency_interval, args.saliency_max_side, args.saliency_ema
    )


class SubjectRankingModel:
    """Prioritizes candidates using detection, saliency, pose, and track continuity."""

    def __init__(self) -> None:
        self.class_bias = {
            "person": 0.22,
            "dog": 0.12,
            "cat": 0.10,
            "car": 0.06,
            "bicycle": 0.02,
            "motorcycle": 0.02,
            "bus": 0.01,
            "truck": 0.01,
        }
        self.feature_weights = {
            "det_conf": 1.35,
            "mask_presence": 0.95,
            "center_affinity": 0.55,
            "face_presence": 0.48,
            "pose_presence": 0.34,
            "saliency_presence": 0.72,
            "saliency_conf": 0.78,
            "tracking_match": 1.05,
            "lock_match": 1.30,
            "speaker_active": 0.22,
            "size_logit": 0.26,
        }

    def predict(
        self,
        *,
        cls_name: str,
        conf: float,
        mask_area: float,
        frame_area: float,
        dist_center: float,
        frame_diag: float,
        has_face: bool,
        has_pose: bool,
        saliency_confidence: float,
        tracking_match: bool,
        lock_match: bool,
        speaker_active: bool,
    ) -> float:
        norm_area = clamp(mask_area / max(frame_area, 1.0), 0.0, 1.0)
        center_affinity = 1.0 - clamp(dist_center / max(frame_diag, 1.0), 0.0, 1.0)
        size_logit = math.log1p(norm_area * 250.0)

        weights = self.feature_weights
        score = (
            self.class_bias.get(cls_name, 0.0)
            + weights["det_conf"] * clamp(conf, 0.0, 1.0)
            + weights["mask_presence"] * math.sqrt(norm_area)
            + weights["center_affinity"] * center_affinity
            + (weights["face_presence"] if has_face else 0.0)
            + (weights["pose_presence"] if has_pose else 0.0)
            + (weights["saliency_presence"] if saliency_confidence > 0.0 else 0.0)
            + weights["saliency_conf"] * clamp(saliency_confidence, 0.0, 1.0)
            + (weights["tracking_match"] if tracking_match else 0.0)
            + (weights["lock_match"] if lock_match else 0.0)
            + (weights["speaker_active"] if speaker_active else 0.0)
            + weights["size_logit"] * size_logit
        )

        return score

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="High-performance AI auto-reframe with single-pass pipeline and robust velocity estimation."
    )
    parser.add_argument("input", help="Source widescreen video file")
    parser.add_argument("output", help="Destination vertical video file")

    parser.add_argument("--seg-model", default="yolo11n-seg.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.30)

    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default="talking_head",
    )
    parser.add_argument("--classes", nargs="+", default=None)

    parser.add_argument("--output-width", type=int, default=1080)
    parser.add_argument("--output-height", type=int, default=1920)

    parser.add_argument("--zoom-alpha", type=float, default=0.035)
    parser.add_argument("--target-alpha", type=float, default=0.08)
    parser.add_argument("--motion-response", type=float, default=0.08)
    parser.add_argument("--motion-damping", type=float, default=0.92)
    parser.add_argument("--zoom-response", type=float, default=0.05)
    parser.add_argument("--zoom-damping", type=float, default=0.90)
    parser.add_argument("--max-missed-frames", type=int, default=40)
    parser.add_argument("--min-subject-hold-frames", type=int, default=14)
    parser.add_argument("--switch-score-threshold", type=float, default=1.20)

    parser.add_argument("--max-step-x", type=float, default=None)
    parser.add_argument("--max-step-y", type=float, default=None)

    parser.add_argument("--min-zoom", type=float, default=None)
    parser.add_argument("--max-zoom", type=float, default=None)

    parser.add_argument("--lock-first-subject", action="store_true")
    parser.add_argument("--two-person-framing", action="store_true")
    parser.add_argument("--two-person-threshold", type=float, default=0.75)

    parser.add_argument("--face-model", default="", help="Path to MediaPipe face detector model.")
    parser.add_argument("--pose-model", default="", help="Path to MediaPipe pose landmarker model.")

    parser.add_argument(
        "--saliency-model",
        choices=["auto", "deepgazemr", "handcrafted"],
        default="handcrafted",
    )
    parser.add_argument(
        "--saliency-device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument("--saliency-max-side", type=int, default=384)
    parser.add_argument(
        "--saliency-trust-repo",
        action="store_true",
        help="Allow torch.hub to execute untrusted code from remote or local repo.",
    )

    parser.add_argument("--speaker-aware-mode", action="store_true")
    parser.add_argument("--speaker-json", default="")

    parser.add_argument("--scene-threshold", type=float, default=3.0)
    parser.add_argument("--min-scene-len", type=int, default=15)

    parser.add_argument(
        "--post-restore",
        action="store_true",
        help="Apply unsharp mask and gentle denoising during FFmpeg stage.",
    )

    parser.add_argument("--video-encoder", default="auto")
    parser.add_argument("--audio-bitrate", default="192k")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset-ffmpeg", default="medium")

    parser.add_argument("--save-debug-preview", action="store_true")
    parser.add_argument("--debug-path", default="")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--nvenc-preset", choices=[f"p{i}" for i in range(1, 8)], default="p5"
    )
    parser.add_argument(
        "--video-bitrate", default="8M", help="VideoToolbox target bitrate"
    )
    parser.add_argument(
        "--lost-hold-seconds",
        type=float,
        default=None,
        help="Overrides --max-missed-frames (legacy frames at 30 fps)",
    )
    parser.add_argument("--subject-hold-seconds", type=float, default=None)
    parser.add_argument(
        "--dead-zone",
        type=float,
        default=None,
        help="Fraction of crop size; talking_head default .06, others .015",
    )
    parser.add_argument("--zoom-dead-zone", type=float, default=0.04)
    parser.add_argument("--zoom-hold-seconds", type=float, default=0.4)
    parser.add_argument("--fixed-zoom", type=float, default=None)
    parser.add_argument(
        "--cue-interval",
        type=float,
        default=0.2,
        help="Seconds between face/pose updates per track; 0 analyzes every frame",
    )
    parser.add_argument(
        "--cue-top-k",
        type=int,
        default=0,
        help="Limit expensive person cues; 0 means all people",
    )
    parser.add_argument(
        "--temp-dir", default=None, help="Location for lossless temporary videos"
    )
    parser.add_argument("--saliency-max-failures", type=int, default=3)
    parser.add_argument(
        "--retina-masks",
        action="store_true",
        help="Opt in to full-resolution masks",
    )
    parser.add_argument("--saliency-interval", type=int, default=3)
    parser.add_argument("--saliency-ema", type=float, default=0.65)
    parser.add_argument(
        "--saliency-amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CUDA FP16 autocast; automatically disabled if inference fails",
    )
    parser.add_argument(
        "--scene-method", choices=["inline", "prepass"], default="inline"
    )
    parser.add_argument(
        "--scene-downscale",
        type=int,
        default=4,
        help="Spatial reduction for optional PySceneDetect prepass",
    )
    parser.add_argument(
        "--encode-mode", choices=["direct", "lossless"], default="direct"
    )
    parser.add_argument(
        "--pan-time",
        type=float,
        default=None,
        help="Critical spring time scale in seconds",
    )
    parser.add_argument("--zoom-time", type=float, default=None)
    parser.add_argument("--target-time", type=float, default=None)
    parser.add_argument(
        "--pan-speed-x",
        type=float,
        default=None,
        help="Source width fractions per second",
    )
    parser.add_argument(
        "--pan-speed-y",
        type=float,
        default=None,
        help="Source height fractions per second",
    )
    parser.add_argument(
        "--zoom-speed", type=float, default=None, help="Zoom units per second"
    )
    parser.add_argument(
        "--prediction-window", type=float, default=0.3, help="Seconds of motion history"
    )
    parser.add_argument("--lookahead-seconds", type=float, default=0.12)
    args = parser.parse_args()

    if (
        args.saliency_interval < 1
        or args.scene_downscale < 1
        or not 0 < args.saliency_ema <= 1
    ):
        parser.error("Invalid saliency interval/EMA or scene downscale")
    for key in (
        "pan_time",
        "zoom_time",
        "target_time",
        "pan_speed_x",
        "pan_speed_y",
        "zoom_speed",
        "prediction_window",
    ):
        value = getattr(args, key)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"{key} must be finite and positive")
    if not 0 <= args.lookahead_seconds <= 1:
        parser.error("lookahead-seconds must be 0..1")
    if (
        args.output_width < 64
        or args.output_height < 64
        or args.output_width % 2
        or args.output_height % 2
    ):
        parser.error("Output dimensions must be even and at least 64")
    if not 0 <= args.conf <= 1 or not 0 <= args.crf <= 51:
        parser.error("conf must be 0..1 and crf/quality must be 0..51")
    for key in (
        "lost_hold_seconds",
        "subject_hold_seconds",
        "zoom_hold_seconds",
        "cue_interval",
    ):
        value = getattr(args, key)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(f"{key} must be finite and nonnegative")
    for key in ("min_zoom", "max_zoom", "fixed_zoom"):
        value = getattr(args, key)
        if value is not None and (not math.isfinite(value) or value < 1):
            parser.error(f"{key} must be finite and >= 1")
    for key in ("dead_zone", "zoom_dead_zone"):
        value = getattr(args, key)
        if value is not None and not 0 <= value <= 0.4:
            parser.error(f"{key} must be between 0 and .4")
    if args.video_encoder not in {
        "auto",
        "libx264",
        "libx265",
        "h264_nvenc",
        "hevc_nvenc",
        "h264_videotoolbox",
        "hevc_videotoolbox",
    }:
        parser.error("Unsupported video encoder")
    for key in ("motion_response", "zoom_response", "zoom_alpha", "target_alpha"):
        if not 0 < getattr(args, key) <= 1:
            parser.error(f"{key} must be in (0, 1]")
    for key in ("motion_damping", "zoom_damping"):
        if not 0 <= getattr(args, key) <= 1:
            parser.error(f"{key} must be in [0, 1]")
    if args.max_missed_frames < 0 or args.min_subject_hold_frames < 0:
        parser.error("Hold frame counts must be nonnegative")
    for key in ("max_step_x", "max_step_y"):
        value = getattr(args, key)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"{key} must be finite and positive")
    if args.cue_top_k < 0 or args.saliency_max_failures < 1:
        parser.error("cue-top-k must be >=0 and saliency-max-failures >=1")
    return args

def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset = PRESETS[args.preset]

    if args.classes is None:
        args.classes = preset["classes"]

    for key in ["max_step_x", "max_step_y", "min_zoom", "max_zoom"]:
        if getattr(args, key) is None:
            setattr(args, key, preset[key])

    if args.min_zoom > args.max_zoom:
        raise ValueError("min-zoom must be <= max-zoom")
    if (
        args.fixed_zoom is not None
        and not args.min_zoom <= args.fixed_zoom <= args.max_zoom
    ):
        raise ValueError("fixed-zoom must lie inside min/max zoom")
    if args.dead_zone is None:
        args.dead_zone = 0.06 if args.preset == "talking_head" else 0.015

    if args.pan_time is None:
        args.pan_time = (
            0.35
            * math.sqrt(0.08 / args.motion_response)
            * max(0.2, args.motion_damping / 0.92)
        )
    if args.zoom_time is None:
        args.zoom_time = (
            0.60
            * math.sqrt(0.05 / args.zoom_response)
            * max(0.2, args.zoom_damping / 0.90)
        )
    if args.target_time is None:
        args.target_time = -(1 / 30) / math.log1p(-min(args.target_alpha, 0.999))
    if args.pan_speed_x is None:
        args.pan_speed_x = args.max_step_x * 30 / 1920
    if args.pan_speed_y is None:
        args.pan_speed_y = args.max_step_y * 30 / 1080
    if args.zoom_speed is None:
        args.zoom_speed = args.zoom_alpha * 30
    return args


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def lerp(a: float, b: float, alpha: float) -> float:
    return a + (b - a) * alpha


def compute_base_crop(
    frame_w: int,
    frame_h: int,
    aspect_w: int = 9,
    aspect_h: int = 16,
) -> tuple[int, int]:
    target_ratio = aspect_w / aspect_h
    frame_ratio = frame_w / frame_h
    if frame_ratio >= target_ratio:
        crop_h = frame_h
        crop_w = int(round(crop_h * target_ratio))
    else:
        crop_w = frame_w
        crop_h = int(round(crop_w / target_ratio))
    return min(crop_w, frame_w), min(crop_h, frame_h)


def current_crop_size(
    base_crop_w: int,
    base_crop_h: int,
    zoom: float,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int]:
    safe_zoom = max(0.1, zoom)
    crop_w = int(round(base_crop_w / safe_zoom))
    crop_h = int(round(base_crop_h / safe_zoom))
    crop_w = max(64, min(crop_w, frame_w))
    crop_h = max(64, min(crop_h, frame_h))
    return crop_w, crop_h


class MediaPipeFaceHelper:
    """Detects faces within cropped person bounding boxes using Tasks API or legacy fallback."""

    def __init__(
        self,
        min_detection_confidence: float = 0.45,
        model_path: Optional[str] = None,
    ):
        self.detector = None
        self.is_tasks = False

        if HAS_MP_TASKS and mp_vision is not None:
            resolved_model = ensure_mediapipe_model(
                model_name_or_path=model_path,
                default_filename="blaze_face_short_range.tflite",
                url=DEFAULT_FACE_MODEL_URL,
            )
            if resolved_model is not None:
                try:
                    base_options = mp_python.BaseOptions(
                        model_asset_path=str(resolved_model)
                    )
                    options = mp_vision.FaceDetectorOptions(
                        base_options=base_options,
                        min_detection_confidence=min_detection_confidence,
                        running_mode=mp_vision.RunningMode.IMAGE,
                    )
                    self.detector = mp_vision.FaceDetector.create_from_options(options)
                    self.is_tasks = True
                    logging.info(
                        "MediaPipe Tasks FaceDetector initialized successfully."
                    )
                except Exception as exc:
                    logging.warning(
                        "MediaPipe Tasks FaceDetector failed to initialize: %s", exc
                    )

        if self.detector is None and legacy_mp_face is not None:
            try:
                self.detector = legacy_mp_face.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=min_detection_confidence,
                )
                self.is_tasks = False
                logging.info("Initialized legacy MediaPipe FaceDetection solution.")
            except Exception as exc:
                logging.warning("Legacy MediaPipe FaceDetection failed: %s", exc)

        if self.detector is None:
            logging.warning(
                "Face detection unavailable; running auto-reframe without face priority."
            )

    def detect_in_person_box(
        self,
        frame_bgr: np.ndarray,
        person_box: tuple[int, int, int, int],
        prepared: Optional[SharedPersonROI] = None,
    ) -> Optional[tuple[int, int, int, int]]:
        if self.detector is None:
            return None

        prepared = prepared or SharedPersonROI(frame_bgr, person_box)
        rgb, x1, y1 = prepared.face_view()
        roi = rgb
        if rgb.size == 0:
            return None

        best = None
        best_area = -1.0

        if self.is_tasks:
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self.detector.detect(mp_image)
                if not result.detections:
                    return None

                for det in result.detections:
                    bbox = det.bounding_box
                    px1 = max(0, int(bbox.origin_x))
                    py1 = max(0, int(bbox.origin_y))
                    pw = max(0, int(bbox.width))
                    ph = max(0, int(bbox.height))
                    px2 = min(roi.shape[1], px1 + pw)
                    py2 = min(roi.shape[0], py1 + ph)

                    area = max(1, px2 - px1) * max(1, py2 - py1)
                    if area > best_area:
                        best_area = area
                        best = (x1 + px1, y1 + py1, x1 + px2, y1 + py2)
            except Exception:
                return None
        else:
            try:
                result = self.detector.process(rgb)
                if not result.detections:
                    return None

                for det in result.detections:
                    bbox = det.location_data.relative_bounding_box
                    fx1 = max(0.0, bbox.xmin)
                    fy1 = max(0.0, bbox.ymin)
                    fw = max(0.0, bbox.width)
                    fh = max(0.0, bbox.height)

                    px1 = int(round(fx1 * roi.shape[1]))
                    py1 = int(round(fy1 * roi.shape[0]))
                    px2 = int(round((fx1 + fw) * roi.shape[1]))
                    py2 = int(round((fy1 + fh) * roi.shape[0]))

                    area = max(1, px2 - px1) * max(1, py2 - py1)
                    if area > best_area:
                        best_area = area
                        best = (x1 + px1, y1 + py1, x1 + px2, y1 + py2)
            except Exception:
                return None

        return best

    def close(self) -> None:
        if self.detector is not None and hasattr(self.detector, "close"):
            try:
                self.detector.close()
            except Exception:
                pass

POSE_LANDMARK_NAMES = {
    0: "nose",
    2: "left_eye",
    5: "right_eye",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}


class MediaPipePoseHelper:
    """Detects body landmarks within cropped person bounding boxes."""

    def __init__(
        self,
        min_detection_confidence: float = 0.35,
        model_path: Optional[str] = None,
    ):
        self.detector = None
        self.is_tasks = False
        self.min_visibility = 0.35

        if HAS_MP_TASKS and mp_vision is not None:
            resolved_model = ensure_mediapipe_model(
                model_name_or_path=model_path,
                default_filename="pose_landmarker_lite.task",
                url=DEFAULT_POSE_MODEL_URL,
            )
            if resolved_model is not None:
                try:
                    base_options = mp_python.BaseOptions(
                        model_asset_path=str(resolved_model)
                    )
                    options = mp_vision.PoseLandmarkerOptions(
                        base_options=base_options,
                        running_mode=mp_vision.RunningMode.IMAGE,
                        num_poses=1,
                        min_pose_detection_confidence=min_detection_confidence,
                        min_pose_presence_confidence=min_detection_confidence,
                        min_tracking_confidence=0.4,
                        output_segmentation_masks=False,
                    )
                    self.detector = mp_vision.PoseLandmarker.create_from_options(
                        options
                    )
                    self.is_tasks = True
                    logging.info(
                        "MediaPipe Tasks PoseLandmarker initialized successfully."
                    )
                except Exception as exc:
                    logging.warning(
                        "MediaPipe Tasks PoseLandmarker failed to initialize: %s", exc
                    )

        if self.detector is None and legacy_mp_pose is not None:
            try:
                self.detector = legacy_mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=0.4,
                )
                self.is_tasks = False
                logging.info("Initialized legacy MediaPipe Pose solution.")
            except Exception as exc:
                logging.warning("Legacy MediaPipe Pose failed: %s", exc)

        if self.detector is None:
            logging.warning(
                "Pose detection unavailable; composition will fall back to mask framing."
            )

    def detect_in_person_box(
        self,
        frame_bgr: np.ndarray,
        person_box: tuple[int, int, int, int],
        pad_ratio: float = 0.12,
        prepared: Optional[SharedPersonROI] = None,
    ) -> Optional[dict]:
        if self.detector is None:
            return None

        prepared = prepared or SharedPersonROI(frame_bgr, person_box, pad_ratio)
        rgb, rx1, ry1 = prepared.pose_view()
        roi = rgb
        if rgb.size == 0:
            return None
        try:
            if self.is_tasks:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self.detector.detect(mp_image)
                if not result.pose_landmarks:
                    return None
                landmarks_list = result.pose_landmarks[0]
            else:
                result = self.detector.process(rgb)
                if not result.pose_landmarks:
                    return None
                landmarks_list = result.pose_landmarks.landmark
        except Exception:
            return None

        roi_h, roi_w = roi.shape[:2]
        points: dict[str, tuple[float, float, float]] = {}
        for idx, lm in enumerate(landmarks_list):
            name = POSE_LANDMARK_NAMES.get(idx)
            if name is None:
                continue
            px = rx1 + lm.x * roi_w
            py = ry1 + lm.y * roi_h
            vis_val = getattr(lm, "visibility", None)
            vis = float(vis_val) if vis_val is not None else 1.0
            points[name] = (px, py, vis)

        if not points:
            return None

        def visible(name: str) -> Optional[tuple[float, float]]:
            p = points.get(name)
            if p is None or p[2] < self.min_visibility:
                return None
            return p[0], p[1]

        left_eye = visible("left_eye")
        right_eye = visible("right_eye")
        nose = visible("nose")
        if left_eye and right_eye:
            eye_y = (left_eye[1] + right_eye[1]) / 2
            eye_cx = (left_eye[0] + right_eye[0]) / 2
        elif nose is not None:
            eye_y = nose[1]
            eye_cx = nose[0]
        else:
            eye_y = None
            eye_cx = None

        mouth_l = visible("mouth_left")
        mouth_r = visible("mouth_right")
        chin_y = max(mouth_l[1], mouth_r[1]) if (mouth_l and mouth_r) else None

        left_sh = visible("left_shoulder")
        right_sh = visible("right_shoulder")
        if left_sh and right_sh:
            shoulder_span = abs(left_sh[0] - right_sh[0])
            shoulder_cx = (left_sh[0] + right_sh[0]) / 2
            shoulder_y = (left_sh[1] + right_sh[1]) / 2
        else:
            shoulder_span = None
            shoulder_cx = None
            shoulder_y = None

        left_hip = visible("left_hip")
        right_hip = visible("right_hip")
        if left_hip and right_hip:
            hip_cx = (left_hip[0] + right_hip[0]) / 2
            hip_y = (left_hip[1] + right_hip[1]) / 2
        else:
            hip_cx = None
            hip_y = None

        lower_body_names = [
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index",
        ]
        lower_body_visible = [
            visible(name) for name in lower_body_names if visible(name) is not None
        ]

        visible_ys = [p[1] for p in points.values() if p[2] >= self.min_visibility]
        visible_xs = [p[0] for p in points.values() if p[2] >= self.min_visibility]
        if not visible_ys or not visible_xs:
            return None

        body_top_candidates = [min(visible_ys)]
        if eye_y is not None:
            body_top_candidates.append(eye_y)

        ear_y_values = [
            points[n][1]
            for n in ("left_ear", "right_ear")
            if n in points and points[n][2] >= self.min_visibility
        ]
        if ear_y_values:
            body_top_candidates.append(min(ear_y_values))

        body_top_y = min(body_top_candidates)
        body_bottom_y = max(visible_ys)
        body_bottom_confident = len(lower_body_visible) >= 2

        if shoulder_cx is not None and hip_cx is not None:
            body_cx = (shoulder_cx + hip_cx) / 2
        elif shoulder_cx is not None:
            body_cx = shoulder_cx
        elif hip_cx is not None:
            body_cx = hip_cx
        elif eye_cx is not None:
            body_cx = eye_cx
        else:
            body_cx = None

        return {
            "eye_y": eye_y,
            "eye_cx": eye_cx,
            "chin_y": chin_y,
            "shoulder_y": shoulder_y,
            "shoulder_span": shoulder_span,
            "hip_y": hip_y,
            "body_top_y": body_top_y,
            "body_bottom_y": body_bottom_y,
            "body_bottom_confident": body_bottom_confident,
            "body_cx": body_cx,
            "body_min_x": min(visible_xs),
            "body_max_x": max(visible_xs),
        }

    def close(self) -> None:
        if self.detector is not None and hasattr(self.detector, "close"):
            try:
                self.detector.close()
            except Exception:
                pass

def detect_scenes(
    video_path: str,
    threshold: float,
    min_scene_len: int,
    downscale: int = 4,
) -> list[int]:
    """Detects cuts to prevent jarring smooth pans across camera scene changes."""
    try:
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.auto_downscale = False
        scene_manager.downscale = max(1, downscale)
        scene_manager.add_detector(
            AdaptiveDetector(
                adaptive_threshold=threshold,
                min_scene_len=min_scene_len,
            )
        )
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        starts = [1]
        for start_time, _ in scene_list:
            frame_num = getattr(start_time, "frame_num", None)
            if frame_num is None:
                frame_num = start_time.get_frames()
            start_frame = int(frame_num) + 1
            starts.append(start_frame)
        return sorted(set(starts))
    except Exception as exc:
        logging.warning("Scene detection skipped (%s); defaulting to scene 1.", exc)
        return [1]


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


def load_speaker_segments(path: str) -> list[dict]:
    if not path:
        return []
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and "segments" in data:
            return data["segments"]
        if isinstance(data, list):
            return data
    except Exception as exc:
        logging.warning("Failed to load speaker diarization JSON: %s", exc)
    return []


def active_speaker_bonus(
    frame_idx: int,
    fps: float,
    segments: list[dict],
    track_id: Optional[int],
    scene_index: int,
) -> float:
    if track_id is None:
        return 1.0
    t = (frame_idx - 1) / max(fps, 1e-6)
    for seg in segments:
        if (
            seg.get("track_id") == track_id
            and seg.get("scene_index") == scene_index
            and seg.get("start", -1) <= t < seg.get("end", -1)
        ):
            return 1.08
    return 1.0

def build_candidates(
    result: Any,
    frame: np.ndarray,
    saliency_map: np.ndarray,
    ranking_model: SubjectRankingModel,
    allowed_class_ids: list[int],
    class_names: dict[int, str],
    face_helper: MediaPipeFaceHelper,
    pose_helper: Optional[MediaPipePoseHelper],
    state: CameraState,
    fps: float,
    frame_idx: int,
    speaker_segments: list[dict],
    cue_top_k: int = 0,
) -> list[Candidate]:
    h, w = frame.shape[:2]
    frame_cx = w / 2.0
    frame_cy = h / 2.0
    frame_area = float(h * w)
    frame_diag = math.hypot(w, h)

    boxes = result.boxes
    candidates: list[Candidate] = []
    if boxes is None or len(boxes) == 0:
        return candidates
    raw = boxes.data
    raw = raw.detach().cpu().numpy() if hasattr(raw, "detach") else np.asarray(raw)

    parsed_rows = []
    priorities = [] if cue_top_k else None
    for i, row in enumerate(raw):
        try:
            cls_id = int(row[-1])
            if cls_id not in allowed_class_ids:
                continue
            conf = float(row[-2])
            x1, y1, x2, y2 = map(float, row[:4])
            track_id = int(row[4]) if len(row) == 7 else None

            parsed_rows.append((i, cls_id, conf, x1, y1, x2, y2, track_id))
            if cue_top_k and cls_id == CLASS_IDS["person"]:
                score = conf * max(0.0, x2 - x1) * max(0.0, y2 - y1)
                is_tracked = (track_id == state.tracked_id and track_id is not None)
                priorities.append((is_tracked, score, i))
        except Exception:
            logging.debug("Row parsing failed", exc_info=True)
            continue

    if not parsed_rows:
        return candidates

    indices = [item[0] for item in parsed_rows]
    mask_data = getattr(getattr(result, "masks", None), "data", None)
    statistics = mask_statistics(mask_data, indices, (h, w))

    if cue_top_k and priorities:
        eligible = {i for _, _, i in sorted(priorities, reverse=True)[:cue_top_k]}
    elif cue_top_k:
        eligible = set()
    else:
        eligible = set(indices)

    for i, cls_id, conf, x1, y1, x2, y2, track_id in parsed_rows:
        try:
            width = max(1.0, x2 - x1)
            height = max(1.0, y2 - y1)
            area = width * height
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            cls_name = class_names.get(cls_id, str(cls_id))

            mask_area, mask_cx, mask_cy, mask_top_y = statistics.get(
                i, (area, cx, cy, y1)
            )

            framing_cx = mask_cx
            framing_cy = mask_cy
            salient_box = extract_saliency_region(
                saliency_map, (int(x1), int(y1), int(x2), int(y2)), frame_shape=(h, w)
            )
            face_box = None
            pose_data: Optional[dict] = None

            if cls_name == "person":
                if i in eligible:
                    shared_roi = SharedPersonROI(
                        frame, (int(x1), int(y1), int(x2), int(y2))
                    )
                    if pose_helper is not None:
                        pose_data = pose_helper.detect_in_person_box(
                            frame,
                            (int(x1), int(y1), int(x2), int(y2)),
                            track_id,
                            prepared=shared_roi,
                        )
                    face_box = face_helper.detect_in_person_box(
                        frame,
                        (int(x1), int(y1), int(x2), int(y2)),
                        track_id,
                        prepared=shared_roi,
                    )

                if face_box is not None:
                    fx1, fy1, fx2, fy2 = face_box
                    framing_cx = (fx1 + fx2) / 2.0
                    framing_cy = fy1 + (fy2 - fy1) * 0.42
                elif pose_data is not None and pose_data.get("body_cx") is not None:
                    framing_cx = float(pose_data["body_cx"])
                    eye_val = pose_data.get("eye_y")
                    shoulder_val = pose_data.get("shoulder_y")
                    if eye_val is not None and shoulder_val is not None:
                        framing_cy = (float(eye_val) + float(shoulder_val)) / 2.0
                    elif eye_val is not None:
                        framing_cy = float(eye_val)
                    elif shoulder_val is not None:
                        framing_cy = float(shoulder_val)
                    else:
                        framing_cy = mask_cy
                elif salient_box is not None:
                    framing_cx = salient_box[0]
                    framing_cy = salient_box[1]
                else:
                    framing_cx = mask_cx
                    framing_cy = mask_top_y + height * 0.28
            elif salient_box is not None:
                framing_cx = salient_box[0]
                framing_cy = salient_box[1]

            dist_center = math.hypot(
                framing_cx - frame_cx,
                framing_cy - frame_cy,
            )

            speaker_is_active = (
                active_speaker_bonus(
                    frame_idx,
                    fps,
                    speaker_segments,
                    track_id,
                    state.current_scene_index,
                )
                > 1.0
                if cls_name == "person"
                else False
            )
            tracking_match = (
                state.tracked_id is not None
                and track_id == state.tracked_id
                and cls_id == state.tracked_cls_id
            )
            lock_match = (
                state.lock_track_id is not None
                and track_id == state.lock_track_id
                and cls_id == state.lock_cls_id
            )
            score = ranking_model.predict(
                cls_name=cls_name,
                conf=conf,
                mask_area=mask_area,
                frame_area=frame_area,
                dist_center=dist_center,
                frame_diag=frame_diag,
                has_face=face_box is not None,
                has_pose=pose_data is not None,
                saliency_confidence=(
                    salient_box[6] if salient_box is not None else 0.0
                ),
                tracking_match=tracking_match,
                lock_match=lock_match,
                speaker_active=speaker_is_active,
            )

            eye_y_val = None
            chin_y_val = None
            body_top_val = None
            body_bottom_val = None
            body_cx_val = None
            shoulder_span_val = None
            body_bottom_confident_val = False
            body_min_x_val = None
            body_max_x_val = None
            if pose_data is not None:
                eye_y_val = pose_data.get("eye_y")
                chin_y_val = pose_data.get("chin_y")
                body_top_val = pose_data.get("body_top_y")
                body_bottom_val = pose_data.get("body_bottom_y")
                body_cx_val = pose_data.get("body_cx")
                shoulder_span_val = pose_data.get("shoulder_span")
                body_bottom_confident_val = bool(
                    pose_data.get("body_bottom_confident", False)
                )
                body_min_x_val = pose_data.get("body_min_x")
                body_max_x_val = pose_data.get("body_max_x")

            candidates.append(
                Candidate(
                    cls_id=cls_id,
                    cls_name=cls_name,
                    track_id=track_id,
                    conf=conf,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    cx=cx,
                    cy=cy,
                    width=width,
                    height=height,
                    area=area,
                    mask_area=mask_area,
                    mask_cx=mask_cx,
                    mask_cy=mask_cy,
                    mask_top_y=mask_top_y,
                    framing_cx=framing_cx,
                    framing_cy=framing_cy,
                    face_box=face_box,
                    score=score,
                    eye_y=eye_y_val,
                    chin_y=chin_y_val,
                    body_top_y=body_top_val,
                    body_bottom_y=body_bottom_val,
                    body_cx=body_cx_val,
                    shoulder_span=shoulder_span_val,
                    body_bottom_confident=body_bottom_confident_val,
                    body_min_x=body_min_x_val,
                    body_max_x=body_max_x_val,
                    salient_x1=(salient_box[2] if salient_box is not None else None),
                    salient_y1=(salient_box[3] if salient_box is not None else None),
                    salient_x2=(salient_box[4] if salient_box is not None else None),
                    salient_y2=(salient_box[5] if salient_box is not None else None),
                    saliency_confidence=(
                        salient_box[6] if salient_box is not None else 0.0
                    ),
                )
            )
        except Exception:
            logging.debug("Candidate cue extraction failed", exc_info=True)
            continue

    candidates.sort(key=lambda c: c.score, reverse=True)
    if candidates:
        scores = np.asarray([c.score for c in candidates], dtype=np.float64)
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()
        for candidate, relative in zip(candidates, weights):
            absolute = 1.0 - math.exp(-max(0.0, candidate.score) / 4.0)
            candidate.rank_confidence = float(0.5 * absolute + 0.5 * relative)
    return candidates

def choose_subject(
    candidates: list[Candidate],
    state: CameraState,
    lock_first_subject: bool,
    min_subject_hold_frames: int,
    switch_score_threshold: float,
    max_missed_frames: int = 0,
) -> Optional[Candidate]:
    if not candidates:
        return None

    if lock_first_subject and state.lock_track_id is not None:
        for candidate in candidates:
            if (
                candidate.track_id == state.lock_track_id
                and candidate.cls_id == state.lock_cls_id
            ):
                return candidate

    best = candidates[0]
    previous = None
    for candidate in candidates:
        if (
            candidate.track_id == state.tracked_id
            and candidate.cls_id == state.tracked_cls_id
        ):
            previous = candidate
            break

    if (
        previous is None
        and state.tracked_id is not None
        and state.missed_frames < max_missed_frames
    ):
        return None

    if previous is not None:
        if previous.track_id == best.track_id and previous.cls_id == best.cls_id:
            return previous

        if state.frames_since_subject_switch < min_subject_hold_frames:
            if previous.score * switch_score_threshold >= best.score:
                return previous

        if previous.score >= best.score * 0.92:
            return previous

    return best


def choose_two_person_pair(
    candidates: list[Candidate],
    threshold: float,
    currently_active: bool,
) -> Optional[tuple[Candidate, Candidate]]:
    """Evaluates two-person group shot with hysteresis to prevent rapid mode flipping."""
    persons = [c for c in candidates if c.cls_name == "person"]
    if len(persons) < 2:
        return None

    first, second = persons[0], persons[1]
    effective_threshold = (threshold - 0.12) if currently_active else threshold
    if second.score < first.score * effective_threshold:
        return None
    return first, second

def derive_candidate_focus_bounds(
    subject: Candidate,
) -> tuple[float, float, float, float]:
    """Derives visual focus bounding box with proper vertical headroom."""
    if (
        subject.cls_name == "person"
        and subject.body_min_x is not None
        and subject.body_max_x is not None
        and subject.body_top_y is not None
        and subject.body_bottom_y is not None
    ):
        left = float(subject.body_min_x)
        right = float(subject.body_max_x)
        top = float(subject.body_top_y)
        bottom = float(subject.body_bottom_y)

        left = min(left, subject.x1)
        right = max(right, subject.x2)
        if not subject.body_bottom_confident:
            bottom = max(bottom, subject.y2)

        height = max(1.0, bottom - top)
        if subject.eye_y is not None:
            headroom = max(0.0, float(subject.eye_y) - top)
            top -= max(height * 0.12, headroom * 1.35)
        else:
            top -= height * 0.15

        if subject.body_bottom_confident:
            bottom += height * 0.08
        else:
            bottom += height * 0.18
    elif subject.cls_name == "person" and subject.face_box is not None:
        fx1, fy1, fx2, fy2 = subject.face_box
        face_w = max(1.0, fx2 - fx1)
        face_h = max(1.0, fy2 - fy1)
        left = min(subject.x1, fx1 - face_w * 1.15)
        right = max(subject.x2, fx2 + face_w * 1.15)
        top = fy1 - face_h * 1.45
        bottom = max(subject.y2, fy2 + face_h * 4.0)
    else:
        left = subject.x1
        right = subject.x2
        top = subject.mask_top_y
        bottom = subject.y2
        height = max(1.0, bottom - top)
        width = max(1.0, right - left)
        left -= width * 0.14
        right += width * 0.14
        top -= height * 0.14
        bottom += height * 0.16

    if (
        subject.salient_x1 is not None
        and subject.salient_y1 is not None
        and subject.salient_x2 is not None
        and subject.salient_y2 is not None
    ):
        saliency_pad_x = max(8.0, (subject.salient_x2 - subject.salient_x1) * 0.18)
        saliency_pad_y = max(8.0, (subject.salient_y2 - subject.salient_y1) * 0.22)
        left = min(left, subject.salient_x1 - saliency_pad_x)
        top = min(top, subject.salient_y1 - saliency_pad_y)
        right = max(right, subject.salient_x2 + saliency_pad_x)
        bottom = max(bottom, subject.salient_y2 + saliency_pad_y)

    return left, top, right, bottom

def compute_observation_from_bounds(
    bounds: tuple[float, float, float, float],
    confidence: float,
    base_crop_w: int,
    base_crop_h: int,
    frame_w: int,
    frame_h: int,
    min_zoom: float,
    max_zoom: float,
    anchor_y: Optional[float] = None,
) -> CameraObservation:
    left, top, right, bottom = bounds
    left = clamp(left, 0.0, frame_w - 1.0)
    top = clamp(top, 0.0, frame_h - 1.0)
    right = clamp(right, left + 1.0, float(frame_w))
    bottom = clamp(bottom, top + 1.0, float(frame_h))

    region_w = max(1.0, right - left)
    region_h = max(1.0, bottom - top)
    pad_x = max(region_w * 0.12, 16.0)
    pad_y = max(region_h * 0.16, 20.0)

    fit_w = min(frame_w, region_w + pad_x * 2.0)
    fit_h = min(frame_h, region_h + pad_y * 2.0)
    zoom_w = base_crop_w / max(1.0, fit_w)
    zoom_h = base_crop_h / max(1.0, fit_h)
    zoom = clamp(min(zoom_w, zoom_h), min_zoom, max_zoom)

    crop_w, crop_h = current_crop_size(
        base_crop_w,
        base_crop_h,
        zoom,
        frame_w,
        frame_h,
    )

    center_x = clamp((left + right) / 2.0, crop_w / 2.0, frame_w - crop_w / 2.0)

    if anchor_y is not None:
        target_center_y = anchor_y + crop_h * 0.15
        center_y = clamp(target_center_y, crop_h / 2.0, frame_h - crop_h / 2.0)
    else:
        center_y = clamp((top + bottom) / 2.0, crop_h / 2.0, frame_h - crop_h / 2.0)

    return CameraObservation(
        center_x=center_x,
        center_y=center_y,
        zoom=zoom,
        confidence=confidence,
    )


def build_single_subject_observation(
    subject: Candidate,
    base_crop_w: int,
    base_crop_h: int,
    frame_w: int,
    frame_h: int,
    min_zoom: float,
    max_zoom: float,
) -> CameraObservation:
    bounds = derive_candidate_focus_bounds(subject)
    confidence = clamp(
        subject.conf * 0.60
        + subject.rank_confidence * 0.20
        + subject.saliency_confidence * 0.20,
        0.2,
        1.0,
    )

    anchor_y = None
    if subject.eye_y is not None:
        anchor_y = subject.eye_y
    elif subject.face_box is not None:
        anchor_y = (
            subject.face_box[1] + (subject.face_box[3] - subject.face_box[1]) * 0.45
        )

    return compute_observation_from_bounds(
        bounds=bounds,
        confidence=confidence,
        base_crop_w=base_crop_w,
        base_crop_h=base_crop_h,
        frame_w=frame_w,
        frame_h=frame_h,
        min_zoom=min_zoom,
        max_zoom=max_zoom,
        anchor_y=anchor_y,
    )


def build_pair_observation(
    first: Candidate,
    second: Candidate,
    base_crop_w: int,
    base_crop_h: int,
    frame_w: int,
    frame_h: int,
    min_zoom: float,
    max_zoom: float,
) -> CameraObservation:
    first_bounds = derive_candidate_focus_bounds(first)
    second_bounds = derive_candidate_focus_bounds(second)
    union_bounds = (
        min(first_bounds[0], second_bounds[0]),
        min(first_bounds[1], second_bounds[1]),
        max(first_bounds[2], second_bounds[2]),
        max(first_bounds[3], second_bounds[3]),
    )
    confidence = clamp((first.conf + second.conf) / 2.0, 0.25, 1.0)
    eyes = [y for y in (first.eye_y, second.eye_y) if y is not None]
    anchor_y = (sum(eyes) / len(eyes)) if eyes else None

    return compute_observation_from_bounds(
        bounds=union_bounds,
        confidence=confidence,
        base_crop_w=base_crop_w,
        base_crop_h=base_crop_h,
        frame_w=frame_w,
        frame_h=frame_h,
        min_zoom=min_zoom,
        max_zoom=max_zoom,
        anchor_y=anchor_y,
    )


def build_global_saliency_observation(
    saliency_map: np.ndarray,
    base_crop_w: int,
    base_crop_h: int,
    frame_w: int,
    frame_h: int,
    min_zoom: float,
    max_zoom: float,
) -> Optional[CameraObservation]:
    region = extract_saliency_region(
        saliency_map, (0, 0, frame_w, frame_h), frame_shape=(frame_h, frame_w)
    )
    if region is None or region[6] < 0.02:
        return None
    return compute_observation_from_bounds(
        bounds=(region[2], region[3], region[4], region[5]),
        confidence=clamp(region[6], 0.12, 0.55),
        base_crop_w=base_crop_w,
        base_crop_h=base_crop_h,
        frame_w=frame_w,
        frame_h=frame_h,
        min_zoom=min_zoom,
        max_zoom=max_zoom,
    )


def apply_camera_motion(
    state: CameraState,
    observation: CameraObservation,
    args: argparse.Namespace,
    base_crop_w: int,
    base_crop_h: int,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int]:
    dt = 1.0 / args.runtime_fps
    desired_zoom = clamp(
        args.fixed_zoom if args.fixed_zoom is not None else observation.zoom,
        args.min_zoom,
        args.max_zoom,
    )
    if state.is_scene_cut:
        state.zoom = state.target_zoom = desired_zoom
        cw, ch = current_crop_size(
            base_crop_w, base_crop_h, state.zoom, frame_w, frame_h
        )
        state.crop_center_x = clamp(observation.center_x, cw / 2.0, frame_w - cw / 2.0)
        state.crop_center_y = clamp(observation.center_y, ch / 2.0, frame_h - ch / 2.0)
        state.target_center_x, state.target_center_y = (
            state.crop_center_x,
            state.crop_center_y,
        )
        state.velocity_x = state.velocity_y = state.zoom_velocity = 0.0
        state.zoom_pending_seconds = 0.0
        state.is_scene_cut = False
        return cw, ch

    cw, ch = current_crop_size(base_crop_w, base_crop_h, state.zoom, frame_w, frame_h)
    confidence = clamp(observation.confidence, 0.15, 1.0)
    alpha = 1.0 - math.exp(-dt / (args.target_time / (0.55 + 0.65 * confidence)))
    x, y = observation.center_x, observation.center_y

    if abs(x - state.crop_center_x) <= cw * args.dead_zone:
        x = state.crop_center_x
        state.target_center_x = x
        state.velocity_x = 0.0
    if abs(y - state.crop_center_y) <= ch * args.dead_zone:
        y = state.crop_center_y
        state.target_center_y = y
        state.velocity_y = 0.0

    if abs(desired_zoom - state.zoom) <= args.zoom_dead_zone:
        state.zoom_pending_seconds = 0.0
    else:
        state.zoom_pending_seconds += dt

    if args.fixed_zoom is not None:
        state.zoom = state.target_zoom = desired_zoom
        state.zoom_velocity = 0.0
    elif state.zoom_pending_seconds + 1e-9 >= args.zoom_hold_seconds:
        state.target_zoom = lerp(state.target_zoom, desired_zoom, alpha)
        state.zoom, state.zoom_velocity = critically_damped_step(
            state.zoom,
            state.target_zoom,
            state.zoom_velocity,
            dt,
            args.zoom_time,
            args.zoom_speed,
        )
    else:
        state.target_zoom = state.zoom
        state.zoom_velocity = 0.0

    bounded = clamp(state.zoom, args.min_zoom, args.max_zoom)
    if bounded != state.zoom:
        state.zoom_velocity = 0.0
        state.zoom = bounded

    cw, ch = current_crop_size(base_crop_w, base_crop_h, state.zoom, frame_w, frame_h)
    state.target_center_x = clamp(
        lerp(state.target_center_x, x, alpha), cw / 2.0, frame_w - cw / 2.0
    )
    state.target_center_y = clamp(
        lerp(state.target_center_y, y, alpha), ch / 2.0, frame_h - ch / 2.0
    )

    nx, state.velocity_x = critically_damped_step(
        state.crop_center_x,
        state.target_center_x,
        state.velocity_x,
        dt,
        args.pan_time,
        frame_w * args.pan_speed_x * (0.35 + 0.65 * confidence),
    )
    ny, state.velocity_y = critically_damped_step(
        state.crop_center_y,
        state.target_center_y,
        state.velocity_y,
        dt,
        args.pan_time,
        frame_h * args.pan_speed_y * (0.35 + 0.65 * confidence),
    )
    state.crop_center_x = clamp(nx, cw / 2.0, frame_w - cw / 2.0)
    state.crop_center_y = clamp(ny, ch / 2.0, frame_h - ch / 2.0)
    if nx != state.crop_center_x:
        state.velocity_x = 0.0
    if ny != state.crop_center_y:
        state.velocity_y = 0.0
    return cw, ch


def crop_frame(
    frame: np.ndarray,
    center_x: float,
    center_y: float,
    crop_w: int,
    crop_h: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    frame_h, frame_w = frame.shape[:2]
    left = int(round(center_x - crop_w / 2.0))
    top = int(round(center_y - crop_h / 2.0))

    left = int(clamp(left, 0, frame_w - crop_w))
    top = int(clamp(top, 0, frame_h - crop_h))
    right = left + crop_w
    bottom = top + crop_h
    return frame[top:bottom, left:right], (left, top, right, bottom)

def draw_debug(
    frame: np.ndarray,
    crop_rect: tuple[int, int, int, int],
    candidates: list[Candidate],
    subject: Optional[Candidate],
    pair: Optional[tuple[Candidate, Candidate]],
    state: CameraState,
    frame_idx: int,
    total_frames: int,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    """Renders diagnostic overlay maintaining clean letterboxed aspect ratio."""
    left, top, right, bottom = crop_rect
    cw = max(1, right - left)
    ch = max(1, bottom - top)

    reframed_crop = cv2.resize(
        frame[top:bottom, left:right],
        (output_width, output_height),
        interpolation=cv2.INTER_LINEAR,
    )

    scale_x = output_width / float(cw)
    scale_y = output_height / float(ch)

    for candidate in candidates[:6]:
        color = (120, 120, 120)
        thickness = 1
        is_sub = (
            subject is not None
            and candidate.track_id == subject.track_id
            and candidate.cls_id == subject.cls_id
        )
        if is_sub:
            color = (0, 255, 0)
            thickness = 2

        vx1 = int((candidate.x1 - left) * scale_x)
        vy1 = int((candidate.y1 - top) * scale_y)
        vx2 = int((candidate.x2 - left) * scale_x)
        vy2 = int((candidate.y2 - top) * scale_y)
        cv2.rectangle(reframed_crop, (vx1, vy1), (vx2, vy2), color, thickness)

        if candidate.face_box is not None:
            fx1, fy1, fx2, fy2 = candidate.face_box
            vfx1 = int((fx1 - left) * scale_x)
            vfy1 = int((fy1 - top) * scale_y)
            vfx2 = int((fx2 - left) * scale_x)
            vfy2 = int((fy2 - top) * scale_y)
            cv2.rectangle(reframed_crop, (vfx1, vfy1), (vfx2, vfy2), (255, 128, 0), 2)

    if pair is not None:
        for candidate in pair:
            pcx = int((candidate.framing_cx - left) * scale_x)
            pcy = int((candidate.framing_cy - top) * scale_y)
            cv2.circle(reframed_crop, (pcx, pcy), 8, (255, 0, 255), -1)

    hud_sub = reframed_crop[10:161, 10:451]
    hud_bg = hud_sub.copy()
    cv2.rectangle(hud_bg, (0, 0), (440, 150), (20, 20, 20), -1)
    cv2.addWeighted(hud_bg, 0.75, hud_sub, 0.25, 0, hud_sub)

    lines = [
        f"Frame: {frame_idx}/{total_frames if total_frames > 0 else '?'}",
        f"Scene: {state.current_scene_index} | Zoom: {state.zoom:.2f}x",
        f"Tracked ID: {state.tracked_id} ({state.tracked_cls_id})",
        f"Hold Count: {state.frames_since_subject_switch} | Missed: {state.missed_frames}",
    ]
    y = 38
    for line in lines:
        cv2.putText(
            reframed_crop,
            line,
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        y += 28

    return reframed_crop

def has_ffmpeg_encoder(encoder_name: str) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
        )
        return encoder_name in result.stdout
    except Exception:
        return False


def build_video_filters(post_restore: bool) -> Optional[str]:
    filters = []
    if post_restore:
        filters.append("hqdn3d=1.2:1.2:6:6")
        filters.append("unsharp=5:5:0.6:5:5:0.0")
    if not filters:
        return None
    return ",".join(filters)


def run_ffmpeg_mux(
    silent_video_path: str,
    source_input_path: str,
    final_output_path: str,
    video_encoder: str,
    audio_bitrate: str,
    crf: int,
    preset: str,
    vf: Optional[str],
    nvenc_preset: str = "p5",
    video_bitrate: str = "8M",
) -> None:
    """Encodes a lossless intermediate with runtime hardware fallback."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")
    supported = {
        "libx264",
        "libx265",
        "h264_nvenc",
        "hevc_nvenc",
        "h264_videotoolbox",
        "hevc_videotoolbox",
    }
    if video_encoder != "auto" and video_encoder not in supported:
        raise ValueError(
            f"Unsupported encoder: {video_encoder}; choose {sorted(supported)}"
        )
    if video_encoder == "auto":
        encoders = [
            e
            for e in ("h264_nvenc", "h264_videotoolbox", "libx264")
            if has_ffmpeg_encoder(e)
        ]
    else:
        encoders = [video_encoder] if has_ffmpeg_encoder(video_encoder) else []
        if video_encoder != "libx264":
            encoders.append("libx264")
    encoders = list(dict.fromkeys(encoders))
    if not encoders:
        raise RuntimeError("No usable video encoder in this FFmpeg build")
    final_path = Path(final_output_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{final_path.stem}-",
        suffix=final_path.suffix or ".mp4",
        dir=final_path.parent,
    )
    os.close(fd)
    partial = Path(name)
    errors = []
    try:
        for encoder in encoders:
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                silent_video_path,
                "-i",
                source_input_path,
                "-map",
                "0:v:0",
                "-map",
                "1:a?",
                "-c:v",
                encoder,
            ]
            if vf:
                cmd += ["-vf", vf]
            if encoder in {"libx264", "libx265"}:
                cmd += ["-crf", str(crf), "-preset", preset]
            elif encoder.endswith("_nvenc"):
                cmd += [
                    "-cq",
                    str(crf),
                    "-preset",
                    nvenc_preset,
                    "-rc",
                    "vbr",
                    "-b:v",
                    "0",
                ]
            else:
                cmd += ["-b:v", video_bitrate]
            cmd += [
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                audio_bitrate,
                "-shortest",
            ]
            if final_path.suffix.lower() in {".mp4", ".mov", ".m4v"}:
                cmd += ["-movflags", "+faststart"]
            cmd += [str(partial)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                os.replace(partial, final_path)
                logging.info("Encoded %s with %s", final_path, encoder)
                return
            errors.append(f"{encoder}: {result.stderr[-4000:]}")
            logging.warning(
                "Encoder %s failed; trying next available encoder. %s",
                encoder,
                result.stderr[-1200:],
            )
        raise RuntimeError("All encoding attempts failed:\n" + "\n".join(errors))
    finally:
        partial.unlink(missing_ok=True)


def reset_for_new_scene(
    state: CameraState,
    frame_w: int,
    frame_h: int,
    min_zoom: float,
) -> None:
    state.crop_center_x = frame_w / 2.0
    state.crop_center_y = frame_h / 2.0
    state.zoom = min_zoom
    state.target_center_x = frame_w / 2.0
    state.target_center_y = frame_h / 2.0
    state.target_zoom = min_zoom
    state.velocity_x = 0.0
    state.velocity_y = 0.0
    state.zoom_velocity = 0.0
    state.tracked_id = None
    state.tracked_cls_id = None
    state.missed_frames = 0
    state.lock_track_id = None
    state.lock_cls_id = None
    state.frames_since_subject_switch = 0
    state.last_framing_cx = None
    state.last_framing_cy = None
    state.last_subject_key = None
    state.framing_vx = 0.0
    state.framing_vy = 0.0
    state.is_scene_cut = True
    state.two_person_active_frames = 0
    state.zoom_pending_frames = 0
    state.zoom_pending_seconds = 0.0
    state.motion_history.clear()

def process_video(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different files")
    if args.save_debug_preview:
        debug_path = (
            Path(args.debug_path)
            if args.debug_path
            else output_path.with_name(output_path.stem + "_debug.mp4")
        )
        if debug_path.resolve() in {input_path.resolve(), output_path.resolve()}:
            raise ValueError("Debug output must differ from input and final output")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found in PATH")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if not math.isfinite(fps) or fps <= 0 or min(frame_w, frame_h) < 64:
        raise ValueError("Invalid video dimensions or FPS")
    args.runtime_fps = fps
    args.reference_dt = 30.0 / fps
    args.max_missed_frames = round(
        fps
        * (
            args.lost_hold_seconds
            if args.lost_hold_seconds is not None
            else args.max_missed_frames / 30.0
        )
    )
    args.min_subject_hold_frames = round(
        fps
        * (
            args.subject_hold_seconds
            if args.subject_hold_seconds is not None
            else args.min_subject_hold_frames / 30.0
        )
    )
    base_crop_w, base_crop_h = compute_base_crop(
        frame_w, frame_h, args.output_width, args.output_height
    )
    allowed_class_ids = [CLASS_IDS[name] for name in args.classes if name in CLASS_IDS]
    if not allowed_class_ids:
        raise ValueError("No valid classes selected for reframing.")

    speaker_segments = load_speaker_segments(args.speaker_json)
    if args.speaker_aware_mode:
        if any(
            not isinstance(seg, dict)
            or not {"start", "end", "track_id", "scene_index"} <= seg.keys()
            for seg in speaker_segments
        ):
            raise ValueError(
                "Speaker segments require start/end seconds, track_id and scene_index (1-based)"
            )
        if not speaker_segments:
            logging.warning(
                "Speaker-aware mode has no mapped segments and will not affect ranking"
            )
    if args.scene_method == "prepass":
        scene_start_set = set(
            detect_scenes(
                str(input_path),
                args.scene_threshold,
                args.min_scene_len,
                args.scene_downscale,
            )
        )
        inline_scene = None
    else:
        scene_start_set = set()
        inline_scene = InlineSceneDetector(args.min_scene_len, args.scene_threshold)

    for model_file in (args.face_model, args.pose_model):
        if model_file and not Path(model_file).is_file():
            raise FileNotFoundError(f"Explicit MediaPipe model not found: {model_file}")
    model = YOLO(args.seg_model)
    class_names = model.names
    face_helper = MediaPipeFaceHelper(
        min_detection_confidence=0.45,
        model_path=args.face_model,
    )
    pose_helper = MediaPipePoseHelper(
        min_detection_confidence=0.35,
        model_path=args.pose_model,
    )
    saliency_helper = build_saliency_helper(args)
    ranking_model = SubjectRankingModel()
    if pose_helper.detector is None:
        pose_helper = None

    face_helper = CueCache(face_helper, args.cue_interval, fps)
    if pose_helper is not None:
        pose_helper = CueCache(pose_helper, args.cue_interval, fps)
    try:
        with (
            tempfile.TemporaryDirectory(dir=args.temp_dir) as tmpdir,
            ExitStack() as resources,
        ):
            tmpdir_path = Path(tmpdir)
            silent_video_path = tmpdir_path / "silent_vertical.mkv"
            debug_video_path = tmpdir_path / "debug_vertical.mkv"

            size = (args.output_width, args.output_height)
            writer = (
                DirectVideoWriter(
                    output_path,
                    input_path,
                    fps,
                    size,
                    args,
                    build_video_filters(args.post_restore),
                )
                if args.encode_mode == "direct"
                else LosslessWriter(str(silent_video_path), fps, size)
            )
            resources.callback(writer.abort)
            if not writer.isOpened():
                raise RuntimeError("Could not create temporary video writer.")

            debug_writer = None
            final_debug_path = None
            if args.save_debug_preview:
                final_debug_path = (
                    Path(args.debug_path)
                    if args.debug_path
                    else output_path.with_name(output_path.stem + "_debug.mp4")
                )
                debug_writer = (
                    DirectVideoWriter(final_debug_path, input_path, fps, size, args)
                    if args.encode_mode == "direct"
                    else LosslessWriter(str(debug_video_path), fps, size)
                )
                resources.callback(debug_writer.abort)
                if not debug_writer.isOpened():
                    raise RuntimeError("Could not create debug preview writer.")

            state = CameraState(
                crop_center_x=frame_w / 2.0,
                crop_center_y=frame_h / 2.0,
                zoom=args.min_zoom,
                target_center_x=frame_w / 2.0,
                target_center_y=frame_h / 2.0,
                target_zoom=args.min_zoom,
                is_scene_cut=True,
            )

            stats = {
                "frames_processed": 0,
                "scene_resets": 0,
                "subject_switches": 0,
                "frames_with_subject": 0,
                "frames_with_face": 0,
                "frames_with_two_person": 0,
            }

            last_subject_key = None
            scene_index = 0

            frames = iter_video_frames(input_path)
            resources.callback(frames.close)

            for frame_idx, frame in frames:
                is_cut = (
                    inline_scene.update(frame, frame_idx)
                    if inline_scene
                    else frame_idx in scene_start_set
                )
                if is_cut:
                    predictor = getattr(model, "predictor", None)
                    for tracker in getattr(predictor, "trackers", []):
                        tracker.reset()
                    face_helper.clear()
                    if pose_helper is not None:
                        pose_helper.clear()
                result = model.track(
                    source=frame,
                    persist=True,
                    tracker=args.tracker,
                    classes=allowed_class_ids,
                    conf=args.conf,
                    retina_masks=args.retina_masks,
                    verbose=False,
                )[0]

                if is_cut:
                    scene_index += 1
                    state.current_scene_index = scene_index
                    reset_for_new_scene(state, frame_w, frame_h, args.min_zoom)
                    saliency_helper.reset_temporal_state()
                    stats["scene_resets"] += 1
                    last_subject_key = None

                face_helper.frame_idx = frame_idx
                if pose_helper is not None:
                    pose_helper.frame_idx = frame_idx
                saliency_map = saliency_helper.compute_map(frame)
                candidates = build_candidates(
                    result=result,
                    frame=frame,
                    saliency_map=saliency_map,
                    ranking_model=ranking_model,
                    allowed_class_ids=allowed_class_ids,
                    class_names=class_names,
                    face_helper=face_helper,
                    pose_helper=pose_helper,
                    state=state,
                    fps=fps,
                    frame_idx=frame_idx,
                    speaker_segments=(
                        speaker_segments if args.speaker_aware_mode else []
                    ),
                    cue_top_k=args.cue_top_k,
                )

                subject = choose_subject(
                    candidates,
                    state,
                    args.lock_first_subject,
                    args.min_subject_hold_frames,
                    args.switch_score_threshold,
                    args.max_missed_frames,
                )

                pair = None
                if args.two_person_framing:
                    is_pair_active = state.two_person_active_frames > 0
                    pair = choose_two_person_pair(
                        candidates,
                        args.two_person_threshold,
                        is_pair_active,
                    )

                if pair is not None and not pair_fits(
                    pair, base_crop_w, base_crop_h, args.fixed_zoom or args.min_zoom
                ):
                    pair = None

                if pair is not None:
                    stats["frames_with_two_person"] += 1
                    state.two_person_active_frames += 1
                    observation = build_pair_observation(
                        pair[0],
                        pair[1],
                        base_crop_w,
                        base_crop_h,
                        frame_w,
                        frame_h,
                        args.min_zoom,
                        args.max_zoom,
                    )
                    crop_w, crop_h = apply_camera_motion(
                        state,
                        observation,
                        args,
                        base_crop_w,
                        base_crop_h,
                        frame_w,
                        frame_h,
                    )
                    state.missed_frames = 0
                    state.tracked_id = None
                    state.tracked_cls_id = None
                    state.frames_since_subject_switch = 0
                    state.last_framing_cx = None
                    state.last_framing_cy = None
                    state.last_subject_key = None
                    state.motion_history.clear()
                    state.framing_vx = 0.0
                    state.framing_vy = 0.0

                elif subject is not None:
                    state.two_person_active_frames = 0
                    previous_subject_key = (state.tracked_id, state.tracked_cls_id)
                    current_subject_key = (subject.track_id, subject.cls_id)
                    if (
                        last_subject_key is not None
                        and current_subject_key != last_subject_key
                    ):
                        stats["subject_switches"] += 1
                    last_subject_key = current_subject_key

                    stats["frames_with_subject"] += 1
                    if subject.face_box is not None:
                        stats["frames_with_face"] += 1

                    if (
                        args.lock_first_subject
                        and state.lock_track_id is None
                        and subject.track_id is not None
                    ):
                        state.lock_track_id = subject.track_id
                        state.lock_cls_id = subject.cls_id

                    observation = build_single_subject_observation(
                        subject=subject,
                        base_crop_w=base_crop_w,
                        base_crop_h=base_crop_h,
                        frame_w=frame_w,
                        frame_h=frame_h,
                        min_zoom=args.min_zoom,
                        max_zoom=args.max_zoom,
                    )

                    subject_key = (subject.track_id, subject.cls_id)
                    if state.last_subject_key != subject_key or state.missed_frames:
                        state.motion_history.clear()
                    now = (frame_idx - 1) / fps
                    state.motion_history.append(
                        (now, subject.framing_cx, subject.framing_cy)
                    )
                    while (
                        state.motion_history
                        and now - state.motion_history[0][0] > args.prediction_window
                    ):
                        state.motion_history.popleft()
                    vx, vy = regression_velocity(state.motion_history)
                    state.framing_vx = clamp(vx, -frame_w, frame_w)
                    state.framing_vy = clamp(vy, -frame_h, frame_h)
                    lookahead = args.lookahead_seconds * clamp(
                        observation.confidence, 0.0, 1.0
                    )
                    observation.center_x = clamp(
                        observation.center_x + state.framing_vx * lookahead,
                        0.0,
                        float(frame_w),
                    )
                    observation.center_y = clamp(
                        observation.center_y + state.framing_vy * lookahead * 0.5,
                        0.0,
                        float(frame_h),
                    )

                    state.last_framing_cx = subject.framing_cx
                    state.last_framing_cy = subject.framing_cy
                    state.last_subject_key = subject_key

                    crop_w, crop_h = apply_camera_motion(
                        state,
                        observation,
                        args,
                        base_crop_w,
                        base_crop_h,
                        frame_w,
                        frame_h,
                    )

                    state.tracked_id = subject.track_id
                    state.tracked_cls_id = subject.cls_id
                    state.missed_frames = 0
                    if current_subject_key == previous_subject_key:
                        state.frames_since_subject_switch += 1
                    else:
                        state.frames_since_subject_switch = 0
                else:
                    state.two_person_active_frames = 0
                    state.missed_frames += 1
                    state.motion_history.clear()
                    state.framing_vx = state.framing_vy = 0.0
                    observation = build_global_saliency_observation(
                        saliency_map=saliency_map,
                        base_crop_w=base_crop_w,
                        base_crop_h=base_crop_h,
                        frame_w=frame_w,
                        frame_h=frame_h,
                        min_zoom=args.min_zoom,
                        max_zoom=args.max_zoom,
                    )

                    if (
                        not state.is_scene_cut
                        and state.missed_frames <= args.max_missed_frames
                    ):
                        state.target_center_x = state.crop_center_x
                        state.target_center_y = state.crop_center_y
                        state.target_zoom = state.zoom
                        state.velocity_x = state.velocity_y = state.zoom_velocity = 0.0
                        crop_w, crop_h = current_crop_size(
                            base_crop_w, base_crop_h, state.zoom, frame_w, frame_h
                        )
                    elif observation is not None:
                        state.tracked_id = state.tracked_cls_id = None
                        crop_w, crop_h = apply_camera_motion(
                            state,
                            observation,
                            args,
                            base_crop_w,
                            base_crop_h,
                            frame_w,
                            frame_h,
                        )
                    else:
                        crop_w, crop_h = current_crop_size(
                            base_crop_w, base_crop_h, state.zoom, frame_w, frame_h
                        )
                        if state.missed_frames > args.max_missed_frames:
                            state.crop_center_x = lerp(
                                state.crop_center_x,
                                frame_w / 2.0,
                                1 - 0.97**args.reference_dt,
                            )
                            state.crop_center_y = lerp(
                                state.crop_center_y,
                                frame_h / 2.0,
                                1 - 0.97**args.reference_dt,
                            )
                            state.zoom = lerp(
                                state.zoom,
                                args.fixed_zoom or args.min_zoom,
                                1 - 0.95**args.reference_dt,
                            )
                            state.target_zoom = lerp(
                                state.target_zoom,
                                args.fixed_zoom or args.min_zoom,
                                1 - 0.95**args.reference_dt,
                            )
                        state.tracked_id = None
                        state.tracked_cls_id = None
                        state.frames_since_subject_switch = 0

                    state.crop_center_x = clamp(
                        state.crop_center_x,
                        crop_w / 2.0,
                        frame_w - crop_w / 2.0,
                    )
                    state.crop_center_y = clamp(
                        state.crop_center_y,
                        crop_h / 2.0,
                        frame_h - crop_h / 2.0,
                    )

                crop_w, crop_h = current_crop_size(
                    base_crop_w, base_crop_h, state.zoom, frame_w, frame_h
                )
                cropped, crop_rect = crop_frame(
                    frame, state.crop_center_x, state.crop_center_y, crop_w, crop_h
                )
                clean_out = cv2.resize(
                    cropped,
                    (args.output_width, args.output_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                writer.write(clean_out)

                if debug_writer is not None:
                    dbg = draw_debug(
                        frame=frame,
                        crop_rect=crop_rect,
                        candidates=candidates,
                        subject=subject,
                        pair=pair,
                        state=state,
                        frame_idx=frame_idx,
                        total_frames=total_frames,
                        output_width=args.output_width,
                        output_height=args.output_height,
                    )
                    debug_writer.write(dbg)

                stats["frames_processed"] += 1

                if frame_idx % 50 == 0:
                    saliency_telemetry = saliency_helper.get_telemetry()
                    logging.info(
                        "Processed %s/%s | scene=%s | zoom=%.2fx | tracked=%s | saliency=%s",
                        frame_idx,
                        total_frames if total_frames > 0 else "?",
                        state.current_scene_index,
                        state.zoom,
                        state.tracked_id,
                        saliency_telemetry.get("active_backend"),
                    )

            writer.release()
            if debug_writer is not None:
                debug_writer.release()

            if args.encode_mode == "lossless":
                vf = build_video_filters(post_restore=args.post_restore)
                run_ffmpeg_mux(
                    silent_video_path=str(silent_video_path),
                    source_input_path=str(input_path),
                    final_output_path=str(output_path),
                    video_encoder=args.video_encoder,
                    audio_bitrate=args.audio_bitrate,
                    crf=args.crf,
                    preset=args.preset_ffmpeg,
                    vf=vf,
                    nvenc_preset=args.nvenc_preset,
                    video_bitrate=args.video_bitrate,
                )

                if debug_writer is not None and final_debug_path is not None:
                    run_ffmpeg_mux(
                        silent_video_path=str(debug_video_path),
                        source_input_path=str(input_path),
                        final_output_path=str(final_debug_path),
                        video_encoder=args.video_encoder,
                        audio_bitrate=args.audio_bitrate,
                        crf=args.crf,
                        preset=args.preset_ffmpeg,
                        vf=None,
                        nvenc_preset=args.nvenc_preset,
                        video_bitrate=args.video_bitrate,
                    )

            saliency_telemetry = saliency_helper.get_telemetry()
            summary = {
                "preset": args.preset,
                "frames_processed": stats["frames_processed"],
                "scene_resets": stats["scene_resets"],
                "frames_with_subject": stats["frames_with_subject"],
                "frames_with_face": stats["frames_with_face"],
                "frames_with_two_person": stats["frames_with_two_person"],
                "subject_switches": stats["subject_switches"],
                "output_width": args.output_width,
                "output_height": args.output_height,
                "post_restore": args.post_restore,
                "encode_mode": args.encode_mode,
                "scene_method": args.scene_method,
                "retina_masks": args.retina_masks,
                "saliency_active_backend": saliency_telemetry.get("active_backend"),
            }
            summary.update({f"saliency_{k}": v for k, v in saliency_telemetry.items()})
            logging.info("Summary: %s", json.dumps(summary, ensure_ascii=False))
    finally:
        face_helper.close()
        if pose_helper is not None:
            pose_helper.close()

def iter_video_frames(path: Path):
    """Decode sequentially so scene boundaries can reset trackers BEFORE tracking."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
            yield index, frame
    finally:
        cap.release()


def pair_fits(pair, base_w: int, base_h: int, zoom: float) -> bool:
    """Reject pairs whose padded focus regions cannot fit in the available crop."""
    a, b = (derive_candidate_focus_bounds(c) for c in pair)
    width = max(a[2], b[2]) - min(a[0], b[0])
    height = max(a[3], b[3]) - min(a[1], b[1])
    return (
        width + 2 * max(width * 0.12, 16) <= base_w / zoom
        and height + 2 * max(height * 0.16, 20) <= base_h / zoom
    )

class CueCache:
    """Cache per track, remapping cached coordinates to the current person box."""

    def __init__(self, helper, interval: float, fps: float):
        self.helper = helper
        self.interval = max(1, round(interval * fps))
        self.frame_idx = 0
        self.cache = {}

    def clear(self):
        self.cache.clear()

    def close(self):
        self.helper.close()

    def detect_in_person_box(self, frame, box, track_id=None, prepared=None):
        previous = self.cache.get(track_id) if track_id is not None else None
        if previous is not None:
            at, old, result = previous
            ow, oh = max(1, old[2] - old[0]), max(1, old[3] - old[1])
            sx, sy = (box[2] - box[0]) / ow, (box[3] - box[1]) / oh
            stable = (
                0.8 <= sx <= 1.25
                and 0.8 <= sy <= 1.25
                and abs(box[0] - old[0]) < ow * 0.2
                and abs(box[1] - old[1]) < oh * 0.2
            )
            if self.frame_idx - at < self.interval and stable and result is not None:

                def x(v):
                    return box[0] + (v - old[0]) * sx

                def y(v):
                    return box[1] + (v - old[1]) * sy

                if isinstance(result, tuple):
                    return tuple(
                        round(v)
                        for v in (
                            x(result[0]),
                            y(result[1]),
                            x(result[2]),
                            y(result[3]),
                        )
                    )
                mapped = dict(result)
                for key, value in result.items():
                    if value is None or isinstance(value, bool):
                        continue
                    if key.endswith("_y"):
                        mapped[key] = y(value)
                    elif key.endswith("_x") or key.endswith("_cx"):
                        mapped[key] = x(value)
                    elif key == "shoulder_span":
                        mapped[key] = value * sx
                return mapped
        result = self.helper.detect_in_person_box(frame, box, prepared=prepared)
        if track_id is not None:
            self.cache[track_id] = (self.frame_idx, box, result)
        if len(self.cache) > 128:
            self.cache = {
                k: v
                for k, v in self.cache.items()
                if self.frame_idx - v[0] <= self.interval * 2
            }
        return result

class LosslessWriter:
    """BGR frames -> lossless FFV1 temporary file; final encode can safely retry."""

    def __init__(self, path: str, fps: float, size: tuple[int, int]):
        self.width, self.height = size
        self.closed = False
        self.error_log = tempfile.TemporaryFile()
        try:
            self.process = subprocess.Popen(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "bgr24",
                    "-s",
                    f"{self.width}x{self.height}",
                    "-r",
                    str(fps),
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "ffv1",
                    "-level",
                    "3",
                    "-pix_fmt",
                    "bgr0",
                    path,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self.error_log,
            )
        except BaseException:
            self.error_log.close()
            raise

    def isOpened(self):
        return not self.closed and self.process.poll() is None

    def _error(self):
        self.error_log.seek(0)
        return self.error_log.read().decode("utf-8", errors="replace")[-6000:]

    def write(self, frame):
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise ValueError(
                "LosslessWriter expects uint8 BGR frames at output resolution"
            )
        try:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            self.process.wait()
            message = self._error()
            self.abort()
            raise RuntimeError(f"Lossless encoding failed: {message}") from exc

    def release(self):
        if self.closed:
            return
        try:
            self.process.stdin.close()
            code = self.process.wait()
            if code:
                raise RuntimeError(f"Lossless encoding failed: {self._error()}")
        finally:
            self.abort()

    def abort(self):
        if self.closed:
            return
        self.closed = True
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        try:
            if self.process.stdin and not self.process.stdin.closed:
                self.process.stdin.close()
        except BrokenPipeError:
            pass
        self.error_log.close()

def mask_statistics(masks, indices: list[int], frame_shape: tuple[int, int]) -> dict:
    """Reduce only selected low-resolution masks; transfer four numbers per mask."""
    if masks is None or not indices:
        return {}
    height, width = frame_shape
    mh, mw = masks.shape[-2:]
    gain = min(mh / height, mw / width)
    pad_w = (mw - round(width * gain)) / 2
    pad_h = (mh - round(height * gain)) / 2
    left, top = round(pad_w - 0.1), round(pad_h - 0.1)
    right, bottom = mw - round(pad_w + 0.1), mh - round(pad_h + 0.1)
    if right <= left or bottom <= top:
        return {}
    sx, sy = width / (right - left), height / (bottom - top)
    valid = [i for i in indices if i < len(masks)]
    output = {}
    if torch is not None and torch.is_tensor(masks):
        # Optimization: Pre-allocate 1D coordinate tensors once outside the batch loop
        # to prevent redundant device allocations on every 16-mask chunk.
        xs = torch.arange(right - left, device=masks.device, dtype=torch.float32)
        ys = torch.arange(bottom - top, device=masks.device, dtype=torch.float32)
        for start in range(0, len(valid), 16):
            ids = valid[start : start + 16]
            binary = masks[ids, top:bottom, left:right] > 0.5
            rows = binary.sum(dim=2, dtype=torch.float32)
            cols = binary.sum(dim=1, dtype=torch.float32)
            count = rows.sum(dim=1)
            denom = count.clamp_min(1)
            cx = (cols * xs).sum(dim=1) / denom
            cy = (rows * ys).sum(dim=1) / denom
            ytop = (rows > 0).to(torch.int32).argmax(dim=1)
            compact = (
                torch.stack(
                    (
                        count * sx * sy,
                        (cx + 0.5) * sx - 0.5,
                        (cy + 0.5) * sy - 0.5,
                        ytop * sy,
                    ),
                    dim=1,
                )
                .cpu()
                .numpy()
            )
            output.update(
                {i: tuple(map(float, v)) for i, v in zip(ids, compact) if v[0] > 0}
            )
    else:
        for i in valid:
            small = masks[i, top:bottom, left:right]
            if torch is not None and torch.is_tensor(small):
                small = small.detach().numpy()
            binary = np.asarray(small > 0.5, dtype=np.uint8)
            moments = cv2.moments(binary, binaryImage=True)
            count = moments["m00"]
            if count:
                output[i] = (
                    count * sx * sy,
                    (moments["m10"] / count + 0.5) * sx - 0.5,
                    (moments["m01"] / count + 0.5) * sy - 0.5,
                    float(binary.any(axis=1).argmax()) * sy,
                )
    return output

class SharedPersonROI:
    """Lazy single BGR->RGB conversion shared by face and pose on a person."""

    def __init__(self, frame, box, pad_ratio=0.12):
        self.frame, self.box, self.pad_ratio = frame, box, pad_ratio
        self.rgb = None

    def ensure(self):
        if self.rgb is not None:
            return
        h, w = self.frame.shape[:2]
        x1, y1, x2, y2 = self.box
        x1, y1 = int(clamp(x1, 0, w - 1)), int(clamp(y1, 0, h - 1))
        x2, y2 = int(clamp(x2, x1 + 1, w)), int(clamp(y2, y1 + 1, h))
        self.box = x1, y1, x2, y2
        px, py = round((x2 - x1) * self.pad_ratio), round((y2 - y1) * self.pad_ratio)
        self.x, self.y = max(0, x1 - px), max(0, y1 - py)
        self.right, self.bottom = min(w, x2 + px), min(h, y2 + py)
        self.rgb = cv2.cvtColor(
            self.frame[self.y : self.bottom, self.x : self.right], cv2.COLOR_BGR2RGB
        )

    def face_view(self):
        self.ensure()
        x1, y1, x2, y2 = self.box
        upper_h = max(1, int((y2 - y1) * 0.65))
        return (
            np.ascontiguousarray(
                self.rgb[y1 - self.y : y1 - self.y + upper_h, x1 - self.x : x2 - self.x]
            ),
            x1,
            y1,
        )

    def pose_view(self):
        self.ensure()
        return self.rgb, self.x, self.y

class SampledSaliency:
    """Low-res refresh plus sparse optical-flow translation and EMA."""

    def __init__(self, backend, interval=3, max_side=320, ema=0.65):
        self.backend = backend
        self.interval = interval
        self.max_side = max_side
        self.ema = ema
        self.reset_temporal_state()
        self.total_frames = self.refreshes = self.propagated = 0

    def reset_temporal_state(self):
        self.previous_gray = self.current_map = None
        self.counter = 0
        self.backend.reset_temporal_state()

    def compute_map(self, frame):
        h, w = frame.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        small = cv2.resize(
            frame,
            (max(32, round(w * scale)), max(32, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        self.total_frames += 1
        self.counter += 1
        if hasattr(self.backend, "observe_frame"):
            self.backend.observe_frame(small)
        propagated = self.current_map
        if propagated is not None and self.previous_gray is not None:
            points = cv2.goodFeaturesToTrack(
                self.previous_gray, maxCorners=40, qualityLevel=0.02, minDistance=12
            )
            if points is not None:
                new, ok, _ = cv2.calcOpticalFlowPyrLK(
                    self.previous_gray, gray, points, None, winSize=(15, 15), maxLevel=2
                )
                if new is not None and ok is not None and np.count_nonzero(ok) >= 4:
                    displacement = (new - points).reshape(-1, 2)[
                        ok.ravel().astype(bool)
                    ]
                    dx, dy = np.median(displacement, axis=0)
                    if (
                        np.isfinite([dx, dy]).all()
                        and abs(dx) < gray.shape[1] * 0.2
                        and abs(dy) < gray.shape[0] * 0.2
                    ):
                        propagated = cv2.warpAffine(
                            propagated,
                            np.float32([[1, 0, dx], [0, 1, dy]]),
                            (gray.shape[1], gray.shape[0]),
                            borderMode=cv2.BORDER_REPLICATE,
                        )
        if self.current_map is None or (self.counter - 1) % self.interval == 0:
            if hasattr(self.backend, "observe_frame"):
                fresh = self.backend.compute_map(small, ingest=False)
            else:
                fresh = self.backend.compute_map(small)
            fresh = cv2.resize(fresh, (gray.shape[1], gray.shape[0]))
            self.current_map = (
                fresh
                if propagated is None
                else self.ema * fresh + (1 - self.ema) * propagated
            )
            self.refreshes += 1
        else:
            self.current_map = propagated
            self.propagated += 1
        self.previous_gray = gray
        return self.current_map

    def get_telemetry(self):
        result = self.backend.get_telemetry()
        result.update(
            sample_frames_total=self.total_frames,
            sample_refreshes=self.refreshes,
            sample_propagated=self.propagated,
            sample_interval=self.interval,
        )
        return result

class InlineSceneDetector:
    """Causal thumbnail cut detector with adaptive histogram and luminance differencing."""

    def __init__(self, min_frames=15, threshold=3.0):
        self.previous = None
        self.previous_hist = None
        self.history = deque(maxlen=30)
        self.last_cut = -min_frames
        self.min_frames, self.threshold = min_frames, threshold

    def update(self, frame, frame_idx):
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
        cv2.normalize(hist, hist, alpha=1, norm_type=cv2.NORM_L1)
        if self.previous is None:
            cut = True
        else:
            delta = float(np.mean(cv2.absdiff(gray, self.previous))) / 255
            distance = cv2.compareHist(
                hist, self.previous_hist, cv2.HISTCMP_BHATTACHARYYA
            )
            baseline = float(np.median(self.history)) if self.history else 0
            adaptive = max(0.10, baseline * self.threshold)
            cut = frame_idx - self.last_cut >= self.min_frames and (
                (delta > adaptive and distance > 0.25) or distance > 0.72
            )
            self.history.append(delta)
        self.previous, self.previous_hist = gray, hist
        if cut:
            self.last_cut = frame_idx
            self.history.clear()
        return cut

def critically_damped_step(current, target, velocity, dt, tau, max_speed):
    """Exact critical spring solution for a fixed target during dt seconds."""
    omega = 2.0 / max(tau, 1e-4)
    offset = current - target
    term = velocity + omega * offset
    decay = math.exp(-omega * dt)
    value = target + (offset + term * dt) * decay
    new_velocity = (velocity - omega * term * dt) * decay
    delta = clamp(value - current, -max_speed * dt, max_speed * dt)
    value = current + delta
    if (target - current) * (target - value) <= 0:
        return target, 0.0
    return value, clamp(new_velocity, -max_speed, max_speed)


def regression_velocity(history):
    """Robust multi-frame velocity in pixels/second using median pairwise slopes."""
    if len(history) < 3:
        return 0.0, 0.0
    a = np.asarray(history, dtype=np.float64)
    # Use k=1 keyword arg for upper triangle offset (second positional arg in np.triu_indices is m, not k)
    i, j = np.triu_indices(len(a), k=1)
    dt = a[j, 0] - a[i, 0]
    valid = dt > 1e-6
    if not valid.any():
        return 0.0, 0.0
    slopes = (a[j[valid], 1:] - a[i[valid], 1:]) / dt[valid, None]
    return tuple(np.median(slopes, axis=0))


def encoder_options(encoder, args):
    if encoder in {"libx264", "libx265"}:
        return ["-crf", str(args.crf), "-preset", args.preset_ffmpeg]
    if encoder.endswith("_nvenc"):
        return [
            "-cq",
            str(args.crf),
            "-preset",
            args.nvenc_preset,
            "-rc",
            "vbr",
            "-b:v",
            "0",
        ]
    return ["-b:v", args.video_bitrate]

def select_live_encoder(args, fps, size, vf):
    """Initializes encoders with a 2-frame test before starting full render."""
    requested = args.video_encoder
    options = (
        ["h264_nvenc", "h264_videotoolbox", "libx264"]
        if requested == "auto"
        else [requested]
    )
    if "libx264" not in options:
        options.append("libx264")
    errors = []
    for encoder in options:
        if not has_ffmpeg_encoder(encoder):
            continue
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={size[0]}x{size[1]}:r={fps}",
            "-frames:v",
            "2",
        ]
        if vf:
            cmd += ["-vf", vf]
        cmd += [
            "-c:v",
            encoder,
            *encoder_options(encoder, args),
            "-pix_fmt",
            "yuv420p",
            "-f",
            "null",
            "-",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, check=False
            )
            if result.returncode == 0:
                logging.info("Selected live encoder: %s", encoder)
                return encoder
            errors.append(f"{encoder}: {result.stderr[-1000:]}")
        except subprocess.TimeoutExpired:
            errors.append(f"{encoder}: initialization timed out")
        logging.warning("Encoder %s failed initialization; trying fallback", encoder)
    raise RuntimeError("No encoder could initialize:\n" + "\n".join(errors))


class DirectVideoWriter(LosslessWriter):
    """Raw BGR pipe directly into final video encoding and source audio mux in one pass."""

    def __init__(self, destination, source, fps, size, args, vf=None):
        self.width, self.height = size
        self.closed = False
        self.frame_count = 0
        self.destination = Path(destination)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.encoder = select_live_encoder(args, fps, size, vf)
        fd, path = tempfile.mkstemp(
            prefix=f".{self.destination.stem}-",
            suffix=self.destination.suffix or ".mp4",
            dir=self.destination.parent,
        )
        os.close(fd)
        self.partial = Path(path)
        self.error_log = tempfile.TemporaryFile()
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-thread_queue_size",
            "64",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            self.encoder,
            *encoder_options(self.encoder, args),
            "-pix_fmt",
            "yuv420p",
        ]
        if vf:
            cmd += ["-vf", vf]
        cmd += ["-c:a", "aac", "-b:a", args.audio_bitrate, "-shortest"]
        if self.destination.suffix.lower() in {".mp4", ".mov", ".m4v"}:
            cmd += ["-movflags", "+faststart"]
        cmd += [str(self.partial)]
        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self.error_log,
            )
        except BaseException:
            self.error_log.close()
            self.partial.unlink(missing_ok=True)
            raise

    def write(self, frame):
        super().write(frame)
        self.frame_count += 1

    def release(self):
        if self.closed:
            return
        try:
            self.process.stdin.close()
            result = self.process.wait()
            if result or self.frame_count == 0:
                raise RuntimeError(
                    f"Direct encoding failed: {self._error()}. "
                    "Retry with --video-encoder libx264 or --encode-mode lossless"
                )
            os.replace(self.partial, self.destination)
        finally:
            self.abort()

    def abort(self):
        super().abort()
        self.partial.unlink(missing_ok=True)

def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    try:
        args = apply_preset(args)
        process_video(args)
        return 0
    except KeyboardInterrupt:
        logging.error("Interrupted by user.")
        return 130
    except Exception as exc:
        logging.exception("Failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
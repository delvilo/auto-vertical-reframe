from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Literal, Optional, TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import numpy as np

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
    head_box: Optional[tuple[float, float, float, float]]
    has_pose: bool = False
    score: float = 0.0
    rank_confidence: float = 0.5
    eye_y: Optional[float] = None
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

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class FrameContext:
    """One-based frame index, nominal video seconds, original pixel geometry."""
    frame_index: int
    timestamp: float
    width: int
    height: int
    scene_index: int


class PoseCues(TypedDict, total=False):
    head_box: Box | None
    has_pose: bool
    eye_y: float | None
    body_cx: float | None
    shoulder_y: float | None
    shoulder_span: float | None
    body_top_y: float | None
    body_bottom_y: float | None
    body_min_x: float | None
    body_max_x: float | None
    body_bottom_confident: bool


@dataclass(frozen=True)
class PoseDetection:
    """CPU COCO-17 (x, y, confidence), in original full-frame pixels."""
    keypoints: np.ndarray
    cues: PoseCues | None


@dataclass(frozen=True)
class PoseObservation:
    keypoints: np.ndarray | None
    inferred_keypoints: np.ndarray | None
    cues: PoseCues | None
    frame: FrameContext
    inferred_at: FrameContext
    source: Literal["inferred", "remapped"]
    track_id: int | None


@dataclass(frozen=True)
class TrackObservation:
    row_index: int
    cls_id: int
    confidence: float
    box: Box
    track_id: int | None
    mask_statistics: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class FrameObservations:
    frame: FrameContext
    tracks: tuple[TrackObservation, ...] = ()
    poses: dict[int, PoseObservation] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendPrediction:
    """Finite 2-D float32 map in [0,1]; spans the complete input image."""
    map: np.ndarray
    backend: str
    status: Literal["predicted", "warmup", "fallback"] = "predicted"
    reason: str | None = None


@dataclass(frozen=True)
class SaliencyResult:
    """Map pixels span frame.width/height; inferred_at is never advanced by flow.

    A None map explicitly means pose-only composition; no saliency is available.
    backend/status describe the last refresh. EMA can retain older content;
    source and history_weight make that mixture explicit. Arrays are CPU data.
    """
    map: np.ndarray | None
    frame: FrameContext
    inferred_at: FrameContext
    backend: str
    status: Literal["predicted", "warmup", "fallback", "skipped"]
    source: Literal["refresh", "ema", "propagated", "pose_only", "cached"]
    reason: str | None = None
    history_weight: float = 0.0

"""Bounded, CPU-only telemetry for continuous video processing.

Frames are one-based; reported time ranges are half-open nominal-FPS ranges,
not measured presentation timestamps. A scene cut starts a new segment without
changing the global reporting window. Logging never resets perception state.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
from typing import Any


SEGMENT_SCHEMA_VERSION = 1
FRAME_PHASES = ("decode", "scene", "segmentation", "pose", "feedback", "saliency",
                "subjects", "camera", "render_write", "bookkeeping")
SEG_COUNTERS = ("thumbnail_builds", "thumbnail_cache_hits", "thumbnail_backfills",
                "thumbnail_deferred_frames", "flow_attempts")
SALIENCY_COUNTERS = ("frames_total", "frames_backend", "frames_fallback", "prediction_requests",
                     "actual_forward_calls", "observed_frames", "sample_frames_total",
                     "sample_refreshes", "sample_propagated", "deepgaze_requests",
                     "handcrafted_predictions", "cache_hits", "tier_transitions",
                     "tier_pose_only_frames", "tier_handcrafted_frames", "tier_deepgaze_frames")
POSE_COUNTERS = ("rois_inferred", "rois_matched", "cache_hits", "primary_only_frames",
                 "full_scan_frames", "rois_skipped")


def counter_snapshot(stats: dict, segmentation: dict, saliency: dict, pose: Any,
                     pipeline_seconds: dict) -> dict:
    """Copy only cumulative scalars/maps, called at reporting boundaries only."""
    result = {key: value for key, value in stats.items() if key != "frames_processed"}
    for name in SEG_COUNTERS:
        result[f"seg_{name}"] = segmentation.get(name, 0)
    for name in ("refresh_reasons", "gate_diagnostics", "timing_seconds"):
        result[f"seg_{name}"] = dict(segmentation.get(name, {}))
    for name in SALIENCY_COUNTERS:
        result[f"saliency_{name}"] = saliency.get(name, 0)
    for name in ("tier_reasons", "timing_seconds", "deepgaze_timing_seconds"):
        result[f"saliency_{name}"] = dict(saliency.get(name, {}))
    for name in POSE_COUNTERS:
        owner = pose.helper if pose is not None and name.startswith("rois_") and name != "rois_skipped" else pose
        result[f"pose_{name}"] = getattr(owner, name) if owner is not None else 0
    result["pipeline_timing_seconds"] = {key: pipeline_seconds[key] for key in FRAME_PHASES}
    return result


def _difference(current: dict, previous: dict) -> dict:
    result = {}
    for key, value in current.items():
        before = previous.get(key, {} if isinstance(value, dict) else 0)
        if isinstance(value, dict):
            result[key] = _difference(value, before)
        else:
            delta = value - before
            # Accumulators never reset. A tiny float subtraction error is harmless;
            # a real negative or nonfinite counter must remain visible as an error.
            if isinstance(delta, float) and -1e-9 < delta < 0:
                delta = 0.0
            if not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta < 0:
                raise ValueError(f"Invalid segment counter delta: {key}={delta!r}")
            result[key] = delta
    return result


class SegmentStats:
    """Accumulate one row and retain one detection index across time windows."""

    def __init__(self, fps: float, interval_seconds: float, baseline: dict):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Segment FPS must be finite and positive")
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("Segment interval must be finite and positive")
        self.fps = fps
        self.interval_seconds = interval_seconds
        self._window_frames = fps * interval_seconds
        self._baseline = deepcopy(baseline)
        self.emitted = 0
        self.pipeline_timing_seconds = dict.fromkeys(FRAME_PHASES, 0.0)
        self._last_detection: int | None = None
        self._last_scene: int | None = None
        self._last_frame: int | None = None
        self._clear_row()

    def _clear_row(self) -> None:
        self.frame_start: int | None = None
        self.frame_end: int | None = None
        self.scene_index: int | None = None
        self.window_index: int | None = None
        self.detector_calls = self.predicted_frames = 0
        self.gaps: Counter = Counter()
        self.intervals: Counter = Counter()

    @property
    def has_frames(self) -> bool:
        return self.frame_start is not None

    def _window(self, zero_based_frame: int) -> int:
        return math.floor(zero_based_frame / self._window_frames + 1e-10)

    def scene_changed(self, scene_index: int) -> bool:
        return self.has_frames and self.scene_index != scene_index

    def record_frame(self, frame_index: int, scene_index: int, decision: dict) -> None:
        """Record a fully rendered frame, even when YOLO found no detections."""
        if self._last_frame is not None and frame_index != self._last_frame + 1:
            raise ValueError("Segment frames must be contiguous")
        if (not isinstance(decision, dict) or decision.get("frame_index") != frame_index
                or not isinstance(decision.get("detected"), bool)
                or not isinstance(decision.get("interval"), int)
                or decision["interval"] < 1):
            raise ValueError("A successful tracker decision is required for each segment frame")
        window_index = self._window(frame_index - 1)
        if self.has_frames and (scene_index != self.scene_index or window_index != self.window_index):
            raise ValueError("Flush the completed segment before recording a new scene/window")
        if scene_index != self._last_scene:
            self._last_detection = None
        if not self.has_frames:
            self.frame_start = frame_index
            self.scene_index, self.window_index = scene_index, window_index
        self.frame_end = self._last_frame = frame_index
        self._last_scene = scene_index
        if decision["detected"]:
            self.detector_calls += 1
            if self._last_detection is not None:
                self.gaps[str(frame_index - self._last_detection)] += 1
            self._last_detection = frame_index
        else:
            self.predicted_frames += 1
        self.intervals[str(decision["interval"])] += 1

    def window_complete(self) -> bool:
        return self.has_frames and self._window(self.frame_end) != self.window_index

    def finish(self, snapshot: dict, reason: str) -> dict | None:
        """Emit a row without retaining rows, arrays, models or frame objects."""
        if not self.has_frames:
            return None
        if reason not in {"interval", "scene", "end"}:
            raise ValueError("Unknown segment end reason")
        values = _difference(snapshot, self._baseline)
        frame_count = self.frame_end - self.frame_start + 1
        wall = sum(values["pipeline_timing_seconds"].values())
        row = {
            "segment_schema_version": SEGMENT_SCHEMA_VERSION,
            "segment_index": self.emitted + 1,
            "scene_index": self.scene_index,
            "window_index": self.window_index,
            "stats_interval": self.interval_seconds,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "start_seconds": (self.frame_start - 1) / self.fps,
            "end_seconds": self.frame_end / self.fps,
            "source_fps": self.fps,
            "frames": frame_count,
            "end_reason": reason,
            "processing_wall_seconds": wall,
            "processing_fps": frame_count / wall if wall > 0 else None,
            "seg_detector_calls": self.detector_calls,
            "seg_predicted_frames": self.predicted_frames,
            "seg_detector_gap_counts": dict(self.gaps),
            "seg_scheduled_interval_counts": dict(self.intervals),
            "timing_kind": "exclusive_host_wall",
            "timing_scope": "completed_frames_excluding_setup_finalization_segment_logging",
            **values,
        }
        self._baseline = deepcopy(snapshot)
        for key, seconds in values["pipeline_timing_seconds"].items():
            self.pipeline_timing_seconds[key] += seconds
        self.emitted += 1
        self._clear_row()
        return row

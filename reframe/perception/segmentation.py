from __future__ import annotations

import logging
import math
import time
from collections import Counter
from dataclasses import replace
from typing import Any
import cv2
import numpy as np
from ultralytics import YOLO
from reframe.config import AppConfig
from reframe.contracts import FrameContext, FrameObservations, TrackObservation
from reframe.perception.motion import SparseMotion
from reframe.runtime import torch, verify_yolo_device

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

def parse_tracks(result: Any, frame_shape: tuple[int, int], allowed_class_ids: list[int]) -> tuple[TrackObservation, ...]:
    h, w = frame_shape
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return ()
    raw = boxes.data
    raw = raw.detach().cpu().numpy() if hasattr(raw, "detach") else np.asarray(raw)

    parsed_rows = []
    for i, row in enumerate(raw):
        try:
            cls_id = int(row[-1])
            if cls_id not in allowed_class_ids:
                continue
            conf = float(row[-2])
            x1, y1, x2, y2 = map(float, row[:4])
            track_id = int(row[4]) if len(row) == 7 else None

            if not all(math.isfinite(v) for v in (conf, x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
                logging.warning("Ignoring invalid detection box at row=%s", i)
                continue

            parsed_rows.append((i, cls_id, conf, x1, y1, x2, y2, track_id))
        except Exception:
            logging.debug("Row parsing failed", exc_info=True)
            continue

    if not parsed_rows:
        return ()

    indices = [item[0] for item in parsed_rows]
    mask_data = getattr(getattr(result, "masks", None), "data", None)
    statistics = mask_statistics(mask_data, indices, (h, w))

    return tuple(TrackObservation(i, cls_id, conf, (x1, y1, x2, y2), track_id, statistics.get(i))
                 for i, cls_id, conf, x1, y1, x2, y2, track_id in parsed_rows)


class SegmentationTracker:
    """Adaptive real detections with prediction-only ByteTrack steps between them.

    Only true detections establish stability/measurement age. Every source frame
    advances the tracker once; a skipped detector call is never an empty update.
    """
    def __init__(self, config: AppConfig, allowed_class_ids: list[int]):
        self.config, self.allowed_class_ids = config, allowed_class_ids
        self.model = YOLO(config.seg_model)
        if self.model.task != "segment":
            raise ValueError(f"--seg-model requires a segmentation model, got {self.model.task!r}")
        self.class_names = self.model.names
        self.actual_device = None
        self.motion = SparseMotion()
        self._counts = Counter()
        self._reasons = Counter()
        self._gate_diagnostics = Counter()
        self._timings = dict.fromkeys(("thumbnail", "scheduler", "frame_change", "model_track",
                                      "parse_masks", "flow", "predict", "bookkeeping"), 0.0)
        self._total_seconds = 0.0
        self._max_age = 0.0
        self._max_interval = 1
        self._clear()

    def _clear(self):
        self.last_decision = None
        self._previous = None
        self._previous_frame = None
        self._gray = None
        self._tracks = ()
        self._measured = ()
        self._measured_context = None
        self._stable_since = None
        self._interval = 1
        self._force = None
        self._primary_id = None

    def reset(self) -> None:
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", []):
            tracker.reset()
        self._clear()

    def _native(self):
        from ultralytics.trackers.byte_tracker import BYTETracker
        trackers = getattr(getattr(self.model, "predictor", None), "trackers", [])
        # BoT-SORT and custom trackers have other clocks (GMC/ReID, etc.).
        return trackers[0] if len(trackers) == 1 and type(trackers[0]) is BYTETracker else None

    def _timed(self, phase, function, *args, **kwargs):
        """Exclusive host wall time; GPU waits remain charged where they occur."""
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            self._timings[phase] += time.perf_counter() - started

    def _build_gray(self, frame):
        gray = self._timed("thumbnail", self._thumbnail, frame)
        self._counts["thumbnail_builds"] += 1
        return gray

    def _remember(self, frame, context, tracks, gray):
        self._tracks, self._previous, self._gray = tracks, context, gray
        # cap.read supplies a new immutable-to-us array per source frame. Retain
        # one reference only when its thumbnail may need to be backfilled later.
        # A precrop view can retain the full 4K backing array (~25 MB host RAM).
        self._previous_frame = frame if gray is None and self.config.seg_max_gap > 1 else None
        if self._previous_frame is not None:
            self._counts["thumbnail_deferred_frames"] += 1

    def _thumbnail(self, frame):
        h, w = frame.shape[:2]
        scale = min(1., 320 / max(h, w))
        size = (max(8, round(w * scale)), max(8, round(h * scale)))
        if self.config.seg_thumbnail_method == "bgr-area":
            small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # Color conversion and area averaging are linear before integer rounding.
        # Resizing one channel avoids processing all three at full resolution;
        # the full-size gray buffer is temporary CPU memory, never a GPU tensor.
        # Rounding can change pixels slightly, so retain the old comparison path.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

    def _changed(self, gray):
        delta = cv2.absdiff(self._gray, gray)
        # Scan the whole inference area, including regions outside tracked boxes.
        tiles = cv2.resize(delta, (8, 6), interpolation=cv2.INTER_AREA)
        return float(delta.mean()) > 18 or float(tiles.max()) > 35

    @staticmethod
    def _motion_envelope(track, horizon):
        """Conservative CPU-only XYAH envelope; never mutate native Kalman state."""
        mean = np.asarray(track.mean)
        covariance = np.asarray(track.covariance)
        box = np.asarray(track.xyxy, dtype=np.float64)
        if (mean.shape != (8,) or covariance.shape != (8, 8)
                or not np.isfinite(mean).all() or not np.isfinite(covariance).all()
                or not np.isfinite(box).all() or np.any(box[2:] <= box[:2])
                or np.any(np.diag(covariance) < 0)):
            return None
        # TrackState.Lost sets height velocity to zero in native multi_predict.
        # Include center/aspect drift and growing velocity uncertainty over all
        # possible intermediate frames, not only the last observed rectangle.
        from ultralytics.trackers.basetrack import TrackState
        end = mean.copy()
        velocity = mean[4:].copy()
        if track.state != TrackState.Tracked:
            velocity[3] = 0
        end[:4] += horizon * velocity
        if end[2] <= 0 or end[3] <= 0:
            return None
        size = np.maximum(box[2:] - box[:2], (end[2] * end[3], end[3]))
        sigma = np.sqrt(np.diag(covariance))
        # XYAH uncertainty expands dimensions as well as the center. Additive
        # standard deviations are deliberately conservative without assuming
        # independent state components or changing Kalman covariance.
        spread = 3 * (sigma[:4] + horizon * sigma[4:])
        width_extra = (max(mean[3], end[3]) * spread[2]
                       + max(mean[2], end[2]) * spread[3] + spread[2] * spread[3])
        margin = spread[:2] + .5 * np.array([width_extra, spread[3]])
        return np.concatenate((np.minimum(mean[:2], end[:2]) - size / 2 - margin,
                               np.maximum(mean[:2], end[:2]) + size / 2 + margin))

    def _unsettled_detail(self, native):
        """Allow only distant lost bystanders; every observed track stays guarded."""
        if getattr(self.config, "seg_skip_policy", "primary") == "all":
            if native.lost_stracks:
                return "all_lost_tracks"
            if any(not t.is_activated for t in native.tracked_stracks):
                return "all_unconfirmed_tracks"
            if {t.track_id for t in native.tracked_stracks} != {t.track_id for t in self._tracks}:
                return "all_track_set_changed"
            return None

        if self._primary_id is None:
            return "primary_unselected"
        by_id = {t.track_id: t for t in native.tracked_stracks}
        observed_ids = {t.track_id for t in self._tracks}
        if (len(by_id) != len(native.tracked_stracks) or len(observed_ids) != len(self._tracks)
                or None in observed_ids):
            return "ambiguous_track_ids"
        primary = by_id.get(self._primary_id)
        if (primary is None or self._primary_id not in observed_ids
                or any(t.track_id == self._primary_id for t in native.lost_stracks)):
            return "primary_missing_or_lost"
        if not primary.is_activated:
            return "primary_unconfirmed"
        if any(not t.is_activated for t in native.tracked_stracks):
            return "secondary_unconfirmed"
        # Never discard a secondary observation or bypass a newly visible ID.
        # Such tracks still need measurement/stability/flow checks as before.
        if by_id.keys() - observed_ids:
            return "secondary_new_or_unreported"
        if observed_ids - by_id.keys():
            return "secondary_missing"
        if not native.lost_stracks:
            return None
        horizon = min(self.config.seg_max_gap, 3)
        primary_envelope = self._motion_envelope(primary, horizon)
        if primary_envelope is None:
            return "primary_uncertain_motion"
        primary_box = np.asarray(primary.xyxy)
        guard = .5 * (primary_box[2:] - primary_box[:2])
        primary_envelope += np.concatenate((-guard, guard))
        for track in native.lost_stracks:
            envelope = self._motion_envelope(track, horizon)
            if envelope is None:
                return "secondary_lost_uncertain"
            if np.all(np.minimum(envelope[2:], primary_envelope[2:])
                      >= np.maximum(envelope[:2], primary_envelope[:2])):
                return "secondary_lost_near_primary"
        # Counts eligible scheduler visits, not lost objects or actual skips:
        # frame-change, age, confidence and full observed-track flow still run.
        self._gate_diagnostics["distant_lost_allowed"] += 1
        return None

    def _metadata_reason(self):
        """Metadata gates precede thumbnails; image and motion checks still follow."""
        if not self._tracks:
            return "no_tracks", None
        if self._force:
            return self._force, None
        native = self._native()
        if native is None:
            return "unsupported_tracker", None
        detail = self._unsettled_detail(native)
        if detail is not None:
            self._gate_diagnostics[detail] += 1
            return "unsettled_tracks", native
        return None, native

    def _refresh_reason(self, context, native):
        # Keep these AFTER the image-change check. A frame change resets measured
        # stability, whereas a scheduled/age refresh can preserve it.
        if any(t.confidence < .45 for t in self._tracks):
            return "low_confidence"
        age = context.timestamp - self._measured_context.timestamp
        if age >= self.config.seg_max_age - 1e-9:
            return "age_limit"
        if context.frame_index - self._measured_context.frame_index >= self._interval:
            return "scheduled"
        for track in native.tracked_stracks:
            if (not np.isfinite(track.mean).all() or not np.isfinite(track.covariance).all()
                    or np.sqrt(np.maximum(0, np.diag(track.covariance)[:2])).max()
                    > .5 * min(track.xyxy[2:] - track.xyxy[:2])):
                return "uncertain_motion"
        return None

    def _update_stability(self, tracks, context):
        previous = {t.track_id: t for t in self._measured}
        stable = (bool(tracks) and None not in previous
                  and {t.track_id for t in tracks} == previous.keys())
        if stable:
            for track in tracks:
                old = np.asarray(previous[track.track_id].box)
                new = np.asarray(track.box)
                size = np.maximum(old[2:] - old[:2], 1)
                if (track.confidence < .45 or np.max(np.abs(new - old) / np.tile(size, 2)) > .12
                        or np.max(np.abs((new[2:] - new[:2]) / size - 1)) > .08):
                    stable = False
                    break
        if not stable or self._stable_since is None:
            self._stable_since = context.timestamp
        duration = context.timestamp - self._stable_since
        self._interval = min(self.config.seg_max_gap, 3 if duration >= .25 else 2 if duration >= .1 else 1)
        self._max_interval = max(self._max_interval, self._interval)
        self._measured, self._measured_context = tracks, context

    def _predict(self, proposals):
        native = self._native()
        # Commit only after all flow proposals pass. Preserve covariance growth;
        # optical flow is a correction, not a new Kalman detector measurement.
        native.frame_id += 1
        # A far-away lost track may now coexist with a predicted frame. Advance
        # its Kalman state too, exactly once per source frame, so re-association
        # uses the correct motion/clock at the next real detection. Native
        # update still owns expiration and matching; no empty detection update.
        tracked_ids = {t.track_id for t in native.tracked_stracks}
        pool = [*native.tracked_stracks,
                *(t for t in native.lost_stracks if t.track_id not in tracked_ids)]
        native.multi_predict(pool)
        by_id = {t.track_id: t for t in native.tracked_stracks}
        output = []
        for track in self._tracks:
            dx, dy, quality = proposals[track.track_id]
            x1, y1, x2, y2 = track.box
            native_track = by_id[track.track_id]
            native_track.mean[:2] = ((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy)
            stats = track.mask_statistics
            if stats is not None:
                stats = (stats[0], stats[1] + dx, stats[2] + dy, stats[3] + dy)
            output.append(replace(track, box=(x1 + dx, y1 + dy, x2 + dx, y2 + dy),
                                  mask_statistics=stats, source="predicted",
                                  prediction_confidence=min(track.prediction_confidence, quality)))
        return tuple(output)

    def track(self, frame: np.ndarray, context: FrameContext | None = None) -> tuple[TrackObservation, ...]:
        """Callers must not mutate a supplied frame before the following call."""
        self.last_decision = None
        started = time.perf_counter()
        before = sum(self._timings.values())
        try:
            return self._track(frame, context)
        finally:
            elapsed = time.perf_counter() - started
            measured = sum(self._timings.values()) - before
            self._timings["bookkeeping"] += max(0., elapsed - measured)
            self._total_seconds += elapsed

    def _track(self, frame, context):
        args = self.config
        if context is None:
            index = self._previous.frame_index + 1 if self._previous else 1
            context = FrameContext(index, (index - 1) / args.runtime_fps,
                                   frame.shape[1], frame.shape[0], 0)
        if (context.width, context.height) != (frame.shape[1], frame.shape[0]):
            raise ValueError("Tracking context must match inference frame dimensions")
        if self._previous and (context.frame_index != self._previous.frame_index + 1
                              or context.timestamp <= self._previous.timestamp
                              or (context.width, context.height, context.scene_index)
                              != (self._previous.width, self._previous.height, self._previous.scene_index)):
            self.reset()
        gray = (self._build_gray(frame)
                if args.seg_max_gap > 1 and args.seg_thumbnail_mode == "eager" else None)
        reason = "every_frame" if args.seg_max_gap == 1 else "initial"
        if args.seg_max_gap > 1 and self._previous:
            reason, native = self._timed("scheduler", self._metadata_reason)
            if reason is None:
                if self._gray is None:
                    if self._previous_frame is None:
                        # Defensive recovery if a caller enables adaptive mode
                        # midstream. Never substitute a non-adjacent gray frame.
                        reason = "missing_history"
                    else:
                        self._gray = self._build_gray(self._previous_frame)
                        self._counts["thumbnail_backfills"] += 1
                else:
                    self._counts["thumbnail_cache_hits"] += 1
                if reason is None:
                    if gray is None:
                        gray = self._build_gray(frame)
                    reason = ("frame_change" if self._timed("frame_change", self._changed, gray)
                              else self._timed("scheduler", self._refresh_reason, context, native))
            if reason is None:
                self._counts["flow_attempts"] += 1
                proposals, reason = self._timed("flow", self.motion.estimate, self._gray, gray,
                                               self._tracks, context.width, context.height)
                if reason is None and any(v[2] < .7 for v in proposals.values()):
                    reason = "flow_low_quality"
                if reason is None:
                    tracks = self._timed("predict", self._predict, proposals)
                    self._counts["predicted_frames"] += 1
                    self._max_age = max(self._max_age, context.timestamp - self._measured_context.timestamp)
                    self._remember(frame, context, tracks, gray)
                    self.last_decision = {"frame_index": context.frame_index, "detected": False,
                                          "reason": None, "interval": self._interval}
                    return tracks
        self._reasons[reason] += 1
        if reason not in {"scheduled", "age_limit", "every_frame"}:
            self._stable_since = None
            self._interval = 1
        result = self._timed("model_track", self.model.track, source=frame, persist=True,
                             tracker=args.tracker, classes=self.allowed_class_ids, conf=args.conf,
                             retina_masks=args.retina_masks, device=args.yolo_device, verbose=False)[0]
        if self.actual_device is None:
            self.actual_device = verify_yolo_device(self.model, args.yolo_device)
        tracks = tuple(replace(t, measured_at=context) for t in
                       self._timed("parse_masks", parse_tracks, result, frame.shape[:2],
                                   self.allowed_class_ids))
        self._counts["detector_calls"] += 1
        self._timed("scheduler", self._update_stability, tracks, context)
        self._remember(frame, context, tracks, gray)
        self._force = None
        self.last_decision = {"frame_index": context.frame_index, "detected": True,
                              "reason": reason, "interval": self._interval}
        return tracks

    def feedback(self, observations: FrameObservations, preferred_track_id: int | None = None) -> None:
        """Weak fresh primary pose requests a refresh; cached pose cannot raise interval."""
        if self._primary_id is not None and preferred_track_id != self._primary_id:
            self._force = "primary_changed"
        self._primary_id = preferred_track_id
        primary = next((t for t in observations.tracks if t.track_id == preferred_track_id), None)
        if preferred_track_id is not None and primary is None:
            self._force = "primary_missing"
        elif primary is not None and primary.cls_id == 0:
            pose = observations.poses.get(primary.row_index)
            if (pose is not None and pose.source == "inferred"
                    and pose.inferred_at == observations.frame
                    and (not pose.cues or not pose.cues.get("has_pose"))):
                self._force = "weak_primary_pose"

    def telemetry(self) -> dict:
        return {"detector_calls": self._counts["detector_calls"],
                "predicted_frames": self._counts["predicted_frames"],
                "flow_seconds": self._timings["flow"],
                "timing_seconds": dict(self._timings),
                "total_seconds": self._total_seconds,
                "thumbnail_mode": self.config.seg_thumbnail_mode,
                "thumbnail_method": self.config.seg_thumbnail_method,
                "thumbnail_builds": self._counts["thumbnail_builds"],
                "thumbnail_cache_hits": self._counts["thumbnail_cache_hits"],
                "thumbnail_backfills": self._counts["thumbnail_backfills"],
                "thumbnail_deferred_frames": self._counts["thumbnail_deferred_frames"],
                "flow_attempts": self._counts["flow_attempts"],
                "max_prediction_age_seconds": self._max_age,
                "max_interval": self._max_interval,
                "skip_policy": getattr(self.config, "seg_skip_policy", "primary"),
                "gate_diagnostics": dict(self._gate_diagnostics),
                "refresh_reasons": dict(self._reasons)}

    def close(self) -> None:
        self._clear()
        self.model = None

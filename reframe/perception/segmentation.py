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
        self._flow_seconds = 0.0
        self._max_age = 0.0
        self._max_interval = 1
        self._clear()

    def _clear(self):
        self._previous = None
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

    @staticmethod
    def _thumbnail(frame):
        h, w = frame.shape[:2]
        scale = min(1., 320 / max(h, w))
        small = cv2.resize(frame, (max(8, round(w * scale)), max(8, round(h * scale))),
                           interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def _changed(self, gray):
        delta = cv2.absdiff(self._gray, gray)
        # Scan the whole inference area, including regions outside tracked boxes.
        tiles = cv2.resize(delta, (8, 6), interpolation=cv2.INTER_AREA)
        return float(delta.mean()) > 18 or float(tiles.max()) > 35

    def _refresh_reason(self, context, gray):
        if not self._tracks:
            return "no_tracks"
        if self._force:
            return self._force
        native = self._native()
        if native is None:
            return "unsupported_tracker"
        if (native.lost_stracks or any(not t.is_activated for t in native.tracked_stracks)
                or {t.track_id for t in native.tracked_stracks} != {t.track_id for t in self._tracks}):
            return "unsettled_tracks"
        if self._changed(gray):
            return "frame_change"
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
        native.multi_predict(native.tracked_stracks)
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
        gray = self._thumbnail(frame) if args.seg_max_gap > 1 else None
        reason = "every_frame" if args.seg_max_gap == 1 else "initial"
        if args.seg_max_gap > 1 and self._previous:
            reason = self._refresh_reason(context, gray)
            if reason is None:
                started = time.perf_counter()
                proposals, reason = self.motion.estimate(self._gray, gray, self._tracks,
                                                          context.width, context.height)
                self._flow_seconds += time.perf_counter() - started
                if reason is None and any(v[2] < .7 for v in proposals.values()):
                    reason = "flow_low_quality"
                if reason is None:
                    tracks = self._predict(proposals)
                    self._counts["predicted_frames"] += 1
                    self._max_age = max(self._max_age, context.timestamp - self._measured_context.timestamp)
                    self._tracks, self._previous, self._gray = tracks, context, gray
                    return tracks
        self._reasons[reason] += 1
        if reason not in {"scheduled", "age_limit", "every_frame"}:
            self._stable_since = None
            self._interval = 1
        result = self.model.track(source=frame, persist=True, tracker=args.tracker,
                                 classes=self.allowed_class_ids, conf=args.conf,
                                 retina_masks=args.retina_masks, device=args.yolo_device,
                                 verbose=False)[0]
        if self.actual_device is None:
            self.actual_device = verify_yolo_device(self.model, args.yolo_device)
        tracks = tuple(replace(t, measured_at=context) for t in
                       parse_tracks(result, frame.shape[:2], self.allowed_class_ids))
        self._counts["detector_calls"] += 1
        self._update_stability(tracks, context)
        self._tracks, self._previous, self._gray = tracks, context, gray
        self._force = None
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
                "flow_seconds": self._flow_seconds,
                "max_prediction_age_seconds": self._max_age,
                "max_interval": self._max_interval,
                "refresh_reasons": dict(self._reasons)}

    def close(self) -> None:
        self.model = None

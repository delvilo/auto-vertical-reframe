from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional
import numpy as np
from ultralytics import YOLO
from reframe.config import AppConfig, CLASS_IDS
from reframe.contracts import (CameraState, FrameContext, FrameObservations, TrackObservation,
                               PoseDetection, PoseObservation, PoseCues)
from reframe.geometry import box_iou
from reframe.runtime import verify_yolo_device

def pose_owner(box, person_boxes) -> Optional[int]:
    """Match to segmentation rows, never to an independent pose tracker ID.

    Every visible person participates, including people excluded by cue-top-k.
    Reject ambiguous overlap rather than borrowing a neighbour's skeleton.
    """
    scores = sorted(((box_iou(box, target), index)
                     for index, target in person_boxes.items()), reverse=True)
    if not scores or scores[0][0] < 0.30:
        return None
    if len(scores) > 1 and scores[0][0] - scores[1][0] < 0.10:
        return None
    return scores[0][1]


def pose_cues(keypoints, person_box, frame_shape, min_confidence) -> PoseCues | None:
    """Convert COCO-17 pixel coordinates to conservative framing cues.

    Head bounds are estimates, not face detections. Keep the segmentation box's
    bottom even when ankles are visible: COCO has neither heels nor toes.
    """
    points = np.asarray(keypoints, dtype=np.float64)
    if points.shape != (17, 3):
        raise ValueError(f"Expected COCO keypoints (17, 3), got {points.shape}")
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = person_box
    pw, ph = max(1.0, x2 - x1), max(1.0, y2 - y1)
    valid = (np.isfinite(points).all(axis=1) & (points[:, 2] >= min_confidence)
             & (points[:, 0] >= max(0, x1 - pw * 0.12))
             & (points[:, 0] < min(width, x2 + pw * 0.12))
             & (points[:, 1] >= max(0, y1 - ph * 0.12))
             & (points[:, 1] < min(height, y2 + ph * 0.12)))
    if valid.sum() < 2:
        return None

    def midpoint(indices):
        selected = [index for index in indices if valid[index]]
        return points[selected, :2].mean(axis=0) if selected else None

    eyes = midpoint((1, 2))
    anchor = eyes if eyes is not None else (points[0, :2] if valid[0] else None)
    shoulders, hips = midpoint((5, 6)), midpoint((11, 12))
    shoulder_span = abs(points[5, 0] - points[6, 0]) if valid[5] and valid[6] else None
    facial = points[:5, :2][valid[:5]]
    head_box = None
    if len(facial) >= 2:
        center = anchor if anchor is not None else facial.mean(axis=0)
        estimates = [float(np.ptp(facial[:, 0])) * 1.5, pw * 0.18]
        if valid[1] and valid[2]:
            estimates.append(float(np.linalg.norm(points[1, :2] - points[2, :2])) * 3.0)
        if valid[3] and valid[4]:
            estimates.append(float(np.linalg.norm(points[3, :2] - points[4, :2])) * 1.2)
        if shoulder_span is not None:
            estimates.append(float(shoulder_span) * 0.38)
        hw = min(max(estimates), pw * 0.75, ph * 0.65)
        hh = hw * 1.25
        head_box = (max(0.0, float(center[0] - hw / 2)),
                    max(0.0, float(center[1] - hh * 0.45)),
                    min(float(width), float(center[0] + hw / 2)),
                    min(float(height), float(center[1] + hh * 0.55)))

    has_body = int(valid[5:].sum()) >= 2 and (shoulders is not None or hips is not None)
    if not has_body and head_box is None:
        return None
    body_center = (shoulders + hips) / 2 if shoulders is not None and hips is not None else (
        shoulders if shoulders is not None else hips
    )
    if body_center is None:
        body_center = anchor
    selected = points[valid, :2]
    return {
        "head_box": head_box,
        "has_pose": bool(has_body),
        "eye_y": float(anchor[1]) if anchor is not None else None,
        "body_cx": float(body_center[0]) if body_center is not None else None,
        "shoulder_y": float(shoulders[1]) if shoulders is not None else None,
        "shoulder_span": float(shoulder_span) if shoulder_span is not None else None,
        "body_top_y": float(min(y1, selected[:, 1].min(),
                                head_box[1] if head_box else y1)) if has_body else None,
        "body_bottom_y": float(max(y2, selected[:, 1].max())) if has_body else None,
        "body_min_x": float(min(x1, selected[:, 0].min())) if has_body else None,
        "body_max_x": float(max(x2, selected[:, 0].max())) if has_body else None,
        "body_bottom_confident": bool(valid[15] and valid[16]),
    }

class YOLOPoseHelper:
    """One persistent COCO-17 model, batched BGR ROIs, no separate face model."""

    def __init__(self, args: AppConfig):
        path = Path(args.pose_model)
        if path.suffix.lower() != ".pt":
            raise ValueError("--pose-model requires a COCO-17 YOLO .pt checkpoint")
        if args.pose_model != "yolo26n-pose.pt" and not path.is_file():
            raise FileNotFoundError(f"Pose checkpoint not found: {path}")
        self.model = YOLO(args.pose_model)
        if self.model.task != "pose":
            raise ValueError(f"Expected pose model, got task={self.model.task!r}")
        shape = getattr(self.model.model.model[-1], "kpt_shape", None)
        if shape is None or tuple(shape) != (17, 3):
            raise ValueError(f"Pose checkpoint must use COCO-17 (17, 3), got {shape}")
        if self.model.names != {0: "person"}:
            raise ValueError("Pose checkpoint must describe COCO person keypoints")
        self.device = args.yolo_device
        self.imgsz, self.conf = args.pose_imgsz, args.pose_conf
        self.keypoint_conf, self.batch_size = args.keypoint_conf, args.pose_batch_size
        self.actual_device = None
        self.rois_inferred = self.rois_matched = 0
        logging.info("YOLO pose loaded: model=%s device=%s imgsz=%s batch_size=%s "
                     "detection_conf=%s keypoint_conf=%s", args.pose_model, self.device,
                     self.imgsz, self.batch_size, self.conf, self.keypoint_conf)

    def detect_many(self, frame, requests, person_boxes):
        """requests: (segmentation row index, person box, track ID)."""
        output = {index: None for index, _, _ in requests}
        height, width = frame.shape[:2]
        prepared = []
        for index, box, track_id in requests:
            x1, y1, x2, y2 = box
            px, py = (x2 - x1) * 0.12, (y2 - y1) * 0.12
            left, top = max(0, math.floor(x1 - px)), max(0, math.floor(y1 - py))
            right, bottom = min(width, math.ceil(x2 + px)), min(height, math.ceil(y2 + py))
            if left >= right or top >= bottom:
                continue
            image = np.ascontiguousarray(frame[top:bottom, left:right])
            prepared.append((index, box, track_id, left, top, image))
            logging.debug("Pose ROI track=%s x=%s y=%s width=%s height=%s BGR device=%s",
                          track_id, left, top, right - left, bottom - top, self.device)
        for start in range(0, len(prepared), self.batch_size):
            batch = prepared[start:start + self.batch_size]
            try:
                results = self.model.predict(source=[item[5] for item in batch],
                                             device=self.device, imgsz=self.imgsz,
                                             conf=self.conf, verbose=False)
                if self.actual_device is None:
                    self.actual_device = verify_yolo_device(self.model, self.device, "pose")
                if len(results) != len(batch):
                    raise RuntimeError("Pose batch result count does not match ROI count")
                self.rois_inferred += len(batch)
                for item, result in zip(batch, results):
                    index, box, track_id, left, top, _ = item
                    if result.boxes is None or len(result.boxes) == 0:
                        continue
                    if result.keypoints is None:
                        raise ValueError("Pose model returned boxes without keypoints")
                    boxes = result.boxes.xyxy.detach().cpu().numpy()
                    points = result.keypoints.data.detach().cpu().numpy()
                    if points.shape != (len(boxes), 17, 3):
                        raise ValueError(f"Invalid pose output shape: {points.shape}")
                    matches = []
                    for row, local_box in enumerate(boxes):
                        global_box = local_box + np.array([left, top, left, top])
                        if not np.isfinite(global_box).all() or pose_owner(global_box, person_boxes) != index:
                            continue
                        matches.append((box_iou(global_box, box), row))
                    matches.sort(reverse=True)
                    if not matches or (len(matches) > 1 and matches[0][0] - matches[1][0] < 0.10):
                        logging.debug("Pose association missing/ambiguous for track=%s", track_id)
                        continue
                    keypoints = points[matches[0][1]].copy()
                    keypoints[:, :2] += np.array([left, top])
                    output[index] = PoseDetection(keypoints, pose_cues(keypoints, box, frame.shape, self.keypoint_conf))
                    self.rois_matched += 1
            except Exception:
                logging.exception("YOLO pose inference failed: tracks=%s device=%s",
                                  [item[2] for item in batch], self.device)
                raise
        return output

    def close(self):
        self.model = None


class PoseCueCache:
    """Batch expired tracks and remap cached head/body cues; never retain tensors."""

    def __init__(self, helper, interval, fps):
        self.helper = helper
        self.interval = max(1, round(interval * fps))
        self.fps = fps
        self.cache = {}
        self.cache_hits = 0

    def clear(self):
        self.cache.clear()

    def close(self):
        self.clear()
        self.helper.close()

    @staticmethod
    def remap_cues(result, old, box):
        if result is None:
            return None
        sx, sy = (box[2] - box[0]) / (old[2] - old[0]), (box[3] - box[1]) / (old[3] - old[1])
        def x(v):
            return box[0] + (v - old[0]) * sx
        def y(v):
            return box[1] + (v - old[1]) * sy
        mapped = dict(result)
        for key, value in result.items():
            if value is None or isinstance(value, bool):
                continue
            if key == "head_box":
                mapped[key] = (x(value[0]), y(value[1]), x(value[2]), y(value[3]))
            elif key.endswith("_y"):
                mapped[key] = y(value)
            elif key.endswith("_x") or key.endswith("_cx"):
                mapped[key] = x(value)
            elif key == "shoulder_span":
                mapped[key] = value * sx
        return mapped

    def get_many(self, frame, requests, person_boxes, context: FrameContext) -> dict[int, PoseObservation]:
        frame_idx = context.frame_index
        output, pending = {}, []
        self.cache = {key: value for key, value in self.cache.items()
                      if 0 <= frame_idx - value[0] < max(2, self.interval * 2)}
        for index, box, track_id in requests:
            previous = self.cache.get(track_id) if track_id is not None else None
            crowded = any(other != index and box_iou(box, other_box) > 0.35
                          for other, other_box in person_boxes.items())
            if previous is not None and not crowded:
                at, old, result = previous
                ow, oh = max(1.0, old[2] - old[0]), max(1.0, old[3] - old[1])
                sx, sy = (box[2] - box[0]) / ow, (box[3] - box[1]) / oh
                stable = (0.8 <= sx <= 1.25 and 0.8 <= sy <= 1.25
                          and abs(box[0] - old[0]) < ow * 0.2
                          and abs(box[1] - old[1]) < oh * 0.2)
                if 0 <= frame_idx - at < self.interval and stable:
                    output[index] = self.remap_observation(result, old, box, context)
                    self.cache_hits += 1
                    continue
            pending.append((index, box, track_id))
        if pending:
            detections = self.helper.detect_many(frame, pending, person_boxes)
            fresh = {}
            for index, box, track_id in pending:
                detected = detections[index]
                points = detected.keypoints if detected is not None else None
                fresh[index] = PoseObservation(points, points, detected.cues if detected else None,
                                               context, context, "inferred", track_id)
            output.update(fresh)
            for index, box, track_id in pending:
                if track_id is not None:
                    self.cache[track_id] = (frame_idx, box, fresh[index])
            while len(self.cache) > 128:
                self.cache.pop(next(iter(self.cache)))
        return output

    @classmethod
    def remap_observation(cls, result: PoseObservation, old, box, context: FrameContext) -> PoseObservation:
        points = None
        if result.keypoints is not None:
            points = result.keypoints.copy()
            points[:, 0] = box[0] + (points[:, 0] - old[0]) * (box[2] - box[0]) / (old[2] - old[0])
            points[:, 1] = box[1] + (points[:, 1] - old[1]) * (box[3] - box[1]) / (old[3] - old[1])
        return PoseObservation(points, result.inferred_keypoints,
                               cls.remap_cues(result.cues, old, box), context,
                               result.inferred_at, "remapped", result.track_id)


def observe_poses(frame: np.ndarray, tracks: tuple[TrackObservation, ...],
                  helper: PoseCueCache | None, state: CameraState, context: FrameContext,
                  cue_top_k: int = 0) -> FrameObservations:
    people = [track for track in tracks if track.cls_id == CLASS_IDS["person"]]
    person_boxes = {track.row_index: track.box for track in people}
    if cue_top_k:
        def priority(track):
            x1, y1, x2, y2 = track.box
            return (track.track_id is not None and track.track_id == state.tracked_id,
                    track.confidence * (x2 - x1) * (y2 - y1), track.row_index)
        people = sorted(people, key=priority, reverse=True)[:cue_top_k]
    requests = [(track.row_index, track.box, track.track_id) for track in people]
    poses = helper.get_many(frame, requests, person_boxes, context) if helper is not None else {}
    return FrameObservations(context, tracks, poses)

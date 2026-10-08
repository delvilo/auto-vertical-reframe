"""Conservative translation proposals from small grayscale frames; no tracking state."""
from __future__ import annotations

import cv2
import numpy as np

from reframe.contracts import TrackObservation


class SparseMotion:
    """Return inference-coordinate translations, or reject the complete proposal.

    Gray inputs share the same geometry and span ``frame_width x frame_height``.
    Inputs and observations are never modified. Quality is the inlier fraction;
    it describes flow consistency, not detector confidence or a new measurement.
    """

    def __init__(self, max_jump_fraction: float = .15):
        self.max_jump_fraction = max_jump_fraction

    @staticmethod
    def _crowded(boxes):
        for i, box in enumerate(boxes):
            for other in boxes[i + 1:]:
                overlap = np.maximum(0, np.minimum(box[2:], other[2:]) - np.maximum(box[:2], other[:2]))
                intersection = float(np.prod(overlap))
                union = np.prod(box[2:] - box[:2]) + np.prod(other[2:] - other[:2]) - intersection
                if intersection / union > .35:
                    return True
        return False

    def estimate(self, previous_gray: np.ndarray, current_gray: np.ndarray,
                 tracks: tuple[TrackObservation, ...], frame_width: int,
                 frame_height: int) -> tuple[dict[int, tuple[float, float, float]], str | None]:
        if (previous_gray is None or current_gray is None or frame_width <= 0 or frame_height <= 0
                or previous_gray.ndim != 2 or current_gray.shape != previous_gray.shape
                or min(previous_gray.shape) < 8 or not np.isfinite(previous_gray).all()
                or not np.isfinite(current_gray).all()):
            return {}, "invalid_flow_frame"
        if previous_gray.dtype != np.uint8 or current_gray.dtype != np.uint8:
            return {}, "invalid_flow_dtype"
        height, width = previous_gray.shape
        if max(height, width) > 320:
            scale = 320 / max(height, width)
            size = (max(8, round(width * scale)), max(8, round(height * scale)))
            previous_gray = cv2.resize(previous_gray, size, interpolation=cv2.INTER_AREA)
            current_gray = cv2.resize(current_gray, size, interpolation=cv2.INTER_AREA)
            height, width = previous_gray.shape
        sx, sy = width / frame_width, height / frame_height
        if not tracks or any(t.track_id is None for t in tracks):
            return {}, "flow_missing_track"
        if len({t.track_id for t in tracks}) != len(tracks):
            return {}, "flow_duplicate_track"
        boxes = np.asarray([t.box for t in tracks], dtype=np.float64)
        if (not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2])):
            return {}, "flow_invalid_box"
        # Two thumbnail pixels protect both old and proposed boxes at crop edges.
        margin = np.array([2 / sx, 2 / sy])
        limit = np.array([frame_width, frame_height]) - margin
        if np.any(boxes[:, :2] < margin) or np.any(boxes[:, 2:] > limit):
            return {}, "flow_boundary"
        if self._crowded(boxes):
            return {}, "flow_crowding"
        proposals = {}
        for track, box in zip(tracks, boxes):
            mask = np.zeros_like(previous_gray)
            scaled = box * np.array([sx, sy, sx, sy])
            inset = np.maximum(2, .07 * (scaled[2:] - scaled[:2]))
            x1, y1 = np.ceil(scaled[:2] + inset).astype(int)
            x2, y2 = np.floor(scaled[2:] - inset).astype(int)
            mask[y1:y2, x1:x2] = 255
            points = cv2.goodFeaturesToTrack(previous_gray, 80, .01, 3, mask=mask, blockSize=3)
            if points is None or len(points) < 6:
                return {}, "flow_feature_poor"
            params = dict(winSize=(15, 15), maxLevel=2,
                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .03))
            try:
                forward, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, points, None, **params)
                if forward is None or not np.isfinite(forward).all():
                    return {}, "flow_invalid_result"
                backward, back_status, _ = cv2.calcOpticalFlowPyrLK(current_gray, previous_gray, forward, None, **params)
            except cv2.error:
                return {}, "flow_solver_failure"
            if backward is None or not np.isfinite(backward).all():
                return {}, "flow_invalid_result"
            origin, moved = points.reshape(-1, 2), forward.reshape(-1, 2)
            valid = ((status.ravel() > 0) & (back_status.ravel() > 0)
                     & (np.linalg.norm(backward.reshape(-1, 2) - origin, axis=1) <= 1.5)
                     & (moved[:, 0] >= 0) & (moved[:, 0] < width)
                     & (moved[:, 1] >= 0) & (moved[:, 1] < height))
            if valid.sum() < 6 or valid.mean() < .6:
                return {}, "flow_unreliable"
            deltas = moved[valid] - origin[valid]
            translation = np.median(deltas, axis=0)
            residual = np.linalg.norm(deltas - translation, axis=1)
            inliers = residual <= 2.
            if (np.median(residual) > 2 or inliers.sum() < 6
                    or inliers.sum() / len(points) < .6):
                return {}, "flow_deformation"
            translation = np.median(deltas[inliers], axis=0) / np.array([sx, sy])
            if np.linalg.norm(translation) > self.max_jump_fraction * min(box[2:] - box[:2]):
                return {}, "flow_jump"
            if np.any(box[:2] + translation < margin) or np.any(box[2:] + translation > limit):
                return {}, "flow_boundary"
            proposals[track.track_id] = (float(translation[0]), float(translation[1]),
                                         float(inliers.sum() / len(points)))
        moved_boxes = boxes + np.array([proposals[t.track_id][:2] * 2 for t in tracks])
        if self._crowded(moved_boxes):
            return {}, "flow_crowding"
        return proposals, None

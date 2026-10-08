"""Inference-only regions; rendering always uses the original video frame."""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
from reframe.contracts import FrameContext, FrameObservations


@dataclass(frozen=True)
class InferenceRegion:
    x: int
    y: int
    width: int
    height: int
    source_width: int
    source_height: int

    @classmethod
    def from_frame(cls, width: int, height: int, mode: str | None = None):
        if mode not in (None, "middle", "left", "right"):
            raise ValueError("precrop must be middle, left or right")
        if min(width, height) < 1:
            raise ValueError("Inference region dimensions must be positive")
        crop_width = width if mode is None else max(1, width // 2)
        x = {None: 0, "left": 0, "middle": (width - crop_width) // 2,
             "right": width - crop_width}[mode]
        return cls(x, 0, crop_width, height, width, height)

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.width, self.y + self.height

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2

    def crop(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[:2] != (self.source_height, self.source_width):
            raise ValueError("Video dimensions changed during precrop")
        # A view avoids copying the full-resolution frame.
        return frame[self.y:self.y + self.height, self.x:self.x + self.width]

    def context(self, frame_index: int, timestamp: float, scene_index: int) -> FrameContext:
        return FrameContext(frame_index, timestamp, self.width, self.height, scene_index)

    def to_source(self, observations: FrameObservations) -> FrameObservations:
        """Translate boxes, mask moments, cues and both keypoint histories once."""
        if (observations.frame.width, observations.frame.height) != (self.width, self.height):
            raise ValueError("Observations do not match inference region")

        def context(value):
            return replace(value, width=self.source_width, height=self.source_height)

        def box(value):
            return (value[0] + self.x, value[1] + self.y,
                    value[2] + self.x, value[3] + self.y)

        def points(value):
            if value is None:
                return None
            result = value.copy()
            result[:, 0] += self.x
            result[:, 1] += self.y
            return result

        tracks = []
        for track in observations.tracks:
            stats = track.mask_statistics
            if stats is not None:
                stats = (stats[0], stats[1] + self.x, stats[2] + self.y, stats[3] + self.y)
            tracks.append(replace(track, box=box(track.box), mask_statistics=stats,
                                  measured_at=context(track.measured_at) if track.measured_at else None))
        poses = {}
        for index, pose in observations.poses.items():
            cues = dict(pose.cues) if pose.cues is not None else None
            if cues is not None:
                for key in ("body_cx", "body_min_x", "body_max_x"):
                    if cues.get(key) is not None:
                        cues[key] += self.x
                for key in ("eye_y", "shoulder_y", "body_top_y", "body_bottom_y"):
                    if cues.get(key) is not None:
                        cues[key] += self.y
                if cues.get("head_box") is not None:
                    cues["head_box"] = box(cues["head_box"])
            poses[index] = replace(pose, keypoints=points(pose.keypoints),
                                   inferred_keypoints=points(pose.inferred_keypoints), cues=cues,
                                   frame=context(pose.frame), inferred_at=context(pose.inferred_at))
        return FrameObservations(context(observations.frame), tuple(tracks), poses)

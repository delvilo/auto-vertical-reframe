from __future__ import annotations

import logging
from collections import deque
import cv2
import numpy as np
from scenedetect import AdaptiveDetector, SceneManager, open_video

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

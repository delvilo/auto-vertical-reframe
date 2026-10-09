"""Ingest every frame, decide refresh, align maps, and report provenance."""
from __future__ import annotations

from typing import Any
import cv2
import numpy as np
from reframe.contracts import FrameObservations, SaliencyResult
from reframe.saliency.base import SaliencyBackend
from reframe.saliency.cache import SaliencyCache
from reframe.saliency.scheduler import FixedIntervalScheduler
from reframe.timing import HostTimings, timed_method


class SaliencyService:
    def __init__(self, backend: SaliencyBackend, interval: int = 3,
                 max_side: int = 320, ema: float = 0.65):
        self.backend = backend
        self.max_side = max_side
        self.scheduler = FixedIntervalScheduler(interval)
        self.cache = SaliencyCache(ema)
        self.total_frames = self.refreshes = self.propagated = 0
        self._loaded = False
        self._timings = HostTimings(("preparation", "observe", "selection_cache",
                                    "cheap", "deepgaze", "bookkeeping"))
        self.reset()

    def reset(self) -> None:
        self.cache.reset()
        self.scheduler.reset()
        self.backend.reset()

    @timed_method("selection_cache")
    def process(self, frame: np.ndarray, observations: FrameObservations) -> SaliencyResult:
        context = observations.frame
        h, w = frame.shape[:2]
        if (w, h) != (context.width, context.height):
            raise ValueError("Frame context does not match image dimensions")
        previous = self.cache.result
        if previous is not None and (
            context.scene_index != previous.frame.scene_index
            or (w, h) != (previous.frame.width, previous.frame.height)
            or context.frame_index <= previous.frame.frame_index
        ):
            self.reset()
        with self._timings.measure("preparation"):
            scale = min(1.0, self.max_side / max(h, w))
            small = cv2.resize(frame, (max(32, round(w * scale)), max(32, round(h * scale))),
                               interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if not self._loaded:
            with self._timings.measure("deepgaze"):
                self.backend.load()
            self._loaded = True
        # Temporal backends must ingest even when their predictions are skipped.
        with self._timings.measure("observe"):
            self.backend.observe(small, context)
        self.total_frames += 1
        propagated = self.cache.propagate(gray)
        if self.scheduler.should_refresh(observations, missing=self.cache.result is None):
            with self._timings.measure("deepgaze"):
                prediction = self.backend.predict(small, context)
            result = self.cache.refresh(prediction, context, propagated, gray.shape)
            self.refreshes += 1
        else:
            result = self.cache.reuse(propagated, context)
            self.propagated += 1
        self.cache.previous_gray = gray
        return result

    def close(self) -> None:
        try:
            self.backend.close()
        finally:
            self.cache.reset()
            self.scheduler.reset()
            self._loaded = False

    def telemetry(self) -> dict[str, Any]:
        result = dict(self.backend.telemetry())
        result.update(sample_frames_total=self.total_frames, sample_refreshes=self.refreshes,
                      sample_propagated=self.propagated, sample_interval=self.scheduler.interval,
                      timing_seconds=dict(self._timings.seconds),
                      total_seconds=self._timings.total,
                      timing_kind="exclusive_host_wall")
        cached = self.cache.result
        if cached is not None:
            result.update(active_backend=cached.backend, map_source=cached.source,
                          prediction_status=cached.status, fallback_reason=cached.reason,
                          map_frame_index=cached.frame.frame_index,
                          inferred_frame_index=cached.inferred_at.frame_index,
                          inferred_timestamp=cached.inferred_at.timestamp,
                          map_width=cached.map.shape[1], map_height=cached.map.shape[0])
        return result

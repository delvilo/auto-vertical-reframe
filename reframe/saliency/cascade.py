"""Pose first, inexpensive visual evidence second, temporal neural inference last.

The inexpensive thumbnail is also the change detector.  No saliency map or
optical flow is computed on the pose-only path.  Neural cache reuse is deliberately
limited to nearly stationary images: an unaligned old map is less useful than a
fresh inexpensive one.  All ages use the original inference time, in video seconds.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any

import cv2
import numpy as np

from reframe.contracts import FrameContext, FrameObservations, SaliencyResult, TrackObservation
from reframe.geometry import box_iou
from reframe.perception.reliability import assess_pose, pose_motion
from reframe.saliency.base import SaliencyBackend
from reframe.saliency.backends.handcrafted import HandcraftedSaliencyHelper
from reframe.saliency.cache import SaliencyCache
from reframe.timing import HostTimings, timed_method
from reframe.frame_pixels import TensorFrame


class CascadeSaliencyService:
    """Conservative deterministic gates for an aggressive three-tier strategy.

    Three consecutive reliable observations permit pose-only framing.  A reliable
    torso needs three geometrically consistent shoulders/hips; a visible face
    is not required.  Scene/geometry changes and rewinds discard all temporal
    state.  Neural cache lifetime is 0.35 video seconds, independently of reuse.
    """

    def __init__(self, deep_backend: SaliencyBackend, interval: int = 3,
                 max_side: int = 384, ema: float = 0.65, *,
                 lock_first_subject: bool = False, two_person_framing: bool = False,
                 keypoint_conf: float = 0.35, pose_max_age: float = 0.2,
                 tracking_max_age: float = 0.1,
                 cheap_backend: SaliencyBackend | None = None):
        self.deep_backend = deep_backend
        self.cheap_backend = cheap_backend or HandcraftedSaliencyHelper()
        self.interval = max(1, interval)
        self.max_side = max_side
        self.lock_first_subject = lock_first_subject
        self.two_person_framing = two_person_framing
        self.keypoint_conf = keypoint_conf
        self.pose_max_age = max(0.0, pose_max_age)
        self.tracking_max_age = tracking_max_age
        self.cache = SaliencyCache(ema)
        self.cache_max_age = 0.35
        self.tier_counts: Counter[str] = Counter()
        self.reason_counts: Counter[str] = Counter()
        self.pose_evidence_counts: Counter[str] = Counter()
        self.risk_counts: Counter[str] = Counter()
        self.risk_escalations = self.cooldown_bypasses = 0
        self.total_frames = self.deepgaze_requests = self.cheap_predictions = 0
        self.tier_transitions = self.cache_hits = 0
        self._timings = HostTimings(("preparation", "observe", "selection_cache",
                                    "cheap", "deepgaze", "bookkeeping"))
        self.reset()

    def reset(self) -> None:
        self.deep_backend.reset()
        self.cheap_backend.reset()
        self.cache.reset()
        self._previous: FrameObservations | None = None
        self._gray: np.ndarray | None = None
        self._cache_gray: np.ndarray | None = None
        self._cache_tracks: tuple[TrackObservation, ...] = ()
        self._last_deep_request: FrameContext | None = None
        self._stable_frames = self._cheap_stable_frames = 0
        self._tier: str | None = None
        self._reason: str | None = None
        self._result: SaliencyResult | None = None
        self._frame_delta = 0.0
        self._active_risks: frozenset[tuple] = frozenset()
        self._observed_tracks: tuple[TrackObservation, ...] = ()
        self._subject_ids: frozenset[int] = frozenset()

    @staticmethod
    def _resize(frame: np.ndarray, max_side: int) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        shape = (max(32, round(w * scale)), max(32, round(h * scale)))
        if isinstance(frame, TensorFrame):
            return frame.resized(shape)
        return frame if shape == (w, h) else cv2.resize(frame, shape, interpolation=cv2.INTER_AREA)

    @staticmethod
    def _change(gray: np.ndarray, previous: np.ndarray | None) -> tuple[float, float]:
        if previous is None or previous.shape != gray.shape:
            return 0.0, 0.0
        delta = cv2.absdiff(gray, previous).astype(np.float32) / 255.0
        # Tile changes detect a new small subject even when global change is low.
        tiles = [float(tile.mean()) for row in np.array_split(delta, 4, axis=0)
                 for tile in np.array_split(row, 4, axis=1) if tile.size]
        return float(delta.mean()), max(tiles, default=0.0)

    @staticmethod
    def _same_tracks(tracks: tuple[TrackObservation, ...],
                     previous: tuple[TrackObservation, ...], *, strict: bool = False) -> bool:
        if any(t.track_id is None for t in tracks + previous):
            return False
        old = {t.track_id: t for t in previous}
        if {t.track_id for t in tracks} != set(old):
            return False
        movement = 0.025 if strict else 0.12
        size_min, size_max = ((0.96, 1.04) if strict else (0.8, 1.25))
        for track in tracks:
            before = old[track.track_id]
            x1, y1, x2, y2 = before.box
            w, h = max(x2 - x1, 1), max(y2 - y1, 1)
            a, b, c, d = track.box
            if (track.cls_id != before.cls_id
                    or abs((a + c - x1 - x2) / 2) > movement * w
                    or abs((b + d - y1 - y2) / 2) > movement * h
                    or not size_min <= (c - a) / w <= size_max
                    or not size_min <= (d - b) / h <= size_max):
                return False
        return True

    def _reliable_pose(self, track: TrackObservation, observations: FrameObservations) -> bool:
        pose = observations.poses.get(track.row_index)
        context = observations.frame
        if (not track.reliable_at(context, self.tracking_max_age)
                or pose is None or pose.keypoints is None or not pose.cues
                or not pose.cues.get("has_pose") or track.confidence < 0.45
                or track.track_id is None or pose.track_id != track.track_id
                or pose.frame != context
                or pose.inferred_at.frame_index > context.frame_index
                or pose.inferred_at.scene_index != context.scene_index
                or (pose.inferred_at.width, pose.inferred_at.height) != (context.width, context.height)
                or not 0 <= context.timestamp - pose.inferred_at.timestamp <= self.pose_max_age + 1e-8):
            return False
        evidence = assess_pose(pose.keypoints, track.box, self.keypoint_conf)
        self.pose_evidence_counts[evidence.reason] += 1
        if evidence.torso_reliable and not evidence.face_reliable:
            self.pose_evidence_counts["torso_without_face"] += 1
        if evidence.torso_reliable and not evidence.limb_reliable:
            self.pose_evidence_counts["torso_with_partial_limbs"] += 1
        return evidence.torso_reliable

    @staticmethod
    def _area(track: TrackObservation) -> float:
        x1, y1, x2, y2 = track.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @classmethod
    def _relevant_tracks(cls, tracks: tuple[TrackObservation, ...],
                         subjects: tuple[TrackObservation, ...]) -> tuple[TrackObservation, ...]:
        """Keep entrants under review without letting a distant small actor churn L1.

        This is only a stability filter. Full-frame image changes and all new
        identities remain risk signals, and neural cache alignment uses all tracks.
        """
        if not subjects:
            return tracks
        identities = {t.track_id for t in subjects}
        relevant = []
        for track in tracks:
            if track.track_id in identities:
                relevant.append(track)
                continue
            for subject in subjects:
                x1, y1, x2, y2 = subject.box
                margin_x, margin_y = (x2 - x1) * .2, (y2 - y1) * .2
                nearby = box_iou(track.box, (x1 - margin_x, y1 - margin_y,
                                            x2 + margin_x, y2 + margin_y)) > 0
                if nearby or cls._area(track) >= .45 * cls._area(subject):
                    relevant.append(track)
                    break
        return tuple(relevant)

    def _subjects(self, observations: FrameObservations, preferred_track_id: int | None
                  ) -> tuple[tuple[TrackObservation, ...], bool, str]:
        people = tuple(t for t in observations.tracks if t.cls_id == 0)
        if not people:
            reason = ("locked_subject_missing" if self.lock_first_subject
                      and preferred_track_id is not None else "no_person_pose")
            return (), False, reason
        preferred = next((t for t in people if t.track_id == preferred_track_id), None)
        if self.lock_first_subject and preferred_track_id is not None and preferred is None:
            return (), False, "locked_subject_missing"
        if self.two_person_framing and len(people) == 2:
            subjects = people
        elif len(people) == 1:
            subjects = people
        elif self.lock_first_subject and preferred is not None:
            subjects = (preferred,)
        else:
            return people, False, "ambiguous_subjects"
        if any(box_iou(subject.box, other.box) > 0.35
               for subject in subjects for other in people if subject.row_index != other.row_index):
            return subjects, False, "overlapping_people"
        # A locked identity is a tracking preference, not evidence that a second
        # person cannot become visually prominent. Retain the saliency cascade.
        if (len(subjects) == 1 and any(
                other.row_index != subjects[0].row_index and other.confidence >= .45
                and self._area(other) >= .60 * self._area(subjects[0]) for other in people)):
            return subjects, False, "competing_person"
        # Other tracked object classes are possible focal subjects too.
        if not self.lock_first_subject and any(t.cls_id != 0 for t in observations.tracks):
            return subjects, False, "competing_object"
        reliable = all(self._reliable_pose(t, observations) for t in subjects)
        return subjects, reliable, "reliable_pose" if reliable else "insufficient_or_stale_pose"

    def _pose_stable(self, subjects: tuple[TrackObservation, ...],
                     observations: FrameObservations) -> bool:
        if self._previous is None:
            return False
        previous_tracks = {t.track_id: t for t in self._previous.tracks}
        for subject in subjects:
            old_track = previous_tracks.get(subject.track_id)
            old_pose = self._previous.poses.get(old_track.row_index) if old_track is not None else None
            pose = observations.poses.get(subject.row_index)
            if old_pose is None or old_pose.keypoints is None or pose is None or pose.keypoints is None:
                return False
            motion = pose_motion(old_pose.keypoints, old_track.box, pose.keypoints,
                                 subject.box, self.keypoint_conf)
            if motion.torso_displacement > .10:
                self.pose_evidence_counts["torso_motion"] += 1
                return False
            if motion.limb_displacement > .18:
                # Reliable large limb motion affects framing. A missing or weak
                # wrist alone must not invalidate a reliable back/shoulder/hip pose.
                self.pose_evidence_counts["reliable_limb_motion"] += 1
                return False
        return True

    def _risks(self, observations: FrameObservations,
               subjects: tuple[TrackObservation, ...], reason: str,
               delta: float, tile: float,
               observed_tracks: tuple[TrackObservation, ...]) -> frozenset[tuple]:
        risks = set()
        relevant = self._relevant_tracks(observations.tracks, subjects)
        identities = tuple(sorted(t.track_id for t in relevant if t.track_id is not None))
        if reason in {"overlapping_people", "locked_subject_missing", "competing_person",
                      "competing_object", "ambiguous_subjects", "insufficient_or_stale_pose"}:
            risks.add((reason, identities))
        if delta >= .12 or tile >= .30:
            risks.add(("severe_image_change",))
        if self._previous is not None:
            before = {t.track_id: t for t in self._previous.tracks}
            subject_ids = {t.track_id for t in subjects if t.track_id is not None}
            risks.update(("subject_missing", identity) for identity in self._subject_ids - subject_ids)
            previous_ids = {t.track_id for t in self._observed_tracks}
            for track in observed_tracks:
                if track.track_id not in previous_ids or track.track_id is None:
                    risks.add(("new_subject", track.track_id))
            for track in relevant:
                old = before.get(track.track_id)
                if old is not None and not self._same_tracks((track,), (old,)):
                    risks.add(("subject_geometry_change", track.track_id))
        return frozenset(risks)

    @staticmethod
    def _cheap_evidence(saliency: np.ndarray, observations: FrameObservations,
                        subjects: tuple[TrackObservation, ...], reason: str) -> bool:
        """Require a distinct, compact focus, and resolve competing subject scores."""
        if reason in {"overlapping_people", "locked_subject_missing"}:
            return False
        values = saliency.astype(np.float32, copy=False)
        spread = float(values.max() - values.min())
        if spread < 0.08:
            return False
        normalized = (values - values.min()) / spread
        bright = (normalized >= 0.65).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
        if count <= 1:
            return False
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = int(np.argmax(areas)) + 1
        occupied = float(areas.sum()) / values.size
        if not 0.002 <= occupied <= 0.30 or float(areas.max()) / areas.sum() < 0.60:
            return False
        tracks = observations.tracks
        if not tracks:
            return True
        focus = (labels == largest).astype(np.float32) * normalized
        total = max(float(focus.sum()), 1e-6)
        h, w = values.shape
        scores = []
        for track in tracks:
            x1, y1, x2, y2 = track.box
            left, right = max(0, int(x1 * w / observations.frame.width)), min(w, int(np.ceil(x2 * w / observations.frame.width)))
            top, bottom = max(0, int(y1 * h / observations.frame.height)), min(h, int(np.ceil(y2 * h / observations.frame.height)))
            score = float(focus[top:bottom, left:right].sum()) / total
            scores.append((score, track.row_index))
        scores.sort(reverse=True)
        if scores[0][0] < 0.65:
            return False
        if reason in {"ambiguous_subjects", "competing_object", "competing_person"}:
            return len(scores) == 1 or scores[0][0] >= 1.5 * max(scores[1][0], 1e-6)
        return scores[0][1] in {t.row_index for t in subjects} if subjects else True

    def _cache_valid(self, context: FrameContext, gray: np.ndarray,
                     tracks: tuple[TrackObservation, ...]) -> bool:
        cached = self.cache.result
        if cached is None or not 0 <= context.timestamp - cached.inferred_at.timestamp < self.cache_max_age:
            return False
        delta, tile = self._change(gray, self._cache_gray)
        return delta < 0.015 and tile < 0.045 and self._same_tracks(tracks, self._cache_tracks, strict=True)

    @timed_method("bookkeeping")
    def _finish(self, result: SaliencyResult, tier: str, reason: str,
                observations: FrameObservations, gray: np.ndarray) -> SaliencyResult:
        if self._tier is not None and tier != self._tier:
            self.tier_transitions += 1
        self._tier, self._reason, self._result = tier, reason, result
        self.tier_counts[tier] += 1
        self.reason_counts[reason] += 1
        self._previous, self._gray = observations, gray
        return result

    @timed_method("selection_cache")
    def process(self, frame: np.ndarray, observations: FrameObservations, *,
                preferred_track_id: int | None = None) -> SaliencyResult:
        context = observations.frame
        observed_tracks = observations.tracks
        h, w = frame.shape[:2]
        if (w, h) != (context.width, context.height):
            raise ValueError("Frame context does not match image dimensions")
        # Accept reliable flow predictions during skipped detector frames, but
        # never let a new pose validate stale segmentation or another scene.
        tracks = tuple(t for t in observations.tracks
                       if t.reliable_at(context, self.tracking_max_age))
        if len(tracks) != len(observations.tracks):
            eligible_rows = {t.row_index for t in tracks}
            observations = replace(observations, tracks=tracks,
                                   poses={i: p for i, p in observations.poses.items()
                                          if i in eligible_rows})
        if self._previous is not None:
            old = self._previous.frame
            if (context.scene_index != old.scene_index or (w, h) != (old.width, old.height)
                    or context.frame_index <= old.frame_index or context.timestamp < old.timestamp):
                self.reset()
                self.risk_counts["scene_geometry_or_rewind"] += 1
                self.risk_escalations += 1
        with self._timings.measure("preparation"):
            small = self._resize(frame, self.max_side)
            cheap_frame = self._resize(small, 160)
            if isinstance(cheap_frame, TensorFrame):
                cheap_frame = cheap_frame.bgr()
            gray = cv2.cvtColor(cheap_frame, cv2.COLOR_BGR2GRAY)
        with self._timings.measure("observe"):
            # Bounded uint8 temporal ingest; loading belongs to predict, at L3.
            self.deep_backend.observe(small, context)
            self.cheap_backend.observe(cheap_frame, context)
        self.total_frames += 1
        subjects, reliable, reason = self._subjects(observations, preferred_track_id)
        delta, tile = self._change(gray, self._gray)
        self._frame_delta = delta
        risks = self._risks(observations, subjects, reason, delta, tile, observed_tracks)
        self._observed_tracks = observed_tracks
        self._subject_ids = frozenset(t.track_id for t in subjects if t.track_id is not None)
        new_risks = risks - self._active_risks
        urgent = bool(new_risks)
        self._active_risks = risks
        if urgent:
            self.risk_escalations += 1
            self.risk_counts.update(risk[0] for risk in new_risks)
            self._stable_frames = self._cheap_stable_frames = 0
            # A geometrically aligned neural map can still describe the wrong
            # subject after new evidence. Do not blend/reuse it or delay review.
            self.cache.reset()
            self._cache_gray, self._cache_tracks = None, ()
        previous_relevant = (() if self._previous is None else self._relevant_tracks(
            self._previous.tracks, tuple(t for t in self._previous.tracks
                                        if t.track_id in self._subject_ids)))
        stable = (self._previous is not None
                  and not urgent
                  and self._same_tracks(self._relevant_tracks(observations.tracks, subjects),
                                        previous_relevant)
                  and self._pose_stable(subjects, observations)
                  and delta < 0.045 and tile < 0.12)
        self._stable_frames = self._stable_frames + 1 if reliable and stable else 0
        if self._stable_frames >= 3:
            # Pose provenance remains real even when the observations are remapped.
            inferred_at = min((observations.poses[t.row_index].inferred_at for t in subjects),
                              key=lambda c: c.timestamp)
            result = SaliencyResult(None, context, inferred_at, "pose", "skipped", "pose_only",
                                    "stable_reliable_pose")
            return self._finish(result, "pose_only", "stable_reliable_pose", observations, gray)

        with self._timings.measure("cheap"):
            prediction = self.cheap_backend.predict(cheap_frame, context)
            self.cheap_predictions += 1
            # Validate without contaminating the neural cache.
            cheap_cache = SaliencyCache()
            cheap = cheap_cache.refresh(prediction, context, None, gray.shape)
            enough = ((reliable and not urgent)
                      or self._cheap_evidence(cheap.map, observations, subjects, reason))
        self._cheap_stable_frames = self._cheap_stable_frames + 1 if enough else 0
        if enough:
            # Hold the last neural map briefly while evidence settles; never call
            # the model merely to maintain a hysteresis delay.
            if (self._tier == "deepgaze" and self._cheap_stable_frames < 3
                    and self._cache_valid(context, gray, observations.tracks)):
                result = replace(self.cache.result, frame=context, source="cached")
                self.cache_hits += 1
                return self._finish(result, "deepgaze", "settling_evidence", observations, gray)
            why = "pose_stabilizing" if reliable else "cheap_evidence_sufficient"
            return self._finish(cheap, "handcrafted", why, observations, gray)

        backend_state = self.deep_backend.telemetry()
        required = backend_state.get("required_window_frames", 0)
        observed = backend_state.get("temporal_window_frames", required)
        if observed < required:
            cheap = replace(cheap, status="warmup", reason="temporal_window")
            return self._finish(cheap, "handcrafted", "temporal_window", observations, gray)
        if self._cache_valid(context, gray, observations.tracks):
            result = replace(self.cache.result, frame=context, source="cached")
            self.cache_hits += 1
            return self._finish(result, "deepgaze", "valid_neural_cache", observations, gray)
        in_cooldown = (self._last_deep_request is not None
                       and context.frame_index - self._last_deep_request.frame_index < self.interval)
        if in_cooldown and not urgent:
            # A moving scene invalidates the old map; use current inexpensive
            # evidence during the neural cooldown instead of returning stale data.
            cheap = replace(cheap, reason="neural_cooldown")
            return self._finish(cheap, "handcrafted", "neural_cooldown", observations, gray)
        if in_cooldown:
            self.cooldown_bypasses += 1
        self._last_deep_request = context
        self.deepgaze_requests += 1
        with self._timings.measure("deepgaze"):
            prediction = self.deep_backend.predict(small, context)
        if prediction.status != "predicted":
            # The backend owns visible diagnostics and fallback reasons.  Do not
            # overwrite a genuine neural inference timestamp with fallback time.
            cheap = cheap_cache.refresh(prediction, context, None, gray.shape)
            return self._finish(cheap, "handcrafted", prediction.reason or prediction.status,
                                observations, gray)
        # EMA is only meaningful if the previous map is still spatially aligned.
        previous = self.cache.result
        aligned = (previous is not None and self._cache_gray is not None
                   and self._change(gray, self._cache_gray)[1] < 0.045
                   and self._same_tracks(observations.tracks, self._cache_tracks, strict=True))
        result = self.cache.refresh(prediction, context, previous.map if aligned else None, small.shape[:2])
        self._cache_gray, self._cache_tracks = gray.copy(), observations.tracks
        return self._finish(result, "deepgaze", reason, observations, gray)

    def telemetry(self) -> dict[str, Any]:
        result = dict(self.deep_backend.telemetry())
        result.update(requested_backend="cascade", tier=self._tier, tier_reason=self._reason,
                      sample_frames_total=self.total_frames, sample_interval=self.interval,
                      sample_refreshes=self.deepgaze_requests, sample_propagated=0,
                      deepgaze_requests=self.deepgaze_requests,
                      handcrafted_predictions=self.cheap_predictions, cache_hits=self.cache_hits,
                      tier_transitions=self.tier_transitions, tier_reasons=dict(self.reason_counts),
                      pose_reliability_reasons=dict(self.pose_evidence_counts),
                      risk_reasons=dict(self.risk_counts), risk_escalations=self.risk_escalations,
                      cooldown_bypasses=self.cooldown_bypasses,
                      active_risks=sorted({risk[0] for risk in self._active_risks}),
                      pose_max_age=self.pose_max_age, tracking_max_age=self.tracking_max_age,
                      frame_change=self._frame_delta,
                      timing_seconds=dict(self._timings.seconds),
                      total_seconds=self._timings.total,
                      timing_kind="exclusive_host_wall")
        for tier in ("pose_only", "handcrafted", "deepgaze"):
            result[f"tier_{tier}_frames"] = self.tier_counts[tier]
        if self._result is not None:
            active = self._result
            result.update(active_backend=active.backend, map_source=active.source,
                          prediction_status=active.status,
                          fallback_reason=active.reason if active.status in {"warmup", "fallback"} else None,
                          map_frame_index=active.frame.frame_index,
                          inferred_frame_index=active.inferred_at.frame_index,
                          inferred_timestamp=active.inferred_at.timestamp,
                          map_width=active.map.shape[1] if active.map is not None else None,
                          map_height=active.map.shape[0] if active.map is not None else None)
            cached = self.cache.result
            result["cache_age_seconds"] = (active.frame.timestamp - cached.inferred_at.timestamp
                                           if cached is not None else None)
        return result

    def close(self) -> None:
        try:
            self.deep_backend.close()
        finally:
            self.cheap_backend.close()
            self.cache.reset()
            self._previous = self._gray = self._cache_gray = self._result = None

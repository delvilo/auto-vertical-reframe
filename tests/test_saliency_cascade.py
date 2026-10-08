"""Decision and inference provenance checks with real OpenCV and no weights."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from reframe.contracts import (BackendPrediction, FrameContext, FrameObservations,
                               PoseObservation, TrackObservation)
from reframe.saliency.base import SaliencyBackend
from reframe.saliency.cascade import CascadeSaliencyService
from reframe.saliency.backends.handcrafted import HandcraftedSaliencyHelper


def context(index, *, scene=1, width=128, height=64):
    return FrameContext(index, (index - 1) / 30, width, height, scene)


def person_observations(index, *, inferred_index=None, track_id=7, scene=1, box=(32, 4, 96, 60)):
    frame = context(index, scene=scene)
    inferred = context(inferred_index if inferred_index is not None else index, scene=scene)
    points = np.zeros((17, 3), np.float32)
    # A reliable torso without any face/head landmarks.
    points[[5, 6, 11, 12]] = [[45, 20, .9], [75, 20, .9], [48, 45, .9], [72, 45, .9]]
    pose = PoseObservation(points, points.copy(), {"has_pose": True, "body_cx": 60.0},
                           frame, inferred, "inferred" if frame == inferred else "remapped", track_id)
    return FrameObservations(frame, (TrackObservation(0, 0, .9, box, track_id),), {0: pose})


class FakeBackend(SaliencyBackend):
    def __init__(self, name="deepgazemr", required=0, focus=False):
        self.name, self.required, self.focus = name, required, focus
        self.events = []
        self.window = []
        self.calls = self.loads = 0

    def load(self):
        self.loads += 1
        raise AssertionError("Cascade must not eagerly load a model")

    def observe(self, frame, context):
        self.events.append(("observe", context.frame_index))
        self.window.append(context.frame_index)
        self.window = self.window[-16:]

    def predict(self, frame, context):
        self.calls += 1
        self.events.append(("predict", context.frame_index))
        result = np.zeros(frame.shape[:2], np.float32)
        if self.focus:
            h, w = result.shape
            result[h // 3:2 * h // 3, 3 * w // 8:5 * w // 8] = 1
        return BackendPrediction(result, self.name)

    def reset(self):
        self.window.clear()

    def close(self):
        self.reset()

    def telemetry(self):
        return {"required_window_frames": self.required,
                "temporal_window_frames": len(self.window),
                "actual_forward_calls": self.calls}


class CascadeTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((64, 128, 3), np.uint8)
        self.deep = FakeBackend()
        self.cheap = FakeBackend("handcrafted")
        self.service = CascadeSaliencyService(self.deep, max_side=128, cheap_backend=self.cheap)
        self.addCleanup(self.service.close)

    def process(self, index, observations=None, frame=None, **kwargs):
        return self.service.process(self.frame if frame is None else frame,
                                    observations or FrameObservations(context(index)), **kwargs)

    def test_reliable_body_without_head_skips_all_saliency_and_model_load(self):
        for index in range(1, 61):
            result = self.process(index, person_observations(index))
        self.assertIsNone(result.map)
        self.assertEqual((result.backend, result.status, result.source), ("pose", "skipped", "pose_only"))
        self.assertEqual(self.deep.calls, 0)
        self.assertEqual(self.deep.loads, 0)
        self.assertEqual(self.cheap.calls, 3)
        self.assertEqual(self.deep.window, list(range(45, 61)))
        with patch.object(cv2, "calcOpticalFlowPyrLK", side_effect=AssertionError("no L1 flow")):
            self.process(61, person_observations(61))
        self.assertEqual(self.service.telemetry()["tier_pose_only_frames"], 58)

    def test_neural_map_keeps_configured_resolution_not_change_thumbnail_size(self):
        service = CascadeSaliencyService(FakeBackend(), max_side=384,
                                         cheap_backend=FakeBackend("handcrafted"))
        self.addCleanup(service.close)
        frame = np.zeros((216, 384, 3), np.uint8)
        result = service.process(frame, FrameObservations(context(1, width=384, height=216)))
        self.assertEqual(result.map.shape, (216, 384))

    def test_remapped_stale_pose_cannot_keep_pose_only(self):
        for index in range(1, 7):
            self.process(index, person_observations(index, inferred_index=1))
        self.assertEqual(self.service.telemetry()["tier"], "pose_only")
        result = self.process(9, person_observations(9, inferred_index=1))
        self.assertEqual(result.backend, "deepgazemr")
        self.assertEqual(self.deep.calls, 1)
        self.assertEqual(self.service.telemetry()["tier_reason"], "insufficient_or_stale_pose")

    def test_temporal_warmup_uses_cheap_maps_and_does_not_load(self):
        self.deep.required = 16
        for index in range(1, 16):
            result = self.process(index)
            self.assertEqual((result.backend, result.status), ("handcrafted", "warmup"))
        self.assertEqual(self.deep.calls, 0)
        result = self.process(16)
        self.assertEqual(result.backend, "deepgazemr")
        self.assertEqual(self.deep.calls, 1)
        self.assertEqual(self.deep.events[-2:], [("observe", 16), ("predict", 16)])

    def test_cache_reuse_does_not_rejuvenate_inference_timestamp(self):
        first = self.process(1)
        for index in range(2, 12):
            result = self.process(index)
            self.assertEqual(result.inferred_at, first.inferred_at)
            self.assertEqual(result.source, "cached")
        self.assertEqual(self.deep.calls, 1)
        self.assertAlmostEqual(self.service.telemetry()["cache_age_seconds"], 10 / 30)
        refreshed = self.process(12)
        self.assertEqual(refreshed.inferred_at.frame_index, 12)
        self.assertEqual(self.deep.calls, 2)

    def test_motion_invalidates_cache_and_cooldown_returns_current_cheap_map(self):
        self.process(1)
        bright = np.full_like(self.frame, 200)
        result = self.process(2, frame=bright)
        self.assertEqual((result.backend, result.reason), ("handcrafted", "neural_cooldown"))
        result = self.process(4, frame=bright)
        self.assertEqual(result.backend, "deepgazemr")
        self.assertEqual(result.inferred_at.frame_index, 4)

    def test_scene_and_geometry_reset_neural_window_and_maps(self):
        self.deep.required = 16
        for index in range(1, 17):
            self.process(index)
        result = self.process(17, FrameObservations(context(17, scene=2)))
        self.assertEqual(result.status, "warmup")
        self.assertEqual(self.deep.window, [17])
        self.assertIsNone(self.service.cache.result)
        image = np.zeros((96, 160, 3), np.uint8)
        self.process(18, FrameObservations(context(18, scene=2, width=160, height=96)), frame=image)
        self.assertEqual(self.deep.window, [18])

    def test_clear_cheap_focus_avoids_neural_inference_without_pose(self):
        self.cheap.focus = True
        for index in range(1, 21):
            result = self.process(index)
        self.assertEqual(result.backend, "handcrafted")
        self.assertEqual(self.service.telemetry()["tier_reason"], "cheap_evidence_sufficient")
        self.assertEqual(self.deep.calls, 0)

    def test_loss_of_cheap_evidence_escalates_to_neural(self):
        self.cheap.focus = True
        self.process(1)
        self.cheap.focus = False
        result = self.process(2)
        self.assertEqual(result.backend, "deepgazemr")
        self.assertEqual(self.deep.calls, 1)

    def test_multiple_people_require_evidence_unless_primary_is_locked(self):
        self.service.lock_first_subject = True
        for index in range(1, 6):
            obs = person_observations(index)
            other = TrackObservation(1, 0, .9, (0, 5, 20, 60), 8)
            obs = replace(obs, tracks=obs.tracks + (other,))
            result = self.process(index, obs, preferred_track_id=7)
        self.assertEqual(result.source, "pose_only")
        result = self.process(6, obs := replace(obs, frame=context(6)), preferred_track_id=9)
        self.assertEqual(result.backend, "deepgazemr")
        self.assertEqual(self.service.telemetry()["tier_reason"], "locked_subject_missing")

    def test_pose_changes_and_new_object_leave_pose_only(self):
        for index in range(1, 5):
            self.process(index, person_observations(index))
        obs = person_observations(5)
        points = obs.poses[0].keypoints.copy()
        points[5:13, 0] += 15
        obs = replace(obs, poses={0: replace(obs.poses[0], keypoints=points)})
        self.assertEqual(self.process(5, obs).backend, "handcrafted")
        for index in range(6, 10):
            self.process(index, person_observations(index))
        obs = person_observations(10)
        obs = replace(obs, tracks=obs.tracks + (TrackObservation(1, 16, .8, (0, 0, 20, 30), 9),))
        self.assertEqual(self.process(10, obs).backend, "deepgazemr")

    def test_two_person_mode_requires_both_poses(self):
        self.service.two_person_framing = True
        for index in range(1, 6):
            obs = person_observations(index)
            obs = replace(obs, tracks=obs.tracks + (TrackObservation(1, 0, .9, (0, 4, 25, 60), 8),))
            result = self.process(index, obs)
        self.assertNotEqual(result.source, "pose_only")

    def test_invalid_cheap_map_is_rejected_before_ranking(self):
        invalid = BackendPrediction(np.full((8, 8), np.nan), "handcrafted")
        with patch.object(self.cheap, "predict", return_value=invalid), self.assertRaises(ValueError):
            self.process(1)

    def test_cheap_motion_uses_adjacent_frames_after_pose_only_gap(self):
        backend = HandcraftedSaliencyHelper()
        first = np.zeros_like(self.frame)
        first[15:35, 10:30] = 200
        backend.observe(first, context(1))
        backend.predict(first, context(1))
        for index in range(2, 22):
            frame = np.zeros_like(first)
            frame[15:35, 10 + index:30 + index] = 200
            backend.observe(frame, context(index))
            if index == 20:
                previous = frame.copy()
        # A direct two-frame baseline has exactly the same motion history.
        reference = HandcraftedSaliencyHelper()
        reference.compute_map(previous)
        expected = reference.compute_map(frame)
        actual = backend.predict(frame, context(21)).map
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        self.assertEqual(backend.frames_backend, 2)
        backend.reset()
        self.assertIsNone(backend._observed_gray_small)


if __name__ == "__main__":
    unittest.main()

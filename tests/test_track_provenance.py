"""Skipped YOLO frames may use flow, but cannot renew model measurement ages."""
from dataclasses import replace
import unittest
from unittest.mock import MagicMock

import numpy as np

from reframe.contracts import (BackendPrediction, CameraState, FrameContext,
                               FrameObservations, PoseDetection, PoseObservation,
                               TrackObservation)
from reframe.perception.pose import PoseCueCache, observe_poses, pose_cues
from reframe.saliency.cascade import CascadeSaliencyService
from reframe.subjects import SubjectRankingModel, build_candidates


def context(index, **changes):
    return replace(FrameContext(index, (index - 1) / 30, 128, 96, 1), **changes)


def person(index, *, measured_index=None, predicted=False, quality=.9):
    measured = context(index if measured_index is None else measured_index)
    return TrackObservation(0, 0, .9, (30., 5., 94., 90.), 7,
                            measured_at=measured,
                            source="predicted" if predicted else "detected",
                            prediction_confidence=quality)


def skeleton():
    points = np.zeros((17, 3), np.float32)
    points[[5, 6, 7, 8, 11, 12, 13, 14]] = [
        [45, 25, .9], [75, 25, .9], [42, 43, .9], [78, 43, .9],
        [48, 55, .9], [72, 55, .9], [48, 73, .9], [72, 73, .9]]
    return points


def observations(index, track, *, pose_index=None):
    points = skeleton()
    now, measured = context(index), context(index if pose_index is None else pose_index)
    pose = PoseObservation(points, points.copy(), {"has_pose": True, "body_cx": 60.},
                           now, measured, "inferred" if now == measured else "remapped", 7)
    return FrameObservations(now, (track,), {0: pose})


def backend(name):
    helper = MagicMock()
    helper.predict.side_effect = lambda image, now: BackendPrediction(
        np.zeros(image.shape[:2], np.float32), name)
    helper.telemetry.return_value = {}
    return helper


class TrackProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((96, 128, 3), np.uint8)
        self.state = CameraState(64, 48, 1, 64, 48, 1, tracked_id=7,
                                 tracked_cls_id=0, lock_track_id=7, lock_cls_id=0,
                                 is_scene_cut=False, frames_since_subject_switch=10)

    def cascade(self):
        deep, cheap = backend("deepgazemr"), backend("handcrafted")
        service = CascadeSaliencyService(deep, cheap_backend=cheap)
        self.addCleanup(service.close)
        return service, deep

    def test_fresh_flow_tracks_support_pose_only_without_neural_escalation(self):
        service, deep = self.cascade()
        for index in range(1, 10):
            detected = ((index - 1) // 3) * 3 + 1
            track = person(index, measured_index=detected, predicted=index != detected)
            result = service.process(self.frame, observations(index, track, pose_index=detected))
        self.assertEqual(result.source, "pose_only")
        deep.predict.assert_not_called()
        self.assertEqual(deep.observe.call_count, 9)
        self.assertEqual(service.telemetry()["tier_pose_only_frames"], 6)

    def test_new_pose_cannot_rescue_stale_or_invalid_segmentation(self):
        invalid = [
            person(6, measured_index=1, predicted=True),
            replace(person(6, predicted=True), prediction_confidence=.69),
            replace(person(6, predicted=True), measured_at=context(6, scene_index=2)),
            replace(person(6, predicted=True), measured_at=context(6, width=256)),
            replace(person(6, predicted=True), measured_at=context(7)),
            replace(person(6, predicted=True), measured_at=None),
            replace(person(6, predicted=True), source="unknown"),
        ]
        for track in invalid:
            with self.subTest(track=track):
                service, _ = self.cascade()
                for index in range(1, 5):
                    service.process(self.frame, observations(index, person(index)))
                self.assertEqual(service.telemetry()["tier"], "pose_only")
                result = service.process(self.frame, observations(6, track))
                self.assertNotEqual(result.source, "pose_only")
                self.assertEqual(service._previous.tracks, ())

    def test_new_segmentation_cannot_rescue_stale_pose(self):
        service, _ = self.cascade()
        for index in range(1, 5):
            service.process(self.frame, observations(index, person(index)))
        result = service.process(self.frame, observations(10, person(10), pose_index=1))
        self.assertNotEqual(result.source, "pose_only")

    def pose_helper(self):
        helper = MagicMock(keypoint_conf=.35)
        helper.detect_many.side_effect = lambda frame, requests, boxes: {
            i: PoseDetection(skeleton(), pose_cues(skeleton(), box, frame.shape, .35))
            for i, box, _ in requests}
        return PoseCueCache(helper, .2, 30)

    def test_flow_keeps_primary_pose_cache_and_both_original_measurement_times(self):
        cache = self.pose_helper()
        first = observe_poses(self.frame, (person(1),), cache, self.state, context(1))
        predicted = person(2, measured_index=1, predicted=True)
        second = observe_poses(self.frame, (predicted,), cache, self.state, context(2))
        cache.helper.detect_many.assert_called_once()
        self.assertEqual(cache.primary_only_frames, 1)
        self.assertEqual(second.poses[0].source, "remapped")
        self.assertEqual(second.poses[0].inferred_at, first.poses[0].inferred_at)
        self.assertEqual(second.tracks[0].measured_at, context(1))
        self.assertEqual(second.tracks[0].source, "predicted")

    def test_invalid_prediction_never_requests_pose_or_enters_ranking(self):
        cache = self.pose_helper()
        invalid = person(8, measured_index=1, predicted=True)
        self.assertFalse(cache.trustworthy_pose(invalid, context(8)))
        result = observe_poses(self.frame, (invalid,), cache, self.state, context(8))
        self.assertEqual(result.tracks, ())
        self.assertEqual(result.poses, {})
        cache.helper.detect_many.assert_not_called()
        self.assertEqual(self.rank(observations(8, invalid)), [])

    def rank(self, value):
        return build_candidates(value, None, SubjectRankingModel(), {0: "person"},
                                 self.state, 30., [])

    def test_prediction_confidence_reduces_rank_without_altering_box_geometry(self):
        detected = self.rank(observations(2, person(2)))[0]
        predicted = self.rank(observations(2, person(2, measured_index=1, predicted=True)))[0]
        self.assertLess(predicted.score, detected.score)
        self.assertAlmostEqual(predicted.conf, .81)
        self.assertEqual(predicted.area, detected.area)
        self.assertEqual(predicted.mask_area, detected.mask_area)
        self.assertEqual(predicted.framing_cx, detected.framing_cx)

    def test_geometry_change_forces_pose_remeasurement(self):
        cache = self.pose_helper()
        cache.get_many(self.frame, [(0, person(1).box, 7)], {0: person(1).box}, context(1))
        wider_frame = np.zeros((96, 256, 3), np.uint8)
        cache.get_many(wider_frame, [(0, person(2).box, 7)], {0: person(2).box}, context(2, width=256))
        self.assertEqual(cache.helper.detect_many.call_count, 2)

    def test_custom_tracking_age_applies_without_a_pose_helper(self):
        track = replace(person(6, measured_index=1, predicted=True), cls_id=2)
        result = observe_poses(self.frame, (track,), None, self.state, context(6),
                               tracking_max_age=.2)
        self.assertEqual(result.tracks, (track,))

    def test_precrop_translates_measurement_geometry_without_renewing_age(self):
        from reframe.precrop import InferenceRegion
        region = InferenceRegion.from_frame(256, 96, "right")
        track = person(3, measured_index=1, predicted=True)
        mapped = region.to_source(observations(3, track))
        self.assertEqual(mapped.tracks[0].measured_at.width, 256)
        self.assertEqual(mapped.tracks[0].measured_at.timestamp, 0)
        self.assertEqual(mapped.tracks[0].measured_at.frame_index, 1)
        self.assertEqual(mapped.tracks[0].box[0], track.box[0] + 128)
        self.assertTrue(mapped.tracks[0].reliable_at(mapped.frame))


if __name__ == "__main__":
    unittest.main()

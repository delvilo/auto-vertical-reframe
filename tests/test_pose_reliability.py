"""Torso/hand separation and independent evidence required for longer caching."""
from dataclasses import replace
import unittest
from unittest.mock import MagicMock

import numpy as np

from reframe.contracts import (CameraState, FrameContext, MaskEvidence,
                               PoseDetection, TrackObservation)
from reframe.perception.pose import PoseCueCache, observe_poses, pose_cues
from reframe.perception.reliability import assess_pose, pose_motion


BOX = (100., 20., 200., 220.)


def skeleton():
    xy = [(150, 45), (143, 40), (157, 40), (130, 44), (170, 44),
          (120, 75), (180, 75), (110, 110), (190, 110), (105, 145),
          (195, 145), (130, 130), (170, 130), (130, 165), (170, 165),
          (130, 200), (170, 200)]
    return np.column_stack((np.asarray(xy, float), np.full(17, .9)))


def context(index, **changes):
    return replace(FrameContext(index, (index - 1) / 30, 500, 300, 1), **changes)


def person(index, **changes):
    now = context(index)
    track = TrackObservation(0, 0, .9, BOX, 91, measured_at=now,
                             mask_evidence=MaskEvidence(now, True, max(2, index), .01))
    return replace(track, **changes)


class TorsoReliabilityTests(unittest.TestCase):
    def test_back_view_and_face_hand_occlusion_keep_strong_torso(self):
        points = skeleton()
        points[:5, 2] = 0
        points[[7, 8, 9, 10, 13, 14, 15, 16], 2] = 0
        points[12, 2] = 0
        reliability = assess_pose(points, BOX)
        self.assertTrue(reliability.torso_reliable)
        self.assertFalse(reliability.face_reliable)
        self.assertEqual(reliability.torso_mask.sum(), 3)

    def test_collapsed_outside_and_weak_torso_never_gain_trust(self):
        samples = []
        collapsed = skeleton()
        collapsed[[5, 6, 11, 12], :2] = (150, 100)
        samples.append(collapsed)
        outside = skeleton()
        outside[[5, 6, 11, 12], :2] = (900, 800)
        samples.append(outside)
        weak = skeleton()
        weak[[5, 6, 11, 12], 2] = .49
        samples.append(weak)
        hand_only = skeleton()
        hand_only[:, 2] = 0
        hand_only[[7, 8, 9, 10], 2] = .99
        samples.append(hand_only)
        for points in samples:
            with self.subTest(points=points):
                self.assertFalse(assess_pose(points, BOX, .2).torso_reliable)

    def test_bad_hands_do_not_block_torso_but_confident_arm_motion_is_preserved(self):
        before = skeleton()
        after = before.copy()
        after[9] = (195, 50, .1)
        motion = pose_motion(before, BOX, after, BOX)
        self.assertEqual(motion.torso_displacement, 0)
        self.assertEqual(motion.limb_displacement, 0)
        after[9, 2] = .9
        motion = pose_motion(before, BOX, after, BOX)
        self.assertTrue(assess_pose(after, BOX).torso_reliable)
        self.assertGreater(motion.limb_displacement, .18)


class AdaptivePoseCacheTests(unittest.TestCase):
    def setUp(self):
        self.points = skeleton()
        self.helper = MagicMock(keypoint_conf=.35)
        self.helper.detect_many.side_effect = self.detect
        self.cache = PoseCueCache(self.helper, .1, 30, pose_max_age=.3,
                                  tracking_profile="balanced")
        self.frame = np.zeros((300, 500, 3), np.uint8)
        self.state = CameraState(200, 120, 1, 200, 120, 1, tracked_id=91,
                                 tracked_cls_id=0, lock_track_id=91, lock_cls_id=0,
                                 is_scene_cut=False, frames_since_subject_switch=20)

    def detect(self, frame, requests, people):
        return {index: PoseDetection(self.points.copy(), pose_cues(self.points, box, frame.shape, .35))
                for index, box, _ in requests}

    def observe(self, index, tracks=None, now=None):
        return observe_poses(self.frame, (person(index),) if tracks is None else tracks,
                             self.cache, self.state, now or context(index))

    def seed(self, until=6):
        for index in range(1, until + 1):
            self.observe(index)

    def test_extension_requires_two_true_poses_and_does_not_renew_timestamps(self):
        self.seed()
        self.assertEqual(self.helper.detect_many.call_count, 2)
        last = None
        for index in range(7, 13):
            last = self.observe(index)
            self.assertEqual(last.poses[0].inferred_at, context(4))
            self.assertTrue(self.cache.trustworthy_pose(last.tracks[0], context(index)))
        self.assertEqual(self.cache.extension_hits, 6)
        self.assertEqual(self.helper.detect_many.call_count, 2)
        self.assertEqual(self.observe(13).poses[0].inferred_at, context(13))
        self.assertEqual(self.helper.detect_many.call_count, 3)
        self.assertEqual(self.cache.telemetry()["extension_reasons"]["extended"], 6)

    def test_missing_stale_and_predicted_masks_cannot_self_validate(self):
        for evidence in (None, MaskEvidence(context(1), True, 100, .0),
                         MaskEvidence(context(7), False, 2, .2),
                         MaskEvidence(context(7), True, 1, .0)):
            with self.subTest(evidence=evidence):
                self.setUp()
                self.seed()
                track = person(7, source="predicted", measured_at=context(6),
                               mask_evidence=evidence)
                observed = self.observe(7, (track,))
                self.assertEqual(observed.poses[0].inferred_at, context(7))
                self.assertEqual(self.cache.extension_hits, 0)

    def test_recent_real_mask_can_support_predicted_track_without_becoming_new(self):
        self.seed()
        evidence = MaskEvidence(context(6), True, 2, .01)
        track = person(7, source="predicted", measured_at=context(6), mask_evidence=evidence)
        observed = self.observe(7, (track,))
        self.assertEqual(observed.poses[0].inferred_at, context(4))
        self.assertEqual(observed.tracks[0].mask_evidence.measured_at, context(6))

    def test_freshened_mask_timestamp_cannot_rejuvenate_predicted_track(self):
        self.seed()
        track = person(7, source="predicted", measured_at=context(6),
                       mask_evidence=MaskEvidence(context(7), True, 10, 0))
        observed = self.observe(7, (track,))
        self.assertEqual(observed.poses[0].inferred_at, context(7))
        self.assertEqual(self.cache.telemetry()["extension_reasons"]["mask_stale_or_changed"], 1)

    def test_new_near_or_prominent_competitor_prevents_extended_reuse(self):
        self.seed()
        new = TrackObservation(1, 0, .9, (350, 30, 390, 140), 92)
        self.assertEqual(self.observe(7, (person(7), new)).poses[0].inferred_at, context(7))
        for box in ((190, 20, 250, 220), (300, 20, 400, 220)):
            with self.subTest(box=box):
                self.setUp()
                competitor = replace(new, box=box)
                for index in range(1, 8):
                    observed = self.observe(index, (person(index), competitor))
                self.assertEqual(observed.poses[0].inferred_at, context(7))
                self.assertEqual(self.cache.extension_hits, 0)

    def test_limb_motion_blocks_extension_without_invalidating_torso(self):
        self.seed(until=3)
        self.points[9] = (190, 80, .9)
        self.observe(4)
        self.observe(5)
        self.observe(6)
        self.assertTrue(assess_pose(self.points, BOX).torso_reliable)
        self.assertEqual(self.observe(7).poses[0].inferred_at, context(7))
        self.assertEqual(self.cache.extension_hits, 0)

    def test_cut_geometry_or_rewind_never_inherits_extended_pose(self):
        for now in (context(7, scene_index=2), context(7, width=600), context(2), context(5)):
            with self.subTest(now=now):
                self.setUp()
                self.seed()
                track = replace(person(now.frame_index), measured_at=now,
                                mask_evidence=MaskEvidence(now, True, 3, 0))
                observed = self.observe(now.frame_index, (track,), now)
                self.assertEqual(observed.poses[0].inferred_at, now)
                self.assertEqual(self.cache.extension_hits, 0)

    def test_zero_hard_age_and_zero_interval_never_reuse_across_frames(self):
        for interval, hard_age in ((.1, 0), (0, .3)):
            with self.subTest(interval=interval, hard_age=hard_age):
                self.cache = PoseCueCache(self.helper, interval, 30, pose_max_age=hard_age,
                                          tracking_profile="balanced")
                before = self.helper.detect_many.call_count
                self.seed(until=8)
                self.assertEqual(self.helper.detect_many.call_count - before, 8)
                self.assertEqual(self.cache.cache_hits, 0)

    def test_conservative_retains_soft_interval_with_larger_hard_limit(self):
        self.cache = PoseCueCache(self.helper, .1, 30, pose_max_age=.3,
                                  tracking_profile="conservative")
        self.seed(until=10)
        self.assertEqual(self.helper.detect_many.call_count, 4)
        self.assertEqual(self.cache.extension_hits, 0)


if __name__ == "__main__":
    unittest.main()

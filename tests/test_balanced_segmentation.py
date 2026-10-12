"""Scheme B uses actual mask/Pose measurements to extend primary-only skips."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from reframe.contracts import FrameContext, FrameObservations
from reframe.perception.motion import SparseMotion
import test_adaptive_segmentation as adaptive
import test_primary_skip_policy as primary_policy


class BalancedSegmentationTests(unittest.TestCase):
    setUp = adaptive.AdaptiveSegmentationTests.setUp
    step = adaptive.AdaptiveSegmentationTests.step
    warm = adaptive.AdaptiveSegmentationTests.warm

    def configure(self, profile="balanced", fps=60.):
        self.config.tracking_profile = profile
        self.config.seg_max_gap = 6
        self.config.runtime_fps = fps

    def test_60fps_extended_gaps_need_stable_real_masks_and_pose(self):
        self.configure()
        self.warm(110)
        gaps = np.diff(self.fake.calls)
        self.assertIn(4, gaps)
        self.assertIn(6, gaps)
        self.assertTrue(np.all(gaps <= 6))
        for context, track in self.history:
            self.assertLessEqual(context.timestamp - track.measured_at.timestamp, .1 + 1e-9)
            self.assertEqual(track.mask_evidence.measured_at, track.measured_at)
            if track.source == "predicted":
                self.assertLess(track.mask_evidence.measured_at.frame_index, context.frame_index)
        metrics = self.tracker.telemetry()
        self.assertGreater(metrics["interval_counts"]["6"], 0)
        self.assertEqual(sum(metrics["measurement_age_counts"].values()), 110)
        self.assertNotIn("over_.1", metrics["measurement_age_counts"])

    def test_seconds_budget_caps_30fps_even_when_gap_six_requested(self):
        self.configure(fps=30.)
        self.warm(65)
        self.assertEqual(max(np.diff(self.fake.calls)), 3)
        self.assertEqual(self.tracker.telemetry()["max_interval"], 3)

    def test_conservative_profile_keeps_three_frame_cap(self):
        self.configure("conservative")
        self.warm(90)
        self.assertEqual(max(np.diff(self.fake.calls)), 3)

    def test_stable_boxes_without_pose_cannot_authorize_extended_skips(self):
        self.configure()
        feedback = self.tracker.feedback

        def no_pose(observations, preferred):
            feedback(FrameObservations(observations.frame, observations.tracks), preferred)

        with patch.object(self.tracker, "feedback", side_effect=no_pose):
            self.warm(100)
        self.assertEqual(max(np.diff(self.fake.calls)), 3)

    def test_old_pose_cannot_become_fresh_by_remapping_to_current_frame(self):
        self.configure()
        feedback = self.tracker.feedback
        original = None

        def old_pose(observations, preferred):
            nonlocal original
            pose = next(iter(observations.poses.values()))
            if original is None:
                original = pose.inferred_at
            poses = {key: replace(value, inferred_at=original, source="remapped")
                     for key, value in observations.poses.items()}
            feedback(replace(observations, poses=poses), preferred)

        with patch.object(self.tracker, "feedback", side_effect=old_pose):
            self.warm(100)
        self.assertLessEqual(max(np.diff(self.fake.calls)), 3)

    def test_explicit_zero_pose_age_does_not_fall_back_to_cue_interval(self):
        self.configure()
        self.config.pose_max_age = 0.0
        self.warm(100)
        self.assertEqual(max(np.diff(self.fake.calls)), 3)

    def test_confident_large_hand_motion_prevents_longer_skips(self):
        self.configure()
        feedback = self.tracker.feedback

        def moving_hand(observations, preferred):
            track = observations.tracks[0]
            pose = observations.poses[track.row_index]
            points = pose.keypoints.copy()
            points[9, 0] = track.box[0] + (.2 if observations.frame.frame_index % 2 else .8) * (track.box[2] - track.box[0])
            feedback(replace(observations, poses={track.row_index: replace(pose, keypoints=points)}), preferred)

        with patch.object(self.tracker, "feedback", side_effect=moving_hand):
            self.warm(100)
        self.assertLessEqual(max(np.diff(self.fake.calls)), 3)

    def test_expired_mask_or_pose_immediately_caps_current_long_interval(self):
        self.configure()
        self.warm(90)
        self.assertEqual(self.tracker._interval, 6)
        self.tracker._pose_feedback = replace(self.tracker._pose_feedback,
                                              inferred_at=self.history[0][0])
        context = replace(self.history[-1][0], frame_index=self.index + 1,
                          timestamp=self.timestamp + 1 / 60)
        self.tracker._refresh_reason(context, self.fake.tracker)
        self.assertEqual(self.tracker._interval, 3)

    def test_dark_frame_and_failed_flow_still_force_detection_after_extension(self):
        self.configure()
        self.warm(90)
        while self.history[-1][1].source != "detected":
            self.step()
        before = len(self.fake.calls)
        with patch.object(SparseMotion, "estimate", return_value=({}, "flow_unreliable")):
            self.assertEqual(self.step().source, "detected")
        self.assertEqual(len(self.fake.calls), before + 1)
        self.warm(70)
        self.assertEqual(self.step(np.zeros_like(self.frame)).source, "detected")
        self.assertGreater(self.tracker.telemetry()["refresh_reasons"]["frame_change"], 0)

    def test_scene_reset_discards_mask_and_pose_continuity(self):
        self.configure()
        self.warm(90)
        self.scene += 1
        track = self.step()
        self.assertFalse(track.mask_evidence.stable)
        self.assertEqual(track.mask_evidence.observations, 1)
        self.assertEqual(self.tracker._interval, 1)

    def test_mask_deformation_and_missing_mask_reset_continuity(self):
        self.warm(35)
        context, old = self.history[-1]
        context = replace(context, frame_index=context.frame_index + 1,
                          timestamp=context.timestamp + 1 / 60)
        distorted = replace(old, source="detected", measured_at=context,
                            mask_statistics=(old.mask_statistics[0] * .5, *old.mask_statistics[1:]))
        output = self.tracker._mask_evidence((distorted,), context)[0]
        self.assertFalse(output.mask_evidence.stable)
        self.assertEqual(output.mask_evidence.observations, 1)
        self.assertGreater(output.mask_evidence.change, .08)
        output = self.tracker._mask_evidence((replace(distorted, mask_statistics=None),), context)[0]
        self.assertIsNone(output.mask_evidence)


class VisibleSecondaryReliefTests(unittest.TestCase):
    setUp = primary_policy.PrimarySkipPolicyTests.setUp
    step = adaptive.AdaptiveSegmentationTests.step
    warm = adaptive.AdaptiveSegmentationTests.warm
    add_native = primary_policy.PrimarySkipPolicyTests.add_native

    def secondary(self, box=(14., 30., 26., 50.), observations=2):
        native = self.add_native(box=box, lost=False)
        primary = self.tracker._tracks[0]
        observation = replace(primary, row_index=1, track_id=native.track_id,
                              box=tuple(native.xyxy), confidence=.2, mask_statistics=None,
                              mask_evidence=None)
        self.tracker._tracks += (observation,)
        self.tracker._measurement_counts[native.track_id] = observations
        return native, observation

    def next_tracks(self):
        context = replace(self.history[-1][0], frame_index=self.index + 1,
                          timestamp=self.timestamp + 1 / 60)
        self.fake.index = context.frame_index
        return context, self.tracker.track(self.frame, context)

    def test_distant_small_secondary_does_not_veto_but_is_not_reported_reliable(self):
        native, original = self.secondary()
        before = len(self.fake.calls)
        context, output = self.next_tracks()
        self.assertEqual(len(self.fake.calls), before)
        self.assertEqual(len(output), 2)
        secondary = next(t for t in output if t.track_id == native.track_id)
        self.assertEqual(secondary.source, "predicted")
        self.assertEqual(secondary.measured_at, original.measured_at)
        self.assertLess(secondary.prediction_confidence, .7)
        self.assertFalse(secondary.reliable_at(context))
        self.assertTrue(output[0].reliable_at(context))

    def test_new_secondary_keeps_detection_veto(self):
        self.secondary(observations=1)
        self.assertEqual(self.tracker._unsettled_detail(self.fake.tracker), "secondary_new_measurement")

    def test_near_or_prominent_low_confidence_secondary_keeps_detection_veto(self):
        for box in ((120., 50., 140., 100.), (5., 15., 45., 90.)):
            with self.subTest(box=box):
                native, _ = self.secondary(box=box)
                self.assertIsNone(self.tracker._unsettled_detail(self.fake.tracker))
                self.assertNotIn(native.track_id, self.tracker._secondary_relief)
                context = replace(self.history[-1][0], frame_index=self.index + 1,
                                  timestamp=self.timestamp + 1 / 60)
                self.assertEqual(self.tracker._refresh_reason(context, self.fake.tracker), "low_confidence")
                self.fake.tracker.tracked_stracks.remove(native)
                self.tracker._tracks = self.tracker._tracks[:1]

    def test_approaching_distant_secondary_keeps_full_flow_gate(self):
        native, _ = self.secondary()
        native.mean[4] = 35.
        self.assertIsNone(self.tracker._unsettled_detail(self.fake.tracker))
        self.assertNotIn(native.track_id, self.tracker._secondary_relief)

    def test_prominent_lost_secondary_still_requests_real_detection(self):
        lost = self.add_native(box=(2., 20., 42., 80.), lost=True)
        lost.covariance[:] = 0  # Isolate prominence from the separate motion guard.
        self.assertEqual(self.tracker._unsettled_detail(self.fake.tracker), "secondary_lost_prominent")

    def test_new_measurement_rechecks_secondary_that_became_prominent(self):
        native, original = self.secondary()
        self.tracker._unsettled_detail(self.fake.tracker)
        self.assertIn(native.track_id, self.tracker._secondary_relief)
        self.tracker._measured += (original,)
        native.mean[:4] = (50., 80., 1., 90.)
        prominent = replace(original, box=tuple(native.xyxy))
        context = replace(self.history[-1][0], frame_index=self.index + 1,
                          timestamp=self.timestamp + 1 / 60)
        self.tracker._update_stability((self.tracker._tracks[0], prominent), context)
        self.assertNotIn(native.track_id, self.tracker._secondary_relief)
        self.assertEqual(self.tracker._interval, 1)

    def test_all_policy_and_conservative_profile_keep_secondary_confidence_gate(self):
        self.secondary()
        for attr, value in (("seg_skip_policy", "all"), ("tracking_profile", "conservative")):
            with self.subTest(option=attr):
                setattr(self.config, attr, value)
                self.assertIsNone(self.tracker._unsettled_detail(self.fake.tracker))
                self.assertFalse(self.tracker._secondary_relief)
                setattr(self.config, attr, "primary" if attr == "seg_skip_policy" else "balanced")


if __name__ == "__main__":
    unittest.main()

"""Primary-focused skips retain all visible people and real ByteTrack clocks."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from ultralytics.trackers.byte_tracker import STrack

import test_adaptive_segmentation as adaptive


class PrimarySkipPolicyTests(unittest.TestCase):
    step = adaptive.AdaptiveSegmentationTests.step
    warm = adaptive.AdaptiveSegmentationTests.warm

    def setUp(self):
        adaptive.AdaptiveSegmentationTests.setUp(self)
        self.fake.box_fractions = (.44, .28, .54, .67)
        self.config.seg_skip_policy = "primary"
        self.warm(35)
        while self.history[-1][1].source != "detected":
            self.step()

    def add_native(self, box=(14., 30., 26., 50.), lost=True, activated=True):
        native = self.fake.tracker
        x1, y1, x2, y2 = box
        track = STrack(np.array([(x1 + x2) / 2, (y1 + y2) / 2,
                                 x2 - x1, y2 - y1, 1.]), .95, 0)
        track.activate(native.kalman_filter, native.frame_id)
        track.is_activated = activated
        if lost:
            track.mark_lost()
            native.lost_stracks.append(track)
        else:
            native.tracked_stracks.append(track)
        return track

    def test_distant_lost_track_allows_more_skips_without_losing_native_time(self):
        lost = self.add_native()
        lost.mean[4] = .4
        old_mean = lost.mean.copy()
        old_covariance = lost.covariance.copy()
        measured_frame = lost.frame_id
        previous_calls = len(self.fake.calls)
        with patch.object(self.fake.tracker, "multi_predict",
                          wraps=self.fake.tracker.multi_predict) as prediction:
            track = self.step()
        self.assertEqual(track.source, "predicted")
        self.assertEqual(len(self.fake.calls), previous_calls)
        self.assertEqual(self.fake.tracker.frame_id, self.index)
        self.assertEqual(lost.frame_id, measured_frame)
        self.assertAlmostEqual(lost.mean[0], old_mean[0] + .4)
        self.assertGreater(lost.covariance[0, 0], old_covariance[0, 0])
        self.assertEqual(sum(item is lost for item in prediction.call_args.args[0]), 1)
        self.warm(19)
        # Lost covariance grows with real source time and may eventually enter
        # the primary guard. Even then the policy retains the safety veto.
        self.assertLess(len(self.fake.calls) - previous_calls, 20)
        self.assertIn(lost, self.fake.tracker.lost_stracks)
        diagnostics = self.tracker.telemetry()["gate_diagnostics"]
        self.assertGreater(diagnostics["distant_lost_allowed"], 0)

    def test_all_policy_retains_previous_global_lost_track_veto(self):
        self.config.seg_skip_policy = "all"
        self.add_native()
        previous_calls = len(self.fake.calls)
        for _ in range(20):
            self.assertEqual(self.step().source, "detected")
        self.assertEqual(len(self.fake.calls) - previous_calls, 20)
        self.assertEqual(self.tracker.telemetry()["gate_diagnostics"]["all_lost_tracks"], 20)

    def test_lost_track_near_primary_forces_detection(self):
        self.add_native(box=(120., 50., 140., 100.))
        self.assertEqual(self.step().source, "detected")
        self.assertEqual(self.tracker.telemetry()["gate_diagnostics"]["secondary_lost_near_primary"], 1)

    def test_fast_approaching_lost_track_forces_detection_before_overlap(self):
        lost = self.add_native()
        self.assertIsNone(self.tracker._unsettled_detail(self.fake.tracker))
        lost.mean[4] = 40.
        self.assertEqual(self.tracker._unsettled_detail(self.fake.tracker), "secondary_lost_near_primary")

    def test_uncertain_lost_state_cannot_be_ignored(self):
        lost = self.add_native()
        lost.covariance[0, 0] = 10000.
        self.assertEqual(self.tracker._unsettled_detail(self.fake.tracker), "secondary_lost_near_primary")
        lost.mean[0] = np.nan
        self.assertEqual(self.tracker._unsettled_detail(self.fake.tracker), "secondary_lost_uncertain")

    def test_missing_or_unconfirmed_primary_forces_detection(self):
        primary = self.fake.tracker.tracked_stracks[0]
        with self.subTest("unselected"):
            selected = self.tracker._primary_id
            self.tracker._primary_id = None
            self.assertEqual(self.tracker._unsettled_detail(self.fake.tracker), "primary_unselected")
            self.tracker._primary_id = selected
        with self.subTest("unconfirmed"):
            primary.is_activated = False
            self.assertEqual(self.tracker._unsettled_detail(self.fake.tracker), "primary_unconfirmed")
            primary.is_activated = True
        with self.subTest("lost"):
            primary.mark_lost()
            self.fake.tracker.tracked_stracks.clear()
            self.fake.tracker.lost_stracks.append(primary)
            self.assertEqual(self.tracker._unsettled_detail(self.fake.tracker), "primary_missing_or_lost")

    def test_unconfirmed_or_new_secondary_forces_detection(self):
        secondary = self.add_native(lost=False, activated=False)
        self.assertEqual(self.tracker._unsettled_detail(self.fake.tracker), "secondary_unconfirmed")
        secondary.is_activated = True
        self.assertEqual(self.tracker._unsettled_detail(self.fake.tracker), "secondary_new_or_unreported")

    def test_missing_observed_secondary_forces_detection(self):
        previous = self.tracker._tracks[0]
        self.tracker._tracks += (replace(previous, row_index=1, track_id=999),)
        self.assertEqual(self.tracker._unsettled_detail(self.fake.tracker), "secondary_missing")

    def test_live_secondary_remains_in_full_prediction_output(self):
        secondary = self.add_native(lost=False)
        primary = self.tracker._tracks[0]
        observation = replace(primary, row_index=1, track_id=secondary.track_id,
                              box=tuple(secondary.xyxy), mask_statistics=None)
        self.tracker._tracks += (observation,)
        self.assertIsNone(self.tracker._unsettled_detail(self.fake.tracker))
        measured_frame = secondary.frame_id
        predicted = self.tracker._predict({primary.track_id: (1., 0., .9),
                                           secondary.track_id: (0., 1., .95)})
        self.assertEqual({t.track_id for t in predicted}, {primary.track_id, secondary.track_id})
        self.assertEqual(predicted[1].measured_at, observation.measured_at)
        self.assertEqual(predicted[1].source, "predicted")
        self.assertAlmostEqual(predicted[1].box[1], observation.box[1] + 1.)
        self.assertEqual(secondary.frame_id, measured_frame)

    def test_distant_lost_allowance_preserves_whole_frame_change_and_age_gates(self):
        self.add_native()
        self.assertEqual(self.step(np.full_like(self.frame, 240)).source, "detected")
        self.assertEqual(self.tracker.telemetry()["refresh_reasons"]["frame_change"], 1)
        self.warm(25)
        while self.history[-1][1].source != "detected":
            self.step()
        self.fake.tracker.lost_stracks.clear()
        self.add_native()
        self.assertEqual(self.step(timestamp=self.timestamp + .101).source, "detected")
        self.assertEqual(self.tracker.telemetry()["refresh_reasons"]["age_limit"], 1)

    def test_lost_expiration_remains_owned_by_next_real_native_update(self):
        lost = self.add_native()
        self.fake.tracker.max_frames_lost = 1
        self.assertEqual(self.step().source, "predicted")
        self.assertEqual(self.step().source, "predicted")
        self.assertEqual(self.step().source, "detected")
        self.assertEqual(self.fake.tracker.frame_id, self.index)
        self.assertIn(lost, self.fake.tracker.removed_stracks)


if __name__ == "__main__":
    unittest.main()

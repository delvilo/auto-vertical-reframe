"""Lazy thumbnails preserve adaptive decisions and adjacent-frame flow inputs."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from reframe.perception import segmentation
import test_adaptive_segmentation as adaptive


class _Harness(unittest.TestCase):
    setUp = adaptive.AdaptiveSegmentationTests.setUp
    step = adaptive.AdaptiveSegmentationTests.step
    warm = adaptive.AdaptiveSegmentationTests.warm


class LazySegmentationTests(_Harness):
    def test_every_frame_baseline_never_builds_thumbnails_in_either_mode(self):
        self.config.seg_max_gap = 1
        with patch.object(self.tracker, "_thumbnail", wraps=self.tracker._thumbnail) as thumbnail:
            for mode in ("lazy", "eager"):
                self.config.seg_thumbnail_mode = mode
                self.warm(8)
            thumbnail.assert_not_called()
        self.assertEqual(self.tracker.telemetry()["thumbnail_builds"], 0)
        self.assertEqual(len(self.fake.calls), self.index)
        self.assertIsNone(self.tracker._gray)
        self.assertIsNone(self.tracker._previous_frame)

    def test_forced_detection_prefix_defers_all_thumbnails(self):
        self.config.seg_thumbnail_mode = "lazy"
        with patch.object(self.tracker, "_thumbnail", wraps=self.tracker._thumbnail) as thumbnail:
            for _ in range(12):
                self.tracker._force = "weak_primary_pose"
                self.step()
            thumbnail.assert_not_called()
        metrics = self.tracker.telemetry()
        self.assertEqual(metrics["thumbnail_builds"], 0)
        self.assertEqual(metrics["thumbnail_deferred_frames"], 12)
        self.assertEqual(metrics["detector_calls"], 12)
        self.assertIs(self.tracker._previous_frame, self.frame)
        self.assertIsNone(self.tracker._gray)

    def test_backfill_uses_last_source_frame_and_builds_each_frame_once(self):
        self.config.seg_thumbnail_mode = "lazy"
        original_thumbnail = self.tracker._thumbnail
        frames = {}
        built = []
        flow_inputs = []
        original_flow = self.tracker.motion.estimate

        def thumbnail(frame):
            index = int(frame[0, 0, 0])
            built.append(index)
            return original_thumbnail(frame)

        def estimate(previous, current, *args):
            index = self.index
            np.testing.assert_array_equal(previous, original_thumbnail(frames[index - 1]))
            np.testing.assert_array_equal(current, original_thumbnail(frames[index]))
            flow_inputs.append((index - 1, index))
            return original_flow(previous, current, *args)

        with patch.object(self.tracker, "_thumbnail", side_effect=thumbnail), \
                patch.object(self.tracker.motion, "estimate", side_effect=estimate):
            for index in range(1, 61):
                frame = self.frame.copy()
                frame[0, 0] = index
                frames[index] = frame
                # Force a second long run after valid thumbnails already exist.
                if index <= 10 or 31 <= index <= 40:
                    self.tracker._force = "weak_primary_pose"
                self.step(frame)
                if index in (10, 40):
                    self.assertIsNone(self.tracker._gray)
                    self.assertIs(self.tracker._previous_frame, frame)
                if self.tracker._gray is not None:
                    self.assertIsNone(self.tracker._previous_frame)
        self.assertEqual(built[:2], [10, 11])
        self.assertIn(40, built)
        self.assertEqual(built[built.index(40) + 1], 41)
        self.assertEqual(len(built), len(set(built)))
        self.assertTrue(any(current > 40 for _, current in flow_inputs))
        metrics = self.tracker.telemetry()
        self.assertEqual(metrics["thumbnail_builds"], len(built))
        self.assertEqual(metrics["thumbnail_backfills"], 2)
        self.assertGreater(metrics["thumbnail_cache_hits"], 0)
        self.assertEqual(metrics["flow_attempts"], len(flow_inputs))

    @staticmethod
    def _scenario(mode):
        harness = _Harness()
        harness.setUp()
        harness.config.seg_thumbnail_mode = mode
        output = []
        try:
            with patch.object(harness.tracker, "_thumbnail", wraps=harness.tracker._thumbnail) as thumbnail:
                for index in range(1, 81):
                    frame = harness.frame
                    if 25 <= index <= 35:
                        harness.tracker._force = "weak_primary_pose"
                    if 42 <= index <= 46:
                        frame = np.full_like(frame, 235)
                    if index == 50:
                        harness.scene += 1
                    if index >= 60:
                        frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_NEAREST)
                    track = harness.step(frame)
                    output.append((track.track_id, track.box, track.source, track.measured_at,
                                   track.prediction_confidence, track.mask_statistics))
                return output, harness.fake.calls, harness.tracker.telemetry(), thumbnail.call_count
        finally:
            harness.tracker.close()
            harness.doCleanups()

    def test_eager_and_lazy_have_identical_decisions_through_discontinuities(self):
        lazy, lazy_calls, lazy_metrics, lazy_builds = self._scenario("lazy")
        eager, eager_calls, eager_metrics, eager_builds = self._scenario("eager")
        self.assertEqual(lazy, eager)
        self.assertEqual(lazy_calls, eager_calls)
        for key in ("refresh_reasons", "detector_calls", "predicted_frames",
                    "max_prediction_age_seconds", "max_interval"):
            self.assertEqual(lazy_metrics[key], eager_metrics[key], key)
        self.assertEqual(eager_builds, 80)
        self.assertLess(lazy_builds, eager_builds)
        self.assertEqual(eager_metrics["thumbnail_builds"], 80)
        self.assertEqual(eager_metrics["thumbnail_backfills"], 0)
        self.assertEqual(eager_metrics["thumbnail_deferred_frames"], 0)

    def test_image_change_keeps_precedence_over_late_confidence_gate(self):
        self.warm(35)
        self.tracker._tracks = tuple(replace(track, confidence=.1) for track in self.tracker._tracks)
        before = self.tracker.telemetry()["refresh_reasons"].get("frame_change", 0)
        self.step(np.full_like(self.frame, 240))
        self.assertEqual(self.tracker.telemetry()["refresh_reasons"]["frame_change"], before + 1)

    def test_reset_and_close_release_raw_and_gray_history(self):
        self.step()
        self.assertIs(self.tracker._previous_frame, self.frame)
        self.assertIsNone(self.tracker._gray)
        self.tracker.reset()
        self.assertIsNone(self.tracker._previous_frame)
        self.assertIsNone(self.tracker._gray)
        self.warm(10)
        self.assertIsNone(self.tracker._previous_frame)
        self.assertIsNotNone(self.tracker._gray)
        self.tracker.close()
        self.assertIsNone(self.tracker._previous_frame)
        self.assertIsNone(self.tracker._gray)
        self.assertIsNone(self.tracker.model)

    def test_early_gates_do_not_build_thumbnails(self):
        self.config.seg_thumbnail_mode = "lazy"
        self.step()
        with patch.object(self.tracker, "_thumbnail", wraps=self.tracker._thumbnail) as thumbnail:
            self.tracker._tracks = ()
            self.step()
            self.assertEqual(self.tracker.telemetry()["refresh_reasons"]["no_tracks"], 1)
            with patch.object(self.tracker, "_native", return_value=None):
                self.step()
            self.assertEqual(self.tracker.telemetry()["refresh_reasons"]["unsupported_tracker"], 1)
            native = self.fake.tracker
            lost = native.tracked_stracks.pop()
            lost.mark_lost()
            native.lost_stracks.append(lost)
            self.step()
            self.assertEqual(self.tracker.telemetry()["refresh_reasons"]["unsettled_tracks"], 1)
            thumbnail.assert_not_called()

    def test_exclusive_timings_reconcile_without_gpu_synchronization(self):
        with patch.object(segmentation.torch.cuda, "synchronize") as synchronize:
            self.warm(40)
            synchronize.assert_not_called()
        metrics = self.tracker.telemetry()
        timings = metrics["timing_seconds"]
        self.assertEqual(set(timings), {"thumbnail", "scheduler", "frame_change", "model_track",
                                        "parse_masks", "flow", "predict", "bookkeeping"})
        self.assertTrue(all(value >= 0 for value in timings.values()))
        for key in ("thumbnail", "model_track", "parse_masks", "flow", "predict"):
            self.assertGreater(timings[key], 0, key)
        self.assertAlmostEqual(sum(timings.values()), metrics["total_seconds"], delta=2e-5)
        self.assertAlmostEqual(timings["flow"], metrics["flow_seconds"], delta=1e-9)
        self.assertGreater(metrics["flow_attempts"], 0)


if __name__ == "__main__":
    unittest.main()

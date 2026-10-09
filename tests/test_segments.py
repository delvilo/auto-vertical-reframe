"""Long-video reporting boundaries and conservation without model weights/CUDA."""
from contextlib import ExitStack, redirect_stderr
import io
import json
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from reframe import cli, config, pipeline
from reframe.contracts import FrameObservations
from reframe.segments import FRAME_PHASES, SegmentStats
from reframe.timing import PhaseTimings


def snapshot(count=0):
    return {"seg_refresh_reasons": {"scheduled": count},
            "pipeline_timing_seconds": {key: count * .01 for key in FRAME_PHASES}}


def decision(index, detected=True, interval=3):
    return {"frame_index": index, "detected": detected,
            "reason": "scheduled" if detected else None, "interval": interval}


class SegmentStatsTests(unittest.TestCase):
    def test_window_and_partial_rows_preserve_actual_cross_window_gap(self):
        stats = SegmentStats(2., 2., snapshot())
        for index in range(1, 5):
            stats.record_frame(index, 1, decision(index, index == 2))
        self.assertTrue(stats.window_complete())
        first = stats.finish(snapshot(1), "interval")
        self.assertEqual((first["frame_start"], first["frame_end"]), (1, 4))
        self.assertEqual((first["start_seconds"], first["end_seconds"]), (0., 2.))
        self.assertEqual(first["seg_detector_calls"], 1)
        self.assertEqual(first["seg_predicted_frames"], 3)
        self.assertEqual(first["seg_detector_gap_counts"], {})
        stats.record_frame(5, 1, decision(5))
        final = stats.finish(snapshot(2), "end")
        self.assertEqual(final["seg_detector_gap_counts"], {"3": 1})
        self.assertEqual(final["seg_scheduled_interval_counts"], {"3": 1})
        self.assertEqual(final["seg_refresh_reasons"], {"scheduled": 1})
        self.assertEqual(final["frames"], 1)
        self.assertIsNone(stats.finish(snapshot(2), "end"))
        self.assertEqual(stats.emitted, 2)
        self.assertEqual(stats.pipeline_timing_seconds["decode"], .02)
        json.dumps(final, allow_nan=False)

    def test_scene_flush_resets_gaps_but_keeps_global_time_window(self):
        stats = SegmentStats(2., 2., snapshot())
        stats.record_frame(1, 1, decision(1))
        stats.record_frame(2, 1, decision(2))
        self.assertTrue(stats.scene_changed(2))
        first = stats.finish(snapshot(2), "scene")
        stats.record_frame(3, 2, decision(3))
        stats.record_frame(4, 2, decision(4))
        second = stats.finish(snapshot(4), "interval")
        self.assertEqual(first["window_index"], second["window_index"])
        self.assertEqual(second["seg_detector_gap_counts"], {"1": 1})
        self.assertEqual(second["start_seconds"], first["end_seconds"])
        stats.record_frame(5, 3, decision(5))
        third = stats.finish(snapshot(5), "end")
        self.assertEqual(third["window_index"], 1)
        self.assertEqual(third["seg_detector_gap_counts"], {})

    def test_fractional_fps_uses_frame_start_for_global_window(self):
        stats = SegmentStats(29.97, 5., snapshot())
        for index in range(1, 151):
            stats.record_frame(index, 1, decision(index))
            self.assertEqual(stats.window_complete(), index == 150)
        first = stats.finish(snapshot(150), "interval")
        stats.record_frame(151, 1, decision(151))
        second = stats.finish(snapshot(151), "end")
        self.assertEqual(first["end_seconds"], second["start_seconds"])
        self.assertEqual(second["seg_detector_gap_counts"], {"1": 1})

    def test_requires_successful_contiguous_decisions_and_flush(self):
        stats = SegmentStats(1., 1., snapshot())
        for invalid in ({}, decision(2), dict(decision(1), detected=None)):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "decision"):
                stats.record_frame(1, 1, invalid)
        stats.record_frame(1, 1, decision(1))
        with self.assertRaisesRegex(ValueError, "contiguous"):
            stats.record_frame(3, 1, decision(3))
        with self.assertRaisesRegex(ValueError, "Flush"):
            stats.record_frame(2, 1, decision(2))

    def test_empty_run_and_zero_time_are_json_finite(self):
        stats = SegmentStats(30., 5., snapshot())
        self.assertIsNone(stats.finish(snapshot(), "end"))
        stats.record_frame(1, 1, decision(1))
        row = stats.finish(snapshot(), "end")
        self.assertIsNone(row["processing_fps"])
        json.dumps(row, allow_nan=False)

    def test_cpu_reporting_module_has_no_heavy_imports(self):
        program = "import sys; import reframe.segments; assert not {'torch','cv2','numpy','ultralytics'} & sys.modules.keys()"
        subprocess.run([sys.executable, "-c", program], check=True)

    def test_phase_snapshot_does_not_advance_accumulator(self):
        timer = PhaseTimings(("setup", "work"), "setup", 10.)
        with patch("reframe.timing.perf_counter", side_effect=[12., 15., 17.]):
            self.assertEqual(timer.snapshot(), {"setup": 2., "work": 0.})
            timer.switch("work")
            self.assertEqual(timer.snapshot(), {"setup": 5., "work": 2.})
        self.assertEqual(timer.seconds, {"setup": 5., "work": 0.})


class SegmentConfigTests(unittest.TestCase):
    def test_default_disabled_and_collector_interval(self):
        self.assertEqual(cli.parse_args(["in.mp4", "out.mp4"]).stats_interval, 0)
        self.assertEqual(cli.parse_args(["in.mp4", "out.mp4", "--stats-interval", "5"]).stats_interval, 5)

    def test_reject_negative_nonfinite_or_invalid_direct_values(self):
        for value in (-1, float("nan"), float("inf"), True, "5", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "stats-interval"):
                config.validate_config(config.AppConfig(input="in.mp4", output="out.mp4", stats_interval=value))
        for value in ("-1", "nan", "inf"):
            with self.subTest(value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.parse_args(["in.mp4", "out.mp4", f"--stats-interval={value}"])


class PipelineSegmentsTests(unittest.TestCase):
    def run_fixture(self, maximum=None, fail_frame=None, report=True):
        extra = ["--stats-interval", "2"] if report else []
        if maximum is not None:
            extra.extend(["--max-frames", str(maximum)])
        args = config.apply_preset(cli.parse_args(["input.mp4", "/tmp/segments-output.mp4", *extra]))
        frame = np.zeros((240, 400, 3), np.uint8)
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.side_effect = {pipeline.cv2.CAP_PROP_FPS: 2.,
                                  pipeline.cv2.CAP_PROP_FRAME_WIDTH: 400,
                                  pipeline.cv2.CAP_PROP_FRAME_HEIGHT: 240,
                                  pipeline.cv2.CAP_PROP_FRAME_COUNT: 11}.__getitem__
        tracker = MagicMock(class_names={0: "person"}, actual_device="cpu")
        counts = {"detector_calls": 0, "predicted_frames": 0, "refresh_reasons": {},
                  "timing_seconds": {"model_track": 0.}, "gate_diagnostics": {}}
        def track(_frame, context):
            detected = context.frame_index in {1, 3, 4, 5, 7, 9, 10}
            tracker.last_decision = decision(context.frame_index, detected)
            counts["detector_calls" if detected else "predicted_frames"] += 1
            counts["refresh_reasons"]["scheduled"] = counts["detector_calls"]
            counts["timing_seconds"]["model_track"] += .01
            return ()  # An actual detector invocation can return no tracks.
        tracker.track.side_effect = track
        tracker.telemetry.side_effect = lambda: counts
        pose = SimpleNamespace(helper=SimpleNamespace(actual_device=None, rois_inferred=0, rois_matched=0),
                               cache_hits=0, primary_only_frames=0, full_scan_frames=0, rois_skipped=0,
                               close=MagicMock(), clear=MagicMock())
        service = MagicMock()
        saliency = {"actual_forward_calls": 0, "tier_deepgaze_frames": 0,
                    "timing_seconds": {"deepgaze": 0.}}
        def process(_frame, observations, **kwargs):
            saliency["actual_forward_calls"] += 1
            saliency["tier_deepgaze_frames"] += 1
            saliency["timing_seconds"]["deepgaze"] += .02
            return SimpleNamespace(map=None)
        service.process.side_effect = process
        service.telemetry.side_effect = lambda: saliency
        writer = MagicMock(encoder="test")
        if fail_frame:
            def write(_frame):
                if tracker.last_decision["frame_index"] == fail_frame:
                    raise RuntimeError("writer failure")
            writer.write.side_effect = write
        replacements = dict(
            iter_video_frames=MagicMock(return_value=((index, frame) for index in range(1, 12))),
            SegmentationTracker=MagicMock(return_value=tracker),
            YOLOPoseHelper=MagicMock(), PoseCueCache=MagicMock(return_value=pose),
            observe_poses=lambda f, t, h, s, c, k, **kw: FrameObservations(c, t),
            build_saliency_helper=MagicMock(return_value=service),
            DirectVideoWriter=MagicMock(return_value=writer),
        )
        with ExitStack() as stack:
            for name, value in replacements.items():
                stack.enter_context(patch.object(pipeline, name, value))
            stack.enter_context(patch.object(pipeline.cv2, "VideoCapture", return_value=capture))
            stack.enter_context(patch.object(pipeline.shutil, "which", return_value="ffmpeg"))
            stack.enter_context(patch.object(pipeline.InlineSceneDetector, "update",
                                            side_effect=lambda frame, index: index in {1, 4, 5, 10}))
            logs = stack.enter_context(self.assertLogs(level="INFO"))
            if fail_frame:
                with self.assertRaisesRegex(RuntimeError, "writer failure"):
                    pipeline.process_video(args)
            else:
                pipeline.process_video(args)
        rows = [json.loads(line.split("Segment: ", 1)[1]) for line in logs.output if "Segment: " in line]
        summary = next((json.loads(line.split("Summary: ", 1)[1])
                        for line in logs.output if "Summary: " in line), None)
        return rows, summary, tracker, pose, writer

    def test_time_windows_scenes_and_partial_end_conserve_counts(self):
        rows, summary, tracker, pose, writer = self.run_fixture()
        self.assertEqual([(r["frame_start"], r["frame_end"], r["scene_index"], r["end_reason"])
                          for r in rows], [(1, 3, 1, "scene"), (4, 4, 2, "interval"),
                                           (5, 8, 3, "interval"), (9, 9, 3, "scene"), (10, 11, 4, "end")])
        self.assertEqual(sum(r["frames"] for r in rows), 11)
        for key in ("seg_detector_calls", "seg_predicted_frames", "scene_resets",
                    "saliency_actual_forward_calls", "saliency_tier_deepgaze_frames"):
            self.assertEqual(sum(r[key] for r in rows), summary[key])
        for key in FRAME_PHASES:
            self.assertAlmostEqual(sum(r["pipeline_timing_seconds"][key] for r in rows),
                                   summary["segment_pipeline_timing_seconds"][key])
            self.assertLessEqual(summary["segment_pipeline_timing_seconds"][key],
                                 summary["pipeline_timing_seconds"][key] + 1e-9)
        self.assertEqual(rows[3]["seg_detector_gap_counts"], {"2": 1})
        self.assertEqual(rows[4]["seg_detector_gap_counts"], {})
        self.assertEqual(tracker.reset.call_count, 4)
        self.assertEqual(pose.clear.call_count, 4)
        self.assertEqual(tracker.telemetry.call_count, len(rows) + 2)
        self.assertEqual(writer.write.call_count, 11)
        self.assertEqual(summary["segments_emitted"], 5)
        self.assertGreater(summary["pipeline_timing_seconds"]["segment_logging"], 0)

    def test_max_frames_flushes_partial_and_exact_window_without_empty_rows(self):
        for maximum, row_count in ((6, 3), (8, 3)):
            with self.subTest(maximum=maximum):
                rows, summary, *_ = self.run_fixture(maximum=maximum)
                self.assertEqual(sum(r["frames"] for r in rows), maximum)
                self.assertEqual(rows[-1]["frame_end"], maximum)
                self.assertEqual(len(rows), row_count)
                self.assertEqual(sum(r["seg_detector_calls"] for r in rows), summary["seg_detector_calls"])

    def test_disabled_reporting_does_not_add_snapshots_or_reset_tracking(self):
        rows, summary, tracker, _, _ = self.run_fixture(report=False)
        self.assertEqual(rows, [])
        self.assertEqual(summary["segments_emitted"], 0)
        self.assertEqual(summary["pipeline_timing_seconds"]["segment_logging"], 0.)
        tracker.telemetry.assert_called_once()
        self.assertEqual(tracker.reset.call_count, 4)

    def test_error_preserves_completed_rows_and_does_not_report_failed_frame(self):
        rows, summary, _, _, writer = self.run_fixture(fail_frame=7)
        self.assertIsNone(summary)
        self.assertEqual([(r["frame_start"], r["frame_end"]) for r in rows], [(1, 3), (4, 4)])
        writer.abort.assert_called_once()


if __name__ == "__main__":
    unittest.main()

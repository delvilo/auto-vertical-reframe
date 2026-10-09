"""Timing conservation and failure/lazy-path checks without CUDA or weights."""
from contextlib import ExitStack, nullcontext
import json
import math
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from reframe.contracts import BackendPrediction, FrameContext, FrameObservations
from reframe.saliency.backends import deepgazemr
from reframe.saliency.cascade import CascadeSaliencyService
from reframe.timing import HostTimings, PhaseTimings
from tests.test_deepgaze_lazy import ArrayTensor, RecordingModel
from tests.test_saliency_cascade import FakeBackend, person_observations


class TimingAccumulatorTests(unittest.TestCase):
    def test_nested_sections_are_exclusive_even_for_same_key_and_exception(self):
        timings = HostTimings(("outer", "inner"))
        with patch("reframe.timing.perf_counter", side_effect=[0., 2., 5., 8., 10., 13.]):
            with self.assertRaisesRegex(RuntimeError, "visible"):
                with timings.measure("outer"):
                    with timings.measure("inner"):
                        with timings.measure("inner"):
                            pass
                    raise RuntimeError("visible")
        self.assertEqual(timings.seconds, {"outer": 5., "inner": 8.})
        self.assertEqual(timings.total, 13.)
        self.assertIsNone(timings._active)

    def test_phase_partition_accounts_for_repeated_phases_and_finish(self):
        timings = PhaseTimings(("setup", "decode", "work"), "setup", 10.)
        with patch("reframe.timing.perf_counter", side_effect=[12., 15., 18., 20.]):
            timings.switch("decode")
            timings.switch("work")
            timings.switch("decode")
            finished = timings.switch(None)
        self.assertEqual(timings.seconds, {"setup": 2., "decode": 5., "work": 3.})
        self.assertEqual(sum(timings.seconds.values()), finished - 10.)


class SaliencyTimingTests(unittest.TestCase):
    def assert_timings(self, telemetry, prefix=""):
        buckets = telemetry[f"{prefix}timing_seconds"]
        self.assertTrue(all(math.isfinite(value) and value >= 0 for value in buckets.values()))
        self.assertEqual(sum(buckets.values()), telemetry[f"{prefix}total_seconds"])
        return buckets

    def test_pose_only_keeps_cpu_window_without_torch_calls_or_deep_load(self):
        # Any torch operation on this object fails; CPU observing remains valid.
        with patch.object(deepgazemr, "torch", object()):
            backend = deepgazemr.DeepGazeMRSaliencyHelper(device="cpu", max_side=128)
            service = CascadeSaliencyService(backend, max_side=128)
            self.addCleanup(service.close)
            frame = np.zeros((64, 128, 3), np.uint8)
            for index in range(1, 31):
                result = service.process(frame, person_observations(index))
        self.assertEqual(result.source, "pose_only")
        self.assertIsNone(backend.model)
        self.assertIsNone(backend.tensor_ring)
        self.assertEqual(backend.ring_count, 16)
        timings = self.assert_timings(service.telemetry())
        self.assertEqual(timings["deepgaze"], 0.)
        nested = self.assert_timings(service.telemetry(), "deepgaze_")
        self.assertGreater(nested["observe"], 0.)
        for key in ("init", "preprocess", "transfer", "forward", "postprocess", "fallback"):
            self.assertEqual(nested[key], 0.)

    def test_cache_hits_do_not_add_neural_invocation_time(self):
        backend = FakeBackend()
        service = CascadeSaliencyService(backend, max_side=128,
                                         cheap_backend=FakeBackend("handcrafted"))
        self.addCleanup(service.close)
        frame = np.zeros((64, 128, 3), np.uint8)
        def process(index):
            context = FrameContext(index, (index - 1) / 30, 128, 64, 1)
            return service.process(frame, FrameObservations(context))
        process(1)
        neural_seconds = service.telemetry()["timing_seconds"]["deepgaze"]
        for index in range(2, 10):
            self.assertEqual(process(index).source, "cached")
        self.assertEqual(backend.calls, 1)
        self.assertEqual(self.assert_timings(service.telemetry())["deepgaze"], neural_seconds)
        before_reset = service.telemetry()["total_seconds"]
        service.reset()
        self.assertEqual(service.telemetry()["total_seconds"], before_reset)

    def test_invalid_cheap_output_keeps_exception_and_unwinds_timings(self):
        cheap = FakeBackend("handcrafted")
        service = CascadeSaliencyService(FakeBackend(), max_side=128, cheap_backend=cheap)
        self.addCleanup(service.close)
        frame = np.zeros((64, 128, 3), np.uint8)
        context = FrameContext(1, 0., 128, 64, 1)
        invalid = BackendPrediction(np.full((8, 8), np.nan), "handcrafted")
        with patch.object(cheap, "predict", return_value=invalid), self.assertRaises(ValueError):
            service.process(frame, FrameObservations(context))
        timings = self.assert_timings(service.telemetry())
        self.assertGreater(timings["cheap"], 0.)
        self.assertGreater(timings["selection_cache"], 0.)
        self.assertIsNone(service._timings._active)

    def test_backend_success_and_failure_report_separate_forward_and_fallback(self):
        model = RecordingModel()
        fake_torch = SimpleNamespace(
            hub=SimpleNamespace(load=MagicMock(return_value=model)),
            empty=lambda shape, **kwargs: ArrayTensor(np.empty(shape, np.float32)),
            from_numpy=ArrayTensor, inference_mode=nullcontext,
            is_tensor=lambda value: isinstance(value, ArrayTensor), float32=np.float32,
        )
        with patch.object(deepgazemr, "torch", fake_torch):
            backend = deepgazemr.DeepGazeMRSaliencyHelper(device="cpu", max_side=128)
            self.addCleanup(backend.close)
            frame = np.zeros((64, 128, 3), np.uint8)
            for _ in range(16):
                backend.observe_frame(frame)
            context = FrameContext(16, .5, 128, 64, 1)
            self.assertEqual(backend.predict(frame, context).backend, "deepgazemr")
            first = self.assert_timings(backend.telemetry(), "deepgaze_")
            for key in ("init", "observe", "preprocess", "transfer", "forward", "postprocess"):
                self.assertGreater(first[key], 0.)
            self.assertEqual(first["fallback"], 0.)
            model.fail = True
            with self.assertLogs(level="WARNING"):
                self.assertEqual(backend.predict(frame, context).status, "fallback")
            second = self.assert_timings(backend.telemetry(), "deepgaze_")
            self.assertGreater(second["forward"], first["forward"])
            self.assertGreater(second["fallback"], 0.)
            self.assertEqual(second["postprocess"], first["postprocess"])
            self.assertEqual(backend.actual_forward_calls, 2)
            self.assertIsNone(backend._timings._active)

    def test_unavailable_model_reports_initialization_and_fallback_cost(self):
        with patch.object(deepgazemr, "torch", None):
            backend = deepgazemr.DeepGazeMRSaliencyHelper(device="cpu", max_side=128)
            self.addCleanup(backend.close)
            frame = np.zeros((64, 128, 3), np.uint8)
            for _ in range(16):
                backend.observe_frame(frame)
            with self.assertLogs(level="WARNING"):
                result = backend.predict(frame, FrameContext(16, .5, 128, 64, 1))
        self.assertEqual(result.status, "fallback")
        timings = self.assert_timings(backend.telemetry(), "deepgaze_")
        self.assertGreater(timings["init"], 0.)
        self.assertGreater(timings["fallback"], 0.)
        self.assertEqual(timings["forward"], 0.)


class PipelineTimingTests(unittest.TestCase):
    def test_pipeline_partitions_elapsed_and_retains_outer_stage_aliases(self):
        from reframe import cli, config, pipeline
        args = config.apply_preset(cli.parse_args(["input.mp4", "/tmp/timing-output.mp4",
                                                  "--max-frames", "1"]))
        frame = np.zeros((240, 400, 3), np.uint8)
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.side_effect = {pipeline.cv2.CAP_PROP_FPS: 30,
                                   pipeline.cv2.CAP_PROP_FRAME_WIDTH: 400,
                                   pipeline.cv2.CAP_PROP_FRAME_HEIGHT: 240,
                                   pipeline.cv2.CAP_PROP_FRAME_COUNT: 1}.__getitem__
        tracker = MagicMock(class_names={0: "person"}, actual_device="cpu")
        tracker.track.return_value = ()
        tracker.telemetry.return_value = {}
        pose = MagicMock()
        pose.helper.actual_device = None
        for key in ("cache_hits", "primary_only_frames", "full_scan_frames", "rois_skipped"):
            setattr(pose, key, 0)
        pose.helper.rois_inferred = pose.helper.rois_matched = 0
        service = MagicMock()
        service.process.return_value = SimpleNamespace(map=None)
        service.telemetry.return_value = {}
        writer = MagicMock(encoder="test")
        frames = ((1, frame) for _ in range(1))
        replacements = dict(
            iter_video_frames=MagicMock(return_value=frames),
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
            stack.enter_context(patch.object(pipeline.InlineSceneDetector, "update", return_value=True))
            logs = stack.enter_context(self.assertLogs(level="INFO"))
            pipeline.process_video(args)
        summary = json.loads(next(line.split("Summary: ", 1)[1]
                                  for line in logs.output if "Summary: " in line))
        timings = summary["pipeline_timing_seconds"]
        self.assertEqual(summary["timing_schema_version"], 2)
        self.assertTrue(all(math.isfinite(value) and value >= 0 for value in timings.values()))
        self.assertAlmostEqual(sum(timings.values()), summary["elapsed_seconds"], places=9)
        for key in ("segmentation", "pose", "saliency", "render_write"):
            self.assertEqual(summary["stage_wall_seconds"][key], timings[key])
        for key in ("setup", "decode", "scene", "feedback", "subjects", "camera", "finalization"):
            self.assertGreater(timings[key], 0.)
        self.assertEqual(summary["frames_processed"], 1)
        writer.write.assert_called_once()


if __name__ == "__main__":
    unittest.main()

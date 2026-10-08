"""Bootstrap and resource lifetime regressions introduced by the module split."""
import contextlib
import io
import json
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
for name in ("cv2", "scenedetect", "ultralytics"):
    try:
        __import__(name)
    except ImportError:
        sys.modules[name] = MagicMock()

from reframe import cli, config, pipeline, runtime
from reframe.contracts import FrameObservations


class BootstrapTests(unittest.TestCase):
    def test_help_works_without_loading_video_or_model_modules(self):
        program = """
import contextlib, io, sys
from reframe.cli import main
try:
    with contextlib.redirect_stdout(io.StringIO()):
        main(['--help'])
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError('help did not exit')
assert not {'torch','ultralytics','cv2','reframe.pipeline'} & sys.modules.keys()
"""
        subprocess.run([sys.executable, "-c", program], check=True)

    def test_native_diagnostics_precede_heavy_imports(self):
        program = """
import builtins, os, sys
os.environ.pop('TORCH_SHOW_CPP_STACKTRACES', None)
original = builtins.__import__
seen = []
def import_guard(name, *args, **kwargs):
    if name in ('cv2', 'torch', 'ultralytics'):
        assert os.environ.get('TORCH_SHOW_CPP_STACKTRACES') == '1'
        seen.append(name)
        raise ImportError('TEST_IMPORT_BLOCK')
    return original(name, *args, **kwargs)
builtins.__import__ = import_guard
from reframe.cli import main
assert main(['--diagnose-env', '--native-debug']) == 1
assert seen
"""
        result = subprocess.run([sys.executable, "-c", program], text=True, capture_output=True, check=True)
        self.assertIn("Native diagnostics:", result.stderr)
        self.assertIn("Traceback", result.stderr)
        self.assertIn("TEST_IMPORT_BLOCK", result.stderr)

    def test_cli_returns_typed_configuration_and_preserves_defaults(self):
        args = cli.parse_args(["in.mp4", "out.mp4", "--no-saliency-amp", "--dead-zone", ".15"])
        self.assertIsInstance(args, config.AppConfig)
        self.assertEqual(args.pose_model, "yolo26n-pose.pt")
        self.assertEqual(args.seg_model, "yolo26n-seg.pt")
        self.assertEqual(args.saliency_interval, 3)
        self.assertFalse(hasattr(args, "saliency_model"))
        self.assertIsNone(args.precrop)
        self.assertFalse(args.post_restore)
        self.assertFalse(args.saliency_amp)
        self.assertEqual(args.dead_zone, .15)
        self.assertTrue(cli.parse_args(["--diagnose-env"]).saliency_amp)

    def test_cli_precrop_choices_and_removed_backend_selection(self):
        for choice in ("middle", "left", "right"):
            with self.subTest(precrop=choice):
                args = cli.parse_args(["in.mp4", "out.mp4", "--precrop", choice])
                self.assertEqual(args.precrop, choice)
        for extra in (["--precrop", "full"], ["--saliency-model", "handcrafted"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as result:
                cli.parse_args(["in.mp4", "out.mp4", *extra])
            self.assertEqual(result.exception.code, 2)
        with self.assertRaisesRegex(ValueError, "precrop"):
            config.validate_config(config.AppConfig(input="in.mp4", output="out.mp4", precrop="invalid"))

    def test_cli_exception_is_terminal_traceback_with_failure_exit(self):
        output = io.StringIO()
        with patch.object(runtime, "resolve_yolo_device", return_value="cpu"), \
                patch.object(runtime, "log_runtime_info"), \
                patch.object(pipeline, "process_video", side_effect=RuntimeError("test failure")), \
                patch.object(cli, "setup_logging"), contextlib.redirect_stderr(output), \
                self.assertLogs(level="ERROR") as logs:
            self.assertEqual(cli.main(["in.mp4", "out.mp4"]), 1)
        self.assertIn("Traceback", "\n".join(logs.output))
        self.assertIn("test failure", "\n".join(logs.output))


class PipelineLifetimeTests(unittest.TestCase):
    def run_pipeline(self, failure=None):
        args = config.apply_preset(cli.parse_args(["input.mp4", "output.mp4", "--max-missed-frames", "0"]))
        capture = MagicMock()
        capture.isOpened.return_value = True
        props = {pipeline.cv2.CAP_PROP_FPS: 30, pipeline.cv2.CAP_PROP_FRAME_WIDTH: 400,
                 pipeline.cv2.CAP_PROP_FRAME_HEIGHT: 240, pipeline.cv2.CAP_PROP_FRAME_COUNT: 3}
        capture.get.side_effect = props.__getitem__
        tracker = MagicMock(class_names={0: "person"}, actual_device="cpu")
        tracker.track.return_value = ()
        pose_cache = MagicMock()
        pose_cache.helper.actual_device = None
        pose_cache.helper.rois_inferred = pose_cache.helper.rois_matched = pose_cache.cache_hits = 0
        pose_cache.primary_only_frames = pose_cache.full_scan_frames = pose_cache.rois_skipped = 0
        saliency = MagicMock()
        saliency.process.return_value = SimpleNamespace(map=np.zeros((64, 64), np.float32))
        saliency.telemetry.return_value = {"active_backend": "handcrafted"}
        events = []
        def observe(frame, tracks, helper, state, context, top_k):
            events.append(("pose", context.frame_index))
            return FrameObservations(context, tracks)
        def saliency_process(frame, observations, **kwargs):
            events.append(("saliency", observations.frame.frame_index))
            return saliency.process.return_value
        saliency.process.side_effect = saliency_process
        writer = MagicMock(encoder="test")
        writer.write.side_effect = failure
        frames = ((i, np.zeros((240, 400, 3), np.uint8)) for i in range(1, 4))
        with patch.object(pipeline.cv2, "VideoCapture", return_value=capture), \
                patch.object(pipeline, "SegmentationTracker", return_value=tracker), \
                patch.object(pipeline, "PoseCueCache", return_value=pose_cache), \
                patch.object(pipeline, "YOLOPoseHelper"), \
                patch.object(pipeline, "observe_poses", side_effect=observe), \
                patch.object(pipeline, "build_saliency_helper", return_value=saliency), \
                patch.object(pipeline, "iter_video_frames", return_value=frames), \
                patch.object(pipeline, "DirectVideoWriter", return_value=writer), \
                patch.object(pipeline, "build_global_saliency_observation", return_value=None), \
                patch.object(pipeline.InlineSceneDetector, "update", side_effect=[True, False, False]), \
                patch.object(pipeline.shutil, "which", return_value="ffmpeg"):
            if failure is None:
                with self.assertLogs(level="INFO") as logs:
                    pipeline.process_video(args)
                summary = json.loads(next(x.split("Summary: ", 1)[1] for x in logs.output if "Summary: " in x))
                self.assertEqual(summary["frames_processed"], 3)
                self.assertEqual(summary["frames_with_subject"], 0)
                self.assertEqual(events, [(kind, i) for i in range(1, 4) for kind in ("pose", "saliency")])
                writer.release.assert_called_once()
            else:
                with self.assertRaises(type(failure)):
                    pipeline.process_video(args)
        for resource in (tracker, pose_cache, saliency):
            resource.close.assert_called_once()
        writer.abort.assert_called_once()
        capture.release.assert_called_once()
        self.assertIsNone(frames.gi_frame)

    def test_empty_scene_exercises_lost_subject_path_and_cleanup(self):
        self.run_pipeline()

    def test_encode_error_and_interrupt_close_all_resources(self):
        for failure in (RuntimeError("encoder failed"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                self.run_pipeline(failure)

    def test_early_pose_initialization_error_closes_already_created_models(self):
        args = config.apply_preset(cli.parse_args(["in.mp4", "out.mp4"]))
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.return_value = 128
        tracker, saliency = MagicMock(), MagicMock()
        with patch.object(pipeline.cv2, "VideoCapture", return_value=capture), \
                patch.object(pipeline, "SegmentationTracker", return_value=tracker), \
                patch.object(pipeline, "build_saliency_helper", return_value=saliency), \
                patch.object(pipeline, "YOLOPoseHelper", side_effect=RuntimeError("pose load")), \
                patch.object(pipeline.shutil, "which", return_value="ffmpeg"), \
                self.assertRaisesRegex(RuntimeError, "pose load"):
            pipeline.process_video(args)
        tracker.close.assert_called_once()
        saliency.close.assert_called_once()

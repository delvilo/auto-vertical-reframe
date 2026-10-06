"""Runtime/backend regression tests; no CUDA or model downloads required."""
import contextlib
import io
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

for module in ("cv2", "scenedetect", "ultralytics", "mediapipe"):
    try:
        __import__(module)
    except ImportError:
        sys.modules[module] = MagicMock()

import numpy as np
import auto_reframe as app


def arguments(*extra):
    with patch.object(sys, "argv", ["auto_reframe.py", "input.mp4", "output.mp4", *extra]):
        return app.parse_args()


class BackendTests(unittest.TestCase):
    def test_cuda_selection_is_explicit(self):
        torch = MagicMock()
        torch.cuda.is_available.return_value = True
        torch.cuda.device_count.return_value = 1
        with patch.object(app, "torch", torch):
            for value in ("auto", "0", "cuda", "cuda:0"):
                self.assertEqual(app.resolve_yolo_device(value), "cuda:0")
            with self.assertRaises(RuntimeError):
                app.resolve_yolo_device("cuda:1")

    def test_requested_cuda_never_silently_becomes_cpu(self):
        with patch.object(app, "torch", None):
            with self.assertRaises(RuntimeError):
                app.resolve_yolo_device("0")
            self.assertEqual(app.resolve_yolo_device("cpu"), "cpu")
            with self.assertLogs(level="WARNING"):
                self.assertEqual(app.resolve_yolo_device("auto"), "cpu")

    def test_actual_cpu_predictor_rejected_for_cuda(self):
        model = SimpleNamespace(predictor=SimpleNamespace(device="cpu"))
        with self.assertRaisesRegex(RuntimeError, "device mismatch"):
            app.verify_yolo_device(model, "cuda:0")

    def test_diagnostic_can_run_without_video_paths(self):
        with patch.object(sys, "argv", ["auto_reframe.py", "--diagnose-env"]):
            self.assertTrue(app.parse_args().diagnose_env)
        with patch.object(sys, "argv", ["auto_reframe.py"]), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                app.parse_args()

    def test_invalid_frame_limit_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            arguments("--max-frames", "0")

    def test_face_inference_exception_is_visible(self):
        helper = object.__new__(app.MediaPipeFaceHelper)
        helper.is_tasks = True
        helper.detector = MagicMock()
        helper.detector.detect.side_effect = RuntimeError("face inference failed")
        prepared = SimpleNamespace(face_view=lambda: (np.zeros((20, 10, 3), np.uint8), 0, 0))
        with patch.object(app, "mp", MagicMock()), self.assertLogs(level="WARNING") as logs:
            self.assertIsNone(helper.detect_in_person_box(None, None, prepared))
        self.assertIn("RuntimeError: face inference failed", "\n".join(logs.output))

    def test_pose_inference_exception_is_visible(self):
        helper = object.__new__(app.MediaPipePoseHelper)
        helper.is_tasks = True
        helper.detector = MagicMock()
        helper.detector.detect.side_effect = RuntimeError("pose inference failed")
        prepared = SimpleNamespace(pose_view=lambda: (np.zeros((30, 10, 3), np.uint8), 0, 0))
        with patch.object(app, "mp", MagicMock()), self.assertLogs(level="WARNING") as logs:
            self.assertIsNone(helper.detect_in_person_box(None, None, prepared=prepared))
        self.assertIn("RuntimeError: pose inference failed", "\n".join(logs.output))


class ProcessOutputTests(unittest.TestCase):
    def test_stderr_is_visible_before_child_exits(self):
        ready = threading.Event()
        class RecordingStream(io.StringIO):
            def write(self, value):
                result = super().write(value)
                if "early error" in self.getvalue():
                    ready.set()
                return result
        stream = RecordingStream()
        command = [sys.executable, "-u", "-c",
                   "import sys; print('early error', file=sys.stderr, flush=True); sys.stdin.readline()"]
        with contextlib.redirect_stderr(stream):
            process, stderr = app.start_ffmpeg(command, stdin=subprocess.PIPE)
            try:
                self.assertTrue(ready.wait(5), "stderr was buffered until process exit")
                self.assertIsNone(process.poll())
                process.stdin.write(b"finish\n")
                process.stdin.close()
                self.assertEqual(process.wait(timeout=5), 0)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                if not process.stdin.closed:
                    process.stdin.close()
                tail = stderr.finish()
        self.assertIn("early error", tail)

    def test_large_stderr_cannot_fill_pipe_and_deadlock(self):
        stream = io.StringIO()
        command = [sys.executable, "-u", "-c",
                   "import sys; sys.stderr.write('x'*300000+'\\nFINAL ERROR\\n'); sys.exit(7)"]
        with contextlib.redirect_stderr(stream):
            result = app.run_ffmpeg(command, timeout=5)
        self.assertEqual(result.returncode, 7)
        self.assertGreater(len(stream.getvalue()), 300000)
        self.assertLessEqual(len(result.stderr), 6000)
        self.assertIn("FINAL ERROR", result.stderr)

    def test_timeout_preserves_early_error(self):
        stream = io.StringIO()
        command = [sys.executable, "-u", "-c",
                   "import sys,time; print('before timeout', file=sys.stderr, flush=True); time.sleep(30)"]
        with contextlib.redirect_stderr(stream), self.assertRaises(subprocess.TimeoutExpired):
            app.run_ffmpeg(command, timeout=1)
        self.assertIn("before timeout", stream.getvalue())

    def test_hevc_preflight_uses_requested_encoder_and_logs_fallback(self):
        args = arguments("--video-encoder", "hevc_nvenc", "--ffmpeg-log-level", "info")
        outcomes = [subprocess.CompletedProcess([], 1, stderr="driver error"),
                    subprocess.CompletedProcess([], 0, stderr="")]
        with patch.object(app, "has_ffmpeg_encoder", return_value=True), \
                patch.object(app, "run_ffmpeg", side_effect=outcomes) as run, \
                self.assertLogs(level="WARNING") as logs:
            self.assertEqual(app.select_live_encoder(args, 30, (320, 240), None), "libx264")
        command = run.call_args_list[0].args[0]
        self.assertEqual(command[command.index("-c:v") + 1], "hevc_nvenc")
        self.assertEqual(command[command.index("-loglevel") + 1], "info")
        self.assertIn("hevc_nvenc failed", "\n".join(logs.output))

    def test_diagnostic_fallback_is_failure_not_gpu_success(self):
        args = arguments("--video-encoder", "hevc_nvenc")
        args.yolo_device = "cpu"
        with patch.object(app, "select_live_encoder", return_value="libx264"), \
                patch.object(app.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)), \
                self.assertLogs(level="ERROR"):
            self.assertEqual(app.diagnose_environment(args), 1)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires FFmpeg")
class EncodingSmokeTests(unittest.TestCase):
    def test_direct_and_lossless_writers_still_create_video(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mkv"
            output = Path(directory) / "output.mp4"
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            writer = app.LosslessWriter(str(source), 2, (64, 64))
            try:
                writer.write(frame)
                writer.write(frame)
                writer.release()
            finally:
                writer.abort()
            muxed = Path(directory) / "muxed.mp4"
            encoder = app.run_ffmpeg_mux(str(source), str(source), str(muxed), "libx264",
                                         "192k", 18, "medium", app.build_video_filters(True))
            self.assertEqual(encoder, "libx264")
            self.assertTrue(muxed.is_file())
            args = arguments("--video-encoder", "libx264")
            writer = app.DirectVideoWriter(output, source, 2, (64, 64), args,
                                           app.build_video_filters(True))
            try:
                writer.write(frame)
                writer.write(frame)
                writer.release()
            finally:
                writer.abort()
            probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                                    "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(output)],
                                   capture_output=True, text=True, check=True)
            self.assertEqual(probe.stdout.strip(), "2")


if __name__ == "__main__":
    unittest.main()

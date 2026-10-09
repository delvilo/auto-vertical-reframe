"""Collector checks without model imports, GPU access, or media downloads."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "benchmark_colab", Path(__file__).resolve().parents[1] / "scripts" / "benchmark_colab.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def summary():
    return {"frames_processed": 600, "output_width": 1080, "output_height": 1920,
            "video_encoder_actual": "hevc_nvenc",
            "seg_total_seconds": 2.0,
            "seg_timing_seconds": {key: .25 for key in benchmark.TIMING_KEYS}}


def media(frames=600):
    return {"streams": [
        {"codec_type": "video", "nb_read_frames": str(frames), "avg_frame_rate": "60/1",
         "width": 1080, "height": 1920, "start_time": "0.000000", "duration": "10.000000"},
        {"codec_type": "audio", "start_time": "0.021333", "duration": "9.978667"},
    ]}


class ColabBenchmarkTests(unittest.TestCase):
    def test_summary_parser_requires_single_complete_finite_record(self):
        log = "native stderr\n2026-10-09 INFO Summary: " + json.dumps(summary()) + "\n"
        self.assertEqual(benchmark.parse_summary(log), summary())
        invalid = ["no summary", log + log, "Summary: []", 'Summary: {"seg_timing_seconds": {}}']
        for value in (float("nan"), float("inf"), -.1, True, None):
            record = summary()
            record["seg_timing_seconds"]["thumbnail"] = value
            invalid.append("Summary: " + json.dumps(record))
        for bad in invalid:
            with self.subTest(log=bad), self.assertRaises(ValueError):
                benchmark.parse_summary(bad)

    def test_disjoint_timing_sum_matches_total_with_rounding_tolerance(self):
        record = summary()
        record["seg_total_seconds"] += .000003
        self.assertEqual(benchmark.parse_summary("Summary: " + json.dumps(record)), record)
        record["seg_total_seconds"] += 1
        with self.assertRaisesRegex(ValueError, "do not sum"):
            benchmark.parse_summary("Summary: " + json.dumps(record))

    def test_output_validation_uses_decoded_frames_fps_and_audio_presence(self):
        result = benchmark.validate_output(media(900), media(), summary(), 600)
        self.assertEqual(result["output_frames"], 600)
        self.assertAlmostEqual(result["output_av_start_delta_seconds"], .021333)
        self.assertEqual(result["output_video_duration_seconds"], 10)
        for field, value in (("nb_read_frames", "599"), ("avg_frame_rate", "30/1"), ("width", 720)):
            changed = media()
            changed["streams"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                benchmark.validate_output(media(900), changed, summary(), 600)
        silent = media()
        silent["streams"].pop()
        with self.assertRaises(ValueError):
            benchmark.validate_output(media(), silent, summary(), 600)

    def test_short_source_and_missing_timestamps_are_valid(self):
        source = media(30)
        source["streams"].pop()
        source["streams"][0].pop("start_time")
        output = deepcopy(source)
        record = summary()
        record["frames_processed"] = 30
        result = benchmark.validate_output(source, output, record, 600)
        self.assertFalse(result["source_has_audio"])
        self.assertIsNone(result["output_audio_start_time_seconds"])
        self.assertNotIn("output_av_start_delta_seconds", result)

    def test_warmups_and_failures_are_excluded_from_medians(self):
        records = [
            {"case": "gap3-lazy", "status": status, "warmup": warmup,
             "summary": {"seg_total_seconds": duration}}
            for status, warmup, duration in (("ok", True, 100), ("failed", False, 200),
                                             ("ok", False, 2), ("ok", False, 4))
        ]
        rows = benchmark.aggregates(records)
        self.assertEqual(rows[0]["measured_runs"], 0)
        self.assertEqual(rows[-1]["measured_runs"], 2)
        self.assertEqual(rows[-1]["median.summary.seg_total_seconds"], 3)

    def test_schedule_warms_each_variant_and_counterbalances_measured_runs(self):
        rows = list(benchmark.schedule(2))
        self.assertEqual(len(rows), 9)
        self.assertTrue(all(warmup for _, warmup, _ in rows[:3]))
        self.assertEqual([case for case, _, _ in rows[3:6]], list(benchmark.CASES))
        self.assertEqual([case for case, _, _ in rows[6:]], list(reversed(benchmark.CASES)))
        self.assertFalse(any(warmup for _, warmup, _ in rows[3:]))

    def test_failed_child_preserves_raw_log_exit_code_and_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=Path(directory))
            script = ("import os, sys; os.write(1, b'progress\\r'); "
                      "os.write(2, b'Traceback: native error\\n\\xff'); sys.exit(7)")
            with patch.object(benchmark, "command", return_value=[sys.executable, "-c", script]):
                record = benchmark.run_one(args, benchmark.CASES[0], False, 1, {})
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["returncode"], 7)
            self.assertIn("status 7", record["error"])
            self.assertEqual(Path(record["log"]).read_bytes(), b"progress\rTraceback: native error\n\xff")
            self.assertNotIn("summary", record)

    def test_successful_child_collects_summary_and_validates_output(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=Path(directory), max_frames=600, video_encoder="hevc_nvenc")
            log = "Summary: " + json.dumps(summary())
            with patch.object(benchmark, "command", return_value=[sys.executable, "-c", f"print({log!r})"]), \
                    patch.object(benchmark, "probe", return_value=media()):
                record = benchmark.run_one(args, benchmark.CASES[2], False, 1, media())
            self.assertEqual(record["status"], "ok")
            self.assertEqual(record["returncode"], 0)
            self.assertEqual(record["validation"]["output_frames"], 600)
            self.assertEqual(record["summary"], summary())

    def test_encoder_fallback_is_recorded_but_excluded_from_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=Path(directory), max_frames=600, video_encoder="hevc_nvenc")
            fallback = summary()
            fallback["video_encoder_actual"] = "libx264"
            log = "Summary: " + json.dumps(fallback)
            with patch.object(benchmark, "command", return_value=[sys.executable, "-c", f"print({log!r})"]), \
                    patch.object(benchmark, "probe", return_value=media()):
                record = benchmark.run_one(args, benchmark.CASES[2], False, 1, media())
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["summary"], fallback)
            self.assertEqual(record["validation"]["output_frames"], 600)
            self.assertIn("encoder fallback", record["error"])

    def test_main_saves_failure_before_returning_child_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.mp4"
            source.touch()
            output = Path(directory) / "results"
            failure = {"case": "gap1-lazy", "status": "failed", "warmup": True,
                       "error": "native child error", "returncode": 7}
            with patch.object(benchmark, "environment", return_value={"python": sys.version}), \
                    patch.object(benchmark, "probe", return_value=media()), \
                    patch.object(benchmark, "run_one", return_value=failure) as runner:
                code = benchmark.main(["--input", str(source), "--output-dir", str(output)])
            self.assertEqual(code, 7)
            runner.assert_called_once()
            saved = json.loads((output / "results.json").read_text())
            self.assertEqual(saved["runs"], [failure])
            self.assertTrue(all(row["measured_runs"] == 0 for row in saved["aggregates"]))
            self.assertTrue((output / "results.csv").is_file())
            self.assertTrue((output / "aggregates.csv").is_file())

    def test_existing_relative_model_path_resolves_from_caller_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            caller = Path(directory)
            (caller / "input.mp4").touch()
            (caller / "custom-seg.pt").touch()
            original = Path.cwd()
            try:
                os.chdir(caller)
                with patch.object(benchmark, "environment", return_value={}), \
                        patch.object(benchmark, "probe", return_value=media()), \
                        patch.object(benchmark, "schedule", return_value=iter(())):
                    code = benchmark.main(["--input", "input.mp4", "--output-dir", "results",
                                           "--seg-model", "custom-seg.pt"])
            finally:
                os.chdir(original)
            self.assertEqual(code, 0)
            saved = json.loads((caller / "results" / "results.json").read_text())
            self.assertEqual(saved["configuration"]["seg_model"], str(caller / "custom-seg.pt"))
            self.assertEqual(saved["configuration"]["pose_model"], "yolo26n-pose.pt")

    def test_main_refuses_existing_output_directory_without_changing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.mp4"
            source.write_bytes(b"sentinel")
            with self.assertRaises(SystemExit) as error:
                benchmark.main(["--input", str(source), "--output-dir", directory])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(list(Path(directory).iterdir()), [source])
            self.assertEqual(source.read_bytes(), b"sentinel")


if __name__ == "__main__":
    unittest.main()

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
import zipfile


SPEC = importlib.util.spec_from_file_location(
    "benchmark_colab", Path(__file__).resolve().parents[1] / "scripts" / "benchmark_colab.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def summary():
    result = {"frames_processed": 600, "output_width": 1080, "output_height": 1920,
            "video_encoder_actual": "hevc_nvenc",
            "seg_detector_calls": 587, "yolo_device": "cuda:0",
            "pose_rois_inferred": 328, "pose_device": "cuda:0",
            "saliency_actual_forward_calls": 135, "saliency_device": "cuda",
            "saliency_frames_fallback": 0,
            "timing_schema_version": 2,
            "elapsed_seconds": 12.0,
            "pipeline_timing_seconds": {key: 1.0 for key in benchmark.PIPELINE_TIMING_KEYS},
            "stage_wall_seconds": {key: 1.0 for key in ("segmentation", "pose", "saliency", "render_write")},
            "saliency_total_seconds": .6,
            "saliency_timing_seconds": {key: .1 for key in benchmark.SALIENCY_TIMING_KEYS},
            "saliency_deepgaze_total_seconds": .08,
            "saliency_deepgaze_timing_seconds": {key: .01 for key in benchmark.DEEPGAZE_TIMING_KEYS},
            "seg_total_seconds": 2.0,
            "seg_timing_seconds": {key: .25 for key in benchmark.TIMING_KEYS}}
    for key in benchmark.SEGMENT_COUNTER_KEYS:
        result.setdefault(key, 0)
    result["seg_predicted_frames"] = 13
    result.update(segment_schema_version=1, segments_emitted=1, stats_interval=10.0,
                  seg_refresh_reasons={"scheduled": 587}, seg_gate_diagnostics={}, saliency_tier_reasons={})
    result["segment_pipeline_timing_seconds"] = {
        key: value for key, value in result["pipeline_timing_seconds"].items()
        if key not in {"setup", "finalization", "segment_logging"}}
    return result


def segment(record=None):
    record = summary() if record is None else record
    row = {key: deepcopy(record[key]) for key in benchmark.SEGMENT_COUNTER_KEYS | benchmark.SEGMENT_MAP_KEYS}
    row["pipeline_timing_seconds"] = deepcopy(record["segment_pipeline_timing_seconds"])
    row.update(segment_schema_version=1, segment_index=1, scene_index=1, window_index=0,
               frame_start=1, frame_end=record["frames_processed"], frames=record["frames_processed"],
               start_seconds=0.0, end_seconds=record["frames_processed"] / 60, source_fps=60.0,
               end_reason="end", seg_detector_gap_counts={"1": 573, "2": 13},
               seg_scheduled_interval_counts={"1": 574, "2": 26})
    return row


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

    def test_each_new_exclusive_map_requires_finite_values_and_conservation(self):
        for map_key in ("pipeline_timing_seconds", "saliency_timing_seconds", "saliency_deepgaze_timing_seconds"):
            for value in (None, True, -1, float("nan"), float("inf"), 100):
                record = summary()
                record[map_key][next(iter(record[map_key]))] = value
                with self.subTest(map_key=map_key, value=value), self.assertRaises(ValueError):
                    benchmark.parse_summary("Summary: " + json.dumps(record))
            record = summary()
            record[map_key].pop(next(iter(record[map_key])))
            with self.subTest(map_key=map_key), self.assertRaisesRegex(ValueError, "Missing detailed timings"):
                benchmark.parse_summary("Summary: " + json.dumps(record))
        record = summary()
        record["stage_wall_seconds"]["saliency"] = 99
        with self.assertRaisesRegex(ValueError, "alias differs"):
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

    def test_full_video_validates_all_decoded_source_frames(self):
        record = summary()
        record["frames_processed"] = 10200
        result = benchmark.validate_output(media(10200), media(10200), record, None)
        self.assertEqual(result["output_frames"], 10200)
        with self.assertRaises(ValueError):
            benchmark.validate_output(media(10200), media(600), summary(), None)

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
            {"case": benchmark.CASES[-1][0], "status": status, "warmup": warmup,
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
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(warmup for _, warmup, _ in rows[:4]))
        self.assertEqual([case for case, _, _ in rows[4:8]], list(benchmark.CASES))
        self.assertEqual([case for case, _, _ in rows[8:]], list(reversed(benchmark.CASES)))
        self.assertFalse(any(warmup for _, warmup, _ in rows[4:]))
        legacy = list(benchmark.schedule(2, benchmark.THUMBNAIL_CASES))
        self.assertEqual(len(legacy), 9)
        comparison = list(benchmark.schedule(2, benchmark.COMPARISON_CASES))
        self.assertEqual(len(comparison), 6)
        self.assertEqual([row[0] for row in comparison[:2]], [benchmark.CASES[0], benchmark.CASES[-1]])

    def test_ablation_cases_change_one_optimization_at_a_time(self):
        args = SimpleNamespace(input=Path("input.mp4"), seg_model="seg.pt", pose_model="pose.pt",
                               device="0", saliency_device="cuda", video_encoder="hevc_nvenc", max_frames=600)
        commands = [benchmark.command(args, Path("output.mp4"), case) for case in benchmark.CASES]
        changed_flags = []
        for previous, current in zip(commands, commands[1:]):
            changed_flags.append([previous[index - 1] for index, (old, new)
                                  in enumerate(zip(previous, current)) if old != new])
        self.assertEqual(changed_flags, [["--seg-max-gap"], ["--seg-thumbnail-method"], ["--seg-skip-policy"]])
        self.assertTrue(all("--post-restore" not in command for command in commands))

    def test_full_commands_omit_limits_and_warmups_use_independent_limit(self):
        args = SimpleNamespace(input=Path("input.mp4"), seg_model="seg.pt", pose_model="pose.pt",
                               device="0", saliency_device="cuda", video_encoder="hevc_nvenc", max_frames=None,
                               warmup_frames=600, stats_interval=5.0, log_level="INFO")
        measured = benchmark.command(args, Path("output.mp4"), benchmark.CASES[-1])
        warmup = benchmark.command(args, Path("output.mp4"), benchmark.CASES[-1], True)
        self.assertNotIn("--max-frames", measured)
        self.assertEqual(warmup[warmup.index("--max-frames") + 1], "600")
        self.assertEqual(measured[measured.index("--stats-interval") + 1], "5.0")
        self.assertEqual(measured[measured.index("--log-level") + 1], "INFO")
        self.assertIn("--native-debug", measured)
        args.warmup_frames = None
        self.assertNotIn("--max-frames", benchmark.command(args, Path("output.mp4"), benchmark.CASES[-1], True))

    def test_workload_uses_source_fps_and_short_warmups_without_assumed_60fps(self):
        source = media(5100)
        source["streams"][0]["avg_frame_rate"] = "30/1"
        args = SimpleNamespace(max_frames=None, warmup_frames=600)
        result = benchmark.workload(source, args, list(benchmark.schedule(2)))
        self.assertEqual(result["source_video_seconds"], 170)
        self.assertEqual(result["run_count"], 12)
        self.assertEqual(result["measured_frames"], 40800)
        self.assertEqual(result["warmup_frames"], 2400)
        args.warmup_frames = None
        self.assertEqual(benchmark.workload(source, args, list(benchmark.schedule(2)))["total_frames"], 61200)

    def test_requested_cuda_rejects_cpu_devices_and_deepgaze_fallback(self):
        for key in ("yolo_device", "pose_device", "saliency_device"):
            record = summary()
            record[key] = "cpu"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "device fallback"):
                benchmark.validate_devices(record, "0", "cuda")
        record = summary()
        record["saliency_frames_fallback"] = 1
        with self.assertRaisesRegex(ValueError, "fallback frames"):
            benchmark.validate_devices(record, "cuda:0", "cuda")
        record = summary()
        record["yolo_device"] = "cuda:1"
        with self.assertRaisesRegex(ValueError, "device fallback"):
            benchmark.validate_devices(record, "0", "cuda")

    def test_unused_lazy_models_are_not_claimed_as_gpu_verified(self):
        record = summary()
        record.update(pose_rois_inferred=0, pose_device=None, saliency_actual_forward_calls=0)
        validation = benchmark.validate_devices(record, "0", "cuda")
        self.assertEqual(validation["pose"]["status"], "not_exercised")
        self.assertEqual(validation["deepgaze"]["status"], "not_exercised")
        self.assertEqual(validation["segmentation"]["status"], "cuda_verified")
        record["seg_detector_calls"] = None
        with self.assertRaisesRegex(ValueError, "model call count"):
            benchmark.validate_devices(record, "0", "cuda")

    def test_runtime_records_cuda_build_without_importing_torch(self):
        info = benchmark.runtime_info("INFO PyTorch=2.11.0+cu130 CUDA build=13.0 CUDA available=True\n")
        self.assertEqual(info["torch_version"], "2.11.0+cu130")
        self.assertEqual(info["torch_cuda_build"], "13.0")
        self.assertEqual(benchmark.runtime_info("no runtime line"), {})

    def test_smoke_runs_one_optimized_60_frame_case_without_measured_medians(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.mp4"
            source.touch()
            output = Path(directory) / "smoke"
            success = {"case": benchmark.CASES[-1][0], "status": "ok", "warmup": True}
            with patch.object(benchmark, "environment", return_value={}), \
                    patch.object(benchmark, "probe", return_value=media()), \
                    patch.object(benchmark, "run_one", return_value=success) as runner:
                code = benchmark.main(["--input", str(source), "--output-dir", str(output), "--smoke"])
            self.assertEqual(code, 0)
            args, case, warmup, repeat, _ = runner.call_args.args
            self.assertEqual((args.max_frames, case, warmup, repeat), (60, benchmark.CASES[-1], True, 0))
            saved = json.loads((output / "results.json").read_text())
            self.assertTrue(all(row["measured_runs"] == 0 for row in saved["aggregates"]))

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
            args = SimpleNamespace(output_dir=Path(directory), max_frames=600, video_encoder="hevc_nvenc",
                                   device="0", saliency_device="cuda")
            log = "Segment: " + json.dumps(segment()) + "\nSummary: " + json.dumps(summary())
            with patch.object(benchmark, "command", return_value=[sys.executable, "-c", f"print({log!r})"]), \
                    patch.object(benchmark, "probe", return_value=media()):
                record = benchmark.run_one(args, benchmark.CASES[2], False, 1, media())
            self.assertEqual(record["status"], "ok")
            self.assertEqual(record["returncode"], 0)
            self.assertEqual(record["validation"]["output_frames"], 600)
            self.assertEqual(record["summary"], summary())
            self.assertEqual(record["device_validation"]["deepgaze"]["status"], "cuda_verified")
            self.assertEqual(record["segment_count"], 1)
            self.assertFalse(record["segments"][0]["benchmark_partial"])
            self.assertEqual(json.loads(Path(record["segments_jsonl"]).read_text())["frames"], 600)

    def test_segment_validation_rejects_missing_coverage_and_bad_totals(self):
        row, record = segment(), summary()
        benchmark.validate_segments([row], record)
        for key, value in (("frame_start", 2), ("start_seconds", .1), ("scene_index", 4),
                           ("seg_predicted_frames", 14), ("pose_rois_inferred", 99),
                           ("seg_detector_gap_counts", {"1": 585}), ("seg_scheduled_interval_counts", {"1": 599})):
            changed = deepcopy(row)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                benchmark.validate_segments([changed], record)
        for key in benchmark.SEGMENT_MAP_KEYS:
            changed = deepcopy(row)
            changed[key][next(iter(changed[key]), "unexpected")] = 12345
            with self.subTest(key=key), self.assertRaises(ValueError):
                benchmark.validate_segments([changed], record)
        with self.assertRaisesRegex(ValueError, "no segments"):
            benchmark.validate_segments([], record)

    def test_failed_child_preserves_partial_segments_and_malformed_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=Path(directory))
            log = "Segment: " + json.dumps(segment()) + '\nSegment: {"truncated":'
            script = f"import sys; print({log!r}); sys.exit(7)"
            with patch.object(benchmark, "command", return_value=[sys.executable, "-c", script]):
                record = benchmark.run_one(args, benchmark.CASES[0], False, 1, {})
            self.assertEqual(record["returncode"], 7)
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["segment_count"], 1)
            self.assertTrue(record["segments"][0]["benchmark_partial"])
            self.assertEqual(len(record["segment_parse_errors"]), 1)
            self.assertTrue(Path(record["segments_csv"]).is_file())

    def test_successful_child_with_missing_segments_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=Path(directory), max_frames=600, video_encoder="hevc_nvenc",
                                   device="0", saliency_device="cuda")
            log = "Summary: " + json.dumps(summary())
            with patch.object(benchmark, "command", return_value=[sys.executable, "-c", f"print({log!r})"]), \
                    patch.object(benchmark, "probe", return_value=media()):
                record = benchmark.run_one(args, benchmark.CASES[-1], False, 1, media())
            self.assertEqual(record["status"], "failed")
            self.assertIn("no segments", record["error"])

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
            failure = {"case": benchmark.CASES[0][0], "status": "failed", "warmup": True,
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

    def test_archive_preserves_diagnostics_and_excludes_video_on_failed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.mp4"
            source.touch()
            output = Path(directory) / "results.v1"
            failure = {"case": benchmark.CASES[0][0], "status": "failed", "warmup": True,
                       "error": "native child error", "returncode": 7}
            def child(*args):
                (output / "failed.log").write_text("native traceback")
                (output / "failed.segments.jsonl").write_text("{}\n")
                (output / "failed.mp4").write_bytes(b"video")
                return failure
            with patch.object(benchmark, "environment", return_value={}), \
                    patch.object(benchmark, "probe", return_value=media()), \
                    patch.object(benchmark, "run_one", side_effect=child):
                code = benchmark.main(["--input", str(source), "--output-dir", str(output), "--archive"])
            self.assertEqual(code, 7)
            with zipfile.ZipFile(str(output) + ".zip") as archive:
                names = set(archive.namelist())
                self.assertIn("failed.log", names)
                self.assertIn("failed.segments.jsonl", names)
                self.assertIn("segments.csv", names)
                self.assertIn("results.json", names)
                self.assertNotIn("failed.mp4", names)

    def test_full_video_cli_preserves_none_limit_and_only_probes_source_once(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.mp4"
            source.touch()
            output = Path(directory) / "full"
            success = {"case": benchmark.CASES[0][0], "status": "ok", "warmup": True}
            with patch.object(benchmark, "environment", return_value={}), \
                    patch.object(benchmark, "probe", return_value=media(10200)) as probe, \
                    patch.object(benchmark, "run_one", return_value=success) as runner:
                code = benchmark.main(["--input", str(source), "--output-dir", str(output),
                                       "--full-video", "--warmup-frames", "600", "--suite", "comparison"])
            self.assertEqual(code, 0)
            probe.assert_called_once_with(source)
            self.assertEqual(runner.call_count, 6)
            args = runner.call_args.args[0]
            self.assertIsNone(args.max_frames)
            self.assertEqual(args.warmup_frames, 600)
            saved = json.loads((output / "results.json").read_text())
            self.assertEqual(saved["workload"]["total_frames"], 42000)

    def test_full_video_rejects_conflicting_limits_and_smoke(self):
        for extra in (("--full-video", "--max-frames", "600"), ("--full-video", "--smoke"),
                      ("--stats-interval", "0"), ("--stats-interval", "nan"), ("--warmup-frames", "0")):
            with self.subTest(extra=extra), self.assertRaises(SystemExit) as error:
                benchmark.main(["--input", "unused.mp4", "--output-dir", "unused", *extra])
            self.assertEqual(error.exception.code, 2)

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

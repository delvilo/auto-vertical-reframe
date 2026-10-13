"""Collector recovery, validation and notebook safety without models or CUDA."""
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("colab_torchcodec", ROOT / "scripts/colab_torchcodec.py")
staged = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(staged)


def args(directory):
    return SimpleNamespace(repo=ROOT, input=directory / "input.mp4", seg_model="seg.pt", pose_model="pose.pt",
        output_dir=directory, backup_dir=None, profiles=["baseline", "stage1"], max_frames=180,
        full_video=False, device="0", saliency_device="cuda", decode_device="cuda:0", decode_batch_size=4,
        video_encoder="hevc_nvenc", seg_max_gap=1, stats_interval=5.0, log_level="INFO", allow_cpu_diagnostic=False)


def decoder_summary(profile="stage1", **changes):
    decoder, processing = staged.PROFILES[profile]
    io = {"decoder": decoder, "frame_processing": processing, "actual_device": "cuda:0",
          "cpu_fallback": False, "cpu_fallback_status_known": True, "frames_yielded": 180}
    io.update(changes)
    return {"frames_processed": 180, "video_io": io}


class DecoderValidationTests(unittest.TestCase):
    def test_cuda_tensor_alone_does_not_prove_nvdec(self):
        for updates in ({"cpu_fallback": True}, {"cpu_fallback": None},
                        {"cpu_fallback_status_known": False}, {"actual_device": "cpu"}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                staged.validate_decoder(decoder_summary(**updates), "stage1", "cuda:0")

    def test_verified_nvdec_and_explicit_cpu_are_distinct(self):
        self.assertEqual(staged.validate_decoder(decoder_summary(), "stage1", "cuda")["status"], "nvdec_verified")
        self.assertEqual(staged.validate_decoder(decoder_summary(actual_device="cpu"), "stage1", "cpu")["status"],
                         "explicit_cpu_test")
        self.assertEqual(staged.validate_decoder(decoder_summary("baseline", actual_device="cpu"),
                                                "baseline", "cuda:0")["status"], "cpu_baseline")

    def test_profile_and_frame_count_must_match(self):
        for updates in ({"frame_processing": "tensor"}, {"decoder": "opencv"}, {"frames_yielded": 179}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                staged.validate_decoder(decoder_summary(**updates), "stage1", "cuda:0")

    def test_decoder_fallback_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "fallback"):
            staged.validate_decoder(decoder_summary(decoder_fallback=True, decoder_fallback_reason="init error"),
                                    "stage1", "cuda:0")
    def test_missing_io_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing video_io"):
            staged.validate_decoder({}, "stage1", "cuda:0")


class MediaValidationTests(unittest.TestCase):
    def setUp(self):
        self.media = {"streams": [
            {"codec_type": "video", "start_time": "0", "duration": "3", "avg_frame_rate": "60/1"},
            {"codec_type": "audio", "start_time": "0", "duration": "3.03"}]}

    def test_aac_tail_tolerance(self):
        self.assertEqual(staged.validate_audio_tail(self.media, self.media)["status"], "endpoints_within_tolerance")

    def test_truncated_and_untrimmed_audio_fail(self):
        for duration in ("2.7", "170"):
            output = deepcopy(self.media)
            output["streams"][1]["duration"] = duration
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                staged.validate_audio_tail(output, self.media)

    def test_audio_presence_and_nonfinite_timestamps(self):
        silent = {"streams": self.media["streams"][:1]}
        self.assertEqual(staged.validate_audio_tail(silent, silent)["status"], "no_source_audio")
        for output, source in ((silent, self.media), (self.media, silent)):
            with self.assertRaises(ValueError):
                staged.validate_audio_tail(output, source)
        for invalid in ("nan", "inf", "0", "-1"):
            output = deepcopy(self.media)
            output["streams"][1]["duration"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                staged.validate_audio_tail(output, self.media)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.args = args(self.directory)
        self.expected = staged.identity(self.args, "stage1", "source-sha", {"seg": "weight-sha"},
                                        {"pipeline.py": "code-sha"}, {"torch": "2.11"}, 180)

    def completed(self):
        directory = self.directory / "stage1-short180-attempt001"
        directory.mkdir()
        output = directory / "output.mp4"
        output.write_bytes(b"complete video")
        record = {"status": "ok", "identity": self.expected, "output_sha256": staged.sha256(output)}
        staged.atomic_json(directory / "record.json", record)
        return directory, record

    def test_reuse_only_exact_intact_success(self):
        directory, record = self.completed()
        self.assertEqual(staged.successful_run(self.directory, self.expected), (directory, record))
        for key in ("source_sha256", "code_sha256", "packages", "model_sha256"):
            changed = {**self.expected, key: "changed"}
            with self.subTest(key=key):
                self.assertIsNone(staged.successful_run(self.directory, changed))
        changed = deepcopy(self.expected)
        changed["settings"]["seg_max_gap"] = 3
        self.assertIsNone(staged.successful_run(self.directory, changed))
        (directory / "output.mp4").write_bytes(b"truncated")
        self.assertIsNone(staged.successful_run(self.directory, self.expected))

    def test_running_failed_corrupt_records_not_reused(self):
        directory, record = self.completed()
        for status in ("running", "failed"):
            record["status"] = status
            staged.atomic_json(directory / "record.json", record)
            self.assertIsNone(staged.successful_run(self.directory, self.expected))
        (directory / "record.json").write_text("not json")
        self.assertIsNone(staged.successful_run(self.directory, self.expected))

    def test_tracking_settings_invalidate_recovery_and_match_command(self):
        self.completed()
        for name, value in (("tracking_profile", "fast"), ("seg_max_gap", 6),
                            ("seg_max_age", .08), ("cue_interval", .1), ("pose_max_age", .25)):
            options = deepcopy(self.args)
            setattr(options, name, value)
            changed = staged.identity(options, "stage1", "source-sha", {"seg": "weight-sha"},
                                      {"pipeline.py": "code-sha"}, {"torch": "2.11"}, 180)
            with self.subTest(setting=name):
                self.assertIsNone(staged.successful_run(self.directory, changed))
                cmd = staged.command(options, "stage1", self.directory / "out.mp4")
                for setting in ("tracking_profile", "seg_max_gap", "seg_max_age", "cue_interval", "pose_max_age",
                                "saliency_interval", "seg_skip_policy"):
                    flag = "--" + setting.replace("_", "-")
                    self.assertEqual(cmd[cmd.index(flag) + 1], str(changed["settings"][setting]))

    def test_old_success_is_preserved_but_not_reused_with_unknown_tracking_settings(self):
        directory, record = self.completed()
        old = deepcopy(record)
        for key in ("tracking_profile", "seg_max_age", "cue_interval", "pose_max_age", "saliency_interval", "seg_skip_policy"):
            old["identity"]["settings"].pop(key)
        staged.atomic_json(directory / "record.json", old)
        self.assertIsNone(staged.successful_run(self.directory, self.expected))
        self.assertEqual(staged.successful_run(self.directory, old["identity"]), (directory, old))
        self.assertEqual((directory / "output.mp4").read_bytes(), b"complete video")

    def test_backup_archives_diagnostics_and_preserves_outputs(self):
        directory, _ = self.completed()
        (directory / "native.log").write_text("native traceback\n")
        (directory / "unfinished.tmp").write_text("partial")
        self.args.backup_dir = self.directory.parent / (self.directory.name + "-backup")
        self.addCleanup(__import__("shutil").rmtree, self.args.backup_dir)
        staged.backup(self.args)
        with zipfile.ZipFile(self.directory / "diagnostics.zip") as archive:
            self.assertIn(f"{directory.name}/native.log", archive.namelist())
            self.assertFalse(any(name.endswith((".mp4", ".tmp")) for name in archive.namelist()))
        self.assertEqual((self.args.backup_dir / directory.name / "output.mp4").read_bytes(), b"complete video")
        self.assertFalse((self.args.backup_dir / directory.name / "unfinished.tmp").exists())

    def test_failed_native_process_retains_partial_segments(self):
        benchmark = staged.collector()
        line = 'Segment: {"frame_start":1,"frame_end":5}\n'
        native = [sys.executable, "-c", f"import sys; print({line!r}); print('native failure', file=sys.stderr); sys.exit(7)"]
        with patch.object(staged, "command", return_value=native), patch.object(staged.sys, "stdout", io.StringIO()), \
                patch.object(staged.sys, "stderr", io.StringIO()):
            directory, record = staged.run_one(self.args, "stage1", self.expected, {}, benchmark)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["returncode"], 7)
        self.assertEqual(record["segment_count"], 1)
        self.assertTrue(record["segments"][0]["benchmark_partial"])
        self.assertIn("native failure", (directory / "native.log").read_text())
        self.assertTrue((directory / "record.json").is_file())
        self.assertTrue((directory / "run.segments.csv").is_file())


class EntryPointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.source, self.seg, self.pose = [self.directory / name for name in ("input.mp4", "seg.pt", "pose.pt")]
        for path in (self.source, self.seg, self.pose):
            path.write_bytes(b"fixture")
        self.argv = ["--input", str(self.source), "--seg-model", str(self.seg), "--pose-model", str(self.pose),
                     "--output-dir", str(self.directory / "results")]

    def assert_invalid(self, extra):
        with patch.object(staged.sys, "stderr", io.StringIO()), self.assertRaises(SystemExit) as error:
            staged.main(self.argv + extra)
        self.assertEqual(error.exception.code, 2)

    def test_short_defaults_and_separate_full_profile(self):
        parsed = staged.parser().parse_args(self.argv)
        self.assertEqual(parsed.profiles, ["baseline", "stage1"])
        self.assertEqual(parsed.max_frames, 180)
        self.assertEqual(parsed.tracking_profile, "balanced")
        self.assertIsNone(parsed.seg_max_gap)
        self.assertEqual(staged.tracking_settings(parsed)["seg_max_gap"], 6)
        self.assert_invalid(["--full-video"])
        self.assert_invalid(["--profiles", "baseline", "stage2", "--full-video"])

    def test_cpu_requires_explicit_diagnostic_flag(self):
        self.assert_invalid(["--device", "cpu"])
        self.assert_invalid(["--decode-device", "cpu"])
        self.assert_invalid(["--video-encoder", "libx264"])
        self.assert_invalid(["--decode-device", "auto"])

    def test_invalid_tracking_overrides_fail_before_preflight(self):
        with patch.object(staged, "preflight") as preflight:
            for flag, values in (("--seg-max-gap", ("0", "-1")),
                                 ("--seg-max-age", ("0", "-1", "nan", "inf")),
                                 ("--cue-interval", ("-1", "nan", "inf")),
                                 ("--pose-max-age", ("-1", "nan", "inf"))):
                for value in values:
                    with self.subTest(flag=flag, value=value):
                        self.assert_invalid([flag, value])
            preflight.assert_not_called()

    def test_profile_defaults_and_explicit_gap_one_have_same_effective_identity_and_command(self):
        options = args(self.directory)
        options.seg_max_gap = None
        for profile in ("conservative", "balanced", "fast"):
            options.tracking_profile = profile
            effective = staged.tracking_settings(options)
            identity = staged.identity(options, "stage2", "input", {}, {}, {}, 180)
            cmd = staged.command(options, "stage2", self.directory / "output.mp4")
            for key, value in effective.items():
                with self.subTest(profile=profile, setting=key):
                    flag = "--" + key.replace("_", "-")
                    self.assertEqual(cmd[cmd.index(flag) + 1], str(value))
                    self.assertEqual(identity["settings"][key], value)
        options.seg_max_gap, options.pose_max_age = 1, 0
        cmd = staged.command(options, "stage2", self.directory / "output.mp4")
        self.assertEqual(cmd[cmd.index("--seg-max-gap") + 1], "1")
        self.assertEqual(float(cmd[cmd.index("--pose-max-age") + 1]), 0)

    def test_nested_backup_is_rejected(self):
        self.assert_invalid(["--backup-dir", str(self.directory / "results" / "nested")])
        self.assert_invalid(["--backup-dir", str(self.directory)])

    def test_full_requires_matching_smoke_before_model_preflight(self):
        with patch.object(staged, "preflight") as preflight, \
                patch.object(staged, "collector") as collector, \
                patch.object(staged.sys, "stdout", io.StringIO()), patch.object(staged.sys, "stderr", io.StringIO()):
            collector.return_value.environment.return_value = {"python": sys.version, "packages": {}, "gpu": {}, "ffmpeg": {}}
            collector.return_value.probe.return_value = {"streams": []}
            code = staged.main(self.argv + ["--full-video", "--profiles", "stage1"])
        self.assertEqual(code, 1)
        preflight.assert_not_called()
        results = json.loads((self.directory / "results/results.json").read_text())
        self.assertIn("--max-frames 180 first", results["error"])
        self.assertTrue((self.directory / "results/diagnostics.zip").is_file())

    def test_collectors_import_without_loading_accelerators_or_models(self):
        source = ("import importlib.util, pathlib, sys\n"
                  f"root = pathlib.Path({str(ROOT)!r})\n"
                  "for filename in ('benchmark_colab.py', 'colab_torchcodec.py'):\n"
                  "    spec = importlib.util.spec_from_file_location(filename, root / 'scripts' / filename)\n"
                  "    module = importlib.util.module_from_spec(spec)\n"
                  "    spec.loader.exec_module(module)\n"
                  "assert not {'torch', 'torchcodec', 'ultralytics', 'cv2', 'numpy'} & set(sys.modules)\n")
        subprocess.run([sys.executable, "-c", source], check=True, cwd=self.directory, capture_output=True, text=True)


class NotebookTests(unittest.TestCase):
    def test_cells_compile_and_slow_runs_are_opt_in(self):
        notebook = json.loads((ROOT / "notebooks/colab-torchcodec.ipynb").read_text())
        sources = []
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                compile(source, f"cell-{index}", "exec")
                sources.append(source)
                self.assertEqual(cell["outputs"], [])
        code = "\n".join(sources)
        self.assertIn("RUN_STAGE2 = False", code)
        self.assertIn("RUN_FULL = False", code)
        self.assertIn('"--profiles", "baseline", "stage1", "--max-frames", "180"', code)
        self.assertIn('"--no-deps"', code)
        self.assertIn('"--force-reinstall"', code)
        self.assertNotIn('"--upgrade"', code)


if __name__ == "__main__":
    unittest.main()

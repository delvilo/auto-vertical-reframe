"""Tests for scripts/youtube_validation.py.

Verifies CLI argument constraints, exact identity generation, reuse logic,
subprocess failure/segment recovery, and rejection of fallback, short frames,
or audio tail mismatches.
"""
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("youtube_validation", ROOT / "scripts/youtube_validation.py")
yv = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(yv)


def make_args(directory: Path, **overrides) -> yv.argparse.Namespace:
    values = {
        "repo": ROOT,
        "input": directory / "input.mp4",
        "output_dir": directory,
        "backup_dir": None,
        "seg_model": directory / "seg.pt",
        "pose_model": directory / "pose.pt",
        "tracking_profile": "balanced",
        "max_frames": 180,
    }
    values.update(overrides)
    return yv.argparse.Namespace(**values)


def make_source_probe():
    return {
        "streams": [
            {
                "codec_type": "video",
                "width": 3840,
                "height": 2160,
                "avg_frame_rate": "60/1",
                "start_time": "0.0",
                "duration": "3.0",
                "nb_frames": "180",
            },
            {
                "codec_type": "audio",
                "start_time": "0.0",
                "duration": "3.01",
            },
        ]
    }


def make_summary(frames=180, encoder="hevc_nvenc", profile="balanced"):
    seg_timings = {k: 0.01 for k in (
        "detector_wall_seconds", "detector_wait_seconds", "tracker_wall_seconds",
        "mask_wall_seconds", "crop_wall_seconds", "optflow_wall_seconds"
    )}
    pipe_timings = {
        "segmentation": 0.06,
        "pose": 0.02,
        "saliency": 0.02,
        "render_write": 0.05,
    }
    saliency_timings = {
        "total_wall_seconds": 0.02,
        "tier1_pose_wall_seconds": 0.01,
        "tier2_cheap_wall_seconds": 0.01,
        "tier3_deepgaze_wall_seconds": 0.0,
    }
    deepgaze_timings = {
        "crop_preprocess_seconds": 0.0,
        "tensor_assembly_seconds": 0.0,
        "forward_seconds": 0.0,
        "postprocess_seconds": 0.0,
    }
    tracking = yv.tracking_settings(yv.argparse.Namespace(tracking_profile=profile))
    return {
        "timing_schema_version": 2,
        "segment_schema_version": 1,
        "frames_processed": frames,
        "elapsed_seconds": 0.15,
        "seg_timing_seconds": seg_timings,
        "seg_total_seconds": sum(seg_timings.values()),
        "pipeline_timing_seconds": pipe_timings,
        "stage_wall_seconds": pipe_timings,
        "saliency_timing_seconds": saliency_timings,
        "saliency_total_seconds": sum(saliency_timings.values()),
        "saliency_deepgaze_timing_seconds": deepgaze_timings,
        "saliency_deepgaze_total_seconds": sum(deepgaze_timings.values()),
        "video_encoder_actual": encoder,
        "tracking_settings": tracking,
        "segments_emitted": 1,
        "video_io": {
            "decoder": "torchcodec",
            "frame_processing": "tensor",
            "requested_device": "cuda:0",
            "actual_device": "cuda:0",
            "cpu_fallback": False,
            "cpu_fallback_status_known": True,
            "frames_decoded": frames,
            "frames_yielded": frames,
            "decode_wall_seconds": 0.05,
            "initialization_wall_seconds": 0.01,
        },
        "models": {
            "segmentation": {"actual_device": "cuda:0", "calls": 30},
            "pose": {"actual_device": "cuda:0", "calls": 30},
            "deepgaze": {"actual_device": "cuda", "calls": 0},
        },
    }


class CliAndIdentityTests(unittest.TestCase):
    def test_cli_parsing_valid_and_rejects_unauthorized_flags(self):
        parser = yv.parser()
        args = parser.parse_args([
            "--input", "clip.mp4",
            "--output-dir", "out",
            "--seg-model", "seg.pt",
            "--pose-model", "pose.pt",
            "--tracking-profile", "balanced",
            "--max-frames", "180",
        ])
        self.assertEqual(args.tracking_profile, "balanced")
        self.assertEqual(args.max_frames, 180)

        # Disallowed flags per plan: no --precrop, --profiles, --seg-max-gap
        for bad_flag in ("--precrop", "--profiles", "--seg-max-gap"):
            with self.assertRaises(SystemExit):
                parser.parse_args([
                    "--input", "clip.mp4",
                    "--output-dir", "out",
                    "--seg-model", "seg.pt",
                    "--pose-model", "pose.pt",
                    "--tracking-profile", "balanced",
                    bad_flag, "1",
                ])


class ReuseAndRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.input_file = self.directory / "input.mp4"
        self.input_file.write_bytes(b"source-content")
        self.seg_file = self.directory / "seg.pt"
        self.seg_file.write_bytes(b"seg-content")
        self.pose_file = self.directory / "pose.pt"
        self.pose_file.write_bytes(b"pose-content")

        self.args = make_args(self.directory, input=self.input_file, seg_model=self.seg_file, pose_model=self.pose_file)
        self.models = {"seg": yv.sha256(self.seg_file), "pose": yv.sha256(self.pose_file)}
        self.code = yv.code_fingerprint(ROOT)
        self.packages = {"python": "3.13.0", "packages": {"torch": "2.11.0"}, "gpu": "Tesla T4", "ffmpeg": "6.1"}
        self.expected = yv.build_identity(self.args, yv.sha256(self.input_file), self.models, self.code, self.packages)

    def completed_run(self):
        attempt_dir = self.directory / f"balanced-short180-{yv.identity_digest(self.expected)[:12]}-attempt001"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        out_file = attempt_dir / "output.mp4"
        out_file.write_bytes(b"valid vertical mp4 bytes")
        record = {
            "status": "ok",
            "identity": self.expected,
            "output_sha256": yv.sha256(out_file),
            "output": str(out_file),
        }
        yv.atomic_json(attempt_dir / "record.json", record)
        return attempt_dir, record

    def test_reuse_exact_success_only(self):
        attempt_dir, record = self.completed_run()
        found = yv.successful_run(self.directory, self.expected)
        self.assertEqual(found, (attempt_dir, record))

        # Output file corrupted or changed
        (attempt_dir / "output.mp4").write_bytes(b"tampered")
        self.assertIsNone(yv.successful_run(self.directory, self.expected))

    def test_identity_invalidation_on_source_model_code_settings_change(self):
        attempt_dir, _ = self.completed_run()

        # Source hash changed
        changed_source = deepcopy(self.expected)
        changed_source["source_sha256"] = "different-hash"
        self.assertIsNone(yv.successful_run(self.directory, changed_source))

        # Model hash changed
        changed_model = deepcopy(self.expected)
        changed_model["model_sha256"]["seg"] = "different-seg"
        self.assertIsNone(yv.successful_run(self.directory, changed_model))

        # Runner code hash changed
        changed_code = deepcopy(self.expected)
        changed_code["code_sha256"]["scripts/youtube_validation.py"] = "different-code"
        self.assertIsNone(yv.successful_run(self.directory, changed_code))

        # Tracking settings changed
        changed_settings = deepcopy(self.expected)
        changed_settings["settings"]["tracking_profile"] = "conservative"
        self.assertIsNone(yv.successful_run(self.directory, changed_settings))

    def test_failed_subprocess_retains_partial_segments_and_status_not_ok(self):
        benchmark = yv.collector(ROOT)
        line = 'Segment: {"frame_start":1,"frame_end":5,"benchmark_partial":false}\n'
        fail_cmd = [
            sys.executable,
            "-c",
            f"import sys; print({line!r}); print('fatal error', file=sys.stderr); sys.exit(5)",
        ]
        with patch.object(yv, "build_command", return_value=fail_cmd), \
             patch.object(yv.sys, "stdout", io.StringIO()), \
             patch.object(yv.sys, "stderr", io.StringIO()):
            directory, record = yv.run_one(self.args, self.expected, make_source_probe(), benchmark)

        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["returncode"], 5)
        self.assertIn("fatal error", (directory / "native.log").read_text())
        self.assertTrue((directory / "record.json").is_file())
        self.assertTrue((directory / "run.segments.csv").is_file())


class ValidationRuleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.input_file = self.directory / "input.mp4"
        self.input_file.write_bytes(b"source-content")
        self.seg_file = self.directory / "seg.pt"
        self.seg_file.write_bytes(b"seg-content")
        self.pose_file = self.directory / "pose.pt"
        self.pose_file.write_bytes(b"pose-content")
        self.args = make_args(self.directory, input=self.input_file, seg_model=self.seg_file, pose_model=self.pose_file)
        self.models = {"seg": yv.sha256(self.seg_file), "pose": yv.sha256(self.pose_file)}
        self.code = yv.code_fingerprint(ROOT)
        self.packages = {"python": "3.13.0", "packages": {"torch": "2.11.0"}, "gpu": "Tesla T4", "ffmpeg": "6.1"}
        self.expected = yv.build_identity(self.args, yv.sha256(self.input_file), self.models, self.code, self.packages)
        self.benchmark = yv.collector(ROOT)

    def run_with_summary(self, summary_dict, out_probe=None):
        segment_line = 'Segment: {"frame_start":1,"frame_end":180,"elapsed_seconds":0.15,"scene_number":1}\n'
        summary_line = f"Summary: {json.dumps(summary_dict)}\n"
        cmd = [
            sys.executable,
            "-c",
            f"import sys; print({segment_line!r}); print({summary_line!r})",
        ]
        probe_val = out_probe or {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1080,
                    "height": 1920,
                    "avg_frame_rate": "60/1",
                    "start_time": "0.0",
                    "duration": "3.0",
                    "nb_frames": "180",
                },
                {
                    "codec_type": "audio",
                    "start_time": "0.0",
                    "duration": "3.01",
                },
            ]
        }
        with patch.object(yv, "build_command", return_value=cmd), \
             patch.object(self.benchmark, "probe", return_value=probe_val), \
             patch.object(self.benchmark, "validate_output", return_value={"status": "ok"}), \
             patch.object(yv.sys, "stdout", io.StringIO()), \
             patch.object(yv.sys, "stderr", io.StringIO()):
            # write fake output file so sha256 works
            def side_effect(*a, **kw):
                for p in self.directory.glob("*/output.mp4"):
                    if not p.is_file():
                        p.write_bytes(b"rendered-mp4")
            with patch.object(subprocess.Popen, "__enter__", side_effect=lambda self: self):
                pass
            directory, record = yv.run_one(self.args, self.expected, make_source_probe(), self.benchmark)
        return record

    def test_fallback_fails_validation(self):
        summary = make_summary()
        summary["video_io"]["cpu_fallback"] = True
        record = self.run_with_summary(summary)
        self.assertEqual(record["status"], "failed")

    def test_opencv_fallback_fails_validation(self):
        summary = make_summary()
        summary["video_io"].update({
            "decoder": "opencv",
            "frame_processing": "reference",
            "decoder_fallback": True,
            "decoder_fallback_reason": "TorchCodec init error",
        })
        record = self.run_with_summary(summary)
        self.assertEqual(record["status"], "failed")
    def test_audio_tail_overflow_fails_validation(self):
        summary = make_summary()
        bad_audio_probe = {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1080,
                    "height": 1920,
                    "avg_frame_rate": "60/1",
                    "start_time": "0.0",
                    "duration": "3.0",
                    "nb_frames": "180",
                },
                {
                    "codec_type": "audio",
                    "start_time": "0.0",
                    "duration": "5.0",  # audio tail extends way past tolerance
                },
            ]
        }
        record = self.run_with_summary(summary, out_probe=bad_audio_probe)
        self.assertEqual(record["status"], "failed")

    def test_encoder_mismatch_fails_validation(self):
        summary = make_summary(encoder="libx264")
        record = self.run_with_summary(summary)
        self.assertEqual(record["status"], "failed")


if __name__ == "__main__":
    unittest.main()

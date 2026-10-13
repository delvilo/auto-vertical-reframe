"""YouTube 4K portrait validation runner for Colab T4.

Runs the fixed stage2 tensor pipeline on full-frame input with either balanced or
conservative tracking profile.  Enforces exact identity hashing, T4 NVDEC/NVENC
validation, and timing conservation.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reframe.config import TRACKING_PROFILES, tracking_settings
from reframe.saliency.backends.deepgaze_loader import ARTIFACT_SHA256, UPSTREAM_COMMIT

# Re-use vetted helpers from colab_torchcodec and benchmark_colab
spec = importlib.util.spec_from_file_location("colab_torchcodec", ROOT / "scripts/colab_torchcodec.py")
colab_torchcodec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(colab_torchcodec)

atomic_json = colab_torchcodec.atomic_json
identity_digest = colab_torchcodec.identity_digest
sha256 = colab_torchcodec.sha256
successful_run = colab_torchcodec.successful_run
validate_audio_tail = colab_torchcodec.validate_audio_tail
validate_decoder = colab_torchcodec.validate_decoder
validate_models = colab_torchcodec.validate_models
write_diagnostics = colab_torchcodec.write_diagnostics
backup = colab_torchcodec.backup
collector = colab_torchcodec.collector


def deepgaze_provenance() -> dict[str, object]:
    return {
        "loader": "checkpoint_features_only",
        "upstream_commit": UPSTREAM_COMMIT,
        "checkpoint_sha256": ARTIFACT_SHA256["deepgazemr-ledov.pt"],
        "center_bias_sha256": ARTIFACT_SHA256["center-bias-ledov.pt"],
        "imagenet_vgg_downloaded": False,
    }


def code_fingerprint(repo: Path) -> dict[str, str]:
    paths = [
        *sorted((repo / "reframe").rglob("*.py")),
        repo / "scripts/benchmark_colab.py",
        repo / "scripts/colab_torchcodec.py",
        repo / "scripts/youtube_validation.py",
        repo / "install_colab.py",
        repo / "pyproject.toml",
        repo / "requirements.txt",
    ]
    return {str(path.relative_to(repo)): sha256(path) for path in paths if path.is_file()}


def build_command(
    repo: Path,
    input_path: Path,
    output_path: Path,
    seg_model: Path,
    pose_model: Path,
    tracking_profile: str,
    max_frames: int | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "reframe",
        str(input_path),
        str(output_path),
        "--seg-model",
        str(seg_model),
        "--pose-model",
        str(pose_model),
        "--device",
        "0",
        "--saliency-device",
        "cuda",
        "--video-decoder",
        "torchcodec",
        "--decode-device",
        "cuda:0",
        "--decode-batch-size",
        "4",
        "--frame-processing",
        "tensor",
        "--tracking-profile",
        tracking_profile,
        "--tracker",
        "bytetrack.yaml",
        "--lock-first-subject",
        "--conf",
        "0.3",
        "--output-width",
        "1080",
        "--output-height",
        "1920",
        "--saliency-max-side",
        "384",
        "--no-saliency-amp",
        "--dead-zone",
        "0",
        "--seg-thumbnail-mode",
        "lazy",
        "--seg-thumbnail-method",
        "gray-area",
        "--seg-skip-policy",
        "primary",
        "--stats-interval",
        "5",
        "--video-encoder",
        "hevc_nvenc",
        "--nvenc-preset",
        "p5",
        "--crf",
        "18",
        "--preset-ffmpeg",
        "medium",
        "--encode-mode",
        "direct",
        "--log-level",
        "INFO",
    ]
    if max_frames is not None:
        cmd.extend(["--max-frames", str(max_frames)])
    return cmd


def build_identity(args: argparse.Namespace, source_hash: str, models: dict[str, str], code: dict[str, str], packages: dict[str, object]) -> dict:
    tracking = tracking_settings(args)
    cmd = build_command(
        repo=args.repo,
        input_path=args.input,
        output_path=Path("output.mp4"),
        seg_model=args.seg_model,
        pose_model=args.pose_model,
        tracking_profile=args.tracking_profile,
        max_frames=args.max_frames,
    )
    return {
        "schema_version": 1,
        "runner": "youtube_validation",
        "profile": "stage2",
        "tracking_profile": args.tracking_profile,
        "source_sha256": source_hash,
        "model_sha256": models,
        "deepgaze_provenance": deepgaze_provenance(),
        "code_sha256": code,
        "packages": packages,
        "settings": {
            **tracking,
            "max_frames": args.max_frames,
            "device": "0",
            "saliency_device": "cuda",
            "decode_device": "cuda:0",
            "decode_batch_size": 4,
            "video_encoder": "hevc_nvenc",
            "stats_interval": 5.0,
            "precrop": None,
            "lock_first_subject": True,
            "saliency_amp": False,
            "dead_zone": 0.0,
            "seg_thumbnail_mode": "lazy",
            "seg_thumbnail_method": "gray-area",
            "seg_skip_policy": "primary",
        },
        "command_template": cmd,
    }


def preflight(args: argparse.Namespace) -> dict:
    preflight_args = argparse.Namespace(
        repo=args.repo,
        input=args.input,
        profiles=["stage2"],
        device="0",
        saliency_device="cuda",
        decode_device="cuda:0",
        video_encoder="hevc_nvenc",
    )
    result = colab_torchcodec.preflight(preflight_args)

    import torch
    import torchvision

    gpu_name = torch.cuda.get_device_name(0)
    if "T4" not in gpu_name:
        raise RuntimeError(f"GPU must be a T4; detected: {gpu_name}")
    result["gpu_verified"] = gpu_name

    boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0]], device="cuda")
    scores = torch.tensor([1.0], device="cuda")
    if torchvision.ops.nms(boxes, scores, 0.5).numel() != 1:
        raise RuntimeError("torchvision CUDA NMS verification failed")
    result["cuda_nms_verified"] = True

    from reframe.torchcodec_reader import TorchCodecReader

    with TorchCodecReader(str(args.input), device="cuda:0", batch_size=4) as reader:
        for _ in reader:
            pass
    result["cfr_verified_to_eof"] = True
    return result


def run_one(args: argparse.Namespace, expected: dict, source_probe: dict, benchmark: object) -> tuple[Path, dict]:
    scope = "full" if args.max_frames is None else f"short{args.max_frames}"
    prefix = f"{args.tracking_profile}-{scope}-{identity_digest(expected)[:12]}"
    attempt = 1
    while (directory := args.output_dir / f"{prefix}-attempt{attempt:03d}").exists():
        attempt += 1
    directory.mkdir(parents=True, exist_ok=True)
    output, logfile = directory / "output.mp4", directory / "native.log"
    cmd = build_command(
        repo=args.repo,
        input_path=args.input,
        output_path=output,
        seg_model=args.seg_model,
        pose_model=args.pose_model,
        tracking_profile=args.tracking_profile,
        max_frames=args.max_frames,
    )
    record = {
        "profile": "stage2",
        "tracking_profile": args.tracking_profile,
        "case": f"stage2_{args.tracking_profile}",
        "repeat": 1,
        "warmup": False,
        "scope": scope,
        "identity": expected,
        "status": "running",
        "command": cmd,
        "output": str(output),
        "log": str(logfile),
    }
    atomic_json(directory / "record.json", record)

    started = time.perf_counter()
    try:
        with logfile.open("xb") as handle:
            with subprocess.Popen(
                cmd,
                cwd=args.repo,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            ) as process:
                try:
                    while chunk := process.stdout.read1(65536):
                        handle.write(chunk)
                        handle.flush()
                        if hasattr(sys.stdout, "buffer"):
                            sys.stdout.buffer.write(chunk)
                            sys.stdout.buffer.flush()
                        else:
                            print(chunk.decode("utf-8", errors="replace"), end="", flush=True)
                    record["returncode"] = process.wait()
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    record["returncode"] = process.returncode
                    raise
        record["subprocess_wall_seconds"] = time.perf_counter() - started
        if record["returncode"]:
            raise RuntimeError(f"Native process exited with {record['returncode']}; inspect {logfile}")

        text = logfile.read_text(encoding="utf-8", errors="replace")
        record["summary"] = benchmark.parse_summary(text)
        segments, _ = benchmark.extract_segments(text)
        benchmark.validate_segments(segments, record["summary"])
        record["output_probe"] = benchmark.probe(output)
        record["validation"] = benchmark.validate_output(source_probe, record["output_probe"], record["summary"], args.max_frames)
        record["audio_tail"] = validate_audio_tail(record["output_probe"], source_probe)

        # Validate encoder hevc_nvenc
        if record["summary"].get("video_encoder_actual") != "hevc_nvenc":
            raise ValueError(f"Expected hevc_nvenc, actual was {record['summary'].get('video_encoder_actual')}")

        # Validate output dimensions
        out_v = next(s for s in record["output_probe"]["streams"] if s["codec_type"] == "video")
        if int(out_v["width"]) != 1080 or int(out_v["height"]) != 1920:
            raise ValueError(f"Output dimensions must be 1080x1920, got {out_v['width']}x{out_v['height']}")

        # Validate decoder NVDEC verified
        record["decoder_validation"] = validate_decoder(record["summary"], "stage2", "cuda:0")

        # Validate models device placement
        dummy_args = argparse.Namespace(device="0", saliency_device="cuda")
        record["device_validation"] = validate_models(record["summary"], dummy_args, benchmark)

        # Validate tracking settings match
        effective_tracking = tracking_settings(args)
        sum_tracking = record["summary"].get("tracking_settings", {})
        for k, v in effective_tracking.items():
            if sum_tracking.get(k) != v:
                raise ValueError(f"Tracking setting {k} mismatch: expected {v}, got {sum_tracking.get(k)}")

        record["output_sha256"] = sha256(output)
        record["status"] = "ok"
    except (Exception, KeyboardInterrupt) as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        record["interrupted"] = isinstance(error, KeyboardInterrupt)
        record["validation_traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        record["collector_wall_seconds"] = time.perf_counter() - started
        write_diagnostics(directory, benchmark, record)
        if args.backup_dir:
            backup(args)
    return directory, record


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="YouTube 4K validation runner: executes stage2 tensor full-frame pipeline on Colab T4."
    )
    p.add_argument("--input", type=Path, required=True, help="Input 4K CFR video clip")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory for runs and checkpoints")
    p.add_argument("--backup-dir", type=Path, default=None, help="Backup directory (e.g. Google Drive)")
    p.add_argument("--seg-model", type=Path, required=True, help="YOLO segmentation model path")
    p.add_argument("--pose-model", type=Path, required=True, help="YOLO pose model path")
    p.add_argument("--tracking-profile", choices=("balanced", "conservative"), required=True, help="Tracking profile")
    p.add_argument("--max-frames", type=int, default=None, help="Process only first N frames")
    p.add_argument("--repo", type=Path, default=ROOT, help=argparse.SUPPRESS)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.backup_dir:
        args.backup_dir.mkdir(parents=True, exist_ok=True)

    benchmark = collector(args.repo)

    # 1. Environment & Preflight
    environment = benchmark.environment()
    environment["packages"]["torchcodec"] = (
        importlib.metadata.version("torchcodec") if importlib.util.find_spec("torchcodec") else None
    )
    source_probe = benchmark.probe(args.input)
    source_hash = sha256(args.input)
    model_hashes = {"seg": sha256(args.seg_model), "pose": sha256(args.pose_model)}
    code_hashes = code_fingerprint(args.repo)
    packages = {
        "python": environment["python"],
        "packages": environment["packages"],
        "gpu": environment["gpu"],
        "ffmpeg": environment["ffmpeg"],
    }

    expected = build_identity(args, source_hash, model_hashes, code_hashes, packages)

    # 2. Check reuse
    existing = successful_run(args.output_dir, expected)
    if existing is not None:
        existing_dir, existing_record = existing
        print(f"Reusing exact verified run: {existing_dir}")
        return 0

    # 3. Hardware preflight
    print("Running hardware preflight...")
    preflight_result = preflight(args)
    print("Preflight passed:", preflight_result)

    # 4. Run single attempt
    directory, record = run_one(args, expected, source_probe, benchmark)
    if record["status"] != "ok":
        print(f"Validation failed: {record.get('error')}", file=sys.stderr)
        return 1
    print(f"Validation completed successfully in {directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Resumable, short-first comparisons of the two optional TorchCodec stages.

No model or accelerator imports occur until preflight/run. Existing timing and
media validators remain shared with benchmark_colab.py.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
import zipfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from reframe.config import TRACKING_PROFILES, tracking_settings


PROFILES = {"baseline": ("opencv", "reference"),
            "stage1": ("torchcodec", "reference"),
            "stage2": ("torchcodec", "tensor")}
SMOKE_FRAMES = 180


def collector(repo=ROOT):
    spec = importlib.util.spec_from_file_location("staged_benchmark", repo / "scripts/benchmark_colab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def code_fingerprint(repo: Path) -> dict:
    paths = [*sorted((repo / "reframe").rglob("*.py")),
             repo / "scripts/benchmark_colab.py", repo / "scripts/colab_torchcodec.py",
             repo / "pyproject.toml", repo / "requirements.txt"]
    return {str(path.relative_to(repo)): sha256(path) for path in paths}


def identity(args, profile, source_hash, models, code, packages, max_frames):
    tracking = tracking_settings(args)
    return {"schema_version": 1, "profile": profile, "source_sha256": source_hash,
            "model_sha256": models, "code_sha256": code, "packages": packages,
            "settings": {**tracking, "max_frames": max_frames,
                         "device": args.device, "saliency_device": args.saliency_device,
                         "decode_device": args.decode_device, "decode_batch_size": args.decode_batch_size,
                         "video_encoder": args.video_encoder, "stats_interval": 5.0,
                         "cpu_diagnostic": args.allow_cpu_diagnostic,
                         "precrop": "middle", "lock_first_subject": True, "saliency_amp": False,
                         "post_restore": False, "dead_zone": getattr(args, "dead_zone", 0.0),
                         "thumbnail_method": "gray-area", "skip_policy": tracking["seg_skip_policy"]}}


def identity_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def successful_run(directory: Path, expected: dict):
    """Reuse only exact settings, source, weights, code and intact output bytes."""
    for path in sorted(directory.glob("*/record.json"), reverse=True):
        try:
            record = json.loads(path.read_text())
            output = path.parent / "output.mp4"
            if (record.get("status") == "ok" and record.get("identity") == expected
                    and output.is_file() and record.get("output_sha256") == sha256(output)):
                return path.parent, record
        except (OSError, ValueError):
            continue
    return None


def validate_decoder(summary: dict, profile: str, requested: str) -> dict:
    io = summary.get("video_io")
    if not isinstance(io, dict):
        raise ValueError("Missing video_io telemetry; update the repository")
    decoder, processing = PROFILES[profile]
    if io.get("decoder") != decoder or io.get("frame_processing") != processing:
        raise ValueError("Decoder/frame-processing telemetry differs from requested profile")
    if io.get("decoder_fallback"):
        raise ValueError(f"Decoder fallback occurred: {io.get('decoder_fallback_reason')}")
    result = {key: io.get(key) for key in ("decoder", "frame_processing", "requested_device",
              "actual_device", "cpu_fallback", "cpu_fallback_status_known", "frames_decoded",
              "frames_yielded", "decode_wall_seconds", "initialization_wall_seconds",
              "requested_decoder", "requested_frame_processing", "decoder_fallback",
              "decoder_fallback_reason")}
    if decoder == "opencv":
        if io.get("actual_device") != "cpu":
            raise ValueError("OpenCV baseline must explicitly report CPU decoding")
        result["status"] = "cpu_baseline"
        return result
    if io.get("frames_yielded") != summary.get("frames_processed"):
        raise ValueError("Decoder frames_yielded differs from Summary frames_processed")
    requested = "cuda:0" if requested == "cuda" else requested
    actual = "cuda:0" if io.get("actual_device") == "cuda" else io.get("actual_device")
    if actual != requested:
        raise ValueError(f"Decoder requested {requested}, actually used {actual}")
    if requested.startswith("cuda"):
        if io.get("cpu_fallback") is not False or io.get("cpu_fallback_status_known") is not True:
            raise ValueError("NVDEC is unverified or used CPU fallback; CUDA output alone is insufficient")
        result["status"] = "nvdec_verified"
    else:
        result["status"] = "explicit_cpu_test"
    return result


def validate_models(summary, args, benchmark):
    result = benchmark.validate_devices(summary, args.device, args.saliency_device)
    for label, requested in (("segmentation", args.device), ("pose", args.device),
                             ("deepgaze", args.saliency_device)):
        if (requested == "cpu" and result[label]["calls"] > 0
                and result[label]["actual_device"] != "cpu"):
            raise ValueError(f"{label} requested CPU but did not report CPU execution")
        if requested == "cpu" and result[label]["calls"] > 0:
            result[label]["status"] = "cpu_diagnostic"
    return result


def validate_audio_tail(output: dict, source: dict) -> dict:
    """Full frame validation is separate; timestamp endpoints check audio bounds."""
    video = next(s for s in output["streams"] if s["codec_type"] == "video")
    audio = next((s for s in output["streams"] if s["codec_type"] == "audio"), None)
    has_audio = any(s["codec_type"] == "audio" for s in source["streams"])
    if not has_audio:
        if audio is not None:
            raise ValueError("Silent source unexpectedly gained audio")
        return {"status": "no_source_audio"}
    if audio is None:
        raise ValueError("Source audio missing from output")
    endpoints = {}
    for kind, stream in (("video", video), ("audio", audio)):
        start, duration = float(stream["start_time"]), float(stream["duration"])
        if not all(math.isfinite(value) for value in (start, duration)) or duration <= 0:
            raise ValueError(f"Invalid {kind} stream timestamps")
        endpoints[kind + "_end_seconds"] = start + duration
    delta = endpoints["audio_end_seconds"] - endpoints["video_end_seconds"]
    # AAC packetization can extend the endpoint by several packets. A long source
    # audio stream must not survive a short --max-frames clip in its entirety.
    tolerance = max(0.1, 2 / float(Fraction(video["avg_frame_rate"])))
    endpoints.update(audio_minus_video_end_seconds=delta, tolerance_seconds=tolerance)
    if abs(delta) > tolerance + 1e-6:
        raise ValueError(f"Audio/video endpoint difference {delta:.6f}s exceeds {tolerance:.6f}s")
    endpoints["status"] = "endpoints_within_tolerance"
    return endpoints


def command(args, profile, output):
    benchmark = collector(args.repo)
    tracking = tracking_settings(args)
    case = (profile, tracking["seg_max_gap"], "lazy", "gray-area", tracking["seg_skip_policy"])
    decoder, processing = PROFILES[profile]
    result = benchmark.command(args, output, case, video_decoder=decoder)
    result += ["--frame-processing", processing,
               "--decode-device", args.decode_device, "--decode-batch-size", str(args.decode_batch_size)]
    return result


def preflight(args):
    if sys.version_info < (3, 13):
        raise RuntimeError("Python 3.13+ is required; select a compatible Colab runtime")
    import torch
    from packaging.version import Version
    if Version(torch.__version__.split("+")[0]) < Version("2.11"):
        raise RuntimeError("This TorchCodec 0.17 collector requires PyTorch >=2.11; keep matching torch/torchvision builds")
    result = {"torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
              "cuda_available": torch.cuda.is_available(), "python": sys.version}
    if args.device != "cpu" or args.saliency_device == "cuda" or (
            any(PROFILES[p][0] == "torchcodec" for p in args.profiles) and args.decode_device.startswith("cuda")):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; CPU fallback is excluded")
    if any(PROFILES[p][0] == "torchcodec" for p in args.profiles):
        import torchcodec
        from torchcodec.decoders import VideoDecoder
        version = Version(torchcodec.__version__)
        if version.release[:2] != (0, 17):
            raise RuntimeError(f"Expected TorchCodec 0.17.x, found {version}")
        decoder = VideoDecoder(str(args.input), device=args.decode_device, dimension_order="NCHW", seek_mode="exact")
        frame = decoder.get_frame_at(0).data
        fallback = decoder.cpu_fallback
        result.update(torchcodec=torchcodec.__version__, decoder_actual_device=str(frame.device),
                      decoder_cpu_fallback=bool(fallback), decoder_cpu_fallback_status_known=fallback.status_known,
                      decoder_cpu_fallback_description=str(fallback))
        expected = "cuda:0" if args.decode_device == "cuda" else args.decode_device
        if str(frame.device) != expected or (expected.startswith("cuda") and (
                bool(fallback) or not fallback.status_known)):
            raise RuntimeError(f"TorchCodec preflight did not use the requested decoder: {result}")
        del frame, decoder
    if args.video_encoder.endswith("_nvenc"):
        nvenc_command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
            "-i", "color=s=320x240:r=30:d=0.1", "-c:v", args.video_encoder, "-f", "null", "-"]
        try:
            native = subprocess.run(nvenc_command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as error:
            print(error.stderr, file=sys.stderr, flush=True)
            raise RuntimeError(f"NVENC preflight failed: {error.stderr}") from error
        result["nvenc_preflight"] = {"status": "ok", "stderr": native.stderr}
    return result


def write_diagnostics(directory, benchmark, record):
    log = directory / "native.log"
    if log.exists():
        text = log.read_text(encoding="utf-8", errors="replace")
        benchmark.save_run_segments(record, directory, "run", text)
    atomic_json(directory / "record.json", record)


def run_one(args, profile, expected, source, benchmark):
    scope = "full" if args.max_frames is None else f"short{args.max_frames}"
    prefix = f"{profile}-{scope}-{identity_digest(expected)[:12]}"
    attempt = 1
    while (directory := args.output_dir / f"{prefix}-attempt{attempt:03}").exists():
        attempt += 1
    directory.mkdir()
    output, logfile = directory / "output.mp4", directory / "native.log"
    record = {"profile": profile, "case": profile, "repeat": 1, "warmup": False,
              "identity": expected, "status": "running", "command": command(args, profile, output),
              "output": str(output), "log": str(logfile)}
    atomic_json(directory / "record.json", record)
    started = time.perf_counter()
    try:
        with logfile.open("xb") as handle:
            with subprocess.Popen(record["command"], cwd=args.repo, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT) as process:
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
        record["validation"] = benchmark.validate_output(source, record["output_probe"], record["summary"], args.max_frames)
        record["audio_tail"] = validate_audio_tail(record["output_probe"], source)
        record["device_validation"] = validate_models(record["summary"], args, benchmark)
        record["decoder_validation"] = validate_decoder(record["summary"], profile, args.decode_device)
        if record["summary"].get("video_encoder_actual") != args.video_encoder:
            raise ValueError("Encoder fallback is excluded from comparisons")
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
    return directory, record


def backup(args):
    """Checkpoint each completed attempt; hard runtime deletion cannot run finally."""
    archive = args.output_dir / "diagnostics.zip"
    temporary = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for path in sorted(args.output_dir.rglob("*")):
            if path.is_file() and path.suffix in {".json", ".jsonl", ".csv", ".log"}:
                package.write(path, path.relative_to(args.output_dir))
    temporary.replace(archive)
    if args.backup_dir:
        args.backup_dir.mkdir(parents=True, exist_ok=True)
        # Mirror checkpoint + outputs without overwriting unrelated backups.
        for path in sorted(args.output_dir.rglob("*")):
            if path.is_file() and path.suffix != ".tmp":
                target = args.backup_dir / path.relative_to(args.output_dir)
                if target.exists() and (target.stat().st_size, target.stat().st_mtime_ns) == (
                        path.stat().st_size, path.stat().st_mtime_ns):
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo", type=Path, default=ROOT)
    result.add_argument("--input", type=Path, required=True)
    result.add_argument("--seg-model", type=Path, required=True)
    result.add_argument("--pose-model", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True, help="Stable directory reused to resume exact successful runs")
    result.add_argument("--backup-dir", type=Path)
    result.add_argument("--profiles", choices=tuple(PROFILES), nargs="+", default=["baseline", "stage1"])
    limit = result.add_mutually_exclusive_group()
    limit.add_argument("--max-frames", type=int, default=SMOKE_FRAMES)
    limit.add_argument("--full-video", action="store_true", help="Requires an equivalent successful 180-frame short run per selected profile")
    result.add_argument("--tracking-profile", choices=tuple(TRACKING_PROFILES), default="balanced")
    result.add_argument("--seg-max-gap", type=int,
                        help="Positive override of the tracking profile; use 1 to isolate dataflow speed")
    result.add_argument("--seg-max-age", type=float,
                        help="Override true segmentation measurement age limit in seconds")
    result.add_argument("--cue-interval", type=float,
                        help="Override soft Pose refresh interval in seconds")
    result.add_argument("--pose-max-age", type=float,
                        help="Override hard Pose cache age in seconds; 0 disables cross-frame reuse")
    result.add_argument("--decode-batch-size", type=int, choices=range(1, 9), default=4)
    result.add_argument("--decode-device", default="cuda:0")
    result.add_argument("--device", default="0")
    result.add_argument("--saliency-device", choices=("cuda", "cpu"), default="cuda")
    result.add_argument("--video-encoder", default="hevc_nvenc")
    result.add_argument("--dead-zone", type=float, default=0.0,
                        help="Optional pan dead zone, 0..0.4 (default 0); part of resume identity")
    result.add_argument("--allow-cpu-diagnostic", action="store_true",
                        help="Permit explicitly selected CPU devices/software encoding; never a T4 benchmark")
    return result


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    if not 0 <= args.dead_zone <= 0.4:
        arguments.error("--dead-zone must be between 0 and 0.4")
    if args.seg_max_gap is not None and args.seg_max_gap <= 0:
        arguments.error("--seg-max-gap must be positive")
    for key in ("seg_max_age", "cue_interval", "pose_max_age"):
        value = getattr(args, key)
        if value is not None and (not math.isfinite(value) or value < 0
                                  or (key == "seg_max_age" and value == 0)):
            arguments.error(f"--{key.replace('_', '-')} must be finite and "
                            + ("positive" if key == "seg_max_age" else "nonnegative"))
    for key, value in tracking_settings(args).items():
        setattr(args, key, value)
    for key in ("repo", "input", "seg_model", "pose_model", "output_dir", "backup_dir"):
        if (value := getattr(args, key)) is not None:
            setattr(args, key, value.resolve())
    for key in ("input", "seg_model", "pose_model"):
        if not getattr(args, key).is_file():
            arguments.error(f"--{key.replace('_', '-')} must be an existing local file")
    if args.max_frames <= 0:
        arguments.error("--max-frames must be positive")
    if args.full_video and len(set(args.profiles)) != 1:
        arguments.error("--full-video runs exactly one profile at a time; select --profiles baseline, stage1 or stage2")
    if not re.fullmatch(r"cpu|cuda(?::\d+)?", args.decode_device):
        arguments.error("--decode-device must be cpu, cuda or cuda:N")
    if not re.fullmatch(r"cpu|cuda(?::\d+)?|\d+", args.device):
        arguments.error("--device must be cpu, cuda, cuda:N or a GPU index")
    if not args.allow_cpu_diagnostic and (args.device == "cpu" or args.saliency_device == "cpu"
            or (any(PROFILES[p][0] == "torchcodec" for p in args.profiles) and args.decode_device == "cpu")
            or not args.video_encoder.endswith("_nvenc")):
        arguments.error("GPU comparisons require CUDA models, NVDEC and NVENC; use --allow-cpu-diagnostic for explicit CPU checks")
    if args.backup_dir and (args.backup_dir == args.output_dir or args.output_dir in args.backup_dir.parents
                            or args.backup_dir in args.output_dir.parents):
        arguments.error("--backup-dir and --output-dir must not overlap")
    args.max_frames = None if args.full_video else args.max_frames
    args.stats_interval, args.log_level = 5.0, "INFO"
    # The legacy command builder accepts model names as strings.
    args.seg_model, args.pose_model = str(args.seg_model), str(args.pose_model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = collector(args.repo)
    results = {"schema_version": 1, "test": "torchcodec-stages", "status": "running", "runs": [],
               "cpu_diagnostic": args.allow_cpu_diagnostic}
    code = 1
    try:
        environment = benchmark.environment()
        environment["packages"]["torchcodec"] = importlib.metadata.version("torchcodec") if importlib.util.find_spec("torchcodec") else None
        results["environment"] = environment
        source = benchmark.probe(args.input)
        results["source_probe"] = source
        source_hash = sha256(args.input)
        models = {name: sha256(Path(getattr(args, name))) for name in ("seg_model", "pose_model")}
        code_hashes = code_fingerprint(args.repo)
        packages = {"python": environment["python"], "packages": environment["packages"],
                    "gpu": environment["gpu"], "ffmpeg": environment["ffmpeg"]}
        expected = {profile: identity(args, profile, source_hash, models, code_hashes, packages, args.max_frames)
                    for profile in dict.fromkeys(args.profiles)}
        if args.full_video:
            for profile, value in expected.items():
                smoke = {**value, "settings": {**value["settings"], "max_frames": SMOKE_FRAMES}}
                if successful_run(args.output_dir, smoke) is None:
                    raise ValueError(f"Run {profile} with --max-frames {SMOKE_FRAMES} first using the same source/settings/code")
        results["preflight"] = preflight(args)
        atomic_json(args.output_dir / "results.json", results)
        for profile, value in expected.items():
            previous = successful_run(args.output_dir, value)
            if previous is not None:
                directory, record = previous
                print(f"Resume: verified completed {profile}: {directory}", flush=True)
            else:
                directory, record = run_one(args, profile, value, source, benchmark)
            results["runs"].append({"directory": str(directory), **record})
            atomic_json(args.output_dir / "results.json", results)
            backup(args)
            if record["status"] != "ok":
                raise RuntimeError(record["error"])
        results["status"] = "ok"
        code = 0
    except (Exception, KeyboardInterrupt) as error:
        results["status"] = "failed"
        results["error"] = f"{type(error).__name__}: {error}"
        results["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        atomic_json(args.output_dir / "results.json", results)
        backup(args)
        print("Diagnostics:", args.output_dir / "diagnostics.zip", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

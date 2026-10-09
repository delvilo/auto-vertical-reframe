"""One audio-tail regression, preserving the existing gap1 visual baseline.

The companion notebook also embeds this runner for use before a GitHub release.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


def collector(repo):
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("audio_tail_benchmark", repo / "scripts/benchmark_colab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Only this process gains a one-case reporting group; the collector is unchanged.
    module.SUITES["audio-tail"] = (module.CASES[0],)
    return module


def endpoints(media):
    result = {}
    for kind in ("video", "audio"):
        stream = next((s for s in media["streams"] if s["codec_type"] == kind), None)
        if stream and stream.get("duration") not in (None, "N/A"):
            result[kind + "_end_seconds"] = float(stream.get("start_time", 0)) + float(stream["duration"])
    if len(result) == 2:
        result["audio_minus_video_end_seconds"] = result["audio_end_seconds"] - result["video_end_seconds"]
    return result


def source_fingerprint(repo):
    paths = [*sorted((repo / "reframe").rglob("*.py")), repo / "scripts/benchmark_colab.py"]
    return {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def mux_smoke(args, benchmark, results):
    """Exercise both production writers with 120 frames and deliberately short audio."""
    import cv2
    import numpy as np
    from reframe.config import AppConfig
    from reframe.video_io import DirectVideoWriter, LosslessWriter, run_ffmpeg_mux

    source = args.output_dir / "short-audio-source.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "info", "-n",
        "-f", "lavfi", "-i", "color=c=gray:s=320x240:r=60:d=2",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=1.866666667",
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "libmp3lame", str(source),
    ], check=True)
    original = benchmark.probe(source)
    results["source_probe"] = original
    if endpoints(original).get("audio_minus_video_end_seconds", 0) >= -.05:
        raise ValueError("Synthetic source must have audio shorter than its video")
    config = AppConfig(video_encoder=args.video_encoder, ffmpeg_log_level="info")
    for mode in ("direct", "lossless"):
        output = args.output_dir / f"mux-{mode}.mp4"
        intermediate = args.output_dir / "silent.mkv"
        record = {"mode": mode, "status": "failed", "output": str(output)}
        results["runs"].append(record)
        if mode == "direct":
            writer = DirectVideoWriter(output, source, 60, (320, 240), config)
        else:
            writer = LosslessWriter(str(intermediate), 60, (320, 240), "info")
        try:
            for index in range(120):
                writer.write(np.full((240, 320, 3), 240 if index >= 112 else 40, dtype=np.uint8))
            writer.release()
        finally:
            writer.abort()
        actual = writer.encoder if mode == "direct" else run_ffmpeg_mux(
            str(intermediate), str(source), str(output), args.video_encoder,
            config.audio_bitrate, config.crf, config.preset_ffmpeg, None,
            ffmpeg_log_level="info")
        record["actual_encoder"] = actual
        if actual != args.video_encoder:
            raise ValueError(f"Encoder fallback: requested {args.video_encoder}, got {actual}")
        record["output_probe"] = benchmark.probe(output)
        record["validation"] = benchmark.validate_output(original, record["output_probe"],
            {"frames_processed": 120, "output_width": 320, "output_height": 240}, None)
        record["endpoints"] = endpoints(record["output_probe"])
        cap = cv2.VideoCapture(str(output))
        means = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                means.append(float(frame.mean()))
        finally:
            cap.release()
        if len(means) != 120 or not all(value > 220 for value in means[-8:]) or means[-9] >= 60:
            raise ValueError("The distinctive last eight video frames were not retained")
        record["tail_marker_verified"] = True
        record["status"] = "ok"
        print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    intermediate.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--mode", choices=("mux", "full"), required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--seg-model", default="yolo26n-seg.pt")
    parser.add_argument("--pose-model", default="yolo26n-pose.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--saliency-device", default="cuda")
    parser.add_argument("--video-encoder", default="hevc_nvenc")
    args = parser.parse_args(argv)
    if args.mode == "full" and (args.input is None or not args.input.is_file()):
        parser.error("--mode full requires an existing --input video")
    args.repo, args.output_dir = args.repo.resolve(), args.output_dir.resolve()
    if args.input:
        args.input = args.input.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.max_frames, args.stats_interval, args.log_level = None, 5.0, "INFO"
    logging.basicConfig(level=logging.INFO)
    benchmark = collector(args.repo)
    results = {"schema_version": 3, "test": "audio-tail", "status": "running",
               "environment": benchmark.environment(), "source_sha256": source_fingerprint(args.repo),
               "configuration": {**{key: str(value) if isinstance(value, Path) else value
                                     for key, value in vars(args).items()}, "suite": "audio-tail"},
               "runs": []}
    results["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    results["audio_patch_diff"] = subprocess.run(
        ["git", "diff", "HEAD", "--", "reframe/video_io.py"], cwd=args.repo,
        check=True, capture_output=True, text=True).stdout
    code = 1
    try:
        if args.mode == "mux":
            mux_smoke(args, benchmark, results)
        else:
            results["source_probe"] = benchmark.probe(args.input)
            case = benchmark.CASES[0]
            results["workload"] = benchmark.workload(results["source_probe"], args, [(case, False, 1)])
            print("Exactly one full gap1 baseline, no model warmups:", results["workload"], flush=True)
            benchmark.save(args.output_dir, results)
            record = benchmark.run_one(args, case, False, 1, results["source_probe"])
            results["runs"].append(record)
            if "output_probe" in record:
                record["endpoints"] = endpoints(record["output_probe"])
            if record["status"] != "ok":
                raise RuntimeError(record["error"])
        results["status"] = "ok"
        code = 0
    except (Exception, KeyboardInterrupt) as error:
        results["status"] = "failed"
        results["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        if args.mode == "full":
            benchmark.save(args.output_dir, results)
        else:
            (args.output_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
        archive = benchmark.archive_results(args.output_dir)
        print("Diagnostic ZIP:", archive, flush=True)
        if args.backup_dir:
            args.backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(archive, args.backup_dir / archive.name)
            # Keep the produced video even if a later metadata check failed.
            if args.mode == "full":
                for path in args.output_dir.glob("*.mp4"):
                    shutil.copy2(path, args.backup_dir / path.name)
            print("Drive backup:", args.backup_dir, flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

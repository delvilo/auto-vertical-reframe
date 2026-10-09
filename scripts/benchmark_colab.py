"""Collect reproducible Colab timings without importing torch or downloading models here."""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
CASES = (("gap1-lazy", 1, "lazy"), ("gap3-eager", 3, "eager"), ("gap3-lazy", 3, "lazy"))
TIMING_KEYS = {"thumbnail", "scheduler", "frame_change", "model_track", "parse_masks",
               "flow", "predict", "bookkeeping"}


def parse_summary(log: str) -> dict:
    summaries = [json.loads(line.partition("Summary: ")[2])
                 for line in log.splitlines() if "Summary: " in line]
    if len(summaries) != 1 or not isinstance(summaries[0], dict):
        raise ValueError("Expected exactly one JSON Summary in the run log")
    summary = summaries[0]
    timings = summary.get("seg_timing_seconds", {})
    if not isinstance(timings, dict) or not TIMING_KEYS.issubset(timings):
        raise ValueError("Missing detailed segmentation timings; update the repository and rerun "
                         "from its directory (this benchmark requires --seg-thumbnail-mode)")
    values = [timings[key] for key in TIMING_KEYS] + [summary.get("seg_total_seconds")]
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Summary contains missing, negative, or non-finite segmentation timings")
    total = summary["seg_total_seconds"]
    if not math.isclose(sum(timings[key] for key in TIMING_KEYS), total,
                        rel_tol=0, abs_tol=1e-4 + 1e-6 * total):
        raise ValueError("Detailed segmentation timings do not sum to seg_total_seconds")
    return summary


def probe(path: Path) -> dict:
    result = subprocess.run([
        "ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    # Keep media metadata, not arbitrary embedded tags or source filenames.
    fields = {"index", "codec_type", "codec_name", "width", "height", "r_frame_rate",
              "avg_frame_rate", "nb_frames", "nb_read_frames", "start_time", "duration",
              "duration_ts", "time_base", "sample_rate", "channels"}
    data = json.loads(result.stdout)
    return {"streams": [{key: value for key, value in stream.items() if key in fields}
                        for stream in data.get("streams", [])],
            "duration": data.get("format", {}).get("duration")}


def validate_output(source: dict, output: dict, summary: dict, max_frames: int) -> dict:
    original = next(stream for stream in source["streams"] if stream["codec_type"] == "video")
    rendered = next(stream for stream in output["streams"] if stream["codec_type"] == "video")
    source_frames = int(original["nb_read_frames"])
    output_frames = int(rendered["nb_read_frames"])
    source_fps = Fraction(original["avg_frame_rate"])
    output_fps = Fraction(rendered["avg_frame_rate"])
    source_audio = any(stream["codec_type"] == "audio" for stream in source["streams"])
    output_audio = any(stream["codec_type"] == "audio" for stream in output["streams"])
    if output_frames != min(source_frames, max_frames) or output_frames != summary["frames_processed"]:
        raise ValueError("Output frame count does not match source limit and Summary")
    if source_fps <= 0 or abs(float(source_fps - output_fps)) > 1e-6:
        raise ValueError(f"Output FPS changed: {source_fps} -> {output_fps}")
    if source_audio != output_audio:
        raise ValueError("Output audio presence differs from input")
    if (rendered["width"], rendered["height"]) != (summary["output_width"], summary["output_height"]):
        raise ValueError("Output dimensions differ from Summary")
    result = {"source_frames": source_frames, "output_frames": output_frames,
            "source_fps": float(source_fps), "output_fps": float(output_fps),
            "source_has_audio": source_audio, "output_has_audio": output_audio}
    # Record stream timestamps for review; matching stream starts cannot prove A/V sync.
    for label, media in (("source", source), ("output", output)):
        starts = {}
        for kind in ("video", "audio"):
            stream = next((stream for stream in media["streams"] if stream["codec_type"] == kind), {})
            for field in ("start_time", "duration"):
                value = stream.get(field)
                number = float(value) if value not in (None, "N/A") else None
                result[f"{label}_{kind}_{field}_seconds"] = number
                if field == "start_time":
                    starts[kind] = number
        if all(value is not None for value in starts.values()):
            result[f"{label}_av_start_delta_seconds"] = starts["audio"] - starts["video"]
    return result


def flatten(data: dict, prefix: str = "") -> dict:
    flattened = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten(value, name))
        else:
            flattened[name] = json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
    return flattened


def aggregates(records: list[dict]) -> list[dict]:
    rows = []
    for name, _, _ in CASES:
        valid = [flatten(record) for record in records
                 if record["case"] == name and record["status"] == "ok" and not record["warmup"]]
        row = {"case": name, "measured_runs": len(valid)}
        keys = set().union(*(record.keys() for record in valid))
        for key in sorted(keys):
            values = [record.get(key) for record in valid]
            if all(isinstance(value, (int, float)) and not isinstance(value, bool)
                   and math.isfinite(value) for value in values):
                row[f"median.{key}"] = statistics.median(values)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save(directory: Path, results: dict) -> None:
    results["aggregates"] = aggregates(results["runs"])
    temporary = directory / "results.json.tmp"
    temporary.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(directory / "results.json")
    write_csv(directory / "results.csv", [flatten(record) for record in results["runs"]])
    write_csv(directory / "aggregates.csv", results["aggregates"])


def environment() -> dict:
    result = {"python": sys.version, "platform": platform.platform()}
    for name, command in {
        "git_revision": ["git", "rev-parse", "HEAD"],
        "git_status": ["git", "status", "--porcelain"],
        "gpu": ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        "ffmpeg": ["ffmpeg", "-version"],
    }.items():
        try:
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=15)
            result[name] = {"returncode": completed.returncode,
                            "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip()}
        except (OSError, subprocess.TimeoutExpired) as error:
            result[name] = {"error": str(error)}
    return result


def schedule(repeats: int):
    for case in CASES:
        yield case, True, 0
    for repeat in range(repeats):
        offset = (repeat // 2) % len(CASES)
        order = CASES[offset:] + CASES[:offset]
        for case in reversed(order) if repeat % 2 else order:
            yield case, False, repeat + 1


def command(args, output: Path, case: tuple) -> list[str]:
    _, gap, mode = case
    return [sys.executable, "-u", "-m", "reframe", str(args.input), str(output),
            "--seg-model", args.seg_model, "--pose-model", args.pose_model,
            "--device", args.device, "--saliency-device", args.saliency_device,
            "--precrop", "middle", "--lock-first-subject", "--dead-zone", "0.06",
            "--conf", "0.3", "--seg-max-gap", str(gap), "--seg-max-age", "0.1",
            "--seg-thumbnail-mode", mode, "--video-encoder", args.video_encoder,
            "--max-frames", str(args.max_frames), "--saliency-trust-repo",
            "--saliency-max-side", "384", "--saliency-interval", "3", "--no-saliency-amp",
            "--native-debug", "--log-level", "DEBUG", "--ffmpeg-log-level", "info"]


def run_one(args, case: tuple, warmup: bool, repeat: int, source: dict) -> dict:
    name = f"{case[0]}-{'warmup' if warmup else f'run{repeat}'}"
    output, logfile = args.output_dir / f"{name}.mp4", args.output_dir / f"{name}.log"
    record = {"case": case[0], "warmup": warmup, "repeat": repeat, "status": "failed",
              "output": str(output), "log": str(logfile), "command": command(args, output, case)}
    print(f"\n=== {name} ({'excluded from medians' if warmup else 'measured'}) ===", flush=True)
    started = time.perf_counter()
    try:
        with logfile.open("xb") as handle:
            with subprocess.Popen(record["command"], cwd=ROOT, stdout=subprocess.PIPE,
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
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    record["returncode"] = process.returncode
                    raise
                record["returncode"] = process.wait()
        record["wall_seconds"] = time.perf_counter() - started
        if record["returncode"]:
            raise RuntimeError(f"Reframe exited with status {record['returncode']}; inspect {logfile}")
        record["summary"] = parse_summary(logfile.read_text(encoding="utf-8", errors="replace"))
        record["output_probe"] = probe(output)
        record["validation"] = validate_output(source, record["output_probe"], record["summary"], args.max_frames)
        actual_encoder = record["summary"].get("video_encoder_actual")
        if args.video_encoder != "auto" and actual_encoder != args.video_encoder:
            raise ValueError(f"Requested encoder {args.video_encoder}, but actual encoder was {actual_encoder}; "
                             "encoder fallback is excluded from benchmark results")
        record["status"] = "ok"
    except (Exception, KeyboardInterrupt) as error:
        record["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, KeyboardInterrupt):
            record["interrupted"] = True
        record.setdefault("wall_seconds", time.perf_counter() - started)
    return record


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory; existing paths are refused")
    parser.add_argument("--repeats", type=positive, default=2)
    parser.add_argument("--max-frames", type=positive, default=600)
    parser.add_argument("--device", default="0")
    parser.add_argument("--saliency-device", default="cuda", choices=("cuda", "cpu", "mps", "auto"))
    parser.add_argument("--video-encoder", default="hevc_nvenc")
    parser.add_argument("--seg-model", default="yolo26n-seg.pt")
    parser.add_argument("--pose-model", default="yolo26n-pose.pt")
    args = parser.parse_args(argv)
    args.input, args.output_dir = args.input.resolve(), args.output_dir.resolve()
    for name in ("seg_model", "pose_model"):
        candidate = Path(getattr(args, name)).expanduser()
        if candidate.is_file():
            setattr(args, name, str(candidate.resolve()))
    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        parser.error(f"Cannot create new output directory: {error}")
    results = {"schema_version": 1, "environment": environment(),
               "configuration": {key: str(value) if isinstance(value, Path) else value
                                 for key, value in vars(args).items()}, "runs": []}
    try:
        results["source_probe"] = probe(args.input)
        save(args.output_dir, results)
        for case, warmup, repeat in schedule(args.repeats):
            record = run_one(args, case, warmup, repeat, results["source_probe"])
            results["runs"].append(record)
            save(args.output_dir, results)
            if record["status"] != "ok":
                print(f"Benchmark stopped: {record['error']}", file=sys.stderr)
                if record.get("interrupted"):
                    return 130
                code = record.get("returncode") or 1
                return 128 - code if code < 0 else code
    except (Exception, KeyboardInterrupt) as error:
        results["error"] = f"{type(error).__name__}: {error}"
        save(args.output_dir, results)
        print(f"Benchmark stopped: {results['error']}", file=sys.stderr)
        return 1
    print(f"\nCollected logs, results.json, results.csv and aggregates.csv in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Collect reproducible Colab timings without importing torch or downloading models here."""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]
CASES = (("gap1-baseline", 1, "lazy", "bgr-area", "all"),
         ("gap3-bgr-all", 3, "lazy", "bgr-area", "all"),
         ("gap3-gray-all", 3, "lazy", "gray-area", "all"),
         ("gap3-gray-primary", 3, "lazy", "gray-area", "primary"))
THUMBNAIL_CASES = (("gap1-lazy", 1, "lazy", "bgr-area", "all"),
                   ("gap3-eager", 3, "eager", "bgr-area", "all"),
                   ("gap3-lazy", 3, "lazy", "bgr-area", "all"))
COMPARISON_CASES = (CASES[0], CASES[-1])
SUITES = {"optimizations": CASES, "comparison": COMPARISON_CASES, "thumbnails": THUMBNAIL_CASES}
TIMING_KEYS = {"thumbnail", "scheduler", "frame_change", "model_track", "parse_masks",
               "flow", "predict", "bookkeeping"}
PIPELINE_TIMING_KEYS = {"setup", "decode", "scene", "segmentation", "pose", "feedback",
                        "saliency", "subjects", "camera", "render_write", "bookkeeping", "finalization"}
SALIENCY_TIMING_KEYS = {"preparation", "observe", "selection_cache", "cheap", "deepgaze", "bookkeeping"}
DEEPGAZE_TIMING_KEYS = {"init", "observe", "preprocess", "transfer", "forward", "postprocess",
                       "fallback", "bookkeeping"}
SEGMENT_COUNTER_KEYS = {"seg_detector_calls", "seg_predicted_frames", "seg_thumbnail_builds",
                        "seg_thumbnail_cache_hits", "seg_thumbnail_backfills", "seg_thumbnail_deferred_frames",
                        "seg_flow_attempts", "pose_rois_inferred", "pose_rois_matched", "pose_cache_hits",
                        "pose_primary_only_frames", "pose_full_scan_frames", "pose_rois_skipped",
                        "scene_resets", "subject_switches", "frames_with_subject", "frames_with_head_cues",
                        "frames_with_pose", "frames_with_two_person",
                        *(f"saliency_{key}" for key in (
                            "frames_total", "frames_backend", "frames_fallback", "prediction_requests",
                            "actual_forward_calls", "observed_frames", "sample_frames_total", "sample_refreshes",
                            "sample_propagated", "deepgaze_requests", "handcrafted_predictions", "cache_hits",
                            "tier_transitions", "tier_pose_only_frames", "tier_handcrafted_frames", "tier_deepgaze_frames"))}
SEGMENT_MAP_KEYS = {"seg_refresh_reasons", "seg_gate_diagnostics", "seg_timing_seconds",
                    "saliency_tier_reasons", "saliency_timing_seconds", "saliency_deepgaze_timing_seconds",
                    "pipeline_timing_seconds"}


def validate_timing_sum(summary: dict, map_key: str, total_key: str, required: set[str]) -> None:
    timings = summary.get(map_key)
    if not isinstance(timings, dict) or not required.issubset(timings):
        raise ValueError(f"Missing detailed timings: {map_key}; update the repository and rerun")
    values = [*timings.values(), summary.get(total_key)]
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value < 0 for value in values):
        raise ValueError(f"Missing, negative, or non-finite timings: {map_key}")
    total = summary[total_key]
    if not math.isclose(sum(timings.values()), total, rel_tol=0, abs_tol=1e-4 + 1e-6 * total):
        raise ValueError(f"Detailed {map_key} timings do not sum to {total_key}")


def parse_summary(log: str) -> dict:
    summaries = [json.loads(line.partition("Summary: ")[2])
                 for line in log.splitlines() if "Summary: " in line]
    if len(summaries) != 1 or not isinstance(summaries[0], dict):
        raise ValueError("Expected exactly one JSON Summary in the run log")
    summary = summaries[0]
    if summary.get("timing_schema_version") != 2:
        raise ValueError("Expected timing_schema_version=2; update the repository and rerun")
    for map_key, total_key, keys in (
        ("seg_timing_seconds", "seg_total_seconds", TIMING_KEYS),
        ("pipeline_timing_seconds", "elapsed_seconds", PIPELINE_TIMING_KEYS),
        ("saliency_timing_seconds", "saliency_total_seconds", SALIENCY_TIMING_KEYS),
        ("saliency_deepgaze_timing_seconds", "saliency_deepgaze_total_seconds", DEEPGAZE_TIMING_KEYS),
    ):
        validate_timing_sum(summary, map_key, total_key, keys)
    stages = summary.get("stage_wall_seconds", {})
    for key in ("segmentation", "pose", "saliency", "render_write"):
        if stages.get(key) != summary["pipeline_timing_seconds"][key]:
            raise ValueError(f"Stage timing alias differs from pipeline timing: {key}")
    return summary


def extract_segments(log: str, strict: bool = True) -> tuple[list[dict], list[str]]:
    """Recover flushed rows after interruption without inventing a final segment."""
    rows, errors = [], []
    for line_number, line in enumerate(log.splitlines(), 1):
        if "Segment: " not in line:
            continue
        try:
            row = json.loads(line.partition("Segment: ")[2])
            if not isinstance(row, dict):
                raise ValueError("Segment must be an object")
            rows.append(row)
        except (ValueError, TypeError) as error:
            message = f"Invalid Segment at log line {line_number}: {error}"
            if strict:
                raise ValueError(message) from error
            errors.append(message)
    return rows, errors


def validate_segments(rows: list[dict], summary: dict) -> None:
    if summary.get("segment_schema_version") != 1:
        raise ValueError("Expected segment_schema_version=1; update the repository and rerun")
    if len(rows) != summary.get("segments_emitted") or not rows:
        raise ValueError("Segment count differs from Summary or no segments were emitted")
    previous_end, previous_seconds, previous_scene = 0, 0.0, 1
    gap_count, detected_scenes = 0, set()
    for index, row in enumerate(rows, 1):
        if row.get("segment_schema_version") != 1:
            raise ValueError("Unknown Segment schema version")
        for key in ("segment_index", "scene_index", "frame_start", "frame_end", "frames",
                    "seg_detector_calls", "seg_predicted_frames"):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Invalid Segment integer: {key}")
        if row["segment_index"] != index or row["frame_start"] != previous_end + 1:
            raise ValueError("Segment frame coverage is not contiguous")
        if row["frames"] <= 0 or row["frame_end"] - row["frame_start"] + 1 != row["frames"]:
            raise ValueError("Segment frame range differs from frame count")
        if row["frames"] != row["seg_detector_calls"] + row["seg_predicted_frames"]:
            raise ValueError("Segment detector and prediction counts do not cover its frames")
        start, end = row.get("start_seconds"), row.get("end_seconds")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in (start, end)):
            raise ValueError("Invalid Segment video timestamps")
        if end <= start or not math.isclose(start, previous_seconds, abs_tol=1e-6, rel_tol=0):
            raise ValueError("Segment time coverage is not contiguous")
        if row["scene_index"] not in {previous_scene, previous_scene + 1}:
            raise ValueError("Segment scene sequence is not contiguous")
        if row.get("end_reason") not in {"interval", "scene", "end"}:
            raise ValueError("Unknown Segment end reason")
        fps = row.get("source_fps")
        if (isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0
                or not math.isclose(start, (row["frame_start"] - 1) / fps, abs_tol=1e-6, rel_tol=0)
                or not math.isclose(end, row["frame_end"] / fps, abs_tol=1e-6, rel_tol=0)):
            raise ValueError("Segment timestamps differ from source FPS/frame range")
        for key in ("seg_detector_gap_counts", "seg_scheduled_interval_counts"):
            counts = row.get(key)
            if not isinstance(counts, dict) or any(
                    not isinstance(gap, str) or not gap.isdecimal() or int(gap) <= 0
                    or isinstance(count, bool) or not isinstance(count, int) or count < 0
                    for gap, count in counts.items()):
                raise ValueError(f"Invalid Segment interval counts: {key}")
        if sum(row["seg_scheduled_interval_counts"].values()) != row["frames"]:
            raise ValueError("Segment scheduled interval counts do not cover its frames")
        gap_count += sum(row["seg_detector_gap_counts"].values())
        if row["seg_detector_calls"]:
            detected_scenes.add(row["scene_index"])
        previous_end, previous_seconds, previous_scene = row["frame_end"], end, row["scene_index"]
    if previous_end != summary["frames_processed"]:
        raise ValueError("Segment frames do not sum to Summary")
    if gap_count != summary["seg_detector_calls"] - len(detected_scenes):
        raise ValueError("Segment actual detector gaps do not cover consecutive detections")
    for key in SEGMENT_COUNTER_KEYS:
        values = [row.get(key) for row in rows]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise ValueError(f"Missing or invalid Segment counter: {key}")
        if sum(values) != summary.get(key):
            raise ValueError(f"Segment counter does not sum to Summary: {key}")
    for key in SEGMENT_MAP_KEYS:
        summary_key = "segment_pipeline_timing_seconds" if key == "pipeline_timing_seconds" else key
        if any(not isinstance(row.get(key), dict) for row in rows) or not isinstance(summary.get(summary_key), dict):
            raise ValueError(f"Missing Segment/Summary map: {key}")
        keys = set().union(*(row[key] for row in rows))
        keys.update(summary[summary_key])
        for item in keys:
            values = [row[key].get(item, 0) for row in rows]
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value < 0 for value in values):
                raise ValueError(f"Invalid Segment map value: {key}.{item}")
            total = summary[summary_key].get(item, 0)
            if (isinstance(total, bool) or not isinstance(total, (int, float))
                    or not math.isfinite(total) or total < 0):
                raise ValueError(f"Invalid Summary map value: {summary_key}.{item}")
            if not math.isclose(sum(values), total, abs_tol=1e-4 + 1e-6 * total, rel_tol=0):
                raise ValueError(f"Segment map does not sum to Summary: {key}.{item}")
            if key == "pipeline_timing_seconds" and total > summary[key].get(item, 0) + 1e-4:
                raise ValueError(f"Segment frame time exceeds full pipeline time: {item}")


def save_run_segments(record: dict, directory: Path, name: str, log: str) -> None:
    rows, errors = extract_segments(log, strict=False)
    if errors:
        record["segment_parse_errors"] = errors
    metadata = {"benchmark_case": record["case"], "benchmark_warmup": record["warmup"],
                "benchmark_repeat": record["repeat"], "benchmark_partial": record["status"] != "ok"}
    record["segments"] = [{**row, **metadata} for row in rows]
    record["segment_count"] = len(rows)
    csv_path, jsonl_path = directory / f"{name}.segments.csv", directory / f"{name}.segments.jsonl"
    record["segments_csv"], record["segments_jsonl"] = str(csv_path), str(jsonl_path)
    write_csv(csv_path, [flatten(row) for row in record["segments"]])
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in record["segments"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def validate_output(source: dict, output: dict, summary: dict, max_frames: int | None) -> dict:
    original = next(stream for stream in source["streams"] if stream["codec_type"] == "video")
    rendered = next(stream for stream in output["streams"] if stream["codec_type"] == "video")
    source_frames = int(original["nb_read_frames"])
    output_frames = int(rendered["nb_read_frames"])
    source_fps = Fraction(original["avg_frame_rate"])
    output_fps = Fraction(rendered["avg_frame_rate"])
    source_audio = any(stream["codec_type"] == "audio" for stream in source["streams"])
    output_audio = any(stream["codec_type"] == "audio" for stream in output["streams"])
    expected_frames = source_frames if max_frames is None else min(source_frames, max_frames)
    if output_frames != expected_frames or output_frames != summary["frames_processed"]:
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


def validate_devices(summary: dict, device: str, saliency_device: str) -> dict:
    """Check exercised models; lazy models without calls are explicitly untested."""
    device = device.strip().lower()
    expected = None
    if device not in {"auto", "cpu", "mps"}:
        index = "0" if device == "cuda" else device.removeprefix("cuda:")
        if not index.isdecimal():
            raise ValueError(f"Unsupported requested YOLO device: {device}")
        expected = f"cuda:{int(index)}"
    result = {}
    for label, count_key, actual_key, requested in (
        ("segmentation", "seg_detector_calls", "yolo_device", expected),
        ("pose", "pose_rois_inferred", "pose_device", expected),
        ("deepgaze", "saliency_actual_forward_calls", "saliency_device",
         "cuda" if saliency_device == "cuda" else None),
    ):
        calls, actual = summary.get(count_key), summary.get(actual_key)
        if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
            raise ValueError(f"Missing or invalid model call count: {count_key}")
        exercised = calls > 0
        result[label] = {"calls": calls, "actual_device": actual,
                         "status": "exercised" if exercised else "not_exercised"}
        if exercised and requested is not None:
            matches = (actual in {"cuda", "cuda:0"} if label == "deepgaze" else actual == requested)
            if not matches:
                raise ValueError(f"{label} requested {requested}, but actual device was {actual}; "
                                 "device fallback is excluded from benchmark results")
            result[label]["status"] = "cuda_verified"
    if saliency_device == "cuda":
        fallbacks = summary.get("saliency_frames_fallback")
        if isinstance(fallbacks, bool) or not isinstance(fallbacks, int) or fallbacks < 0:
            raise ValueError("Missing or invalid DeepGaze fallback count")
        if fallbacks:
            raise ValueError(f"DeepGaze recorded {fallbacks} fallback frames; inspect the native log")
    return result


def runtime_info(log: str) -> dict:
    match = re.search(r"PyTorch=(\S+) CUDA build=(\S+) CUDA available=(\S+)", log)
    return ({"torch_version": match[1], "torch_cuda_build": match[2],
             "torch_cuda_available": match[3]} if match else {})


def flatten(data: dict, prefix: str = "") -> dict:
    flattened = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten(value, name))
        else:
            flattened[name] = json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
    return flattened


def aggregates(records: list[dict], cases: tuple = CASES) -> list[dict]:
    rows = []
    for name, *_ in cases:
        valid = [flatten({key: value for key, value in record.items() if key != "segments"}) for record in records
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
    cases = SUITES[results["configuration"].get("suite", "optimizations")]
    results["aggregates"] = aggregates(results["runs"], cases)
    temporary = directory / "results.json.tmp"
    temporary.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(directory / "results.json")
    write_csv(directory / "results.csv", [flatten({key: value for key, value in record.items()
                                                  if key != "segments"}) for record in results["runs"]])
    write_csv(directory / "aggregates.csv", results["aggregates"])
    write_csv(directory / "segments.csv", [flatten(segment)
              for record in results["runs"] for segment in record.get("segments", [])])


def archive_results(directory: Path) -> Path:
    """Package diagnostics, including partial failed runs, without copying large videos."""
    destination = Path(str(directory) + ".zip")
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix in {".log", ".json", ".csv", ".jsonl"}:
                archive.write(path, arcname=path.name)
    return destination


def environment() -> dict:
    result = {"python": sys.version, "platform": platform.platform()}
    result["packages"] = {}
    for name in ("torch", "torchvision", "ultralytics", "numpy", "lap", "scenedetect",
                 "opencv-python", "opencv-python-headless", "opencv-contrib-python"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
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


def schedule(repeats: int, cases: tuple = CASES):
    for case in cases:
        yield case, True, 0
    for repeat in range(repeats):
        offset = (repeat // 2) % len(cases)
        order = cases[offset:] + cases[:offset]
        for case in reversed(order) if repeat % 2 else order:
            yield case, False, repeat + 1


def frame_limit(args, warmup: bool = False) -> int | None:
    warmup_frames = getattr(args, "warmup_frames", None)
    if warmup and warmup_frames is not None:
        return warmup_frames
    return getattr(args, "max_frames", None)


def command(args, output: Path, case: tuple, warmup: bool = False) -> list[str]:
    _, gap, mode, method, policy = case
    result = [sys.executable, "-u", "-m", "reframe", str(args.input), str(output),
            "--seg-model", args.seg_model, "--pose-model", args.pose_model,
            "--device", args.device, "--saliency-device", args.saliency_device,
            "--precrop", "middle", "--lock-first-subject", "--dead-zone", "0.06",
            "--conf", "0.3", "--seg-max-gap", str(gap), "--seg-max-age", "0.1",
            "--seg-thumbnail-mode", mode, "--seg-thumbnail-method", method,
            "--seg-skip-policy", policy, "--video-encoder", args.video_encoder,
            "--saliency-trust-repo",
            "--saliency-max-side", "384", "--saliency-interval", "3", "--no-saliency-amp",
            "--stats-interval", str(getattr(args, "stats_interval", 5.0)),
            "--native-debug", "--log-level", getattr(args, "log_level", "INFO"), "--ffmpeg-log-level", "info"]
    limit = frame_limit(args, warmup)
    if limit is not None:
        result.extend(("--max-frames", str(limit)))
    return result


def run_one(args, case: tuple, warmup: bool, repeat: int, source: dict) -> dict:
    suffix = "smoke" if getattr(args, "smoke", False) else "warmup" if warmup else f"run{repeat}"
    name = f"{case[0]}-{suffix}"
    output, logfile = args.output_dir / f"{name}.mp4", args.output_dir / f"{name}.log"
    record = {"case": case[0], "warmup": warmup, "repeat": repeat, "status": "failed",
              "output": str(output), "log": str(logfile), "max_frames": frame_limit(args, warmup),
              "command": command(args, output, case, warmup)}
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
        log = logfile.read_text(encoding="utf-8", errors="replace")
        record["summary"] = parse_summary(log)
        record["runtime"] = runtime_info(log)
        record["output_probe"] = probe(output)
        record["validation"] = validate_output(source, record["output_probe"], record["summary"],
                                                 frame_limit(args, warmup))
        actual_encoder = record["summary"].get("video_encoder_actual")
        if args.video_encoder != "auto" and actual_encoder != args.video_encoder:
            raise ValueError(f"Requested encoder {args.video_encoder}, but actual encoder was {actual_encoder}; "
                             "encoder fallback is excluded from benchmark results")
        record["device_validation"] = validate_devices(record["summary"], args.device, args.saliency_device)
        for model, validation in record["device_validation"].items():
            if validation["status"] == "not_exercised":
                print(f"{model}: no inference in this clip; GPU execution remains untested", flush=True)
        if getattr(args, "stats_interval", 5.0) > 0:
            rows, _ = extract_segments(log)
            validate_segments(rows, record["summary"])
        record["status"] = "ok"
    except (Exception, KeyboardInterrupt) as error:
        record["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, KeyboardInterrupt):
            record["interrupted"] = True
        record.setdefault("wall_seconds", time.perf_counter() - started)
    finally:
        if logfile.exists():
            try:
                save_run_segments(record, args.output_dir, name, logfile.read_text(encoding="utf-8", errors="replace"))
            except Exception as error:
                record["status"] = "failed"
                record.setdefault("error", f"Failed to save segments: {type(error).__name__}: {error}")
    return record


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def positive_seconds(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def workload(source: dict, args, runs: list[tuple]) -> dict:
    video = next(stream for stream in source["streams"] if stream["codec_type"] == "video")
    source_frames, fps = int(video["nb_read_frames"]), Fraction(video["avg_frame_rate"])
    if source_frames <= 0 or fps <= 0:
        raise ValueError("Input must have positive decoded frame count and FPS")
    totals = {"warmup_frames": 0, "measured_frames": 0}
    for _, warmup, _ in runs:
        limit = frame_limit(args, warmup)
        totals["warmup_frames" if warmup else "measured_frames"] += (
            source_frames if limit is None else min(source_frames, limit))
    return {"source_frames": source_frames, "source_fps": float(fps),
            "source_video_seconds": float(source_frames / fps), "run_count": len(runs),
            "warmup_runs": sum(warmup for _, warmup, _ in runs),
            "measured_runs": sum(not warmup for _, warmup, _ in runs),
            **totals, "total_frames": sum(totals.values())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory; existing paths are refused")
    parser.add_argument("--repeats", type=positive, default=2)
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument("--max-frames", type=positive, help="Defaults to 600, or 60 with --smoke")
    limits.add_argument("--full-video", action="store_true", help="Process every source frame in measured runs")
    parser.add_argument("--warmup-frames", type=positive,
                        help="Limit each excluded warmup independently; otherwise use the measured run length")
    parser.add_argument("--stats-interval", type=positive_seconds, default=5.0,
                        help="Segment statistics interval in video seconds (default: 5)")
    parser.add_argument("--log-level", choices=("INFO", "DEBUG"), default="INFO",
                        help="INFO avoids per-ROI debug volume; all child stdout/stderr is preserved")
    parser.add_argument("--archive", action="store_true",
                        help="Create a sibling ZIP of diagnostic logs/JSON/CSV, including failed runs; exclude videos")
    parser.add_argument("--suite", choices=tuple(SUITES), default="optimizations")
    parser.add_argument("--smoke", action="store_true",
                        help="Run the fully optimized case once; exclude it from measured medians")
    parser.add_argument("--device", default="0")
    parser.add_argument("--saliency-device", default="cuda", choices=("cuda", "cpu", "mps", "auto"))
    parser.add_argument("--video-encoder", default="hevc_nvenc")
    parser.add_argument("--seg-model", default="yolo26n-seg.pt")
    parser.add_argument("--pose-model", default="yolo26n-pose.pt")
    args = parser.parse_args(argv)
    if args.full_video and args.smoke:
        parser.error("--full-video cannot be combined with --smoke; use a separate short smoke first")
    if args.max_frames is None and not args.full_video:
        args.max_frames = 60 if args.smoke else 600
    args.input, args.output_dir = args.input.resolve(), args.output_dir.resolve()
    for name in ("seg_model", "pose_model"):
        candidate = Path(getattr(args, name)).expanduser()
        if candidate.is_file():
            setattr(args, name, str(candidate.resolve()))
    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")
    if args.archive and Path(str(args.output_dir) + ".zip").exists():
        parser.error(f"Archive already exists: {args.output_dir}.zip")
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        parser.error(f"Cannot create new output directory: {error}")
    results = {"schema_version": 3, "environment": environment(),
               "configuration": {key: str(value) if isinstance(value, Path) else value
                                 for key, value in vars(args).items()}, "runs": []}
    try:
        results["source_probe"] = probe(args.input)
        runs = ([(CASES[-1], True, 0)] if args.smoke
                else list(schedule(args.repeats, SUITES[args.suite])))
        results["workload"] = workload(results["source_probe"], args, runs)
        budget = results["workload"]
        print(f"Source: {budget['source_frames']} frames at {budget['source_fps']:g} FPS "
              f"({budget['source_video_seconds']:.2f} video seconds). "
              f"Scheduled: {budget['warmup_runs']} warmups + {budget['measured_runs']} measured runs, "
              f"{budget['total_frames']} frames total. "
              "Estimated processing seconds = total frames / observed processing FPS, plus startup and probes.",
              flush=True)
        save(args.output_dir, results)
        for case, warmup, repeat in runs:
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
    finally:
        if args.archive:
            try:
                print(f"Diagnostic archive: {archive_results(args.output_dir)}", flush=True)
            except (OSError, zipfile.BadZipFile) as error:
                print(f"Archive creation failed; original files remain in {args.output_dir}: {error}", file=sys.stderr)
    print(f"\nCollected raw logs, per-run segments, results.json/CSV, aggregates.csv and segments.csv in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import fields
from reframe.config import AppConfig, PRESETS, apply_preset, validate_config

def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stderr,
        force=True,
    )

def parse_args(argv: list[str] | None = None) -> AppConfig:
    parser = argparse.ArgumentParser(
        description="High-performance AI auto-reframe with single-pass pipeline and robust velocity estimation."
    )
    parser.add_argument("input", nargs="?", help="Source widescreen video file")
    parser.add_argument("output", nargs="?", help="Destination vertical video file")
    parser.add_argument("--diagnose-env", action="store_true",
                        help="Print runtime/GPU diagnostics and test the requested encoder, then exit")
    parser.add_argument("--native-debug", action="store_true",
                        help="Enable PyTorch C++ tracebacks and Python fault diagnostics; keep stderr visible")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Process only the first N frames for a short diagnostic render")
    parser.add_argument("--device", default="auto",
                        help="YOLO inference device: auto, cpu, mps, cuda, cuda:N, or GPU index N")

    parser.add_argument("--seg-model", default="yolo26n-seg.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.30)

    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default="talking_head",
    )
    parser.add_argument("--classes", nargs="+", default=None)

    parser.add_argument("--output-width", type=int, default=1080)
    parser.add_argument("--output-height", type=int, default=1920)

    parser.add_argument("--zoom-alpha", type=float, default=0.035)
    parser.add_argument("--target-alpha", type=float, default=0.08)
    parser.add_argument("--motion-response", type=float, default=0.08)
    parser.add_argument("--motion-damping", type=float, default=0.92)
    parser.add_argument("--zoom-response", type=float, default=0.05)
    parser.add_argument("--zoom-damping", type=float, default=0.90)
    parser.add_argument("--max-missed-frames", type=int, default=40)
    parser.add_argument("--min-subject-hold-frames", type=int, default=14)
    parser.add_argument("--switch-score-threshold", type=float, default=1.20)

    parser.add_argument("--max-step-x", type=float, default=None)
    parser.add_argument("--max-step-y", type=float, default=None)

    parser.add_argument("--min-zoom", type=float, default=None)
    parser.add_argument("--max-zoom", type=float, default=None)

    parser.add_argument("--lock-first-subject", action="store_true")
    parser.add_argument("--two-person-framing", action="store_true")
    parser.add_argument("--two-person-threshold", type=float, default=0.75)

    parser.add_argument("--pose-model", default="yolo26n-pose.pt",
                        help="Official YOLO26n pose weights or local COCO-17 person .pt checkpoint")
    parser.add_argument("--pose-imgsz", type=int, default=640)
    parser.add_argument("--pose-conf", type=float, default=0.25,
                        help="Pose person detection confidence threshold")
    parser.add_argument("--keypoint-conf", type=float, default=0.35,
                        help="Minimum confidence for a framing keypoint")
    parser.add_argument("--pose-batch-size", type=int, default=4,
                        help="Maximum person ROIs per pose inference batch")

    parser.add_argument(
        "--saliency-model",
        choices=["auto", "deepgazemr", "handcrafted"],
        default="handcrafted",
    )
    parser.add_argument(
        "--saliency-device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument("--saliency-max-side", type=int, default=384)
    parser.add_argument(
        "--saliency-trust-repo",
        action="store_true",
        help="Allow torch.hub to execute untrusted code from remote or local repo.",
    )

    parser.add_argument("--speaker-aware-mode", action="store_true")
    parser.add_argument("--speaker-json", default="")

    parser.add_argument("--scene-threshold", type=float, default=3.0)
    parser.add_argument("--min-scene-len", type=int, default=15)

    parser.add_argument(
        "--post-restore",
        action="store_true",
        help="Apply unsharp mask and gentle denoising during FFmpeg stage.",
    )

    parser.add_argument("--video-encoder", default="auto")
    parser.add_argument("--ffmpeg-log-level", default="warning",
                        choices=["error", "warning", "info", "verbose", "debug"],
                        help="FFmpeg stderr verbosity; output is streamed to the terminal")
    parser.add_argument("--audio-bitrate", default="192k")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset-ffmpeg", default="medium")

    parser.add_argument("--save-debug-preview", action="store_true")
    parser.add_argument("--debug-path", default="")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--nvenc-preset", choices=[f"p{i}" for i in range(1, 8)], default="p5"
    )
    parser.add_argument(
        "--video-bitrate", default="8M", help="VideoToolbox target bitrate"
    )
    parser.add_argument(
        "--lost-hold-seconds",
        type=float,
        default=None,
        help="Overrides --max-missed-frames (legacy frames at 30 fps)",
    )
    parser.add_argument("--subject-hold-seconds", type=float, default=None)
    parser.add_argument(
        "--dead-zone",
        type=float,
        default=None,
        help="Fraction of crop size; talking_head default .06, others .015",
    )
    parser.add_argument("--zoom-dead-zone", type=float, default=0.04)
    parser.add_argument("--zoom-hold-seconds", type=float, default=0.4)
    parser.add_argument("--fixed-zoom", type=float, default=None)
    parser.add_argument(
        "--cue-interval",
        type=float,
        default=0.2,
        help="Seconds between pose/head updates per track; 0 analyzes every frame",
    )
    parser.add_argument(
        "--cue-top-k",
        type=int,
        default=0,
        help="Limit expensive person cues; 0 means all people",
    )
    parser.add_argument(
        "--temp-dir", default=None, help="Location for lossless temporary videos"
    )
    parser.add_argument("--saliency-max-failures", type=int, default=3)
    parser.add_argument(
        "--retina-masks",
        action="store_true",
        help="Opt in to full-resolution masks",
    )
    parser.add_argument("--saliency-interval", type=int, default=3)
    parser.add_argument("--saliency-ema", type=float, default=0.65)
    parser.add_argument(
        "--saliency-amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CUDA FP16 autocast; automatically disabled if inference fails",
    )
    parser.add_argument(
        "--scene-method", choices=["inline", "prepass"], default="inline"
    )
    parser.add_argument(
        "--scene-downscale",
        type=int,
        default=4,
        help="Spatial reduction for optional PySceneDetect prepass",
    )
    parser.add_argument(
        "--encode-mode", choices=["direct", "lossless"], default="direct"
    )
    parser.add_argument(
        "--pan-time",
        type=float,
        default=None,
        help="Critical spring time scale in seconds",
    )
    parser.add_argument("--zoom-time", type=float, default=None)
    parser.add_argument("--target-time", type=float, default=None)
    parser.add_argument(
        "--pan-speed-x",
        type=float,
        default=None,
        help="Source width fractions per second",
    )
    parser.add_argument(
        "--pan-speed-y",
        type=float,
        default=None,
        help="Source height fractions per second",
    )
    parser.add_argument(
        "--zoom-speed", type=float, default=None, help="Zoom units per second"
    )
    parser.add_argument(
        "--prediction-window", type=float, default=0.3, help="Seconds of motion history"
    )
    parser.add_argument("--lookahead-seconds", type=float, default=0.12)
    parser.set_defaults(**{field.name: getattr(AppConfig(), field.name) for field in fields(AppConfig)})
    args = AppConfig(**vars(parser.parse_args(argv)))
    try:
        validate_config(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def configure_native_debug(argv: list[str]) -> None:
    if "--native-debug" in argv:
        import faulthandler
        os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
        faulthandler.enable()
        print("Native diagnostics: TORCH_SHOW_CPP_STACKTRACES=1; faulthandler enabled",
              file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    configure_native_debug(arguments)
    args = parse_args(arguments)
    setup_logging(args.log_level)
    try:
        # Keep --help and native bootstrap independent of heavy dependency imports.
        from reframe.runtime import resolve_yolo_device, log_runtime_info, diagnose_environment
        args = apply_preset(args)
        args.yolo_device = resolve_yolo_device(args.device)
        log_runtime_info(args)
        if args.diagnose_env:
            return diagnose_environment(args)
        from reframe.pipeline import process_video
        process_video(args)
        return 0
    except KeyboardInterrupt:
        logging.error("Interrupted by user.")
        return 130
    except Exception as exc:
        logging.exception("Failed: %s", exc)
        return 1

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import fields
from reframe.config import (
    AppConfig, PRESETS, PROFILE_DEFAULT, TRACKING_PROFILES, apply_preset, validate_config,
)


LEGACY_OPTIONS = {
    "--motion-response": "--pan-time",
    "--motion-damping": "--pan-time",
    "--zoom-response": "--zoom-time",
    "--zoom-damping": "--zoom-time",
    "--target-alpha": "--target-time",
    "--zoom-alpha": "--zoom-speed",
    "--max-step-x": "--pan-speed-x",
    "--max-step-y": "--pan-speed-y",
    "--max-missed-frames": "--lost-hold-seconds",
    "--min-subject-hold-frames": "--subject-hold-seconds",
    "--saliency-trust-repo": "remove this flag; no longer needed",
}


class _LegacyAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True if self.nargs == 0 else values)
        flag = self.option_strings[0]
        existing = getattr(namespace, "legacy_flags", ())
        if flag not in existing:
            namespace.legacy_flags = (*existing, flag)


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
    parser.add_argument("--max-frames", type=int,
                        help="Process only the first N frames for a short diagnostic render")
    parser.add_argument("--stats-interval", type=float,
                        help="Emit Segment JSON every N video seconds and at scene cuts; "
                             "0 disables it (default). Does not reset tracking or caches.")
    parser.add_argument("--precrop", choices=["middle", "left", "right"],
                        help="Infer on the middle, left, or right half-width at full height; "
                             "omit to infer on the full frame. Output still crops the full source.")
    parser.add_argument("--device",
                        help="YOLO inference device: auto, cpu, mps, cuda, cuda:N, or GPU index N")
    parser.add_argument("--video-decoder", choices=("opencv", "torchcodec"),
                        help="Optional TorchCodec exact sequential decoding; default: opencv")
    parser.add_argument("--decode-device",
                        help="TorchCodec device: cpu or cuda:N (default cuda:0); CUDA fallback is fatal")
    parser.add_argument("--decode-batch-size", type=int,
                        help="TorchCodec bounded decode batch, 1..8 source frames (default 4)")
    parser.add_argument("--frame-processing", choices=("reference", "tensor"),
                        help="Stage 1 keeps NumPy models/render (reference, default); "
                             "stage 2 uses tensor model inputs and rendering (requires TorchCodec)")

    parser.add_argument("--seg-model")
    parser.add_argument("--tracker")
    parser.add_argument("--tracking-profile", choices=tuple(TRACKING_PROFILES),
                        help="Coordinated segmentation/Pose scheduling: balanced (default), "
                             "conservative, or fast; explicit scheduling flags override profile defaults")
    parser.add_argument("--seg-thumbnail-mode", choices=("lazy", "eager"),
                        help="Build tracking thumbnails on demand (lazy, default), or every "
                             "adaptive frame (eager, timing comparison)")
    parser.add_argument("--seg-thumbnail-method", choices=("gray-area", "bgr-area"),
                        help="Convert to grayscale before area resizing (gray-area, default), "
                             "or resize BGR first (bgr-area, reference)")
    parser.add_argument("--seg-skip-policy", choices=("primary", "all"),
                        help="Permit skipping with distant lost secondary tracks when a primary "
                             "is established (primary, default), or require every track settled (all)")
    parser.add_argument("--seg-max-gap", type=int,
                        help="Maximum adaptive YOLO segmentation interval in source frames "
                             "(profile default: balanced/fast 6, conservative 3); "
                             "1 runs segmentation every frame")
    parser.add_argument("--seg-max-age", type=float,
                        help="Maximum time between real segmentation measurements in video seconds "
                             "(default: 0.1); must be finite and positive")
    parser.add_argument("--conf", type=float)

    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
    )
    parser.add_argument("--classes", nargs="+")

    parser.add_argument("--output-width", type=int)
    parser.add_argument("--output-height", type=int)

    for flag in ("--zoom-alpha", "--target-alpha", "--motion-response",
                 "--motion-damping", "--zoom-response", "--zoom-damping",
                 "--max-step-x", "--max-step-y"):
        parser.add_argument(flag, type=float, action=_LegacyAction, help=argparse.SUPPRESS)
    for flag in ("--max-missed-frames", "--min-subject-hold-frames"):
        parser.add_argument(flag, type=int, action=_LegacyAction, help=argparse.SUPPRESS)
    parser.add_argument("--switch-score-threshold", type=float)

    parser.add_argument("--min-zoom", type=float)
    parser.add_argument("--max-zoom", type=float)

    parser.add_argument("--lock-first-subject", action="store_true")
    parser.add_argument("--two-person-framing", action="store_true")
    parser.add_argument("--two-person-threshold", type=float)

    parser.add_argument("--pose-model",
                        help="Official YOLO26n pose weights or local COCO-17 person .pt checkpoint")
    parser.add_argument("--pose-imgsz", type=int)
    parser.add_argument("--pose-conf", type=float,
                        help="Pose person detection confidence threshold")
    parser.add_argument("--keypoint-conf", type=float,
                        help="Minimum confidence for a framing keypoint")
    parser.add_argument("--pose-batch-size", type=int,
                        help="Maximum person ROIs per pose inference batch")

    parser.add_argument(
        "--saliency-device",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument("--saliency-max-side", type=int)
    parser.add_argument(
        "--saliency-trust-repo",
        action=_LegacyAction, nargs=0,
        help=argparse.SUPPRESS,
    )

    parser.add_argument("--speaker-aware-mode", action="store_true")
    parser.add_argument("--speaker-json")

    parser.add_argument("--scene-threshold", type=float)
    parser.add_argument("--min-scene-len", type=int)

    parser.add_argument(
        "--post-restore",
        action="store_true",
        help="Opt in to FFmpeg unsharp and denoising filters (disabled by default for speed).",
    )

    parser.add_argument("--video-encoder")
    parser.add_argument("--ffmpeg-log-level",
                        choices=["error", "warning", "info", "verbose", "debug"],
                        help="FFmpeg stderr verbosity; output is streamed to the terminal")
    parser.add_argument("--audio-bitrate")
    parser.add_argument("--crf", type=int)
    parser.add_argument("--preset-ffmpeg")

    parser.add_argument("--save-debug-preview", action="store_true")
    parser.add_argument("--debug-path")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--nvenc-preset", choices=[f"p{i}" for i in range(1, 8)]
    )
    parser.add_argument(
        "--video-bitrate", help="VideoToolbox target bitrate"
    )
    parser.add_argument(
        "--lost-hold-seconds",
        type=float,
        help="Seconds to retain a temporarily missing subject (default 40/30 seconds)",
    )
    parser.add_argument("--subject-hold-seconds", type=float,
                        help="Minimum seconds before a subject switch (default 14/30 seconds)")
    parser.add_argument(
        "--dead-zone",
        type=float,
        help="Optional pan dead zone as a fraction of crop size; default 0 for every preset",
    )
    parser.add_argument("--zoom-dead-zone", type=float)
    parser.add_argument("--zoom-hold-seconds", type=float)
    parser.add_argument("--fixed-zoom", type=float)
    parser.add_argument(
        "--cue-interval",
        type=float,
        help="Soft Pose refresh interval in video seconds (profile default); "
             "stable reliable poses may be reused up to pose-max-age; 0 analyzes every frame",
    )
    parser.add_argument("--pose-max-age", type=float,
                        help="Hard age limit for cached Pose in video seconds (profile default); "
                             "0 forbids cross-frame reuse; may be shorter than cue-interval")
    parser.add_argument(
        "--cue-top-k",
        type=int,
        help="Limit person cues during discovery; 0 means all. Stable subjects use priority updates.",
    )
    parser.add_argument(
        "--temp-dir", help="Location for lossless temporary videos"
    )
    parser.add_argument("--saliency-max-failures", type=int)
    parser.add_argument(
        "--retina-masks",
        action="store_true",
        help="Opt in to full-resolution masks",
    )
    parser.add_argument("--saliency-interval", type=int)
    parser.add_argument("--saliency-ema", type=float)
    parser.add_argument(
        "--saliency-amp",
        action=argparse.BooleanOptionalAction,
        help="CUDA FP16 autocast; automatically disabled if inference fails",
    )
    parser.add_argument(
        "--scene-method", choices=["inline", "prepass"]
    )
    parser.add_argument(
        "--scene-downscale",
        type=int,
        help="Spatial reduction for optional PySceneDetect prepass",
    )
    parser.add_argument(
        "--encode-mode", choices=["direct", "lossless"]
    )
    parser.add_argument(
        "--pan-time",
        type=float,
        help="Critical spring time scale in seconds",
    )
    parser.add_argument("--zoom-time", type=float)
    parser.add_argument("--target-time", type=float)
    parser.add_argument(
        "--pan-speed-x",
        type=float,
        help="Source width fractions per second",
    )
    parser.add_argument(
        "--pan-speed-y",
        type=float,
        help="Source height fractions per second",
    )
    parser.add_argument(
        "--zoom-speed", type=float, help="Zoom units per second"
    )
    parser.add_argument(
        "--prediction-window", type=float, help="Seconds of motion history"
    )
    parser.add_argument("--lookahead-seconds", type=float)
    defaults = AppConfig()
    parser_defaults = {field.name: getattr(defaults, field.name) for field in fields(AppConfig)}
    # Keep omitted values unresolved until the requested profile is known.
    # An explicit override equal to another profile's default still wins.
    parser_defaults.update({key: PROFILE_DEFAULT for key in TRACKING_PROFILES["balanced"]})
    parser.set_defaults(**parser_defaults)
    try:
        args = AppConfig(**vars(parser.parse_args(argv)))
        validate_config(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.legacy_flags:
        changes = "; ".join(f"{flag} -> {LEGACY_OPTIONS[flag]}" for flag in args.legacy_flags)
        print(f"Legacy CLI options accepted: {changes}. Explicit modern options take precedence.",
              file=sys.stderr, flush=True)
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

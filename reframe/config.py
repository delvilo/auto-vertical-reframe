from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


CLASS_IDS = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
    "cat": 15,
    "dog": 16,
}

PRESETS = {
    "talking_head": {
        "classes": ["person"],
        "min_zoom": 1.00,
        "max_zoom": 1.85,
        "max_step_x": 6.0,
        "max_step_y": 4.0,
    },
    "sports": {
        "classes": ["person", "car", "bicycle", "motorcycle"],
        "min_zoom": 1.00,
        "max_zoom": 1.35,
        "max_step_x": 12.0,
        "max_step_y": 8.0,
    },
    "pets": {
        "classes": ["dog", "cat", "person"],
        "min_zoom": 1.00,
        "max_zoom": 1.55,
        "max_step_x": 10.0,
        "max_step_y": 7.0,
    },
    "cars": {
        "classes": ["car", "truck", "bus", "motorcycle", "person"],
        "min_zoom": 1.00,
        "max_zoom": 1.30,
        "max_step_x": 11.0,
        "max_step_y": 7.0,
    },
}


@dataclass
class AppConfig:
    """CLI configuration plus resolved runtime values; no argparse crosses modules."""
    input: str | None = None
    output: str | None = None
    diagnose_env: bool = False
    native_debug: bool = False
    max_frames: int | None = None
    device: str = 'auto'
    seg_model: str = 'yolo26n-seg.pt'
    tracker: str = 'bytetrack.yaml'
    conf: float = 0.3
    preset: str = 'talking_head'
    classes: list[str] | None = None
    output_width: int = 1080
    output_height: int = 1920
    zoom_alpha: float = 0.035
    target_alpha: float = 0.08
    motion_response: float = 0.08
    motion_damping: float = 0.92
    zoom_response: float = 0.05
    zoom_damping: float = 0.9
    max_missed_frames: int = 40
    min_subject_hold_frames: int = 14
    switch_score_threshold: float = 1.2
    max_step_x: float | None = None
    max_step_y: float | None = None
    min_zoom: float | None = None
    max_zoom: float | None = None
    lock_first_subject: bool = False
    two_person_framing: bool = False
    two_person_threshold: float = 0.75
    pose_model: str = 'yolo26n-pose.pt'
    pose_imgsz: int = 640
    pose_conf: float = 0.25
    keypoint_conf: float = 0.35
    pose_batch_size: int = 4
    saliency_model: str = 'deepgazemsdb'
    saliency_device: str = 'auto'
    saliency_max_side: int = 384
    saliency_center_bias: str = 'mit1003'
    saliency_screen_inches: float = 24.0
    saliency_viewing_distance_cm: float = 60.0
    saliency_pixel_per_dva: float | None = None
    speaker_aware_mode: bool = False
    speaker_json: str = ''
    scene_threshold: float = 3.0
    min_scene_len: int = 15
    post_restore: bool = False
    video_encoder: str = 'auto'
    ffmpeg_log_level: str = 'warning'
    audio_bitrate: str = '192k'
    crf: int = 18
    preset_ffmpeg: str = 'medium'
    save_debug_preview: bool = False
    debug_path: str = ''
    log_level: str = 'INFO'
    nvenc_preset: str = 'p5'
    video_bitrate: str = '8M'
    lost_hold_seconds: float | None = None
    subject_hold_seconds: float | None = None
    dead_zone: float | None = None
    zoom_dead_zone: float = 0.04
    zoom_hold_seconds: float = 0.4
    fixed_zoom: float | None = None
    cue_interval: float = 0.2
    cue_top_k: int = 0
    temp_dir: str | None = None
    saliency_max_failures: int = 3
    retina_masks: bool = False
    saliency_interval: int = 3
    saliency_ema: float = 0.65
    saliency_amp: bool = False
    scene_method: str = 'inline'
    scene_downscale: int = 4
    encode_mode: str = 'direct'
    pan_time: float | None = None
    zoom_time: float | None = None
    target_time: float | None = None
    pan_speed_x: float | None = None
    pan_speed_y: float | None = None
    zoom_speed: float | None = None
    prediction_window: float = 0.3
    lookahead_seconds: float = 0.12
    yolo_device: str = "cpu"
    runtime_fps: float = 30.0
    reference_dt: float = 1.0


def validate_config(args: AppConfig) -> None:
    if not args.diagnose_env and (not args.input or not args.output):
        raise ValueError("input and output are required unless --diagnose-env is used")
    if args.pose_imgsz < 32 or args.pose_imgsz % 32 or args.pose_batch_size < 1:
        raise ValueError("pose-imgsz must be a positive multiple of 32; pose-batch-size must be >= 1")
    for key in ("pose_conf", "keypoint_conf"):
        value = getattr(args, key)
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{key} must be finite and in (0, 1]")
    if Path(args.pose_model).suffix.lower() != ".pt":
        raise ValueError("pose-model requires a COCO-17 YOLO .pt checkpoint")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("max-frames must be >= 1")
    if args.saliency_model not in {"auto", "deepgazemsdb", "handcrafted"}:
        raise ValueError("saliency-model must be deepgazemsdb, handcrafted or auto")
    if args.saliency_center_bias not in {"mit1003", "uniform"}:
        raise ValueError("saliency-center-bias must be mit1003 or uniform")
    for key in ("saliency_screen_inches", "saliency_viewing_distance_cm", "saliency_pixel_per_dva"):
        value = getattr(args, key)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{key} must be finite and positive")
    if args.saliency_max_side < 32:
        raise ValueError("saliency-max-side must be >= 32 (handcrafted only)")

    if (
        args.saliency_interval < 1
        or args.scene_downscale < 1
        or not 0 < args.saliency_ema <= 1
    ):
        raise ValueError("Invalid saliency interval/EMA or scene downscale")
    for key in (
        "pan_time",
        "zoom_time",
        "target_time",
        "pan_speed_x",
        "pan_speed_y",
        "zoom_speed",
        "prediction_window",
    ):
        value = getattr(args, key)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{key} must be finite and positive")
    if not 0 <= args.lookahead_seconds <= 1:
        raise ValueError("lookahead-seconds must be 0..1")
    if (
        args.output_width < 64
        or args.output_height < 64
        or args.output_width % 2
        or args.output_height % 2
    ):
        raise ValueError("Output dimensions must be even and at least 64")
    if not 0 <= args.conf <= 1 or not 0 <= args.crf <= 51:
        raise ValueError("conf must be 0..1 and crf/quality must be 0..51")
    for key in (
        "lost_hold_seconds",
        "subject_hold_seconds",
        "zoom_hold_seconds",
        "cue_interval",
    ):
        value = getattr(args, key)
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"{key} must be finite and nonnegative")
    for key in ("min_zoom", "max_zoom", "fixed_zoom"):
        value = getattr(args, key)
        if value is not None and (not math.isfinite(value) or value < 1):
            raise ValueError(f"{key} must be finite and >= 1")
    for key in ("dead_zone", "zoom_dead_zone"):
        value = getattr(args, key)
        if value is not None and not 0 <= value <= 0.4:
            raise ValueError(f"{key} must be between 0 and .4")
    if args.video_encoder not in {
        "auto",
        "libx264",
        "libx265",
        "h264_nvenc",
        "hevc_nvenc",
        "h264_videotoolbox",
        "hevc_videotoolbox",
    }:
        raise ValueError("Unsupported video encoder")
    for key in ("motion_response", "zoom_response", "zoom_alpha", "target_alpha"):
        if not 0 < getattr(args, key) <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    for key in ("motion_damping", "zoom_damping"):
        if not 0 <= getattr(args, key) <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
    if args.max_missed_frames < 0 or args.min_subject_hold_frames < 0:
        raise ValueError("Hold frame counts must be nonnegative")
    for key in ("max_step_x", "max_step_y"):
        value = getattr(args, key)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{key} must be finite and positive")
    if args.cue_top_k < 0 or args.saliency_max_failures < 1:
        raise ValueError("cue-top-k must be >=0 and saliency-max-failures >=1")


def apply_preset(args: AppConfig) -> AppConfig:
    preset = PRESETS[args.preset]

    if args.classes is None:
        args.classes = list(preset["classes"])

    for key in ["max_step_x", "max_step_y", "min_zoom", "max_zoom"]:
        if getattr(args, key) is None:
            setattr(args, key, preset[key])

    if args.min_zoom > args.max_zoom:
        raise ValueError("min-zoom must be <= max-zoom")
    if (
        args.fixed_zoom is not None
        and not args.min_zoom <= args.fixed_zoom <= args.max_zoom
    ):
        raise ValueError("fixed-zoom must lie inside min/max zoom")
    if args.dead_zone is None:
        args.dead_zone = 0.06 if args.preset == "talking_head" else 0.015

    if args.pan_time is None:
        args.pan_time = (
            0.35
            * math.sqrt(0.08 / args.motion_response)
            * max(0.2, args.motion_damping / 0.92)
        )
    if args.zoom_time is None:
        args.zoom_time = (
            0.60
            * math.sqrt(0.05 / args.zoom_response)
            * max(0.2, args.zoom_damping / 0.90)
        )
    if args.target_time is None:
        args.target_time = -(1 / 30) / math.log1p(-min(args.target_alpha, 0.999))
    if args.pan_speed_x is None:
        args.pan_speed_x = args.max_step_x * 30 / 1920
    if args.pan_speed_y is None:
        args.pan_speed_y = args.max_step_y * 30 / 1080
    if args.zoom_speed is None:
        args.zoom_speed = args.zoom_alpha * 30
    return args

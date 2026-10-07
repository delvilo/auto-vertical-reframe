from __future__ import annotations

import hashlib
import importlib.metadata
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path
from reframe import __version__
from reframe.video_io import select_live_encoder, build_video_filters
try:
    import torch
except Exception:
    torch = None
    logging.warning("PyTorch import failed", exc_info=True)

def resolve_yolo_device(requested: str) -> str:
    """Choose an explicit backend; a requested unavailable GPU must not become CPU."""
    device = requested.strip().lower()
    if device == "cpu":
        return device
    if device == "auto":
        if torch is not None and torch.cuda.is_available():
            return "cuda:0"
        logging.warning("YOLO auto device: CUDA unavailable; selecting CPU")
        return "cpu"
    if device == "mps":
        if (torch is None or not hasattr(torch.backends, "mps")
                or not torch.backends.mps.is_available()):
            raise RuntimeError("Requested YOLO MPS device is unavailable")
        return device
    index = "0" if device == "cuda" else device.removeprefix("cuda:")
    if not index.isdecimal():
        raise ValueError("--device must be auto, cpu, mps, cuda, cuda:N, or GPU index N")
    if (torch is None or not torch.cuda.is_available()
            or int(index) >= torch.cuda.device_count()):
        raise RuntimeError(f"Requested YOLO CUDA device {requested!r} is unavailable")
    return f"cuda:{int(index)}"


def log_runtime_info(args) -> None:
    package = Path(__file__).resolve().parent
    hashes = {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(package.rglob("*.py"))}
    fingerprint = hashlib.sha256("\n".join(f"{name}:{sha}" for name, sha in hashes.items()).encode()).hexdigest()
    logging.info("Package=auto-vertical-reframe version=%s root=%s sha256=%s", __version__, package, fingerprint)
    logging.info("Module SHA256: %s", hashes)
    logging.info("Python=%s version=%s cwd=%s", sys.executable, sys.version.split()[0], Path.cwd())
    logging.info("Command: %s", shlex.join([sys.executable, *sys.argv]))
    logging.info("YOLO model=%s requested_device=%s selected_device=%s",
                 args.seg_model, args.device, args.yolo_device)
    logging.info("YOLO pose model=%s selected_device=%s; encoder requested=%s",
                 args.pose_model, args.yolo_device, args.video_encoder)
    if torch is not None:
        logging.info("PyTorch=%s CUDA build=%s CUDA available=%s",
                     torch.__version__, torch.version.cuda, torch.cuda.is_available())
        if args.yolo_device.startswith("cuda:"):
            index = int(args.yolo_device.split(":")[1])
            logging.info("YOLO GPU=%s capability=%s", torch.cuda.get_device_name(index),
                         torch.cuda.get_device_capability(index))
    if args.native_debug and torch is not None:
        logging.info("Native diagnostics: TORCH_SHOW_CPP_STACKTRACES=%s cuDNN=%s enabled=%s",
                     os.environ.get("TORCH_SHOW_CPP_STACKTRACES", "<unset>"),
                     torch.backends.cudnn.version(), torch.backends.cudnn.enabled)
        logging.info("PyTorch build configuration:\n%s", torch.__config__.show())
    logging.debug("Resolved arguments: %s", vars(args))


def verify_yolo_device(model, expected: str, label: str = "segmentation") -> str:
    """Report the predictor's actual backend after lazy model initialization."""
    actual = str(model.predictor.device)
    logging.info("YOLO %s actual inference device=%s (selected=%s)", label, actual, expected)
    if actual != expected and not (expected == "mps" and actual == "mps:0"):
        raise RuntimeError(f"YOLO device mismatch: selected {expected}, actual {actual}")
    return actual


def diagnose_environment(args) -> int:
    """Run bounded probes in the same CLI interpreter without loading video/model files."""
    failed = False
    for key in ("LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "LD_PRELOAD"):
        logging.info("%s=%s", key, os.environ.get(key, "<unset>"))
    for name in ("torch", "torchvision", "ultralytics", "scenedetect", "lap",
                 "numpy", "opencv-python", "opencv-contrib-python", "opencv-python-headless"):
        try:
            logging.info("Package %s=%s", name, importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:
            logging.info("Package %s not installed", name)
    for command in (["nvidia-smi"], ["ffmpeg", "-hide_banner", "-version"]):
        logging.info("Diagnostic command: %s", shlex.join(command))
        try:
            result = subprocess.run(command, timeout=30, check=False)
            logging.info("Diagnostic exit=%s", result.returncode)
            # CPU-only machines need not have nvidia-smi.
            failed |= result.returncode != 0 and command[0] == "ffmpeg"
        except (OSError, subprocess.TimeoutExpired):
            logging.warning("Diagnostic command failed", exc_info=True)
            failed |= command[0] == "ffmpeg"
    if args.yolo_device.startswith("cuda:"):
        try:
            x = torch.ones((32, 32), device=args.yolo_device)
            y = x @ x
            torch.cuda.synchronize(args.yolo_device)
            logging.info("CUDA matmul device=%s result=%s (expected 32)", y.device, y[0, 0].item())
            image = torch.ones((1, 3, 64, 64), device=args.yolo_device)
            kernel = torch.ones((8, 3, 3, 3), device=args.yolo_device)
            convolution = torch.nn.functional.conv2d(image, kernel)
            torch.cuda.synchronize(args.yolo_device)
            logging.info("CUDA convolution device=%s result=%s (expected 27) cuDNN=%s enabled=%s",
                         convolution.device, convolution[0, 0, 0, 0].item(),
                         torch.backends.cudnn.version(), torch.backends.cudnn.enabled)
        except Exception:
            logging.exception("CUDA computation probe failed")
            failed = True
    try:
        encoder = select_live_encoder(args, 30, (320, 240), build_video_filters(args.post_restore))
        if args.video_encoder != "auto" and encoder != args.video_encoder:
            logging.error("Requested encoder failed diagnostic: requested=%s fallback=%s",
                          args.video_encoder, encoder)
            failed = True
    except Exception:
        logging.exception("Encoder diagnostic failed")
        failed = True
    logging.info("Diagnostics finished; no video was processed")
    return int(failed)



# auto-vertical-reframe

[English](README.md) | [繁體中文](README.zh-TW.md)

Scene-aware vertical auto-reframe CLI that turns horizontal footage into 9:16 video without losing the subject.

---

## Highlights

- **Cross-Scene Subject Tracking**: Seamlessly track people, pets, and vehicles through cuts with YOLO26 instance segmentation and ByteTrack.
- **Multi-Cue Framing**: Integrates head cues, pose keypoints, and saliency maps with subject scoring and a smoothed camera path.
- **Four Tuned Presets**: Built-in presets (`talking_head`, `sports`, `pets`, `cars`) with calibrated zoom and pan limits.
- **Three-Tier Composition**: Pose-first composition (L1) → lightweight handcrafted saliency (L2) → on-demand DeepGaze MR (L3).
- **Inference Region Restriction (`--precrop`)**: Accelerates high-resolution footage via middle, left, or right half-width inference while cropping from the full-resolution source.

---

## Overview

Vertical platforms (Reels, Shorts, TikTok) require 9:16 video, but most high-quality content is shot horizontally. Auto Vertical Reframe reads a video, detects subjects per scene, ranks candidates using multi-model signals, and drives a virtual camera (pan + zoom) through a smoothed path optimizer. It emits a ready-to-publish MP4 via FFmpeg.

Naive center-cropping loses the subject the moment they move; manual reframing is tedious for long footage. Auto Vertical Reframe combines segmentation, pose/head cues, saliency, tracking continuity, and scene detection so each shot gets its own framing decision without relying on a static center crop.

---

## Features

- **Per-Scene Subject Selection**: Uses PySceneDetect (`AdaptiveDetector`) for shot transitions, resets, and re-evaluation.
- **YOLO26 Instance Segmentation**: Configurable target classes and detection confidence.
- **Adaptive Segmentation Intervals**: Progressively skips YOLO calls using sparse optical-flow correction while preserving every source frame.
- **YOLO26n-pose Keypoints**: 17 COCO keypoints with conservative head bounds estimation and batched person ROI inference.
- **Two-Person Framing Mode**: Automatically widens composition when a secondary subject crosses a spatial threshold.
- **Pose-First Three-Tier Composition**: Stable subjects compose directly from pose and segmentation masks; uncertain scenes escalate to DeepGaze MR.
- **Lean On-Demand DeepGaze MR**: Loads only the feature extraction checkpoint (avoiding the 548 MB ImageNet VGG download) across a continuous 16-frame temporal window.
- **Staged TorchCodec Video Pipeline**: Supports NVDEC hardware decoding and end-to-end GPU Tensor image processing with explicit CPU fallback detection.
- **Detailed Telemetry Diagnostics**: Reports tier decisions, real model invocations, cache ages, and hardware placement in the final Summary.
- **Camera Motion Controls**: Subject lock, min/max zoom, max step-per-frame, per-axis damping, and dead zones.
- **FFmpeg Encoding Control**: Custom video encoder (including `hevc_nvenc`), CRF, audio bitrate, and optional post-processing filters.
- **Debug Preview Export**: Generates frame-by-frame visual diagnostic overlays.

`auto_reframe.py` serves as a compatible CLI entry point. The core `reframe` package cleanly separates perception, saliency, composition, and video I/O; see [Architecture](#architecture).

---

## Tech Stack

- **Language:** Python 3.13+
- **Object Detection & Tracking:** Ultralytics YOLO26, ByteTrack (`lap`)
- **Pose & Head Cues:** YOLO26n-pose (`yolo26n-pose.pt`)
- **Scene Detection:** PySceneDetect
- **ML Runtime:** PyTorch 2.11+ (CUDA, CPU, MPS)
- **Video Processing:** FFmpeg (CLI) and optional TorchCodec (NVDEC)

---

## Quick Start

Requires Python 3.13+ and system-installed `ffmpeg`. Install Python packages inside your environment:

```bash
python3 -m pip install -e .
auto_reframe.py input.mp4 output.mp4 --device auto
```

`requirements.txt` pins Ultralytics and tracking dependencies. It uses a single OpenCV provider (`opencv-python`); installing headless/contrib packages alongside it can overwrite shared `cv2` bindings. PyTorch/TorchVision requires matching system CUDA builds.

Development targets **Python 3.13+** as the compatibility baseline (`requires-python >=3.13`). Python 3.12 and older are not supported (see [AGENTS.md](AGENTS.md)).

---

### TorchCodec: Two-Stage Activation

The default decoder remains OpenCV. The TorchCodec path requires installing the matching PyTorch / CUDA wheel (e.g. TorchCodec 0.17.0 cu130) with FFmpeg shared libraries. Fully validated on a Google Colab Tesla T4 GPU on 2026-10-13 (Python 3.13.15, PyTorch 2.11.0+cu130, TorchCodec 0.17.0+cu130, NVDEC hardware decoding verified with zero CPU fallback, HEVC NVENC encoding):

```bash
# Stage 1: NVDEC hardware decoding, reference NumPy model/render path
auto_reframe.py input.mp4 stage1.mp4 --device 0 --saliency-device cuda \
  --video-decoder torchcodec --decode-device cuda:0 --decode-batch-size 4

# Stage 2: Tensor preprocessing on GPU, DeepGaze small tensor window, GPU crop/resize
auto_reframe.py input.mp4 stage2.mp4 --device 0 --saliency-device cuda \
  --video-decoder torchcodec --decode-device cuda:0 --decode-batch-size 4 \
  --frame-processing tensor
```

Both stages preserve audio synchronization, source frame counts, and presentation timestamps; `precrop` only restricts the inference region. Stage 2 supports PyTorch `.pt` YOLO checkpoints and ByteTrack. Variable frame rate (VFR) inputs are rejected to avoid playback speed distortion.

See [T4 Validation Specification](docs/t4-validation.md) and [AI Agent Handoff](docs/agent-handoff.md) for full execution guidelines. The latest hardware reports are available at [T4 GPU Verification Report](docs/reports/t4-2026-10-13-gpu-verified.md) and [YouTube 4K Portrait Validation Report](docs/reports/youtube-validation-2026-10-13-3079cd0d9549.md).

Scheme B tracking profiles and pose caching are fully integrated; `--dead-zone` defaults to `0` across all presets.

---

### Colab Environment Setup & Installation

Run in a Google Colab terminal or code cell:

```python
!python3 -u /content/auto-vertical-reframe/install_colab.py
```

The installer performs the following actions:
1. Uninstalls legacy `mediapipe` and cleans cached weights.
2. Removes overlapping OpenCV distributions and installs a single provider.
3. Freezes current PyTorch / TorchVision versions (preserving `+cu130` tags).
4. Verifies imports and CUDA status in a clean Python subprocess.
5. Installs the `reframe` package in editable mode with the `auto_reframe.py` console script.

If packages were already imported in the current session, restart the Colab kernel after installation.

---

### YOLO26n-pose Framing Mechanism

- `yolo26n-seg.pt` handles segmentation, ByteTrack IDs, and background objects.
- `yolo26n-pose.pt` processes batches of person ROIs to produce 17 keypoints for head and torso cues.
- Prioritizes the locked primary subject when tracking is stable; expands pose estimation during scene cuts or subject competition.
- Head bounding boxes are estimated from high-confidence facial keypoints (nose, eyes, ears) rather than a separate face detector; falls back to person bounding boxes when keypoints are missing.
- When the subject is turned or occluded, 3 geometrically sound shoulder/hip keypoints provide reliable torso evidence.
- Corrupted model files or missing GPUs fail fast with full tracebacks.

| Option | Default / Purpose |
| :--- | :--- |
| `--pose-model` | `yolo26n-pose.pt`; accepts local COCO-17 person `.pt` checkpoints |
| `--pose-imgsz` | `640`; positive multiple of 32 |
| `--pose-conf` | `0.25`; detection confidence threshold for person boxes |
| `--keypoint-conf` | `0.35`; minimum confidence threshold for individual keypoints |
| `--pose-batch-size` | `4`; maximum person ROIs per inference batch |
| `--cue-interval` | `0.2`s in balanced profile; soft update interval, deferred when stable |
| `--pose-max-age` | `0.35`s in balanced profile; hard cache expiration; `0` disables reuse |
| `--device` | `auto`; `0` enforces primary CUDA GPU, `cpu` forces CPU |

---

### Dynamic Segmentation & Optical Flow Correction

Under `--tracking-profile balanced`, the YOLO segmentation interval progressively extends from **1 → 2 → 3 → 4 → 6 frames** as detections remain stable. Intervals exceeding 3 frames require compatible real poses and stable mask evidence. Intermediate frames are corrected via sparse KLT optical flow and motion extrapolation. Scene cuts, lost tracks, optical flow failures, or subject intersections trigger immediate re-detection.

| Option | Default / Purpose |
| :--- | :--- |
| `--tracking-profile` | `balanced`; options: `conservative`, `balanced`, `fast` |
| `--seg-max-gap` | `6` for balanced/fast, `3` for conservative; `1` disables skipping for baseline comparison |
| `--seg-max-age` | `0.1`s; maximum elapsed video time between real YOLO segmentations |
| `--seg-thumbnail-mode` | `lazy` (builds thumbnails only when flow or scene checks require them) or `eager` |
| `--seg-thumbnail-method` | `gray-area` (grayscale conversion before area resize, default) or `bgr-area` |
| `--seg-skip-policy` | `primary` (deprioritizes distant, small secondary subjects) or `all` |

---

### Scheme C: Three-Tier Composition & On-Demand DeepGaze MR

Each frame determines its composition tier based on tracking stability, pose freshness, image motion, and saliency cache validity:

| Tier | Decision & Processing Flow |
| :--- | :--- |
| **L1 `pose_only`** | Primary subject is stable with reliable body keypoints; composes directly from pose and masks without running saliency models. |
| **L2 `handcrafted`** | Partial pose cues or minor scene changes; uses low-resolution spectral residual and motion differentials. |
| **L3 `deepgaze`** | High ambiguity, lost tracking, or non-human focus targets; invokes DeepGaze MR neural inference. |

Maintains a **16-frame circular ring buffer** of low-resolution frames in memory. DeepGaze is only loaded and executed when entering L3 with a complete 16-frame temporal window.

| Option | Behavior |
| :--- | :--- |
| `--saliency-device` | `auto`; specify `cuda` on GPU runtimes |
| `--saliency-max-side` | `384`; maximum long-edge dimension for saliency processing |
| `--saliency-interval` | `3`; base refresh interval when saliency is required |
| `--saliency-amp` | FP16 autocast enabled by default; use `--no-saliency-amp` for FP32 |
| `--post-restore` | **Disabled by default** (bypasses FFmpeg unsharp and denoise filters) |

---

### `--precrop`: Inference Region Restriction

Retains full height and takes half the width (~50% pixel area) to accelerate high-resolution footage:

| Option | Horizontal Span | Inference Window for 3840×2160 Source |
| :--- | :--- | :--- |
| Omitted | 0%–100% | x=0–3840, height 2160 (Full Frame) |
| `--precrop left` | 0%–50% | x=0–1920, height 2160 |
| `--precrop middle` | 25%–75% | x=960–2880, height 2160 |
| `--precrop right` | 50%–100% | x=1920–3840, height 2160 |

Output video is always cropped from the full-resolution source; crop windows can pan beyond the inference region.

---

### Testing & Documentation Index

- [AI Agent Handoff](docs/agent-handoff.md): Architectural decisions, implementation status, and continuation guidelines.
- [T4 Validation Specification](docs/t4-validation.md): GPU validation criteria, commands, and verification gates.
- [T4 GPU Verification Report (2026-10-13)](docs/reports/t4-2026-10-13-gpu-verified.md): Complete end-to-end telemetry and measurements on a physical Tesla T4 GPU.
- [YouTube 4K Portrait Validation Report (2026-10-13)](docs/reports/youtube-validation-2026-10-13-3079cd0d9549.md): 4K source screening, human calibration gate approval, and batch processing results.

Dedicated runners:
- 4K Portrait Runner: [`scripts/youtube_validation.py`](scripts/youtube_validation.py), [`notebooks/colab-youtube-validation.ipynb`](notebooks/colab-youtube-validation.ipynb)
- Staged T4 Collector: [`scripts/colab_torchcodec.py`](scripts/colab_torchcodec.py), [`notebooks/colab-torchcodec.ipynb`](notebooks/colab-torchcodec.ipynb)

---

## Architecture

| Module | Responsibility |
| :--- | :--- |
| `reframe/cli.py`, `config.py` | Early diagnostics before heavy imports; typed `AppConfig` parsing, defaults, and validation |
| `contracts.py`, `geometry.py`, `precrop.py` | Data contracts, spatial geometry, and inference-to-source coordinate mapping |
| `perception/segmentation.py`, `motion.py`, `pose.py` | Adaptive segmentation, KLT optical flow, mask summaries, batched pose inference, and cue caching |
| `saliency/base.py`, `factory.py`, `backends/` | Backend lifecycle management (`load`, `observe`, `predict`, `reset`, `close`, `telemetry`) |
| `saliency/cascade.py`, `cache.py`, `regions.py` | Pose-driven tier routing, ring buffer observation, static cache reuse, EMA, and saliency regions |
| `subjects.py`, `camera.py` | Candidate enrichment/ranking, subject locking, smoothed camera path optimization |
| `scenes.py`, `video_io.py`, `debug.py`, `runtime.py` | Scene cut detection, FFmpeg subprocess management, visual debug overlays, and runtime telemetry |
| `pipeline.py` | Resource ownership and single-pass streaming orchestration |

Every frame proceeds through: Segmentation observation/prediction → Priority pose update → Three-tier saliency routing → Candidate ranking → Camera composition → Frame encoding. Ranking consumes observations and never invokes models directly.

---

### Local Regression Checks

Run after dependency installation:

```bash
python3 -m unittest discover -s tests -v
python3 -m unittest test_security test_security_reframe -v
python3 -c 'import test_torch_load; test_torch_load.test_torch_load_weights_only()'
```

Tests cover geometry alignment, keypoint freshness, temporal ring ingestion, backend fallbacks, scene cut resets, and FFmpeg pipe handling.

---

## Fixed-Camera Example (Tripod Mode)

To produce reframed vertical video that behaves like a **fixed camera mounted on a tripod**—locking onto the opening subject without digital zoom pumping:

```bash
python auto_reframe.py input.mp4 output_fixed.mp4 \
    --preset talking_head \
    --lock-first-subject \
    --fixed-zoom 1.10 \
    --dead-zone 0.08 \
    --pan-time 0.55
```

### Key Parameter Effects:
1. **`--lock-first-subject`**: Locks the initial detected primary subject's `track_id`. The camera will not switch subjects even if passersby enter the background or talk nearby.
2. **`--fixed-zoom 1.10`**: Locks the virtual camera magnification at $1.10\times$, completely disabling dynamic digital zoom to eliminate "zoom pumping" when subjects lean forward or gesture.
3. **`--dead-zone 0.08`**: Defines an 8% dead zone for crop width and height. Minor subject movements within this boundary do not pan the camera.
4. **`--pan-time 0.55`**: When the subject exits the dead zone, smoothly pans to re-center with gentle ease-in-out spring damping, avoiding sudden camera jerks.

---

## Status & License

This implementation is derived from [KazKozDev/auto-vertical-reframe](https://github.com/KazKozDev/auto-vertical-reframe).

Licensed under the MIT License — see [LICENSE](LICENSE).

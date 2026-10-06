# auto-vertical-reframe
Scene-aware vertical auto-reframe CLI that turns horizontal footage into 9:16 video without losing the subject.


## Highlights

- Tracks people, pets, and vehicles through scene cuts with YOLOv11 segmentation and ByteTrack.
- Face-, pose-, and saliency-aware framing with a subject ranking model and smoothed camera path.
- Four tuned presets (`talking_head`, `sports`, `pets`, `cars`) with sensible zoom and motion limits.
- Fast `handcrafted` saliency by default, with optional slow/experimental `deepgazemr` model saliency.
- One-click macOS launcher (`run_verthor.command`) with native dialogs for video, preset, saliency mode, and debug preview.

## Demo

| Source (16:9) | Auto Vertical Reframe output (9:16) |
| :---: | :---: |
| <img src="assets/demo_source.gif" alt="Source clip" width="320"> | <img src="assets/demo_vertical.gif" alt="Auto Vertical Reframe output" width="180"> |

Full-quality files: [assets/demo_source.mp4](assets/demo_source.mp4), [assets/demo_vertical.mp4](assets/demo_vertical.mp4).

## Overview

Vertical platforms (Reels, Shorts, TikTok) demand 9:16 video, but most source material is shot horizontally. Auto Vertical Reframe reads a video, detects subjects per scene, ranks candidate subjects using model signals, and drives a virtual camera (pan + zoom) through a smoothed path optimizer. It emits a ready-to-publish MP4 via ffmpeg.

Naive center-cropping loses the subject the moment they move. Manual reframing is tedious for long footage. Auto Vertical Reframe combines segmentation, face/pose cues, saliency, tracking continuity, and scene detection so each shot gets its own framing decision without relying on a static center crop.

## Features

- Per-scene subject selection via PySceneDetect (`AdaptiveDetector`).
- YOLOv11 instance segmentation with configurable classes and confidence.
- MediaPipe face detection and pose landmarks for framing cues.
- Two-person framing mode when a second subject crosses a spatial threshold.
- `handcrafted` saliency mode by default: fast and usually best for simple single-subject videos.
- Optional `deepgazemr` saliency mode: slower, experimental, useful to try on complex or ambiguous scenes.
- Automatic fallback from `deepgazemr` to `handcrafted` if model loading or inference fails.
- Saliency telemetry in logs and final summary: requested backend, active backend, model loaded state, fallback frames, and device.
- Subject lock, min/max zoom, max step-per-frame, and per-axis motion damping.
- Post-processing via ffmpeg: configurable encoder, CRF, audio bitrate, and optional unsharp/denoise pass.
- Debug preview export for inspecting crop decisions frame-by-frame.


Core logic lives in `src/verthor/auto_reframe.py` as a single pipeline with `Candidate`, `CameraObservation`, and `CameraState` dataclasses.

## Tech Stack

- **Language:** Python 3.11+
- **Detection & tracking:** Ultralytics YOLOv11, ByteTrack (`lap`)
- **Pose & face:** MediaPipe 0.10
- **Scene detection:** PySceneDetect
- **ML runtime:** PyTorch 2.2+
- **Encoding:** ffmpeg (external)

## Quick Start

Prerequisites: Python 3.11+ and `ffmpeg` in `PATH` (`brew install ffmpeg` on macOS).


On macOS you can instead double-click `run_verthor.command` — it provisions the venv and prompts for input, preset, debug preview, and saliency mode via native dialogs. If a non-video file is accidentally passed to the launcher, it opens the file picker again instead of trying to process it.


### 執行範例與參數解析

### Colab T4：裝置確認與即時診斷

YOLO/PyTorch 推論、MediaPipe 人臉／姿態分析、FFmpeg 編碼各自使用不同後端。
`--device auto` 在 CUDA 可用時明確選擇 `cuda:0`，否則記錄原因並使用 CPU。
`--device 0` 強制使用第一張 GPU；無法使用時直接報錯，不會悄悄改成 CPU。
第一幀推論後會輸出 `YOLO actual inference device=...`，結尾 Summary 會列出
實際模型、YOLO 裝置，以及要求／實際選用的編碼器。

先在與影片相同的工作目錄，使用目前執行的 Python 與程式做環境診斷：

```python
!python3 -u /usr/local/bin/auto_reframe.py --diagnose-env --device 0 --video-encoder hevc_nvenc --post-restore --native-debug --log-level DEBUG --ffmpeg-log-level info
```

此模式不讀取影片或下載模型。它列出程式路徑與 SHA256、Python、套件、GPU 資訊，
實測 CUDA 矩陣乘法及卷積，再使用指定編碼器與後處理濾鏡做兩幀測試。
如果要求 `hevc_nvenc` 卻只能回退 `libx264`，診斷會回傳失敗狀態。
一般影片處理仍保留原有的編碼器回退行為，並明確記錄失敗原因及實際編碼器。

接著用明確的模型名稱與原本參數，先跑 90 幀：

```python
!python3 -u /usr/local/bin/auto_reframe.py 'vv110.mp4' 'vv110V_test.mp4' --seg-model yolo26n-seg.pt --device 0 --lock-first-subject --dead-zone 0.15 --post-restore --video-encoder hevc_nvenc --max-frames 90 --native-debug --log-level DEBUG --ffmpeg-log-level info
```

`--max-frames` 只限制這次輸出的影片長度；正式處理時移除它並更換輸出檔名。
輸入檔名需完全一致，例如 `vv110.mp4` 與 `v110.mp4` 是不同檔案。
若曾把程式複製到 `/usr/local/bin/auto_reframe.py`，更新 Git checkout 後也要重新複製：

```python
import shutil
shutil.copy2('/content/auto-vertical-reframe/auto_reframe.py', '/usr/local/bin/auto_reframe.py')
```

| 訊息／參數 | 判讀或用途 |
| --- | --- |
| `cudart_stub.cc: Could not find cuda drivers...` | 可能是 MediaPipe 匯入時的 CUDA runtime 探測訊息；需和 PyTorch 實測、實際 YOLO 裝置分開判斷。`libcuda.so.1` 可載入不代表每個套件都能按名稱載入它需要的 `libcudart`。 |
| `Created TensorFlow Lite XNNPACK delegate for CPU` | MediaPipe 在此程式明確使用 CPU delegate，這是資訊訊息；不決定 YOLO 或 NVENC 的裝置。 |
| `NORM_RECT without IMAGE_DIMENSIONS` | 來自 MediaPipe 套件內部的 landmark projection graph。Tasks 的 Python `detect()` 沒有同名 `IMAGE_DIMENSIONS` 參數；不能憑這句警告判定外部裁切圖形狀有錯。原始警告保留，`DEBUG` 額外列出傳入 ROI 的尺寸。 |
| `--native-debug` | 在匯入原生套件前啟用 `dso_loader=2`，並印出採用的環境參數；不修改 CUDA 函式庫搜尋路徑。 |
| `--log-level DEBUG` | 列出完整 CLI 設定、ROI 尺寸、FFmpeg 子程序命令、PID 與退出狀態。 |
| `--ffmpeg-log-level info` | 提高 FFmpeg 詳細程度；原始 stderr 即時送到 terminal，同時保留有界的錯誤摘要供例外使用。 |

原生 MediaPipe／TensorFlow 警告不會被過濾；face/pose 初始化與推論例外會附 traceback 輸出。
`--native-debug` 提供更多載入證據，並不承諾消除套件內部警告。

### 固定鏡頭範例

如果您希望重構後的直式畫面**如同架在三腳架上的固定鏡頭**，鎖定開場主角且不隨意推拉變焦，可以直接在終端機執行以下指令：

```bash
python auto_reframe.py input.mp4 output_fixed.mp4 \
    --preset talking_head \
    --lock-first-subject \
    --fixed-zoom 1.10 \
    --dead-zone 0.08 \
    --pan-time 0.55 \
    --post-restore

```

也可以透過更新後的範例腳本直接執行測試：

```bash
python run_examples.py --recipe fixed_camera

```

#### 關鍵參數效果說明：

1. **`--lock-first-subject`（鎖定開場主體）**：
在開頭檢測到主要人物後立即鎖定其 `track_id`。後續即使背景有路人走過、或有其他人入鏡交談，鏡頭也絕不切換目標。
2. **`--fixed-zoom 1.10`（固定縮放倍率）**：
強制將虛擬攝影機鎖定在 $1.10\times$ 倍率，完全關閉動態數位變焦，杜絕主體稍微前傾或揮手時造成的「鏡頭拉風箱 / 呼吸效應（Zoom Pumping）」。
3. **`--dead-zone 0.08`（擴大防抖死區）**：
將中心死區從預設的 $0.06$ 放寬到 $0.08$（畫面寬度 $\pm 8\%$）。只要人物在該範圍內微幅晃動，攝影機位移速度直接歸零，保持絕對靜止。
4. **`--pan-time 0.55`（極柔和阻尼時間常數）**：
當主體真的產生大幅度位移走出死區時，攝影機不會突然暴衝，而是以極度絲滑的阻尼過渡（Ease-in-out）平移補位。


## Status

This implementation is derived from https://github.com/KazKozDev/auto-vertical-reframe

MIT — see [LICENSE](LICENSE)

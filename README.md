# auto-vertical-reframe
Scene-aware vertical auto-reframe CLI that turns horizontal footage into 9:16 video without losing the subject.


## Highlights

- Tracks people, pets, and vehicles through scene cuts with YOLO26 segmentation and ByteTrack.
- Head-, pose-, and saliency-aware framing with a subject ranking model and smoothed camera path.
- Four tuned presets (`talking_head`, `sports`, `pets`, `cars`) with sensible zoom and motion limits.
- Full-frame DeepGaze MSDB spatial saliency by default, with configurable MIT1003/uniform centre bias.

## Overview

Vertical platforms (Reels, Shorts, TikTok) demand 9:16 video, but most source material is shot horizontally. Auto Vertical Reframe reads a video, detects subjects per scene, ranks candidate subjects using model signals, and drives a virtual camera (pan + zoom) through a smoothed path optimizer. It emits a ready-to-publish MP4 via ffmpeg.

Naive center-cropping loses the subject the moment they move. Manual reframing is tedious for long footage. Auto Vertical Reframe combines segmentation, head/pose cues, saliency, tracking continuity, and scene detection so each shot gets its own framing decision without relying on a static center crop.

## Features

- Per-scene subject selection via PySceneDetect (`AdaptiveDetector`).
- YOLO26 instance segmentation with configurable classes and confidence.
- YOLO26n-pose COCO-17 keypoints with conservative head estimates and batched person ROIs.
- Two-person framing mode when a second subject crosses a spatial threshold.
- `deepgazemsdb` uses original-size frames, RGB 0–255 input and generalization parameters (`dataset=None`).
- Explicit `handcrafted` remains available; `auto` permits logged handcrafted fallback.
- Default MSDB failures stop with a traceback; no hidden CPU fallback or automatic downscaling.
- Saliency telemetry in logs and final summary: requested backend, active backend, model loaded state, fallback frames, and device.
- Subject lock, min/max zoom, max step-per-frame, and per-axis motion damping.
- Post-processing via ffmpeg: configurable encoder, CRF, audio bitrate, and optional unsharp/denoise pass.
- Debug preview export for inspecting crop decisions frame-by-frame.


`auto_reframe.py` is a thin compatible entry point. The installed `reframe` package
separates perception, saliency, composition, and video I/O; see [Architecture](#architecture).

## Tech Stack

- **Language:** Python 3.11+
- **Detection & tracking:** Ultralytics YOLO26, ByteTrack (`lap`)
- **Pose & head cues:** YOLO26n-pose (`yolo26n-pose.pt`)
- **Scene detection:** PySceneDetect
- **ML runtime:** PyTorch 2.6+ (CUDA, CPU or MPS)
- **Encoding:** ffmpeg (external)

## Quick Start

Python 3.11+ and an external `ffmpeg` executable are required. Install Python packages
with the interpreter that will run the application:

```bash
python3 -m pip install -e .
auto_reframe.py input.mp4 output.mp4 --device auto
```

`requirements.txt` pins Ultralytics to the version observed in the working Colab
runtime and lists the direct inference/tracking dependencies. It uses one OpenCV
provider (`opencv-python`); installing headless/contrib providers alongside it can
overwrite the same `cv2` files. PyTorch/torchvision requirements accept compatible
CUDA builds already present in Colab. FFmpeg and NVIDIA drivers are not pip packages.
A fresh GPU environment needs a matching CUDA-enabled PyTorch/torchvision installation.

### Colab：從 MediaPipe 遷移

先將 `/content/auto-vertical-reframe` 切換到要使用的版本，再執行：

```python
!python3 -u /content/auto-vertical-reframe/install_colab.py
```

安裝程式會：

1. 使用同一個 Python 卸載 `mediapipe`，移除本程式曾下載的兩個舊模型快取。
2. 清除四種可能重疊的 OpenCV distributions，再依 `requirements.txt` 安裝單一 provider。
3. 使用暫存 constraints 保留現有 torch／torchvision 的完整版本（含 `+cu130` 等標記）；
   若依賴不相容會停止並顯示錯誤。
4. 在新 Python 程序驗證套件可匯入、MediaPipe 已不可匯入，印出版本與 CUDA 狀態。
5. 用 editable 模式安裝完整 `reframe` 套件與 `auto_reframe.py` 命令。
   Colab 會在 `/usr/local/bin` 產生命令入口，影片可以放在其他目錄。

拆分後不能再只 `cp auto_reframe.py /usr/local/bin/`。請保留 checkout；
editable 安裝會直接使用其中的模組，後續 `git pull` 後的新程序即可使用新程式。
如 `requirements.txt` 或套件設定有變更，需重新執行安裝程式。
也可在 checkout 使用 `python3 auto_reframe.py`，或安裝後使用 `python3 -m reframe`。

每個 pip 子程序都保留 stdout、stderr，失敗時安裝會停止。
不移除其他程式共用的 TensorFlow／protobuf 等依賴。
若先前 notebook cell 匯入過相關套件，安裝後重新啟動 Colab session，
再掛載 Drive／回到影片目錄；不要重跑含有 `mediapipe` 的舊安裝 cell。

### YOLO26n-pose 的構圖方式

- `yolo26n-seg.pt` 持續負責分割、ByteTrack ID 與原本的非人物類別。
- `yolo26n-pose.pt` 每次處理一批 BGR 人物 ROI，由同一份 17 點結果產生身體與頭部線索。
- 使用分割人物框匹配姿態結果。即使設定 `--cue-top-k`，配對仍會考慮所有人物；
  重疊且無法可靠歸屬的姿態不會套用到目標人物。
- 頭部框由鼻、眼、耳等可信點估算，不是獨立人臉偵測。點位不足時使用人物框／遮罩構圖。
- COCO 沒有嘴角、腳跟或腳尖；移除未使用的 `chin_y`，並保留分割框下緣以保護腳部。
- 場景切換清除快取；人物快速移動或重疊會重新推論。`--cue-interval 0` 可逐幀更新。
- 模型格式錯誤、推論例外或指定 GPU 不可用時，輸出 traceback 並失敗退出。
  單幀沒有可信關鍵點是正常結果，會使用已有的分割構圖資訊。

| 參數 | 預設／用途 |
| --- | --- |
| `--pose-model` | `yolo26n-pose.pt`；也接受本機 COCO-17、person 類別的 `.pt` 模型 |
| `--pose-imgsz` | `640`；32 的正整數倍 |
| `--pose-conf` | `0.25`；姿態人物框的偵測門檻 |
| `--keypoint-conf` | `0.35`；個別關鍵點的可信度門檻 |
| `--pose-batch-size` | `4`；每批最多處理的人物 ROI 數 |
| `--cue-interval` | `0.2` 秒；每個追蹤 ID 的姿態快取更新間隔 |
| `--device` | `auto`；分割與姿態共用。`0` 強制第一張 CUDA GPU，`cpu` 使用 CPU |

舊 `--face-model` 已移除；`.task`／`.tflite` 不再是可接受的姿態模型。
官方模型首次使用時自動下載；離線執行請預先準備兩個 `.pt` 權重。
姿態模型只在允許的類別包含 `person` 時載入。

### 第二階段：DeepGaze MSDB

MSDB 已取代 MR 並成為預設顯著性後端。預設使用 **MIT1003 center bias**，
模型資料集參數仍為 **`dataset=None`**（各資料集平均參數）；這兩項設定彼此獨立。
授權狀態記錄為 **尚未確認**，詳見 [第三方模型說明](THIRD_PARTY_MODELS.md)。
模型授權獨立於本專案；不將未確認的授權標示為 MIT 或可商用。

| 參數 | 預設／用途 |
| --- | --- |
| `--saliency-model` | `deepgazemsdb`；另可選 `handcrafted`、允許回退的 `auto` |
| `--saliency-center-bias` | `mit1003`；可選 `uniform`，uniform 不下載 MIT1003 範本 |
| `--saliency-screen-inches` | `24`；16:9 螢幕旋轉為 9:16 後的對角線尺寸 |
| `--saliency-viewing-distance-cm` | `60` |
| `--saliency-pixel-per-dva` | 預設由螢幕與裁切比例計算；指定時表示**原始輸入影格**每度視角的像素數 |
| `--saliency-device` | `auto`；指定 `cuda` 而 CUDA 不可用會報錯 |
| `--saliency-amp` | 預設關閉，FP32 基準；可明確選擇 CUDA FP16 |
| `--saliency-interval` | 保留既有 `3`；以 `1` 逐幀推論 |
| `--saliency-max-side` | 僅供 handcrafted 使用；對 MSDB 不做縮圖 |

MSDB 收到原始大小的完整 BGR 影格，轉成 RGB、float32，數值維持 0–255；
不先縮圖、不除以 255。內建 CLIP/DINOv2、多尺度處理和模型正規化保持上游行為。
MSDB 是單張影像模型，第一幀即可推論；舊 MR 的 16 幀暖機、緩衝區和 Hub loader 已移除。
`--saliency-model deepgazemr` 與 `--saliency-trust-repo` 不再接受。

觀看尺度假設影片完整呈現在直立螢幕中，依比例縮放、不拉伸；1080×1920 會填滿 9:16 螢幕。
以畫面中央的一度視角計算，24 吋／60 公分約為 **37.8435 輸出像素／度**。
輸入模型的數值為：

```text
input_pixel_per_dva = output_pixel_per_dva × view_crop_height / output_height
```

Colab log 中的原始影片為 **3840×2160、60 fps**；保留完整 2160 高度時，約為
**42.5739 原始像素／度**。若來源為 1920×1080、保留 1080 高度，則約為 21.2869。
裁切高度降至 540 時，約為 10.6435。這是採用指定播放條件的工程近似，尚未用眼動資料校準。
顯著性計算先於本幀構圖，因此使用**更新構圖前的鏡頭裁切高度**；固定 zoom 則使用指定倍率，
場景切換使用重設後的鏡頭狀態。模型的完整影格還包括最終裁切之外的區域。
`--saliency-pixel-per-dva` 可直接覆寫此近似值，且不隨 zoom 變動。

首次執行會下載 CLIP、DINOv2、MSDB head 與 MIT1003 範本；`requirements.txt`
固定 DeepGaze、OpenAI CLIP 的來源 commit，DINO 也使用固定 commit。
安裝完成不代表權重已下載；離線使用前須完成一次模型載入。

完整影格與觀看尺度會影響 MSDB 內部多尺度的大小，**尚未保證能放入 T4 顯存**。
發生 CUDA OOM、載入失敗或非有限預測時，預設會保留 traceback 並失敗退出，
不偷偷縮圖、不切 CPU、不改用其他模型。只有指定 `--saliency-model auto` 才會記錄並回退
到 handcrafted；不要把這種輸出當成 MSDB 成功測試。

先依下方同步步驟取得 `codex/modular-saliency-stage1` 分支的 **0.3.0** 版本，
重新執行 `install_colab.py`。以下 cell 先測 3 幀，再測 90 幀；前者失敗時會停止，
不執行長測試。保留第一階段的 interval=3，便於同條件比較；不縮小來源影格。

```bash
%%bash
set -euo pipefail
cd /content/drive/MyDrive/video/1080p/crop

for frames in 3 90; do
  if [ "$frames" -eq 3 ]; then label=smoke; else label=test; fi
  log="/content/reframe-msdb-${label}.log"
  {
    git -C /content/auto-vertical-reframe rev-parse HEAD
    nvidia-smi
    ffprobe -v error -select_streams v:0 \
      -show_entries stream=width,height,avg_frame_rate,duration -of json vv110.mp4
    python3 -u -m reframe \
      vv110.mp4 "vv110V_msdb_${label}.mp4" \
      --seg-model yolo26n-seg.pt --pose-model yolo26n-pose.pt --device 0 \
      --lock-first-subject --dead-zone 0.15 --post-restore \
      --video-encoder hevc_nvenc --max-frames "$frames" \
      --saliency-model deepgazemsdb --saliency-device cuda \
      --saliency-center-bias mit1003 \
      --saliency-screen-inches 24 --saliency-viewing-distance-cm 60 \
      --saliency-interval 3 --no-saliency-amp \
      --native-debug --log-level DEBUG --ffmpeg-log-level info
    ffprobe -v error -count_frames \
      -show_entries stream=index,codec_type,codec_name,width,height,avg_frame_rate,nb_read_frames,duration \
      -of json "vv110V_msdb_${label}.mp4"
  } 2>&1 | tee "$log"
done
```

比較 uniform 時，只改 `--saliency-center-bias uniform`，並更換影片／log 檔名。
檢查 `MSDB actual input` 是否為原始解析度，以及 Summary 的
`saliency_active_backend=deepgazemsdb`、`saliency_frames_fallback=0`。
Summary 同時列出全程 `wall_seconds`、`processing_fps`、模型載入時間、累計推論時間，
以及 CUDA allocator 的程序峰值（包含同程序 YOLO，並非整張 GPU 的使用量）。
推論時間包含張量搬移及輸出同步，載入／下載另計；第一次推論也包含在統計中。
比較速度時須固定影片、zoom、精度與 interval，並區分首次下載與快取已存在的執行。
本階段不新增姿態驅動的跳幀或其他速度優化。

### Colab T4：診斷與短片驗證

不讀取影片／下載模型的環境診斷：

```python
!python3 -u /usr/local/bin/auto_reframe.py --diagnose-env --device 0 --video-encoder hevc_nvenc --post-restore --native-debug --log-level DEBUG --ffmpeg-log-level info
```

此模式列出 Python、套件與 GPU，執行 CUDA 矩陣乘法、卷積與兩幀 NVENC 測試。
它不代表模型已實測。以下 cell 明確選擇 handcrafted，重現第一階段基準並測試 90 幀，同步保留 terminal 輸出、
日誌和失敗狀態；請在影片所在目錄執行：

```bash
%%bash
set -euo pipefail
python3 -u /usr/local/bin/auto_reframe.py \
  'vv110.mp4' 'vv110V_pose_test.mp4' --saliency-model handcrafted \
  --seg-model yolo26n-seg.pt --pose-model yolo26n-pose.pt --device 0 \
  --lock-first-subject --dead-zone 0.15 --post-restore \
  --video-encoder hevc_nvenc --max-frames 90 \
  --native-debug --log-level DEBUG --ffmpeg-log-level info \
  2>&1 | tee /content/reframe-yolo-pose-test.log
```

確認兩行 `YOLO segmentation actual inference device=cuda:0` 與
`YOLO pose actual inference device=cuda:0`，以及最終 Summary 的
`video_encoder_actual=hevc_nvenc`。如果片段未偵測到人物，pose 不會推論，
`pose_device` 為 null；這不能視為 pose GPU 測試成功。
Summary 另外列出 `frames_with_head_cues`、`frames_with_pose`、
`pose_rois_inferred`、`pose_rois_matched`、`pose_cache_hits`。
`frames_with_head_cues` 不可直接當作舊版的「人臉偵測率」。

### 草稿 PR #29：Colab 同步與安裝

此分支整合第一階段模組拆分與第二階段 MSDB 更新。完整套件版本為 0.3.0；
不能只複製入口 script。第一階段的實測結果見 [COLAB_VALIDATION.md](COLAB_VALIDATION.md)。
請先確認 checkout 沒有尚未保存的修改；
以下命令會在有衝突時停止，不會使用強制覆寫：

```bash
%%bash
set -euo pipefail
{
  cd /content/auto-vertical-reframe
  git fetch origin refs/heads/codex/modular-saliency-stage1:refs/remotes/origin/codex/modular-saliency-stage1
  git switch codex/modular-saliency-stage1 || git switch --track origin/codex/modular-saliency-stage1
  git merge --ff-only origin/codex/modular-saliency-stage1
  git rev-parse HEAD
  python3 -u install_colab.py
} 2>&1 | tee /content/reframe-msdb-install.log
```

更新後請執行上方 MSDB 測試 cell。若要重現第一階段基準，在影片目錄執行：

```bash
%%bash
set -euo pipefail
python3 -u /usr/local/bin/auto_reframe.py \
  'vv110.mp4' 'vv110V_modular_test.mp4' --saliency-model handcrafted \
  --seg-model yolo26n-seg.pt --pose-model yolo26n-pose.pt --device 0 \
  --lock-first-subject --dead-zone 0.15 --post-restore \
  --video-encoder hevc_nvenc --max-frames 90 \
  --native-debug --log-level DEBUG --ffmpeg-log-level info \
  2>&1 | tee /content/reframe-modular-test.log
```

日誌會列出套件路徑、版本、所有 Python 模組 SHA256 與整體指紋，
可辨識 editable checkout 實際使用的程式。除了 GPU 與編碼器，也請確認畫面構圖、
音畫同步、scene resets、subject switches 與原版比較。實際 T4／權重測試需在 Colab 執行。

## Architecture

| Module | Responsibility |
| --- | --- |
| `reframe/cli.py`, `config.py` | Native diagnostics before heavy imports; CLI parsing into typed `AppConfig`, defaults and validation |
| `contracts.py`, `geometry.py` | Frame/track/pose/saliency data contracts, camera state and shared geometry |
| `perception/segmentation.py`, `pose.py` | Segmentation/tracking, compact mask statistics, batched pose inference and cue cache |
| `saliency/base.py`, `factory.py`, `backends/` | One backend lifecycle: `load`, `observe`, `predict`, `reset`, `close`, `telemetry` |
| `saliency/service.py`, `scheduler.py`, `cache.py`, `regions.py` | Every-frame ingestion, fixed refresh policy, optical-flow/EMA map alignment, saliency regions |
| `subjects.py`, `camera.py` | Candidate enrichment/ranking, subject selection, framing and smoothed camera motion |
| `scenes.py`, `video_io.py`, `debug.py`, `runtime.py` | Scene cuts, decoding/encoding and live stderr, overlay, devices and diagnostics |
| `pipeline.py` | Resource ownership and single-pass orchestration |

Each frame runs segmentation, pose observation, saliency, candidate ranking, then
camera composition and encoding. Ranking consumes observations and never calls a
model. `argparse.Namespace` stays out of the processing layers.

`FrameContext` uses one-based frame indices, `(index - 1) / fps` nominal timestamps,
original width/height and scene index. `PoseObservation` retains all 17 `(x,y,confidence)`
points on CPU, the original inference points/context, remapped current points/context,
track ID and `inferred`/`remapped` source. Missing poses are recorded too, so cached
misses do not look like a new inference. Remapping never advances the inference time.

Saliency backends receive BGR uint8 images (original resolution for MSDB, reduced
resolution for handcrafted) and return finite 2-D float32
maps in `[0,1]`, spanning the whole input image. The service retains original geometry
and inference time separately from the frame to which flow aligns the map. `source`
distinguishes refresh, EMA and propagation; backend/status identify the last refresh
and report an explicit `auto` fallback truthfully. EMA may include older map content.

The scheduler still refreshes on scene frames 1, 4, 7, ... by default. Every frame
reaches `observe`, including skipped predictions. MSDB is spatial and needs no
16-frame warmup or temporal ring. A scene cut clears tracking/pose/map/temporal state. `ExitStack` closes models,
saliency state, frame decoding and encoder processes on success, error or interruption.

Phase 2 replaces the MR backend with MSDB. ViNet and SAM2 are excluded.
Pose-driven adaptive scheduling and speed optimizations remain future work.

### Local regression checks

The local phase-2 results and their limits are recorded in [MSDB_VALIDATION.md](MSDB_VALIDATION.md).

After installing dependencies, run:

```bash
python3 -m unittest discover -s tests -v
python3 -m unittest test_security test_security_reframe -v
python3 -c 'import test_torch_load; test_torch_load.test_torch_load_weights_only()'
```

Tests exercise pose geometry/association, raw-keypoint freshness, temporal ingestion,
saliency map geometry/flow/EMA, backend warmup/fallback, scene resets, cleanup on
exceptions/interrupts, CLI bootstrap order and live FFmpeg stderr. Video smoke tests
use actual FFmpeg with controlled model outputs. Model weights and CUDA are not
required for these checks; saliency numeric tests require real OpenCV. They do not
replace the Colab T4/YOLO/NVENC or real DeepGaze MSDB validation.

`--native-debug` 在匯入 PyTorch 前啟用 `TORCH_SHOW_CPP_STACKTRACES=1` 與
Python faulthandler，並印出採用的設定及 PyTorch build 資訊。
`DEBUG` 顯示 ROI 尺寸、批次參數、FFmpeg 命令／PID／退出碼；
所有原生錯誤與 FFmpeg stderr 都保留，不以降低 log level 隱藏問題。
指定編碼器無法初始化時，一般處理會記錄並回退，`--diagnose-env` 則回報失敗。

確認短片構圖與音畫同步後，移除 `--max-frames 90` 並更換輸出檔名處理完整影片。
以相同影片比較速度與峰值顯存；姿態與分割共用 GPU，不保證整體處理速度一定提高。

官方介面參考：[Pose](https://docs.ultralytics.com/tasks/pose/)、
[NumPy/BGR 輸入](https://docs.ultralytics.com/modes/predict/)。

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

Project code: MIT — see [LICENSE](LICENSE). Third-party model terms are separate;
DeepGaze MSDB licensing remains unconfirmed — see [THIRD_PARTY_MODELS.md](THIRD_PARTY_MODELS.md).

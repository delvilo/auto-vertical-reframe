# auto-vertical-reframe
Scene-aware vertical auto-reframe CLI that turns horizontal footage into 9:16 video without losing the subject.


## Highlights

- Tracks people, pets, and vehicles through scene cuts with YOLO26 segmentation and ByteTrack.
- Head-, pose-, and saliency-aware framing with a subject ranking model and smoothed camera path.
- Four tuned presets (`talking_head`, `sports`, `pets`, `cars`) with sensible zoom and motion limits.
- Automatic three-tier composition: reliable pose, lightweight saliency, then on-demand DeepGaze MR.
- Optional half-width inference region (`--precrop middle|left|right`) with full-resolution source output.

## Overview

Vertical platforms (Reels, Shorts, TikTok) demand 9:16 video, but most source material is shot horizontally. Auto Vertical Reframe reads a video, detects subjects per scene, ranks candidate subjects using model signals, and drives a virtual camera (pan + zoom) through a smoothed path optimizer. It emits a ready-to-publish MP4 via ffmpeg.

Naive center-cropping loses the subject the moment they move. Manual reframing is tedious for long footage. Auto Vertical Reframe combines segmentation, head/pose cues, saliency, tracking continuity, and scene detection so each shot gets its own framing decision without relying on a static center crop.

## Features

- Per-scene subject selection via PySceneDetect (`AdaptiveDetector`).
- YOLO26 instance segmentation with configurable classes and confidence.
- Adaptive segmentation intervals with sparse optical-flow correction between YOLO calls; output keeps every source frame.
- YOLO26n-pose COCO-17 keypoints with conservative head estimates and batched person ROIs.
- Two-person framing mode when a second subject crosses a spatial threshold.
- Stable subjects compose directly from pose and segmentation; uncertain scenes use lightweight saliency before DeepGaze MR.
- DeepGaze MR loads only its feature checkpoint when needed and a full 16-frame window is available; warmup and failures visibly use handcrafted output.
- Optional staged TorchCodec decoding and tensor preprocessing, with explicit CPU fallback and transfer diagnostics.
- Tier decisions, actual neural calls, cache age and backend/device state are reported in the final summary.
- Subject lock, min/max zoom, max step-per-frame, and per-axis motion damping.
- Post-processing via ffmpeg: configurable encoder, CRF, audio bitrate, and optional unsharp/denoise pass.
- Debug preview export for inspecting crop decisions frame-by-frame.


`auto_reframe.py` is a thin compatible entry point. The installed `reframe` package
separates perception, saliency, composition, and video I/O; see [Architecture](#architecture).

## Tech Stack

- **Language:** Python 3.13+
- **Detection & tracking:** Ultralytics YOLO26, ByteTrack (`lap`)
- **Pose & head cues:** YOLO26n-pose (`yolo26n-pose.pt`)
- **Scene detection:** PySceneDetect
- **ML runtime:** PyTorch 2.6+ (CUDA, CPU or MPS)
- **Encoding:** ffmpeg (external)

## Quick Start

Python 3.13+ and an external `ffmpeg` executable are required. Install Python packages
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

後續開發以 **Python 3.13+** 為相容基準，套件 metadata 也設定 `requires-python >=3.13`。
不要求支援 Python 3.12 或更舊版本；後續套件、CUDA build 與新 Python 版本仍需實測。
這項基準記錄在 [AGENTS.md](AGENTS.md)，供以後的程式改動遵循。

### TorchCodec：分兩階段啟用

預設仍使用 OpenCV。TorchCodec 路徑需另裝與 PyTorch／CUDA 相符的 wheel，
並提供 FFmpeg shared libraries；目前本機驗證為 Python 3.13.5、PyTorch 2.11.0+cpu、
TorchCodec 0.17.0+cpu。T4／NVDEC 的正確性、速度與記憶體峰值須另行實測。

```bash
# 第一階段：NVDEC 解碼，模型與輸出維持 NumPy reference 路徑
auto_reframe.py input.mp4 stage1.mp4 --device 0 --saliency-device cuda \
  --video-decoder torchcodec --decode-device cuda:0 --decode-batch-size 4
# 第二階段：另啟用 GPU precrop／YOLO 前處理、DeepGaze 小圖窗口、裁切與縮放
auto_reframe.py input.mp4 stage2.mp4 --device 0 --saliency-device cuda \
  --video-decoder torchcodec --decode-device cuda:0 --decode-batch-size 4 \
  --frame-processing tensor
```

兩階段均保留 FFmpeg／NVENC 音訊合成與原始幀數、FPS；`precrop` 仍只限制推論區域。
第二階段目前支援 PyTorch `.pt` YOLO 模型及 ByteTrack。CPU 僅取得分析小圖、追蹤資訊及最終輸出影像；
`--save-debug-preview` 會額外下載完整原圖。GPU 色彩轉換與插值可能改變部分模型結果，需比較視覺焦點。
變動幀率來源會明確拒絕，避免套用固定 FPS 後改變播放速度。

請使用 [T4 測試規格](docs/t4-validation.md) 先短測再測長片；
[AI agent 交接文件](docs/agent-handoff.md) 記錄目前實作、測試入口及下一步。
已完成本機驗證與歷史量測保留在 [驗證報表](docs/reports/implementation-validation-2026-10-12.md)，
新 GPU 路徑仍須依規格集中驗證。

方案 B 的主角排程、有限姿態快取與追蹤 profile 已接入，設定與相容遷移見
[交接文件](docs/agent-handoff.md)。離線雙向插值 C 尚未實作。
`--dead-zone` 所有 preset 與測試收集器已預設為 `0`；可自行指定非零值。

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
- 追蹤穩定時優先更新已選定主角；首次選角、主角遺失或人物重疊時擴大姿態分析。
  雙人構圖會保留第二人的姿態需求。更新後仍使用所有分割框檢查歸屬。
- 使用分割人物框匹配姿態結果。即使設定 `--cue-top-k`，配對仍會考慮所有人物；
  重疊且無法可靠歸屬的姿態不會套用到目標人物。
- 頭部框由鼻、眼、耳等可信點估算，不是獨立人臉偵測。點位不足時使用人物框／遮罩構圖。
- COCO 沒有嘴角、腳跟或腳尖；移除未使用的 `chin_y`，並保留分割框下緣以保護腳部。
- 場景切換清除快取；人物快速移動或重疊會重新推論。`--cue-interval 0` 可逐幀更新。
- 背面／遮臉不等於 Pose 無效：至少 3 個幾何合理且信心不低於 `max(0.5, keypoint_conf)` 的肩髖點
  可提供軀幹證據。手腕／手肘信心不足不直接否決軀幹；可靠的大幅肢體動作仍會阻止延長快取。
- 模型格式錯誤、推論例外或指定 GPU 不可用時，輸出 traceback 並失敗退出。
  單幀沒有可信關鍵點是正常結果，會使用已有的分割構圖資訊。

| 參數 | 預設／用途 |
| --- | --- |
| `--pose-model` | `yolo26n-pose.pt`；也接受本機 COCO-17、person 類別的 `.pt` 模型 |
| `--pose-imgsz` | `640`；32 的正整數倍 |
| `--pose-conf` | `0.25`；姿態人物框的偵測門檻 |
| `--keypoint-conf` | `0.35`；個別關鍵點的可信度門檻 |
| `--pose-batch-size` | `4`；每批最多處理的人物 ROI 數 |
| `--cue-interval` | balanced 預設 `0.2` 秒；軟更新時間，穩定且有獨立證據時才可延後 |
| `--pose-max-age` | balanced 預設 `0.35` 秒；原始 Pose 推論的硬期限，`0` 禁止跨幀沿用 |
| `--device` | `auto`；分割與姿態共用。`0` 強制第一張 CUDA GPU，`cpu` 使用 CPU |

舊 `--face-model` 已移除；`.task`／`.tflite` 不再是可接受的姿態模型。
官方模型首次使用時自動下載；離線執行請預先準備兩個 `.pt` 權重。
姿態模型只在允許的類別包含 `person` 時載入。

### 動態分割間隔與光流校正

預設 `--tracking-profile balanced`，在可信的真實偵測連續穩定後，將 YOLO 分割間隔由
**1 → 2 → 3 → 4 → 6 幀**逐步延長。超過 3 幀額外要求兩次相容的真實 Pose、
近期穩定的真實遮罩摘要及可靠主角追蹤；仍受來源 FPS 與 0.1 秒量測期限限制。
中間影格以運動預測與低解析度稀疏 KLT 光流校正人物位置；預測框與重映射的姿態
不會重新計算「最後一次真實偵測」時間，也不能靠互相引用升高穩定度。
切鏡、追蹤遺失、光流不可靠、人物交錯或畫面變化等事件會要求提前刷新。
每次跳過前都會檢查偵測年齡，達到 `--seg-max-age` 時必須重新偵測。

| 參數 | 預設／用途 |
| --- | --- |
| `--tracking-profile` | `balanced`；也可選 conservative／fast，顯式進階參數優先 |
| `--seg-max-gap` | balanced／fast 為 `6`，conservative 為 `3`；來源幀數上限。`1` 關閉跳幀，提供逐幀比較基準 |
| `--seg-max-age` | `0.1`；距上次真實分割的影片時間上限（秒），必須是有限正數；與幀數上限一起生效 |
| `--seg-thumbnail-mode` | `lazy`；確定需要畫面判斷或光流時才建立縮圖。`eager` 保留原先每幀縮圖方式，供同版本對照 |
| `--seg-thumbnail-method` | `gray-area`；先轉灰階，再以 `INTER_AREA` 縮小單通道影像。`bgr-area` 保留先縮小 BGR 再轉灰階的方式，供對照 |
| `--seg-skip-policy` | `primary`；符合條件的遠處低風險次要人物可降低優先度。`all` 保留所有追蹤目標皆須穩定的條件 |

60 FPS 下 6 幀為 0.1 秒，30 FPS 下預設期限將間隔限制在 3 幀；開場及不穩定片段會提前偵測。
balanced 穩定至少 0.5 秒才評估 4 幀、1 秒才評估 6 幀；fast 可於 0.5 秒評估 6 幀。
conservative 保留最高 3 幀與較短 Pose 快取，仍使用本版共同可靠度及風險保護，不等同舊版程式。
跳幀僅支援已釘選版本的原生 `BYTETracker`；其他 tracker 會回到逐幀偵測並記錄原因。
ByteTrack 的預測時鐘會逐個來源影格前進，跳過 YOLO 不等於偵測到空畫面。

**輸出仍保留所有處理範圍內的原始影格、FPS 與音訊時序。** 切鏡檢查、姿態快取時效判斷、
Saliency 三級策略、DeepGaze 的 16 幀來源時間窗口、相機平滑及編碼仍逐幀執行。
`--precrop` 的推論範圍與原圖輸出定義不變；`--post-restore` 仍預設關閉。
光流也有成本，追蹤變差還可能增加 Pose／DeepGaze 呼叫，因此要比較整體耗時與構圖品質，
不能將省下的 YOLO 呼叫比例當成整體加速倍率。

`lazy` 先檢查既有的「沒有追蹤目標、已要求刷新、不支援的 tracker、追蹤未穩定」條件；
已確定要跑 YOLO 的影格可延後縮圖。後續畫面變化、信心、偵測年齡與間隔判斷維持原順序。
需要畫面比較或光流時，才補建恰好相鄰的 N−1 與 N 幀縮圖；已建立的縮圖直接重用，
每幀最多建立一次，切鏡、重置及結束會清除快取。長邊仍為 320，預設先轉灰階，
再以 `INTER_AREA` 縮小單通道影像；這會改變整數捨入位置，需重新比較判斷與畫面品質，
不保證與 `bgr-area` 逐像素相同。YOLO、Pose 與 DeepGaze 的模型輸入及精度設定不變。
此方式保留上一幀原始影像的唯讀參照；半幅 precrop 是 view，可能暫時持有整張 4K BGR 影像，
約增加 25 MB 主機記憶體。實際節省張數需扣除補建上一幀縮圖的次數。

`primary` 依實際候選間隔的運動範圍、3σ 不確定性及主角安全距離檢查次要人物。
balanced／fast 可放行經多次真實量測、面積小且遠離主角的旁人；沒有通過光流的旁人
只提供明確低可信的預測，不冒充可靠構圖證據。新人物、未確認 ID、靠近／交錯或面積明顯者仍需評估。
整個推論區域保留影像變化檢查，真實 YOLO 量測期限保留全區重新掃描。
`seg_gate_diagnostics` 是門檻通過／失敗的計數，不等於實際跳幀數。

Pose 延長快取需兩次相容真實姿態、近期真實遮罩連續、身分穩定及無鄰近／搶鏡人物；
預測遮罩只保留原始量測時間。`tracking_settings` 記錄展開後設定；
收集器每五秒 Segment 另有量測年齡分布、Pose 延長／拒絕原因與顯著性風險升級計數。
各 profile 數值、舊旗標遷移與 T4 待測入口見 [交接文件](docs/agent-handoff.md)。

### 方案 C：三級自動構圖與按需 DeepGaze MR

`--saliency-model` 已移除；舊指令帶有此參數時會報錯，請刪除整組參數和值。
每幀依追蹤、姿態的原始推論時間、畫面變化及顯著圖快取品質選擇處理層級：

| 層級 | 判斷與處理 |
| --- | --- |
| L1 `pose_only` | 主角 ID、人物框及畫面連續穩定，且有新鮮可靠的身體關鍵點，直接以姿態、分割框與遮罩構圖；不執行顯著性模型或舊顯著圖光流 |
| L2 `handcrafted` | 姿態部分缺失或場景略有變化，先使用低解析度頻譜殘差與運動差分；可靠快取可短期沿用 |
| L3 `deepgaze` | L2 仍無法解除主體歧義、追蹤遺失或非人物焦點等不確定性時，才要求 DeepGaze MR 推論 |

可靠姿態不要求每幀都看見臉；肩、髖等身體關鍵點足夠時即可支持 L1。
快取姿態的重映射不會更新其原始推論時間；過期時先重新取得姿態再判斷層級。
連續穩定後才降低層級，明顯變化會立即重新評估。L1 不會因為 DeepGaze 快取過期而
強制執行神經模型，也不會把過期顯著圖加入主角評分。

每幀的低解析度 uint8 影像都加入 **16 幀 ring**，包含 L1 與 L2；
reference 路徑使用 CPU BGR，tensor 路徑保留解碼裝置上的小圖 RGB，不保留整批原始影片。
只有 L3 且時序窗口完整時才載入 DeepGaze 模型、上傳必要影格至裝置並執行推論。
載入後保留模型，後續只上傳新增影格；切鏡會清除舊時序與顯著圖，重新收集 16 幀。
窗口不足時使用 L2 暖機；模型載入或推論失敗會保留 warning／traceback 並回退。
首次 L3 的模型下載／載入延遲仍可能明顯，應與暖機後耗時分開比較。
模型直接 strict 載入既有 DeepGaze checkpoint（約 76.6 MiB，已含 VGG 特徵層）與 center bias（約 7 MiB），
不建立 VGG classifier、不另下載約 548 MiB 的完整 VGG19，也不執行遠端 Hub Python。
權重來源固定 commit 並驗證 SHA-256；已存在的相同快取會重用。這主要降低冷啟動成本，
不代表每次 DeepGaze forward 會變快。

| 設定 | 行為 |
| --- | --- |
| `--saliency-trust-repo` | 舊命令相容選項；新載入器不執行 Hub 程式碼，不需此參數 |
| `--saliency-device` | `auto`；Colab 可指定 `cuda`，實際裝置以 Summary 為準 |
| `--saliency-max-side` | `384`；限制顯著性縮圖長邊，保留既有縮放方式 |
| `--saliency-interval` | `3`；需要顯著性時的基本更新間隔，實際呼叫由三級策略及快取決定 |
| `--saliency-amp` | 保留既有預設開啟；可用 `--no-saliency-amp` 比較 FP32 |
| `--post-restore` | **預設關閉**，省去 FFmpeg 降噪與銳化；仍保留手動開啟選項 |

模型未載入、神經推論次數為 0，可能代表姿態與 L2 已足夠，不是執行失敗。
需要驗證 L3 時，應選擇遮擋、多人交錯或無可靠人物姿態的片段，確認
`saliency_model_loaded=true`、`saliency_frames_backend>0` 及實際裝置；
最後一幀可能已回到 L1/L2，因此不要要求最後的 active backend 必須是 DeepGaze。

### `--precrop`：限定推論區域

保留完整高度、只取一半寬度，約為原畫面 50% 面積；奇數寬度向下取整至整數像素。

| 選項 | 水平方向範圍 | 3840×2160 來源的推論區域 |
| --- | --- | --- |
| 未指定 | 0%–100% | x=0–3840，高度 2160 |
| `--precrop left` | 0%–50% | x=0–1920，高度 2160 |
| `--precrop middle` | 25%–75% | x=960–2880，高度 2160 |
| `--precrop right` | 50%–100% | x=1920–3840，高度 2160 |

分割／追蹤、姿態及顯著性僅分析所選區域；結果換算回原圖座標，輸出仍從原始影格
裁切，輸出裁切框可以延伸到推論區域外。區域外的主體不會被偵測；主角離開時不會
自動擴大為全畫面搜尋。沒有主體時以所選區域作為初始／備援構圖範圍。

本次沒有啟用固定像素預算或 YOLO 半精度。分割偵測現在使用上述動態間隔；
半幅區域仍採既有模型縮放規則，
不保證總運算量減半；直向區域在固定長邊下甚至可能增加 DeepGaze 輸入像素。
實際收益需比較相同影片的總耗時、神經推論呼叫數、峰值顯存及構圖品質。

### 測試與 AI agent 交接

- [AI agent 交接文件](docs/agent-handoff.md)：使用者決定、實作狀態、相容參數與接手順序。
- [T4 測試規格](docs/t4-validation.md)：環境檢查、可複製命令、各階段驗收、續作與結果保存。
- [目前實作／CPU 驗證報表](docs/reports/implementation-validation-2026-10-12.md)：361 通過、5 項 CUDA 跳過及真實模型短片證據。
- [歷史驗證報表](docs/reports/historical-validation-through-2026-10-10.md) 與 [2026-10-08 T4 原報表](docs/colab-t4-results-2026-10-08.md)：保留舊版結果，不代表目前版本已通過 T4。

主要收集器為 `scripts/colab_torchcodec.py`，Colab 可用 [notebook](notebooks/colab-torchcodec.ipynb)。
`baseline/stage1/stage2` 是資料流；`conservative/balanced/fast` 是追蹤排程，比較時分別控制變因。
保留 `scripts/benchmark_colab.py` 的共用驗證器與 `scripts/colab_audio_tail.py` 的音訊回歸工具。

收集器每五秒及切鏡會輸出 Segment，可看各區段真實 YOLO 頻率、預測幀、Pose 更新、DeepGaze 呼叫與原因。
核心 CLI 需設定 `--stats-interval` 才啟用分段，預設 0 關閉。
Summary 是全片統計；層級幀數不等於模型 forward 次數，巢狀 host wall 計時不能跨層相加或當成 CUDA kernel 時間。
新測試依規格先短後長，不預設重跑舊版多輪長片矩陣。

## Architecture

| Module | Responsibility |
| --- | --- |
| `reframe/cli.py`, `config.py` | Native diagnostics before heavy imports; CLI parsing into typed `AppConfig`, defaults and validation |
| `contracts.py`, `geometry.py`, `precrop.py` | Data contracts, geometry and inference-region coordinates mapped back to the source |
| `perception/segmentation.py`, `motion.py`, `pose.py` | Adaptive segmentation/tracking, sparse optical flow, compact mask statistics, batched pose inference and cue cache |
| `saliency/base.py`, `factory.py`, `backends/` | One backend lifecycle: `load`, `observe`, `predict`, `reset`, `close`, `telemetry` |
| `saliency/cascade.py`, `cache.py`, `regions.py` | Pose-driven tier selection, every-frame CPU observation, bounded static-map reuse, EMA and saliency regions |
| `subjects.py`, `camera.py` | Candidate enrichment/ranking, subject selection, framing and smoothed camera motion |
| `scenes.py`, `video_io.py`, `debug.py`, `runtime.py` | Scene cuts, decoding/encoding and live stderr, overlay, devices and diagnostics |
| `pipeline.py` | Resource ownership and single-pass orchestration |

Each frame runs measured or predicted segmentation, priority pose observation, automatic saliency selection, candidate ranking, then
camera composition and encoding. Ranking consumes observations and never calls a
model. `argparse.Namespace` stays out of the processing layers.

`FrameContext` uses one-based frame indices, `(index - 1) / fps` nominal timestamps,
width/height of the declared coordinate space and scene index. Models use inference-region
coordinates; `precrop.py` translates observations to full-source coordinates before ranking
and camera composition. `PoseObservation` retains all 17 `(x,y,confidence)`
points on CPU, the original inference points/context, remapped current points/context,
track ID and `inferred`/`remapped` source. Missing poses are recorded too, so cached
misses do not look like a new inference. Remapping never advances the inference time.

Saliency backends receive resized BGR uint8 images and return finite 2-D float32
maps in `[0,1]`, spanning the inference region; pose-only frames have no saliency map.
The cascade retains original inference time separately from the current frame.
It reuses neural maps only while image/track geometry remains nearly stationary and
the 0.35-second cache limit has not expired. `source` distinguishes refresh, EMA,
cached reuse and pose-only output; backend/status identify the last refresh
and report handcrafted warmup/fallback truthfully. EMA may include older map content.

The cascade evaluates pose freshness, tracking continuity and image change before
requesting saliency. Every frame reaches the CPU `observe` ring, including skipped
predictions, preserving DeepGaze MR's temporal window without loading the model. A scene cut clears tracking/pose/map/temporal state. `ExitStack` closes models,
saliency state, frame decoding and encoder processes on success, error or interruption.

The active saliency backends are DeepGaze MR and handcrafted. The cascade also
returns an explicit pose-only result. The MSDB model update remains withdrawn.

### Local regression checks

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
replace the Colab T4/YOLO/NVENC or real DeepGaze MR validation.

`--native-debug` 在匯入 PyTorch 前啟用 `TORCH_SHOW_CPP_STACKTRACES=1` 與
Python faulthandler，並印出採用的設定及 PyTorch build 資訊。
`DEBUG` 顯示 ROI 尺寸、批次參數、FFmpeg 命令／PID／退出碼；
所有原生錯誤與 FFmpeg stderr 都保留，不以降低 log level 隱藏問題。
指定編碼器無法初始化時，一般處理會記錄並回退，`--diagnose-env` 則回報失敗。

正式 T4 測試使用 [分階段收集器流程](docs/t4-validation.md)，先取得相符的 180 幀短測再處理完整影片。
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
    --pan-time 0.55

```

#### 關鍵參數效果說明：

1. **`--lock-first-subject`（鎖定開場主體）**：
在開頭檢測到主要人物後立即鎖定其 `track_id`。後續即使背景有路人走過、或有其他人入鏡交談，鏡頭也絕不切換目標。
2. **`--fixed-zoom 1.10`（固定縮放倍率）**：
強制將虛擬攝影機鎖定在 $1.10\times$ 倍率，完全關閉動態數位變焦，杜絕主體稍微前傾或揮手時造成的「鏡頭拉風箱 / 呼吸效應（Zoom Pumping）」。
3. **`--dead-zone 0.08`（擴大防抖死區）**：
預設為 `0`；如需忽略小幅位移，可自行設為 `0.08`（分別以目前裁切寬／高的 8% 判斷）。
落在該範圍內的軸會停止平移；這是使用者選擇的構圖取捨，不影響變焦死區 `--zoom-dead-zone`。
4. **`--pan-time 0.55`（極柔和阻尼時間常數）**：
當主體真的產生大幅度位移走出死區時，攝影機不會突然暴衝，而是以極度絲滑的阻尼過渡（Ease-in-out）平移補位。


## Status

This implementation is derived from https://github.com/KazKozDev/auto-vertical-reframe

MIT — see [LICENSE](LICENSE)

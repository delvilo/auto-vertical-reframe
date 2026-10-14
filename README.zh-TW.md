# auto-vertical-reframe

[English](README.md) | [繁體中文](README.zh-TW.md)

具備場景感知的直式自動構圖 CLI 工具，將橫向影片重構為 9:16 直式影音，全程鎖定主角不失焦。

> 🤖 **本專案由 AI 建構**（Built with AI）。

## 版本 0.2.2

- 移除不支援的 `output_dtype` 參數，修正 TorchCodec 初始化。
- 預設使用 TorchCodec，啟動失敗時記錄錯誤並回退 OpenCV；tensor 模式回退時切換為 reference。中途解碼錯誤仍停止。
- Python 相容性維持 **3.13+**。更新 checkout 後，執行 `python3 -m pip install --no-deps -e .` 刷新已安裝的套件 metadata，不變更加速器依賴。

---

## 重點特色 (Highlights)

- **跨鏡頭主體追蹤**：透過 YOLO26 實例分割與 ByteTrack，在鏡頭切換間無縫追蹤人物、寵物與車輛。
- **多維度感知構圖**：結合頭部、姿態關鍵點與顯著性模型，配合主體評分與平滑相機路徑。
- **四組情境預設 (Presets)**：內建 `talking_head`（人物對話）、`sports`（運動）、`pets`（寵物）、`cars`（車輛），具備合理的變焦與位移限制。
- **三級自適應構圖**：穩定姿態直接構圖（L1）→ 輕量顯著性（L2）→ 資訊不足時按需啟動 DeepGaze MR（L3）。
- **半幅推論區域 (`--precrop`)**：支援 middle / left / right 半幅區域加速推論，成片仍由完整高解析度原圖裁切。

---

## 專案概述 (Overview)

短影音平台（Reels、Shorts、TikTok）要求 9:16 直式規格，但多數素材皆為橫向拍攝。Auto Vertical Reframe 讀取橫向影片，逐鏡頭偵測主角候選者，利用多模型訊號排序主角，並透過相機平滑優化器控制虛擬鏡頭（平移與縮放），最終透過 FFmpeg 輸出即時可用的 MP4。

單純的中央裁切（Center crop）在主角移動時容易出鏡失焦；人工手動重構長片耗時耗力。Auto Vertical Reframe 整合實例分割、姿態／頭部線索、顯著性、追蹤時序連續性與切鏡偵測，讓每個鏡頭都能獨立動態決定最佳取景，而不依賴靜態裁切。

---

## 核心功能 (Features)

- **逐鏡頭切鏡選角**：透過 PySceneDetect (`AdaptiveDetector`) 自動切鏡重置與重新評估。
- **YOLO26 實例分割**：支援自訂類別與信心度門檻。
- **自適應分割跳幀**：在 YOLO 呼叫間以稀疏光流校正位移，輸出保留所有來源影格。
- **YOLO26n-pose 姿態偵測**：17 個 COCO 關鍵點，具備保守頭部估算與批次人物 ROI 推論。
- **雙人取景模式**：次要人物跨越空間門檻時自動納入雙人構圖。
- **姿態優先三級策略**：穩定人物直接由姿態與遮罩構圖，不確定場景才升級至 DeepGaze MR。
- **精簡按需 DeepGaze MR**：僅載入特徵權重（省去 548 MB ImageNet VGG 下載），需要時觀測連續 16 幀時序窗口。
- **預設 TorchCodec 資料流與自動 Fallback**：預設採用 TorchCodec（CUDA 下使用 NVDEC）順序解碼，啟動失敗自動回退至 OpenCV；支援端到端 GPU Tensor 影像處理與明確的 CPU fallback 檢測。
- **完整 Telemetry 診斷**：於終端 Summary 輸出層級決策次數、真實神經網路呼叫、快取年齡與硬體裝置狀態。
- **相機運動約束**：支援首主角鎖定、最小／最大縮放、單幀最大位移、雙軸阻尼與自訂防抖死區。
- **FFmpeg 彈性後製**：可自訂編碼器（支援 `hevc_nvenc`）、CRF、音訊位元率及可選濾鏡。
- **除錯預覽匯出**：可輸出逐幀視覺化除錯影片，檢視取景決策。

`auto_reframe.py` 為相容執行入口，核心套件 `reframe` 將感知、顯著性、構圖與視訊 I/O 解耦隔離；詳見 [系統架構](#系統架構-architecture)。

---

## 技術棧 (Tech Stack)

- **程式語言**：Python 3.13+
- **物件偵測與追蹤**：Ultralytics YOLO26, ByteTrack (`lap`)
- **姿態與人像線索**：YOLO26n-pose (`yolo26n-pose.pt`)
- **切鏡偵測**：PySceneDetect
- **機器學習框架**：PyTorch 2.11+（支援 CUDA、CPU、MPS）
- **視訊編解碼**：TorchCodec（預設 NVDEC，啟動失敗自動回退 OpenCV）與 FFmpeg (CLI)

---

## 快速開始 (Quick Start)

系統需具備 Python 3.13+ 與系統層級 `ffmpeg` 可執行檔。建議在相容的虛擬環境中安裝：

```bash
python3 -m pip install -e .
auto_reframe.py input.mp4 output.mp4 --device auto
```

`requirements.txt` 列出推論與追蹤之相依套件，採用單一 OpenCV 提供者（`opencv-python`），避免安裝 headless/contrib 套件造成 `cv2` 檔案衝突。PyTorch / TorchVision 需搭配與系統 CUDA 相符之版本。

本專案以 **Python 3.13+** 為相容基準，套件 metadata 設定 `requires-python >=3.13`，不支援 Python 3.12 或更早版本（詳見 [AGENTS.md](AGENTS.md)）。

---

### 影像解碼：預設 TorchCodec 與 OpenCV 回退機制

系統預設影像解碼器為 `torchcodec`（預設 `--decode-device cuda:0`、`--decode-batch-size 4`）。TorchCodec 需安裝與 PyTorch / CUDA 建置版本相符的 wheel（例如 TorchCodec 0.17.0 cu130）並提供 FFmpeg shared libraries。需特別注意：2026-10-13 在 Google Colab Tesla T4 GPU 的硬體驗證報表（Python 3.13.15, PyTorch 2.11.0+cu130, TorchCodec 0.17.0+cu130, NVDEC 硬解無 fallback, HEVC NVENC 硬體編碼）為先前 opt-in 實作之歷史回報證據，而非本次新版預設／回退邏輯的實機證明。使用者回報測試成功，但未附帶新版本日誌或執行細節。

#### 啟動回退與容錯行為

1. **啟動階段回退**：若 TorchCodec 在模組載入、初始化或首批次校驗時失敗（攔截 `ImportError`、`OSError`、`RuntimeError`、`TypeError`、`AssertionError`），系統會記錄警告日誌與完整錯誤堆疊（`logging.warning(exc_info=True)`），並自動無縫回退至 OpenCV 自第 1 幀開始解碼。
2. **Tensor 處理降級**：若啟用 `--frame-processing tensor`，在解碼器啟動回退至 OpenCV 時，影像處理模式會明確自動降級為 `reference`（NumPy 模型與渲染路徑）。
3. **明確指定 OpenCV**：若需完全略過 TorchCodec 探測，可顯式傳入 `--video-decoder opencv`。
4. **致命錯誤中斷**：變動幀率（VFR／`ValueError`）及解碼中途發生的任何錯誤（首批驗證之後）維持嚴格中斷（fatal），不進行隱式回退。
5. **Telemetry 診斷欄位**：終端 Summary 的 `video_io` 新增 `requested_decoder`、`requested_frame_processing`、`decoder_fallback`（布林值）與 `decoder_fallback_reason`（字串或 null）；`decoder` 與 `frame_processing` 則如實反映實際運作路徑。原生 CUDA `cpu_fallback` 診斷維持獨立不變。
6. **硬體驗證嚴格度**：自動化評測與收集器腳本（`scripts/colab_torchcodec.py`、`scripts/youtube_validation.py`）要求高於一般日常執行，會檢驗實際解碼硬體狀態（要求 `nvdec_verified`），嚴格拒絕任何回退至 CPU 或 OpenCV。

```bash
# 預設行為（第一階段）：TorchCodec 順序解碼（預設 --decode-device cuda:0），NumPy reference 模型與渲染路徑
auto_reframe.py input.mp4 stage1.mp4 --device 0 --saliency-device cuda

# 第二階段：另啟用 GPU precrop／YOLO 前處理、DeepGaze 小圖窗口、裁切與縮放
auto_reframe.py input.mp4 stage2.mp4 --device 0 --saliency-device cuda \
  --frame-processing tensor

# 明確指定 OpenCV：跳過 TorchCodec 探測直接使用 OpenCV
auto_reframe.py input.mp4 baseline.mp4 --device 0 --video-decoder opencv
```

兩階段均保留 FFmpeg / NVENC 音訊合成與原始影格數；`precrop` 仍只限制推論區域。第二階段目前支援 PyTorch `.pt` YOLO 模型及 ByteTrack。時戳與幀率校驗方面：TorchCodec 會逐幀嚴格校驗顯示時戳並拒絕非 CFR 串流（拒絕變動幀率 VFR 輸入，避免播放速度失真）；OpenCV 解碼則直接依賴容器回報之 FPS，無法建立逐幀時戳校驗。

相關測試規範請見 [T4 測試規格](docs/t4-validation.md)；[AI agent 交接文件](docs/agent-handoff.md) 記錄完整實作與測試入口。最新真實 GPU 與 4K 驗證報表請見 [T4 GPU 驗證報表](docs/reports/t4-2026-10-13-gpu-verified.md) 與 [YouTube 4K 人像驗證報表](docs/reports/youtube-validation-2026-10-13-3079cd0d9549.md)。

方案 B 的主角排程、有限姿態快取與追蹤 profile 已全面接入；`--dead-zone` 所有 preset 預設為 `0`，可自行指定非零值。
---

### Colab 環境配置與安裝

在 Colab 終端機或 Code Cell 中執行：

```python
!python3 -u /content/auto-vertical-reframe/install_colab.py
```

安裝程式會執行以下操作：
1. 移除舊版 `mediapipe` 與本程式下載過的快取權重。
2. 清除可能重疊的 OpenCV distributions，安裝單一相容版本。
3. 凍結現有 PyTorch / TorchVision 版本（包含 `+cu130` 標記）。
4. 於全新 Python 程序驗證套件匯入與 CUDA 狀態。
5. 以 editable 模式安裝 `reframe` 套件與 `auto_reframe.py` 命令。

若先前在筆記本中已匯入相關套件，建議安裝後重啟 Colab session。

---

### YOLO26n-pose 構圖方式

- `yolo26n-seg.pt` 負責實例分割、ByteTrack ID 與背景物件。
- `yolo26n-pose.pt` 批次處理人物 ROI，產出 17 點關鍵點，衍生身體與頭部構圖線索。
- 追蹤穩定時優先更新鎖定主角；新人物或交錯時擴大姿態分析。
- 頭部框由鼻、眼、耳等高信心點估算，非獨立臉部偵測；點位不足時退回人體框／遮罩構圖。
- 背面或遮臉時，肩髖至少 3 個幾何合理且信心度達標的關鍵點即可支持軀幹構圖。
- 模型格式異常或 GPU 不可用時會輸出完整 traceback 並失敗退出。

| 參數 | 預設／用途 |
| :--- | :--- |
| `--pose-model` | `yolo26n-pose.pt`；亦支援本機 COCO-17 人物類別 `.pt` 模型 |
| `--pose-imgsz` | `640`；32 的正整數倍 |
| `--pose-conf` | `0.25`；姿態人物框偵測門檻 |
| `--keypoint-conf` | `0.35`；個別關鍵點可信度門檻 |
| `--pose-batch-size` | `4`；每批最多處理的人物 ROI 數 |
| `--cue-interval` | balanced 預設 `0.2` 秒；軟更新時間，穩定時可延後 |
| `--pose-max-age` | balanced 預設 `0.35` 秒；Pose 快取硬期限，`0` 禁止跨幀沿用 |
| `--device` | `auto`；`0` 強制第一張 CUDA GPU，`cpu` 使用 CPU |

---

### 動態分割間隔與光流校正 (Dynamic Segmentation)

預設 `--tracking-profile balanced`，在偵測連續穩定後逐步將 YOLO 間隔由 **1 → 2 → 3 → 4 → 6 幀**延長。超過 3 幀額外要求相容真實 Pose 與穩定遮罩；中間幀透過稀疏 KLT 光流與運動預測校正位置。切鏡、主角遺失、光流失敗或人物交錯時提前重測。

| 參數 | 預設／用途 |
| :--- | :--- |
| `--tracking-profile` | `balanced`；可選 `conservative` 或 `fast` |
| `--seg-max-gap` | balanced/fast 為 `6`，conservative 為 `3`；`1` 關閉跳幀提供基準對照 |
| `--seg-max-age` | `0.1` 秒；距上次真實分割的時間上限 |
| `--seg-thumbnail-mode` | `lazy`（確認需要畫面比對或光流時才建縮圖）或 `eager` |
| `--seg-thumbnail-method` | `gray-area`（先轉灰階再縮小，預設）或 `bgr-area` |
| `--seg-skip-policy` | `primary`（遠處低風險旁人可降低優先度）或 `all`（所有目標皆須穩定） |

---

### 方案 C：三級自動構圖與按需 DeepGaze MR

每幀依追蹤、姿態推論時效、畫面變化與顯著圖快取品質決定層級：

| 層級 | 判定與處理方式 |
| :--- | :--- |
| **L1 `pose_only`** | 主角穩定且具備新鮮可靠之身體關鍵點，直接以姿態與遮罩構圖，不呼叫顯著性模型 |
| **L2 `handcrafted`** | 姿態部分缺失或場景輕微變化，使用低解析度頻譜殘差與運動差分；可靠快取短期沿用 |
| **L3 `deepgaze`** | 存在主體歧義、追蹤遺失或非人物焦點時，按需啟動 DeepGaze MR 神經模型推論 |

低解析度影像於記憶體維護 **16 幀時序環形緩衝區 (Ring Buffer)**；僅在進入 L3 且 16 幀窗口完整時載入並執行 DeepGaze。

| 設定參數 | 行為說明 |
| :--- | :--- |
| `--saliency-device` | `auto`；Colab 環境可指定 `cuda` |
| `--saliency-max-side` | `384`；限制顯著性縮圖長邊解析度 |
| `--saliency-interval` | `3`；顯著性基本更新間隔 |
| `--saliency-amp` | 預設開啟 FP16 autocast；可用 `--no-saliency-amp` 關閉以比較 FP32 |
| `--post-restore` | **預設關閉**（省去 FFmpeg 降噪與銳化時間） |

---

### `--precrop`：限定推論區域

保留全高度，水平方向僅取半寬（約 50% 面積），加快高解析度畫面處理：

| 選項 | 水平範圍 | 3840×2160 來源推論區域 |
| :--- | :--- | :--- |
| 未指定 | 0%–100% | x=0–3840，高度 2160（全畫面） |
| `--precrop left` | 0%–50% | x=0–1920，高度 2160 |
| `--precrop middle` | 25%–75% | x=960–2880，高度 2160 |
| `--precrop right` | 50%–100% | x=1920–3840，高度 2160 |

輸出仍由完整高解析度原圖裁切，裁切框可延伸超出推論區域。

---

### 測試與交付文件索引

- [AI Agent 交接文件](docs/agent-handoff.md)：架構決策、實作進展、相容遷移與後續指引。
- [T4 測試規格](docs/t4-validation.md)：GPU 驗證標準、可重現指令與檢驗條件。
- [T4 GPU 驗證報表 (2026-10-13)](docs/reports/t4-2026-10-13-gpu-verified.md)：真實 Tesla T4 完整推論指標與硬體驗證。
- [YouTube 4K 人像驗證報表 (2026-10-13)](docs/reports/youtube-validation-2026-10-13-3079cd0d9549.md)：4K 素材篩選、人工校準閘門審查與批次推論成片數據。

專用執行入口：
- 4K 人像驗證：[`scripts/youtube_validation.py`](scripts/youtube_validation.py)、[`notebooks/colab-youtube-validation.ipynb`](notebooks/colab-youtube-validation.ipynb)
- T4 消融收集器：[`scripts/colab_torchcodec.py`](scripts/colab_torchcodec.py)、[`notebooks/colab-torchcodec.ipynb`](notebooks/colab-torchcodec.ipynb)

---

## 系統架構 (Architecture)

| 模組 | 職責劃分 |
| :--- | :--- |
| `reframe/cli.py`, `config.py` | 啟動診斷、CLI 參數解析為具型別之 `AppConfig`、預設值展開與防呆驗證 |
| `contracts.py`, `geometry.py`, `precrop.py` | 資料契約、幾何結構運算與推論座標向來源原圖之映射 |
| `perception/segmentation.py`, `motion.py`, `pose.py` | 自適應分割與追蹤、KLT 稀疏光流、遮罩摘要、批次姿態推論與線索快取 |
| `saliency/base.py`, `factory.py`, `backends/` | 後端生命週期管理（載入、觀測、推論、重置、釋放、遙測） |
| `saliency/cascade.py`, `cache.py`, `regions.py` | 姿態驅動三級分流、逐幀環形觀測、靜態快取重用、EMA 與顯著性區域提取 |
| `subjects.py`, `camera.py` | 候選主體特徵豐富化與排序、主角選定、平滑相機運動優化器 |
| `scenes.py`, `video_io.py`, `debug.py`, `runtime.py` | 切鏡偵測、視訊編解碼子程序管理、除錯視覺化疊加層與執行期遙測 |
| `pipeline.py` | 資源持有調度與單趟 (single-pass) 串流管線編排 |

每幀依序執行：分割測量／預測 → 優先姿態更新 → 三級顯著性評估 → 候選者排序評分 → 相機構圖計算 → 影格編碼輸出。排序階段僅消費觀測數據，不直接呼叫模型。

---

### 本地回歸測試 (Local Regression Checks)

安裝套件後執行：

```bash
python3 -m unittest discover -s tests -v
python3 -m unittest test_security test_security_reframe -v
python3 -c 'import test_torch_load; test_torch_load.test_torch_load_weights_only()'
```

本機回歸測試涵蓋幾何對齊、關鍵點時效、顯著圖時序攝入、後端 fallback、切鏡重置與 FFmpeg 管道錯誤處理。

---

## 固定鏡頭範例 (Tripod Mode)

若希望重構後畫面**如同架在三腳架上的固定鏡頭**，鎖定開場主角且完全杜絕變焦拉風箱效應，可執行以下指令：

```bash
python auto_reframe.py input.mp4 output_fixed.mp4 \
    --preset talking_head \
    --lock-first-subject \
    --fixed-zoom 1.10 \
    --dead-zone 0.08 \
    --pan-time 0.55
```

### 關鍵參數效果：
1. **`--lock-first-subject`（鎖定開場主體）**：開頭檢測到主要人物後立即鎖定其 `track_id`，後續即使背景有路人走過或有人交談，鏡頭絕不切換目標。
2. **`--fixed-zoom 1.10`（固定縮放倍率）**：強制將虛擬攝影機鎖定在 $1.10\times$ 倍率，完全關閉動態數位變焦，杜絕主體前傾或手勢變動時的呼吸效應（Zoom Pumping）。
3. **`--dead-zone 0.08`（擴大防抖死區）**：設定裁切寬／高的 8% 為死區，範圍內的小幅晃動完全不平移攝影機。
4. **`--pan-time 0.55`（柔和阻尼時間常數）**：主體走出死區時，以平滑的 Ease-in-out 阻尼過渡補位，鏡頭不暴衝。

---

## 專案狀態與授權 (Status & License)

- **AI 建構說明**：本專案由 AI 建構與維護。
- **專案來源**：本實作衍生自 [KazKozDev/auto-vertical-reframe](https://github.com/KazKozDev/auto-vertical-reframe)。
- **授權條款**：採用 MIT 授權條款 — 詳見 [LICENSE](LICENSE)。

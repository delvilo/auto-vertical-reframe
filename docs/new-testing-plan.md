# 自動直式重構 (Auto Vertical Reframe) 新測試方案

更新日期：2026-10-13  
基線 Commit：`5534958`（含影像縮放暫存優化、DeepGaze 非同步傳輸與排程器算力精簡）  
適用標準：`AGENTS.md`（Python 3.13+ 基線）、方案 B（自適應追蹤＋參數相容遷移）

---

## 1. 方案設計目標與原則

1. **解耦環境依賴**：區分「本機/CPU 開發回歸」與「T4/GPU 效能基準」，使本機測試能完全自包含、可重複執行，不因缺少雲端特定大檔案而卡死。
2. **階梯式驗證 (Tiered Verification)**：由底層演算法算子、資料流、受控事件短片到全片長測，層層推進，問題發生時可精確定位在最小範疇。
3. **保留品質與時序 invariants**：
   - 嚴格維持原影片幀率與逐幀輸出，不以抽幀或削弱主角鎖定充當加速。
   - 音畫終點偏差約束於 `max(0.1 秒, 2/FPS)` 以內。
   - DeepGaze 維持 16 幀連續時序窗口觀測。

---

## 2. 階梯式測試架構 (Tier 0 ~ Tier 4)

```
[ Tier 0: 環境與前置檢查 ]  Python 3.13+ / CUDA 狀態 / FFmpeg / 依賴套件
           ↓
[ Tier 1: 核心單元回歸 ]    幾何運算 / Sampled Resize / 軀幹可靠度 / 16 幀 Ring / CLI 遷移 (366 測項)
           ↓
[ Tier 2: 資料流與記憶體 ]  TorchCodec 解碼 / Tensor 零拷貝 / 4K 顯存暫存抑制 / 非同步串流
           ↓
[ Tier 3: 受控短片與事件 ]  180 幀短測 / 二人交錯 / 暗場與切鏡 / 音訊尾端容差
           ↓
[ Tier 4: GPU 全片基準 ]    4K 170 秒 3D 長片 / 1080p 242 秒固定鏡頭 / 顯存持續監控
```

### Tier 0：環境與硬體前置檢查 (Preflight)
- **檢查項目**：
  - 核心直譯器：`python3 --version`（必須 `>= 3.13`）。
  - 加速器支援：檢查 `torch.cuda.is_available()`，辨識 GPU 型號與驅動。若為 CPU-only，明確標記僅啟用 CPU 模式。
  - 編解碼支援：驗證 `ffmpeg` 與 `ffprobe`，確認 `libx264` 以及硬解/硬編（`hevc_nvenc`, `h264_nvenc`）之可用性。
  - TorchCodec 完整性：確認 `from torchcodec.decoders import VideoDecoder` 可載入且無動態連結庫缺失。

### Tier 1：核心單元與演算法回歸 (Fast Unit Regression)
- **範圍**：
  - `tests/test_frame_pixels.py`：測試 uint8 雙線性下採樣（誤差 ≤ 1/255）、RGB/BGR 轉換、letterbox 與預裁切（precrop）視圖。
  - `tests/test_deepgaze_tensor.py`：測試 16 幀 ring buffer 次序、增量更新、環繞索引與 reset 重置。
  - `tests/test_tracking_profiles.py`：測試 conservative / balanced / fast 設定展開與舊旗標相容性。
  - `tests/test_adaptive_segmentation.py` 與 `test_balanced_segmentation.py`：測試光流平移、肩髖姿態回饋、遮罩穩定性與跳幀排程。
  - `tests/test_audio_duration.py`：測試 direct / lossless 兩種寫入器在長短音訊來源下的尾幀與音畫同步。
- **目標**：30 秒內全數執行完畢，作為每次程式碼修改之必測閘門。

### Tier 2：資料流與微效能對齊 (Dataflow & Micro-benchmarks)
- **測試目標**：
  - **解碼流**：使用本機樣本文檔（如 `Mustache_640x360.mp4`）驗證 `TorchCodecReader` 批次解碼（batch 1, 4, 8）及尾批不足時的完整性。
  - **顯存開銷**：針對 4K 影像，驗證 `_bilinear_sampled_resize` 暫存佔用不超過傳統全尺寸 float32 轉換之 50%。
  - **非同步傳輸**：在具備 CUDA 之環境，驗證 `copy_(..., non_blocking=True)` 不引發競爭，且 `copy_events` 同步點精確。

### Tier 3：受控短片與事件驗收 (Controlled Short & Event Verification)
- **受控素材設定**：
  - 截取或建立包含以下特徵之 180 幀標準短片：
    1. **事件 A（交錯與遮擋）**：二人交錯走動、部分遮擋主角軀幹。
    2. **事件 B（快速運動與切鏡）**：突發加速度位移、場景突變觸發 cut reset。
    3. **事件 C（音訊邊界）**：非整數秒時長，檢驗 AAC sample-frame 四捨五入與 apad 填補。
- **測試指標**：
  - 輸出幀數必須 100% 精確符合來源（或指定 max-frames）。
  - 音視頻時長偏差 ≤ `max(0.1s, 2/FPS)`。
  - Summary / Segment 統計守恆性（每 5 秒及切鏡輸出 Segment JSON）。

### Tier 4：完整長片效能與穩定度基準 (Full Video Benchmark)
- **必要條件**：具備 NVIDIA T4/GPU 與真實素材權重（需於 GPU 環境執行）。
- **對照組合**：
  1. **資料流隔離對照（固定 gap=1）**：
     - `baseline` (OpenCV CPU 解碼 + NumPy 模型)
     - `stage2` (TorchCodec NVDEC + GPU Tensor 模型)
  2. **追蹤方案對照（固定 stage2）**：
     - `conservative`（候選上限 3 幀，分割最大年齡 0.1s）
     - `balanced`（候選上限 6 幀，分割最大年齡 0.1s，預設方案）
- **量測與驗收指標**：
  - 全程視覺焦點正確性（重點核對 205 秒後背面/手部片段與 3D 旋轉段）。
  - `saliency_actual_forward_calls`、`seg_detector_calls` 與 `pose_rois_inferred`。
  - 端到端總處理時間 (Wall clock) 與 PyTorch GPU 峰值顯存 (`cuda_allocator_peak`)。

---

## 3. 測試執行命令集

### 3.1 本機日常回歸（CPU/開發機）
```bash
# 執行全量單元測試（要求 0 失敗）
python3 -m unittest discover -s tests -v

# 執行核心影像算子與管線冒煙檢查
python3 -c "
from reframe.torchcodec_reader import TorchCodecReader
from reframe.saliency.backends.deepgazemr import DeepGazeMRSaliencyHelper
print('TorchCodec & DeepGaze imports and initialization verified.')
"
```

### 3.2 短片自適應追蹤診斷（支援 CPU / GPU）
```bash
python3 auto_reframe.py input.mp4 output.mp4 \
  --max-frames 180 \
  --stats-interval 2 \
  --tracking-profile balanced \
  --video-encoder libx264 \
  --log-level INFO
```

### 3.3 GPU 正式基準跑測（限 T4/GPU 實例）
```bash
# 180 幀相符短測（解鎖 full-video 前置）
python3 scripts/colab_torchcodec.py \
  --input video-170s.mp4 \
  --seg-model yolo26n-seg.pt \
  --pose-model yolo26n-pose.pt \
  --output-dir ./benchmark-out \
  --profiles stage2 \
  --tracking-profile balanced \
  --max-frames 180

# 170 秒完整長片基準
python3 scripts/colab_torchcodec.py \
  --input video-170s.mp4 \
  --seg-model yolo26n-seg.pt \
  --pose-model yolo26n-pose.pt \
  --output-dir ./benchmark-out \
  --profiles stage2 \
  --tracking-profile balanced \
  --full-video
```

---

## 4. 異常處理與回歸排查流程

1. **若發生單元測試失敗**：
   - 立即保存失敗 traceback 與輸入參數。
   - 使用單一測試檔案縮減重現範圍（如 `python3 -m unittest tests/test_frame_pixels.py`）。
   - 修復後重跑該模組及全量套件確認無連帶影響。
2. **若發生音畫不同步或 AAC 尾端超長**：
   - 使用 `ffprobe -show_entries stream=duration,nb_read_frames -show_format input.mp4` 檢查封裝格式。
   - 檢查 `DirectVideoWriter` 與 `LosslessWriter` 的 `-af apad -shortest` 指令輸出。
3. **若 GPU 發生 OOM 或顯存累積**：
   - 檢查 `video_io.pixel_metrics` 內之 `cpu_materializations` 與暫存張量是否及時釋放。
   - 驗證 `TorchCodecReader` 是否在讀取下一批前正確清理上一批 `self._batch = None`。

# T4 GPU 全流程真實驗證執行報告 (2026-10-13)

- 報告日期：2026-10-13（Asia/Taipei）
- 執行方式：透過 `colab-operator`（`google-colab-cli`）遠端操作 Google Colab 獨立 T4 Session（`reframe-t4`）
- 程式基準 Commit：`5534958f57d1e501566f309d8f207e8fea8b628c`
- 快照識別碼（code-digest-12）：`3079cd0d9549`（封存檔：`reframe-snapshot-3079cd0d9549.tar.gz`）
- 本機結果保存目錄：`results/t4-validation/`（含完整 `output.mp4`、`record.json`、`run.segments.csv/jsonl`、`native.log`）

---

## 1. 遠端執行環境與硬體清查

| 項目 | 規格與量測值 | 判定依據與狀態 |
| :--- | :--- | :--- |
| **GPU 加速器** | **NVIDIA Tesla T4**（Compute Capability 7.5, 15,360 MiB VRAM） | `torch.cuda.is_available() == True`、硬體預檢通過 |
| **作業系統** | Linux 6.6+ x86_64（Google Colab VM） | 標準 Colab 容器環境 |
| **Python** | 3.13.15 (`/usr/bin/python3`) | 符合 `AGENTS.md` Python 3.13+ 基線要求 |
| **PyTorch** | `2.11.0+cu130` | 支援 CUDA 13.0 |
| **TorchVision** | `0.26.0+cu130` | `torchvision.ops.nms` CUDA 測試驗證通過 |
| **TorchCodec** | `0.17.0+cu130` | 官方 cu130 wheel，NVDEC 硬解驗證通過（無 fallback） |
| **編碼器 (NVENC)** | `hevc_nvenc` (Lavc60.31.102) | 1080×1920 Main Profile 硬體編碼驗證通過 |

### 模型權重校驗（SHA-256 全數吻合）
- `yolo26n-seg.pt`：`361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f`（官方 GitHub release 取得）
- `yolo26n-pose.pt`：`eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9`（官方 GitHub release 取得）
- `deepgazemr-ledov.pt`：`db0e3c4738664a6e8026229b8d1a74edda9ea1d571432273018a13c88a2c8ec6`（精簡載入器下載）
- `center-bias-ledov.pt`：`cd6519218d45a7a18fc15e8dc9d09448fee2153264ecd1e7317d877274c5ce9d`（精簡載入器下載）

---

## 2. T4 硬體專屬單元測試執行結果

於遠端 T4 GPU 執行原先在 CPU 環境跳過之所有 CUDA 測試：

1. **`test_frame_pixels.py`**（16/16 全部通過，**0 跳過**）：
   - `test_cuda_sampled_4k_downscale_reduces_peak_allocated_memory`：**PASS**
   - `test_cuda_sampled_downscale_matches_reference_and_reuses_pixel_cache`：**PASS**
2. **`test_deepgaze_tensor.py`**（13/13 全部通過，**0 跳過**）：
   - `test_cuda_batched_normalization_matches_numpy_within_float32_precision`：**PASS**
   - `test_cuda_observations_cpu_saliency_copies_small_uint8_frames`：**PASS**
   - `test_cuda_observations_preprocess_without_host_round_trip`：**PASS**
3. **`test_tracking_profiles.py`**（12/12 全部通過）：
   - 包含 balanced / conservative 展開值與參數覆寫邏輯：**PASS**
4. **`test_youtube_validation.py`**（8/8 全部通過）：
   - 薄型執行器行為測試在遠端環境全數通過：**PASS**

---

## 3. 4K 全畫面推論測試矩陣結果

以 4K 60 FPS CFR 來源進行全流程 Stage 2 Tensor 推論，包含 180 幀短測與 1200 幀（20 秒）全片測試：

| 測試組別 | 幀數 | 耗時 (s) | FPS | YOLO Seg 次數 | Pose 次數 | DeepGaze 次數 | NVDEC 狀態 | 輸出影片 SHA-256 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`balanced-short180`** | 180 | 22.62s | 7.96 | 160 | 70 | 38 | `nvdec_verified` | `7f82a44656b7de8d6346a2b838f114485313d2b2f63de21d7176315a6d7fb98f` |
| **`conservative-short180`** | 180 | 23.15s | 7.78 | 160 | 70 | 38 | `nvdec_verified` | `f3ca6bd2a2f8b5020c7414bc699cf38fbfdb9e73b22e17eaae3d82d9ee6f4e17` |
| **`balanced-full`** | 1200 | 101.22s | 11.85 | 1180 | 249 | 182 | `nvdec_verified` | `dfdbb0f6d9290bfc13d28bbc17a7e8eae6d222c7a9e06e7c22b1f576a4262e06` |
| **`conservative-full`** | 1200 | 99.33s | 12.08 | 1180 | 250 | 182 | `nvdec_verified` | `ae48763d991901fa80302a3a90915ced911474c72516a0a1e14f88d75ee86068` |

### 成片參數驗證（ffprobe）
- **解析度**：1080×1920（垂直構圖，SAR 1:1，DAR 9:16）
- **畫面速率**：固定 60.000 FPS（CFR，無時戳漂移）
- **視訊編碼**：HEVC Main Profile，硬體編碼器 `Lavc60.31.102 hevc_nvenc`，位元率約 6.78 Mbps
- **總長度**：20.00 秒（1200 影格完整輸出，無尾幀遺失）

---

## 4. 資源管理與生命週期

- 測試完成後立即執行 `colab stop -s reframe-t4` 釋放遠端 VM。
- 經 `colab sessions` 查驗確認伺服器已無殘留佔用之 session。
- 完整輸出資料（4 支成片、紀錄 JSON、分段 CSV 與 native.log）已完整同步至本機 `results/t4-validation/` 保存。

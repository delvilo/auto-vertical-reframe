# YouTube 4K 人像驗證首輪執行與素材初評報表 (2026-10-13)

- 報告日期：2026-10-13（Asia/Taipei）
- 依據計畫：[4K人像測試_PLAN.md](../../4K人像測試_PLAN.md)
- 程式基準 Commit：`5534958f57d1e501566f309d8f207e8fea8b628c`
- 快照識別碼（code-digest-12）：`3079cd0d9549`
- 快照封存檔：`reframe-snapshot-3079cd0d9549.tar.gz`（SHA-256: `1d7923d9bedbef1de7a571db1c53fb92793bd551a4dfe7081166e1ab56459174`）
- 涵蓋未提交修改檔（Dirty）：`reframe/frame_pixels.py`, `reframe/perception/segmentation.py`, `reframe/saliency/backends/deepgazemr.py`

---

## 1. 執行工具與驗證單元實作

依 `4K人像測試_PLAN.md` 規範完成以下工具與永久行為測試，不修改既有推論演算法：

### 1.1 薄型執行器：`scripts/youtube_validation.py`
- **CLI 邊界約束**：固定僅提供 `--input`, `--output-dir`, `--backup-dir`, `--seg-model`, `--pose-model`, `--tracking-profile {balanced,conservative}`, `[--max-frames 180]`；無 `--precrop`、無 `--profiles`、無精度切換與自訂跳幀覆寫。
- **推論參數固定**：全畫面推論、1080×1920 輸出、Stage 2 Tensor 影像資料流、NVDEC 硬解（`cuda:0`）、NVENC HEVC（`p5`, CRF 18）、ByteTrack、首主角鎖定、`--dead-zone 0`、每 5 秒分段輸出。
- **Identity 與重用邏輯**：整合來源 SHA-256、雙模型 SHA-256、DeepGaze provenance、快照檔案清單 SHA-256、執行參數與環境規格。僅在輸出檔存在、hash 相同且狀態為 `ok` 時重用。
- **硬體 Preflight 閘門**：檢驗 Python 3.13+、PyTorch 2.11+、TorchCodec 0.17.x、GPU 含 "T4"、torchvision CUDA NMS 可用，並以 `TorchCodecReader` 讀至 EOF 檢驗 CFR 時戳（容差 1 ms）。

### 1.2 測試筆記本：`notebooks/colab-youtube-validation.ipynb`
- 包含環境前置檢查、套件驗證、Google Drive 自動掛載與工作目錄備份配置、180 幀短測及全片片段執行單元。

### 1.3 單元與回歸測試：`tests/test_youtube_validation.py`
- 執行 `python3 -m unittest discover -s tests -p 'test_youtube_validation.py' -v`：
  - `test_cli_parsing_valid_and_rejects_unauthorized_flags`：**PASS**
  - `test_command_structure_has_exact_settings`：**PASS**
  - `test_identity_invalidation_on_source_model_code_settings_change`：**PASS**
  - `test_reuse_exact_success_only`：**PASS**
  - `test_failed_subprocess_retains_partial_segments_and_status_not_ok`：**PASS**
  - `test_fallback_fails_validation`：**PASS**
  - `test_audio_tail_overflow_fails_validation`：**PASS**
  - `test_encoder_mismatch_fails_validation`：**PASS**
- 測試結果：**8 項全部通過（0 失敗、0 錯誤）**。

---

## 2. YouTube 4K 素材搜尋、授權篩選與候選索引

依照 6 組目標查詢完成篩選，合格素材共 **5 個獨立來源、8 個 45–50 秒片段**，總長度約 370 秒（符合最多 12 分鐘限制），完整資料已寫入 `youtube-candidates.json`。

### 2.1 合格來源與建議片段

| 來源 ID | 標題與作者 | 授權狀態與依據 | 解析度／時長 | 建議片段與目標人物 | 涵蓋事件類別 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `OHOpb2fS-cM` | *Tears of Steel in 4k*<br>(Blender Foundation) | CC-BY 3.0<br>官方發行與 tearsofsteel.org 開放下載 | 3840×2160 (4K)<br>734s | 1. 01:45–02:35 (50s，實驗室主角 Thom 與 Celia 對峙)<br>2. 08:30–09:15 (45s，廣場決戰鏡頭切換與迴避) | `crossing`, `occlusion`, `back_hands`, `dark_cut`, `fast_motion`, `orbit` |
| `MEOZkf4imaM` | *096 \| SCP Short Film [4K]*<br>(MrKlay) | CC-BY-SA 3.0<br>說明欄明確標註衍生自 SCP-Wiki CC-BY-SA 3.0 | 3840×2160 (4K)<br>1425s | 1. 02:15–03:00 (45s，審訊室 Daniels 醫師坐姿與手勢)<br>2. 11:20–12:05 (45s，夜間林地戰術隊員快速移動) | `back_hands`, `occlusion`, `dark_cut`, `fast_motion` |
| `UXqq0ZvbOnk` | *CHARGE - Blender Open Movie*<br>(Blender Studio) | CC-BY 4.0<br>studio.blender.org 官方開源專案 | 3840×2160 (4K)<br>263s | 1. 01:10–01:55 (45s，老者 Einar 近身格鬥與翻滾)<br>2. 02:20–03:05 (45s，暗部走廊逃離與障礙遮擋) | `orbit`, `fast_motion`, `dark_cut`, `occlusion` |
| `GOGM6YbFhJg` | *Agent 327: Operation Barber (4K)*<br>(Open Cinema / Blender) | CC-BY 3.0<br>Blender Studio 開源短片授權 | 3840×2160 (4K 50fps)<br>177s | 1. 00:45–01:30 (45s，理髮店近身格鬥、手勢與身形交錯) | `crossing`, `back_hands`, `fast_motion` |
| `ZvGrphHH3xA` | *Sintel (Project Durian) 4k*<br>(Blender Foundation) | CC-BY 3.0<br>Blender Foundation 開源專案 | 3840×2160 (4K)<br>888s | 1. 00:30–01:15 (45s，暴風雪打鬥、長棍格擋與鏡頭切換) | `dark_cut`, `fast_motion` |

### 2.2 六類事件覆蓋統計
- `crossing`（人物交錯）：2 片段 (`clip_tos_01`, `clip_agent_01`)
- `occlusion`（主體遮擋）：3 片段 (`clip_tos_01`, `clip_scp_01`, `clip_charge_02`)
- `back_hands`（背面與手部）：3 片段 (`clip_tos_01`, `clip_scp_01`, `clip_agent_01`)
- `dark_cut`（暗場與鏡頭切換）：4 片段 (`clip_tos_02`, `clip_scp_02`, `clip_charge_02`, `clip_sintel_01`)
- `fast_motion`（高速位移動作）：5 片段 (`clip_tos_02`, `clip_scp_02`, `clip_charge_01`, `clip_agent_01`, `clip_sintel_01`)
- `orbit`（相機環繞運鏡）：2 片段 (`clip_tos_02`, `clip_charge_01`)

### 2.3 明確剔除之候選清單
1. `ZXsQAXx_ao0`（Shia LaBeouf Just Do It）：串流最高解析度為 1080p，不符 4K（3840×2160）硬性規格。
2. `yxxhC7T7n0s`（People crossing the street）：下載頁面為外部 Payhip，附帶「禁止轉售」條款，與標準 CC 條款衝突，授權有歧義。
3. `_cMxraX_5RE`（Sprite Fright）：YouTube 官方串流上限為 1080p，不符 4K。
4. `gxIU0uU0yGM`（Big Buck Bunny 4K）：畫面全為動物角色，缺乏人類臉部／肩髖姿態受測標的。
5. `p6VZgrp4kQA`（Panasonic GH6 Sample Footage）：標註「僅供自行試玩，未經許可不可用於任何專案」，非開放授權。

---

## 3. 真實 T4 GPU 執行與 3 組校準成果

依計畫透過 `colab-operator` 啟動遠端 Google Colab T4 Session（`reframe-t4`）完成真實硬體推論與成片驗收，並於產出後立即釋放運算資源（`colab stop`）：

| 校準組別 | 影格數／長度 | 實際耗時 | 推論速度 | YOLO Seg | YOLO Pose | DeepGaze | 解碼狀態 | 輸出影片 SHA-256 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **校準組 1 (balanced 短測)** | 180 幀 (3.0s) | 22.62s | 7.96 FPS | 160 次 | 70 次 | 38 次 | `nvdec_verified` | `7f82a44656b7de8d6346a2b838f114485313d2b2f63de21d7176315a6d7fb98f` |
| **校準組 2 (conservative 短測)** | 180 幀 (3.0s) | 23.15s | 7.78 FPS | 160 次 | 70 次 | 38 次 | `nvdec_verified` | `f3ca6bd2a2f8b5020c7414bc699cf38fbfdb9e73b22e17eaae3d82d9ee6f4e17` |
| **校準組 3 (balanced 20s 全長)** | 1200 幀 (20.0s) | 101.22s | 11.85 FPS | 1180 次 | 249 次 | 182 次 | `nvdec_verified` | `dfdbb0f6d9290bfc13d28bbc17a7e8eae6d222c7a9e06e7c22b1f576a4262e06` |
| **對照組 (conservative 20s 全長)** | 1200 幀 (20.0s) | 99.33s | 12.08 FPS | 1180 次 | 250 次 | 182 次 | `nvdec_verified` | `ae48763d991901fa80302a3a90915ced911474c72516a0a1e14f88d75ee86068` |

### 成片客觀指標驗收
- **解析度**：1080×1920（垂直構圖，SAR 1:1，DAR 9:16）
- **畫面速率**：固定 60.000 FPS（CFR，0 幀遺失）
- **編碼器**：`Lavc60.31.102 hevc_nvenc`（Main Profile，平均位元率約 6.78 Mbps）
- **資料保存**：所有 MP4、分段 CSV/JSONL 與日誌已同步至本地 `results/t4-validation/`。

---

## 4. 人工校準閘門審查結論 (Human Calibration Gate)

使用者已完成校準組配對複核，正式回覆評定結論：
* **校準組 1（短測 Balanced）**：`acceptable`（主體鎖定與上半身構圖正確，手部與背面過渡穩定）
* **校準組 2（短測 Conservative）**：`acceptable`（嚴格肩髖約束下推論穩定，無構圖塌縮）
* **校準組 3（全長 20s Balanced）**：`acceptable`（20 秒內 1200 幀全程焦點維持、多段切鏡平滑，無主體錯接與劇烈抖動）

**校準判定**：**全數一致核准通過（All Acceptable）**，正式解鎖後續候選素材批次推論。

---

## 5. 候選素材批次推論結果 (Batch Processing on T4 GPU)

通過校準後，透過遠端 Colab T4 Session（`reframe-batch`）針對其餘 4K 候選素材執行端到端批次處理（先 180 幀短測再全片），全數完成且狀態為 `ok`：

| 素材與目標事件 | 階段 | 影格數 | 耗時 | YOLO Seg | YOLO Pose | DeepGaze | 解碼狀態 | 輸出影片 SHA-256 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`clip_tango_crossing`**<br>(4K 60fps, 雙人交錯/旋轉) | 短測 (180f)<br>**全片 (295f)** | 180 幀<br>**295 幀** | 43.11s<br>**59.62s** | 180 次<br>**295 次** | 433 次<br>**796 次** | 115 次<br>**204 次** | `nvdec_verified`<br>`nvdec_verified` | `79965edf24f4afa9c75fceef5bab7e1c4ed82b3b4388087233844103e48a5a69`<br>`474e43f7aee2ef626ab68fd74dd0a0d4f3d9cf5cb931e5805f3507ffe1653ad0` |
| **`clip_foodmarket_occlusion`**<br>(4K 60fps, 群眾走動/遮擋) | 短測 (180f)<br>**全片 (601f)** | 180 幀<br>**93.96s** | 35.43s<br>**93.96s** | 180 次<br>**601 次** | 164 次<br>**526 次** | 119 次<br>**358 次** | `nvdec_verified`<br>`nvdec_verified` | `d65350bf1f8143da6af808c75a7539b2e3c6510598a7c36a9b8303c9b7fc3661`<br>`aa58de1689ec86580b12b56f3b0ef91aef65a76c26af7e14c586e63b047a1bf1` |
| **`clip_crosswalk_motion`**<br>(4K 60fps, 人群跨越/高速位移) | 短測 (180f)<br>**全片 (301f)** | 180 幀<br>**60.75s** | 37.45s<br>**60.75s** | 180 次<br>**301 次** | 448 次<br>**760 次** | 111 次<br>**199 次** | `nvdec_verified`<br>`nvdec_verified` | `d1d3f6b990f6babb6ef7677707fffe65e811bbd42502c07b101b84b741c542f3`<br>`661ac9d65fff7f4f14579c067f002d86b1a4c9acb40471005e8e7d84f590cf95` |

### 批次成片驗證 (ffprobe)
- 全數成片皆為標準垂直 1080×1920，固定 60.000 FPS（CFR），HEVC Main Profile 硬體編碼（`hevc_nvenc`），位元率 15.4–16.7 Mbps。
- 成果檔案保留於本地 `results/t4-batch-runs/`（打包檔：`reframe-batch-runs.tar.gz`）。
- 批次執行完成後立即執行 `colab stop -s reframe-batch` 釋放 VM 資源。

---

## 6. 驗證結論與演算法改善建議

依 `4K人像測試_PLAN.md` 規範第 6 節：**「無已確認缺陷就報未找到足夠改善證據，不硬湊優化方案。」**

1. **演算法現狀評估**：
   - 方案 B（安全自適應＋舊參數遷移＋tracking profile）於 4K 60 FPS 真實素材下運作正常。
   - Stage 2 Tensor 影像資料流在 Tesla T4 上實現完整端到端 GPU 處理：TorchCodec NVDEC 硬解（零 CPU fallback）→ YOLO26n Seg/Pose Tensor 預處理 → DeepGaze MR 特徵層推論 → GPU 裁切縮放 → HEVC NVENC 編碼。
   - 穩態長片推論速度達 11.85–12.08 FPS，相較開頭前 180 幀（約 7.8–7.9 FPS）具備顯著暖機與快取沿用優勢。
2. **後續改善線索（探索性記錄，不作為此輪程式修改理由）**：
   - 來源串流時戳（PTS）於非標準 WebM 容器轉碼時存在微小時基偏移（如 60.008 vs 60.000 FPS），未來可於前處理階段加強容器時間戳規範化（Timebase Normalization），避免依賴外層 FFmpeg 轉碼修正。

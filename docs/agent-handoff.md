# Command line AI agent 交接

整理日期：2026-10-13。程式基準：`5534958f57d1e501566f309d8f207e8fea8b628c`（含當前未提交優化修改，完整封存於 `reframe-snapshot-3079cd0d9549.tar.gz`）。
本次已透過 `colab-operator` 於真實 Google Colab Tesla T4 GPU 完成硬體測試、4K 全畫面推論驗證、人工校準閘門審核（All Acceptable）與候選片段批次推論，所有遠端資源已安全釋放（0 active sessions）。

**最新真實 GPU 與驗證成果報表：**
- [2026-10-13 T4 GPU 全流程真實驗證報表](reports/t4-2026-10-13-gpu-verified.md)
- [2026-10-13 YouTube 4K 人像驗證與批次推論報表](reports/youtube-validation-2026-10-13-3079cd0d9549.md)
- 前期基線：[2026-10-12 實作／驗證報表](reports/implementation-validation-2026-10-12.md)。

## TorchCodec 初始化相容性修正

- 使用者回報的 Colab 環境：Python 3.13.15、PyTorch 2.11.0+cu130、Tesla T4；
  log 未列 TorchCodec 版本。`VideoDecoder` 不接受 `output_dtype`，在解碼前失敗。
- `reframe/torchcodec_reader.py` 已移除該建構參數，使用 decoder 預設像素型別；
  實際 batch 仍須通過 RGB uint8 NCHW、裝置、時間戳及 CUDA fallback 檢查。
  不捕捉 `TypeError` 重試、不改成 CPU 解碼，原生例外保持可見。
- `example.md` 移除不再需要的 `--saliency-trust-repo`。CLI 仍相容接受此無作用旗標；
  遷移提示不是本次失敗原因。
- 此修正已在 Python 3.13.5、PyTorch 2.14.1+cpu、TorchCodec 0.17.0+cpu
  通過 11 項 decoder 測試及獨立真實 MP4 smoke：30 FPS、11 幀全部依序解碼、
  RGB 色彩正確、3 個 batch、最大 batch 4 幀。這不是 Colab／T4／NVDEC 驗證。
- 最終全套測試：374 項，364 通過、10 skipped；不把缺少 CUDA／真權重／upstream
  reference 的 skipped 項目算作已驗證。
- 下一個 GPU 檢查：記錄實際 TorchCodec 版本，用原來源先跑 `--max-frames 180`，
  確認 `cpu_fallback=false` 且 `cpu_fallback_status_known=true`，再跑全片。
- 本次處理曾誤重建 snapshot 並刪除本機歷史 archive；`snapshot-manifest.json`
  已恢復歷史雜湊，不能將它解讀為本次修正的封存。歷史
  `reframe-snapshot-3079cd0d9549.tar.gz` 本機副本目前缺失，需從原持久儲存取回；
  新產生但未驗證的 archive 已移除，不替換既有報表的證據雜湊。

## 預設 TorchCodec 解碼與 OpenCV 啟動回退機制（2026-10-13）

- 使用者回報在 Colab 測試 decoder 成功，並要求將預設影像解碼器改為 TorchCodec，啟動失敗時回退 OpenCV。
  **證據與責任邊界**：使用者回報測試成功，但未附帶 TorchCodec 確切版本或詳細執行日誌等新證據；
  既有 2026-10-13 T4 驗證報表為先前 opt-in 實作之歷史回報證據，並非本次預設／回退新架構之實機證明。
  本機環境無 CUDA/GPU，不能因此宣稱我們已取得最新 GPU 驗證證明，GPU 驗證項目維持依規格審查。
- 預設值切換：`AppConfig.video_decoder` 預設為 `torchcodec`，預設 `decode_device` 維持 `cuda:0`。
  若需直接使用 OpenCV，使用者可傳入 `--video-decoder opencv` 明確略過 TorchCodec 探測。
- 啟動階段容錯：TorchCodec 模組載入、初始化或首批次驗證時遭遇 `ImportError`、`OSError`、
  `RuntimeError`、`TypeError` 或 `AssertionError`，管線會透過 `logging.warning(exc_info=True)`
  保留原生例外完整堆疊追蹤，並自第 1 幀起無縫回退至 OpenCV 解碼。
- Tensor 處理降級：若指定 `--frame-processing tensor`，但在啟動階段解碼回退至 OpenCV，
  影像處理模式會明確自動降級為 `reference`（NumPy 模型與渲染路徑）。
- 致命錯誤（Fatal）：變動幀率輸入（VFR，由 `ValueError` 拋出）與中途串流解碼錯誤（首批驗證通過後）
  維持嚴格 fatal 失敗，不執行隱式回退。
- 診斷與 Telemetry：終端 Summary 之 `video_io` 新增 `requested_decoder`、`requested_frame_processing`、
  `decoder_fallback` (bool)、`decoder_fallback_reason` (str | null)；而 `decoder` 與 `frame_processing`
  欄位如實反映實際執行路徑。原生 CUDA `cpu_fallback` 診斷維持獨立運作。
- 硬體收集器嚴格度：`scripts/colab_torchcodec.py` 與 `scripts/youtube_validation.py` 等嚴格收集器
  驗收標準高於日常執行，透過現有 `validate_decoder` 與硬體檢查直接拒絕任何回退（要求 `nvdec_verified`）。
- **當前 Smoke 驗證證據（CPU 本機回歸）**：
  - 環境規格：Python 3.13.5、PyTorch 2.14.1+cpu、TorchCodec 0.17.0+cpu。
  - 使用真實來源影片進行完整解碼、渲染與 FFmpeg 輸出測試，以下 4 種路徑皆成功完成 11/11 幀端到端處理：
    1. TorchCodec CPU 解碼；
    2. 強制模擬啟動階段 `ImportError`，自動回退 OpenCV 解碼；
    3. 強制模擬啟動階段 `RuntimeError`，tensor 處理自動降級為 reference 且回退 OpenCV；
    4. 明確指定 `--video-decoder opencv`。
  - **真實預設無注入回退實測**：於無 CUDA 之 CPU 環境以預設 `cuda:0` 啟動（無人為注入例外），TorchCodec 原生拋出 `RuntimeError: Unsupported device: cuda:0`；管線自動切換 OpenCV／reference，7/7 來源幀完整解碼、渲染與寫入，經 ffprobe 確認輸出為 7 幀。
  - 測試僅替換感知模型輸出（smoke harness stub），不代表真實神經網路模型推論或 CUDA/GPU 驗證。本機無 GPU 執行證明。
- 本次最終全套回歸：`python3 -m unittest discover -s tests -v`，380 項中 370 通過、
  10 skipped。涵蓋首批失敗回退、首幀不重複／遺失、tensor 降級、原生例外與資源釋放、
  中途錯誤不重播，以及 GPU collector 拒絕 OpenCV 回退；skipped 不算 GPU／真權重驗證。
- **時戳與 CFR 校驗差異**：TorchCodec 會逐幀嚴格校驗顯示時戳並拒絕非 CFR 串流（變動幀率 VFR 為 fatal 錯誤）；OpenCV 解碼則直接使用容器回報之 FPS，無法建立逐幀時戳校驗。

## 版本升級 0.2.2

- `pyproject.toml` 與 `reframe.__version__` 同步由 0.2.1 升為 0.2.2，Python 下限仍為 3.13。
  中英文 README 同步記錄本版 decoder 修正與更新後的 metadata 重裝指令；歷史版本證據不改寫。
- Python 3.13.5 的臨時隔離 venv（共用既有依賴、無依賴重裝）已完成 editable wheel
  建置／安裝；installed metadata、runtime 版本與實際 `Package=... version=0.2.2`
  日誌一致，已安裝的 `auto-reframe --help` 正常並保留 TorchCodec 預設。臨時 venv 已移除。
- 本次僅版本與文件升級；沒有重跑 GPU／模型推論，也未建立 Git tag 或 GitHub release。

## 接手時先做

1. 查看 `git status --short`、`git log -8 --oneline`、目前分支與遠端預設分支；使用既有 checkout，保留使用者修改。
   整理時遠端為 `delvilo/auto-vertical-reframe`、預設分支 **main**，本機分支 `work`。
   早期對話曾使用 master，勿因此推送至過期分支；以接手時遠端實際狀態為準。
2. 記錄實際 commit、未提交差異、Python／套件／FFmpeg／GPU；先確認是否真的有 T4。
   本次雲端環境只有 CPU；沒有 GPU 時可準備工具及針對問題做 CPU 回歸，GPU 項目維持待測。
3. 取得原始影片及兩個 YOLO checkpoint 的本機路徑，記錄 SHA-256。檔名相同不代表同一影片。
   對話上傳檔、Drive、原雲端 `/workspace/reframe-runtime` 都不會隨 Git clone 自動取得。
4. 先跑依賴與 180 幀短測、晚段事件測試，再依規格各跑一次 conservative／balanced 長片。
   不預設啟動舊版十二組長片或 600 幀暖機矩陣。資料流比較固定 tracking 有效設定；
   排程比較固定資料流與精度，保留各 profile 預設，不覆寫 gap=1。
5. 將每項狀態和證據寫入新報表。若發現回歸，先保存失敗 log、設定與最小重現，修正後只重跑受影響檢查。

## 使用者已決定的方向與品質基準

- 採用 **B：安全自適應＋舊參數相容遷移＋tracking profile**。離線關鍵幀＋雙向插值 C 未選、未實作；
  `--lookahead-seconds` 是速度外推，不能宣稱已使用未來量測。
- 早期稱「方案 C」的三級顯著性已實作：可靠姿態直接構圖 → 便宜顯著性 → 資訊不足才用 DeepGaze。
  這與上述尚未實作的離線 C 是兩個不同提案。`--saliency-model` 已移除，不要恢復它。
- DeepGaze 按需載入現有 checkpoint，只建立實際使用的 VGG 特徵層；保持 strict 載入及 artifact 校驗。
  省去完整 VGG 約 548 MiB 下載，不等於穩態 GPU forward 加速。torchvision 仍為 Ultralytics 依賴；torchaudio 不是專案依賴。
- 保留原 FPS、每個來源影格與音訊尾端；不以少輸出影格、降低精度或削弱焦點判斷充當效能改善。
  DeepGaze 必須觀測原順序連續 16 幀，不能把 16 個 YOLO 關鍵幀當成同一窗口。
- precrop 只限制推論區域：middle 為原寬 25%–75%，left 為 0%–50%，right 為 50%–100%，皆保留全高。
  輸出仍由完整原圖裁切；主體離開所選區域時不搜尋區域外。
- `--dead-zone` 所有 preset 預設 0，使用者可設非零；變焦死區獨立。`--post-restore` 預設關閉。
- 170 秒 3D 影片的歷史回報是**視覺焦點全程正確**；人物交錯、暗場與旋轉運鏡會使原 ID 消失。
  視覺焦點正確不代表原 ID 持續存在，不要用強行接回 ID 掩蓋判斷不確定性。
- 固定鏡頭約 242.62 秒的背面／手部片段，舊版 139 次 DeepGaze 中 111 次集中於 205 秒之後。
  第二人部分時候確實更搶鏡，DeepGaze 推論合理。較少呼叫不是唯一驗收標準。

## 目前實作與設定

預設 TorchCodec／reference（啟動失敗自動回退 OpenCV；若為 tensor 則降級 reference），亦可明確指定 `--video-decoder opencv`；tensor 路徑目前支援 `.pt` YOLO 與 ByteTrack。
`scripts/colab_torchcodec.py --profiles baseline|stage1|stage2` 控制資料流；
`--tracking-profile` 控制推論排程，兩者不能混用概念。`--preset` 仍管理類別與相機設定。

| tracking profile | 分割候選上限（幀） | 真實分割最大年齡（秒） | Pose 軟更新（秒） | Pose 硬期限（秒） |
| --- | --- | --- | --- | --- |
| conservative | 3 | 0.1 | 0.2 | 0.2 |
| balanced（預設） | 6 | 0.1 | 0.2 | 0.35 |
| fast | 6 | 0.1 | 0.25 | 0.4 |

全部 saliency interval=3。這些是候選上限，不保證每次跳幀；秒齡另限制為
`max(1, floor(seg_max_age × 實際來源 FPS + 1e-9))` 幀，與 profile／gap 上限取較小值。
0.1 秒在 30／60 FPS 為 3／6 幀，在 29.97／59.94 FPS 為 2／5 幀；不要先四捨五入 FPS。conservative 仍最多 3 幀。
顯式參數覆寫優先，與旗標先後順序無關。`--seg-max-gap 1` 只關閉分割跳幀；
`--cue-interval 0` 每幀更新排程選中的有效 Pose ROI，仍受主角優先／cue-top-k 限制；`--pose-max-age 0` 禁止跨幀沿用。

安全條件已接入，不要在測試前另行放寬：

- 肩髖至少 3 點、同時涵蓋肩／髖、幾何不塌縮，信心至少 `max(0.5, keypoint_conf)`。
  缺臉／不可靠手腕不等於整個 Pose 失效；可靠的大幅肢體動作仍阻止延長。
- gap >3 需要兩次相容真實 Pose 與近期穩定真實遮罩。balanced 穩定 0.5 秒評估 4 幀、1 秒評估 6 幀；fast 0.5 秒評估 6 幀。
  遮罩是面積／質心／頂部摘要，不是完整形狀匹配；光流主要處理平移，不是 3D 運鏡估計。
- 真實量測時間由 segmentation、Pose、neural cache 各自保存；光流、框映射與預測遮罩不刷新它。
  balanced／fast 在弱 Pose 但遮罩可靠時可退回最多 3 幀；conservative 的當幀真實弱 Pose 會要求下一幀偵測。
  證據不足、主角消失、光流失敗或畫面變化仍補偵測。
- 遠處小型已確認旁人可繼續低信任預測，不作為可靠構圖／延長證據。新人物、靠近、交錯或搶鏡仍重新評估。
- 新風險能撤銷 neural cache／降級等待，必要時越過冷卻；同一持續風險不每幀重置冷卻。
  便宜方法足夠時仍可停在 L2；沒有跨暗場／長遮擋的 ID 強接。

| 舊旗標（相容接受、一般 help 隱藏） | 現代入口／既有換算 |
| --- | --- |
| motion-response／motion-damping | `--pan-time`，沿用原 response/damping 換算，不直接當秒數 |
| zoom-response／zoom-damping | `--zoom-time`，沿用原換算 |
| target-alpha | `--target-time`，`-(1/30)/log(1-alpha)`，保留原邊界處理 |
| zoom-alpha | `--zoom-speed`，×30 |
| max-step-x／max-step-y | `--pan-speed-x/y`，分別 ×30/1920、×30/1080 |
| max-missed-frames／min-subject-hold-frames | `--lost-hold-seconds`／`--subject-hold-seconds`，÷30 |
| saliency-trust-repo | 不再需要，相容接受但無作用 |

現代參數優先；每次命令只印一則合併遷移提示，Summary `legacy_flags` 可供追查。

## 程式與測試入口

| 範圍 | 入口 |
| --- | --- |
| profile／CLI 遷移 | [config.py](../reframe/config.py)、[cli.py](../reframe/cli.py) |
| 共用身體／手部可靠度、Pose | [reliability.py](../reframe/perception/reliability.py)、[pose.py](../reframe/perception/pose.py) |
| 真實遮罩、分割／光流排程 | [contracts.py](../reframe/contracts.py)、[segmentation.py](../reframe/perception/segmentation.py) |
| 三級顯著性／DeepGaze | [saliency](../reframe/saliency)、[pipeline.py](../reframe/pipeline.py) |
| 4K 驗證執行器／Colab 入口 | [youtube_validation.py](../scripts/youtube_validation.py)、[notebook](../notebooks/colab-youtube-validation.ipynb) |
| 新 T4 收集器／Colab 入口 | [colab_torchcodec.py](../scripts/colab_torchcodec.py)、[notebook](../notebooks/colab-torchcodec.ipynb) |
| 共用 validator／歷史消融測試 | [benchmark_colab.py](../scripts/benchmark_colab.py)，新收集器仍依賴此檔 |
| 針對音訊尾端回歸 | [colab_audio_tail.py](../scripts/colab_audio_tail.py)、[既有 notebook](../notebooks/colab-audio-tail.ipynb) |
| 本機與 4K 回歸測試 | [tests](../tests)，`python -m unittest discover -s tests -v` |
音訊 notebook 含使用者既有輸出，保留原檔；其中舊安裝／修補 cell 需先核對現有版本，不能盲目重套。
目前主路徑採新收集器，不需要同時再跑舊音訊全長測試矩陣。

開發基準 Python **3.13+**。原雲端可用 `/workspace/venvs/reframe-py313/bin/python`；
其他機器自行使用相容 venv，此路徑不是 repo 要求。測試真權重時指定 `REFRAME_MODEL_DIR`；
`DEEPGAZE_REFERENCE_REPO` 指向已展開、通過測試檔案 SHA 校驗的 upstream
`7beab05d96a36eb7b343995a10d5e91d46e52ddf`，不是空 checkout。
另設 `TORCH_HOME` 使用既有權重快取，避免混入冷下載。只有刻意冷啟動測試使用另一獨立目錄。
未提供真模型／reference 時會有額外 skips，不能直接宣稱重現 361 通過／5 CUDA skips。

## 如何讀數據與繼續工作

Summary 是整片統計；本次收集器固定每 5 秒及切鏡輸出 Segment 區段差值。
核心 CLI 依 `--stats-interval` 設定，預設 0 關閉。先看有效 `tracking_settings`，再對照同一時間段。

| 欄位 | 判讀注意 |
| --- | --- |
| seg_detector_calls／seg_predicted_frames | 真實 YOLO 與預測影格；省略 YOLO 不代表省略輸出幀 |
| seg_detector_gap_counts／seg_interval_counts | 前者是實際偵測間距，後者是排程間隔分布 |
| seg_mask_evidence_counts／seg_pose_feedback_counts | 真實遮罩穩定性與姿態回饋原因 |
| seg_measurement_age_counts | 以 track 觀測計數，不是影片幀數 |
| pose_extension_hits／extension_reasons／refresh_reasons／cache_age_bins（pose_ 前綴） | 以 ROI 更新／沿用計數，區分軟延期與硬到期 |
| saliency_risk_escalations／cooldown_bypasses／risk_reasons（saliency_ 前綴） | 一個事件可有多個原因，不可把 reason 總數當事件數 |
| saliency_tier_*／saliency_actual_forward_calls | L3 幀可沿用快取，不等於 DeepGaze forward 次數；0 calls 為 not_exercised |
| subject_switches／locked_subject_missing | 前者不是原 ID 遺失次數；後者是層級原因，並非所有遺失影格的完整計數 |
| pipeline／saliency 各層 wall timing | 同層依 validator 核對；模型項目嵌套，不能跨層相加。host wall 不是 CUDA kernel 時間 |

原生 stdout/stderr、traceback、FFmpeg 錯誤必須保留。總時間需同時觀察初始化、解碼、輸出、縮圖、排程及模型。
資料流效能與排程效能分開比較；沒有相符對照時只報觀測值，不報加速百分比。

新結果寫入 `docs/reports/t4-YYYY-MM-DD-<short-sha>.md`，另保存小型 JSON 指標。
模型、MP4、大型 log／ZIP 放持久儲存，報表記錄路徑、SHA 與可取得性；不得提交憑證或假設暫存目錄永久存在。
每次停止前記錄：已完成的測試 ID／精確版本與設定／證據／失敗原因／下一個命令。
收集器只重用完整成功的工作，失敗從第一幀重跑；詳細規則見 [T4 規格](t4-validation.md)。
程式變更後更新驗證範圍；CPU 通過、CUDA skip 或未觸發模型都不能標成 GPU 通過。

## 文件清理與保留證據

本次將八份分散的舊進度／測試作業說明移出工作樹：原 `MR_VALIDATION.md`、
`colab-audio-tail.md`、`colab-long-video.md`、`colab-torchcodec.md`、`pipeline-optimization-options.md`、
`scheme-b-progress.md`、`torchcodec-progress.md`、`t4-validation-pending.md`。
已完成結果整理為報表；待測責任全部合併至本文件與 T4 規格，舊文字仍可由 Git 歷史查閱。

- [早期驗證報表](reports/historical-validation-through-2026-10-10.md)：已完成的歷史量測，不是現版操作指令。
- [2026-10-08 T4 原報表](colab-t4-results-2026-10-08.md)：保留原始版本上下文，其中舊旗標不可直接重跑。
- [目前實作與 CPU 報表](reports/implementation-validation-2026-10-12.md)：已完成範圍、證據 SHA 與限制。

保留所有 regression tests、三個收集／測試腳本、安裝程式及兩個 notebook；這些仍有測試用途。
沒有清除影片、模型、上傳檔或原雲端 runtime 證據。

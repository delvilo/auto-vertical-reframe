# TorchCodec 方案②：分階段工作與續作紀錄

更新日期：2026-10-12（Asia/Taipei）

## 已授權目標

使用 TorchCodec GPU 解碼與影像前處理，保留 FFmpeg／NVENC 編碼與已修正的音訊尾端處理。
分兩階段交付，並先精簡 DeepGaze：直接載入現有 checkpoint，避免額外下載完整 VGG19。
torchvision 仍是 Ultralytics 的依賴；torchaudio 不是本專案依賴。

## 開始點

- 基準 commit：`032cba5`，包含使用者在 Colab 更新的音訊測試 notebook。
- Python 3.13+；本機 `/workspace/venvs/reframe-py313/bin/python`，PyTorch 2.11 CPU，沒有 T4。
- T4 已驗證基準：3D 影片 10,252 幀／60 FPS；固定鏡頭影片 14,557 幀／60 FPS。
- 3D 影片視覺焦點正確；背面、手遮臉／身體、手部進出畫面都可能降低 Pose 品質。
  DeepGaze 的使用合理，不應為降低呼叫數而放寬主體判斷。
- 持續保留 native stdout/stderr、模型裝置與計時；不宣稱本機 CPU 測試證明 T4 加速。

## 里程碑

| 里程碑 | 工作 | 狀態 |
| --- | --- | --- |
| M0 | 交接紀錄、確認遠端基準 | 完成 |
| M1 | 只建立所需 VGG 特徵層；strict 載入 DeepGaze checkpoint 與 center bias | 完成，CPU 真實權重比對通過 |
| M2／第一階段 | TorchCodec 可選解碼器、準確幀序、小容量緩衝、共用縮圖與傳輸診斷 | 完成，CPU 端到端通過；NVDEC 待 Colab |
| M3／第二階段 | GPU 裁切／縮放與模型前處理；小尺寸 CPU 分析資料，FFmpeg 音訊輸出 | 完成，CPU 整合通過；CUDA 待 Colab |
| M4 | Colab 分階段收集器、回歸紀錄與提交 | 完成，CPU 三組／續作／備份驗證通過 |
| M5 | 大圖縮小時只取所需像素，減少浮點暫存 | 實作完成，CPU 幾何通過；CUDA 待測 |
| M6 | DeepGaze 小圖窗口合併預處理／複製 | 實作完成，本機回歸通過；T4 延後 |

每個里程碑獨立 commit；各次提交更新本檔。先以可選方式提供 GPU 路徑，T4 實測後再決定預設值。
GPU 解碼不得以 CPU fallback 冒充成功；所有路徑保留原始 FPS、全數尾幀及 precrop 座標定義。

2026-10-12 後續決定：先持續優化，最後才集中進行 T4 測試，不再要求每個提交立即長片測試。
各階段的 GPU 驗證責任保留於 [T4 待驗證紀錄](t4-validation-pending.md)。
M5／M6 的大圖縮放暫存與 DeepGaze 窗口優化已完成本機階段。
下一輪推論排程方案見 [影像資料流優化方案](pipeline-optimization-options.md)：A／B／C 尚待選擇，
不要把討論方案當成已授權全面實作；本輪僅實作明確要求的 dead-zone 預設值變更。

## 續作方式

1. 先閱讀本檔、`AGENTS.md`，查看 `git status --short` 與最近 commits。
2. 使用現有 checkout，保留使用者的 notebook 修改；不要重建已完成的里程碑。
3. 依下方最新紀錄接續第一個未完成項目。驗證只重跑受新修改影響或尚未解決的項目。
4. 若沒有 CUDA，完成 CPU 邏輯／權重比對與 Colab 測試入口，清楚標記 GPU 驗證待執行。

## 最新工作紀錄

- 2026-10-12：全 preset 的 `--dead-zone` 改為 0，保留手動非零值；兩個 Colab 收集器同步，
  TorchCodec resume identity 加入 dead-zone，Summary 記錄平移／縮放死區。
  原本 hardcoded 0.06 可用參數重現；比較新舊速度與品質時需對齊有效值。
  排程／Pose／DeepGaze 本輪未改邏輯，參數尚未移除；A／B／C 與精簡建議已保存，
  T4-D0 與後續 P1–P5 責任列入待驗證紀錄。
  本機 Python 3.13 的既有 collector／TorchCodec collector／CLI 與 pipeline 生命期測試共 52 項全部通過。
  另直接驗證四種 preset 預設與非零覆寫、三入口非法範圍／NaN／Inf 拒絕、命令傳遞及 resume identity 區別。
  未執行新的 GPU 或長片測試，T4 狀態保持待測。

- M0：已 fast-forward 至 `032cba5`；未改動使用者的既有 notebook。
- DeepGaze 已知事實：`deepgazemr-ledov.pt` 約 76.6 MiB，已含全部 32 個 VGG 特徵層權重張量。
  Center bias 約 7.0 MiB；額外的 548.1 MiB ImageNet VGG19 checkpoint 可省略。
- M1：25 項相關測試通過。既有 checkpoint 的 47 個 state tensors 與 pinned upstream 一致；
  16×3×64×96、16×3×96×64 兩組輸出逐元素相同（max_abs_error=0），保留 float64 center bias。
  嚴格檢查缺少／多餘 key 與錯誤 shape；不建立 classifier、不執行遠端 Python 程式碼。
  主要收益是冷啟動下載、載入與 CPU 峰值記憶體；不宣稱降低穩態 GPU forward 時間。
- M2：新增可選 `--video-decoder torchcodec --decode-device cpu|cuda:N`，每批 1–8 幀。
  拒絕未驗證 CUDA 硬體解碼／CPU fallback，驗證 CFR 時戳，保留原始 FPS 與幀序。
  預設仍 OpenCV；第一階段模型與輸出維持 NumPy，鏡頭小圖使用 tensor resize。
  Summary 新增 `video_io` 解碼、緩衝與像素傳輸診斷。
  11 個 reader、10 個像素／幾何、2 個設定測試通過；CPU 端到端影片輸出 90/90 幀、60 FPS，含音訊。
  實際非整數 area 縮圖採像素面積積分，測試與 OpenCV 差異不超過 1/255；不使用差異較大的 adaptive-average area。
- M3：`--frame-processing tensor` 已接上 YOLO segmentation／Pose、DeepGaze 16 幀小圖窗口、
  precrop 與輸出裁切／縮放。ByteTrack 保留原始 ROI 座標；拒絕依賴真實 CPU 畫素的其他 tracker。
  271 項回歸測試執行、267 通過／4 跳過（包含 2 個 CUDA 測試；沒有 GPU）。
  現有模型整合測試確認連續 ID 與不同尺寸 Pose ROI；真實 DeepGaze NumPy／tensor 窗口與 saliency map 相同。
  實際 CPU tensor 影片輸出 90/90 幀、60 FPS，YOLO 90 次、Pose 11 個 ROI、DeepGaze 4 次，無回退；
  此測試 saliency max-side=128、gap=3，只驗證功能，不能與先前不同設定比較速度。
  另測右側 precrop／debug／lossless 三幀輸出成功。獨立程式審查未找到阻擋問題。
  Summary 的 CUDA allocator peak 僅涵蓋 PyTorch、為程序生命期峰值，不含所有外部 NVDEC／NVENC 配置。
- M4：新增 `notebooks/colab-torchcodec.ipynb`、`scripts/colab_torchcodec.py` 與
  `docs/colab-torchcodec.md`。預設 baseline／stage1 各 180 幀一次，stage2 另啟用；
  完整影片一次只跑一組，必須有同來源／模型／程式／環境／設定的成功短測。
  相同成功紀錄與成片 SHA-256 驗證後才跳過；失敗保留 native log、Segment、traceback，從該組第一幀重跑。
  三組一致使用 gray-area／primary 排程；只改解碼／影像處理路徑。保留 torchvision，不新增 torchaudio。
  TorchCodec CUDA wheel 安裝採 `--no-deps --force-reinstall`，不替换既有 torch／torchvision。
- 最終驗證：288 項測試執行，286 通過／2 個 CUDA 測試跳過；包含真實 DeepGaze checkpoint
  與 pinned upstream 的逐元素相同比對。新增 notebook 七個 code cells 可編譯、未偽造執行輸出。
  Colab 收集器在 CPU 跑 baseline／stage1／stage2，均為 90/90 幀、60 FPS；
  影片 1.500 秒／音訊 1.493 秒。再次執行三組皆跳過，備份 results.json 與來源逐 byte 相同。
  這組 collector 測試沒有觸發 DeepGaze，明確標成 not_exercised；M3 的另一組實測已有 4 次 forward。
- 驗證檔案：`/workspace/reframe-runtime/validation/torchcodec-final-tests.log`、
  `torchcodec-stage2.log`、`torchcodec-collector-cpu-20261012/`、`torchcodec-collector-resume-20261012.log`。
  極短 12 幀測試暴露既有 FFmpeg AAC 音訊超尾；收集器確實拒絕，未放寬完整影片的音訊驗證。
- 續作環境：已實際執行 `/workspace/reframe-runtime/setup-torchcodec-cloud.sh`，
  pip check、CPU encoder 與 11 個 TorchCodec reader 測試通過；安裝及啟動指引已更新環境設定草稿。
  草稿尚需在環境設定儲存並發布，才成為後續任務的設定；不影響目前程式與 Git 提交。
- 原 M1–M4 的外部驗證保留，依最新決定延後至優化收斂後；細項及省時執行方式見 T4 待驗證紀錄。
  沒有 T4 實測前，不改預設值、不宣稱 GPU 加速比例或無 CUDA 問題。
- M6：tensor window 每次最多兩段連續批次，直接正規化至現有 float32 ring，不配置額外 float batch。
  原觀測幀数、時間順序、增量／環繞更新、NumPy 路徑與 lazy load 保持。
  新增 batch 計數供最終 T4 對照；原跨裝置 copies 計數仍表示影格數。
  本輪 `test_deepgaze*.py` 執行 21 項，18 通過／3 CUDA 跳過。
- M5：CUDA 雙線性大幅縮小採先取樣 uint8 再轉小型 float 暫存，限面積 ≤1/8、两軸不放大、來源最大邊 ≤8192。
  CPU 與放大維持原路徑。座標快取上限 16，不留全域影像；新增 `bilinear_sampled_calls`。
  CPU 對原實作畫素差至多 1/255；GPU 誤差、峰值與速度全部保留 T4 待驗證，未當作已加速。
  實際新窗口 pipeline 輸出 32 幀／60 FPS、真實 DeepGaze 1 次、16 幀一次更新、0 fallback；
  log：`/workspace/reframe-runtime/validation/tensor-batched-window.log`。
- M5／M6 合併後完整回歸：298 項執行、293 通過／5 個 CUDA 專用測試跳過。
  包含現有 YOLO 模型與真實 DeepGaze／upstream 精確比對；不代表任何新 CUDA 路徑已在 T4 通過。
  log：`/workspace/reframe-runtime/validation/tensor-optimizations-final-tests.log`；
  resize helper 的 CPU 量測／暫存估算：同目錄 `tensor-resize-cpu-20261012.json`。

## 已建立的程式里程碑

- `e785221`：M0 續作紀錄。
- `0c9b154`：M1 精簡 DeepGaze checkpoint 載入。
- `3b476df`：M2 可選 TorchCodec 解碼。
- `51c45e0`：M3 tensor 模型輸入與輸出處理。
- `f9ac46a`：M4 Colab 分階段測試。
- `d28920a`：保留各階段 T4 項目，延後集中測試。
- `46bcbfb`：M6 DeepGaze 小圖窗口批次預處理。
- M5：含本次紀錄的大圖縮小暫存優化提交（以 `git log` 查詢 SHA）。

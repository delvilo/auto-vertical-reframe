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
| M4 | Colab 分階段收集器、回歸紀錄與提交 | 待實作 |

每個里程碑獨立 commit；各次提交更新本檔。先以可選方式提供 GPU 路徑，T4 實測後再決定預設值。
GPU 解碼不得以 CPU fallback 冒充成功；所有路徑保留原始 FPS、全數尾幀及 precrop 座標定義。

## 續作方式

1. 先閱讀本檔、`AGENTS.md`，查看 `git status --short` 與最近 commits。
2. 使用現有 checkout，保留使用者的 notebook 修改；不要重建已完成的里程碑。
3. 依下方最新紀錄接續第一個未完成項目。驗證只重跑受新修改影響或尚未解決的項目。
4. 若沒有 CUDA，完成 CPU 邏輯／權重比對與 Colab 測試入口，清楚標記 GPU 驗證待執行。

## 最新工作紀錄

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
- 接續工作：M4 Colab 收集器、notebook／說明與測試，完成後推送；T4 短測／完整影片視覺比較仍待使用者執行。

# TorchCodec 方案②：分階段工作與續作紀錄

更新日期：2026-10-11（Asia/Taipei）

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
| M1 | 只建立所需 VGG 特徵層；strict 載入 DeepGaze checkpoint 與 center bias | 待實作 |
| M2／第一階段 | TorchCodec 可選解碼器、準確幀序、小容量緩衝、共用縮圖與傳輸診斷 | 待實作 |
| M3／第二階段 | GPU 裁切／縮放與模型前處理；小尺寸 CPU 分析資料，FFmpeg 音訊輸出 | 待實作 |
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
- 接續工作：M1 與 M2 的獨立模組。

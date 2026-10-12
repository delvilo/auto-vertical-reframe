# 方案 B：實作與續作紀錄

更新：2026-10-12（Asia/Taipei）。起點：`124a1da`。

使用者已選定 **B 安全自適應＋舊參數相容遷移＋追蹤 profile**。
離線雙向插值 C 不在本輪範圍。T4 測試仍延後集中進行，每階段責任保留於
[T4 待驗證紀錄](t4-validation-pending.md)。

## 執行順序

1. 共用軀幹／臉／肢體可靠度、真實遮罩量測來源；背面與手部異常分開判斷。
2. 主角風險排程、受真實分割年齡限制的間隔；Pose 軟更新與硬期限分離。
3. 顯著性風險升級與穩定降級，保留有效第二人顯著性及原順序 16 幀窗口。
4. conservative／balanced／fast profile、顯式覆寫優先、舊參數相容遷移。
5. 接入 Summary／五秒 Segment／收集器 identity，驗證並記錄 T4 待測。

## 本輪狀態

正在實作。先查看 `git status --short`、最近提交與各項測試結果；不要覆蓋使用者的音訊 notebook。
Python 3.13 開發環境：`/workspace/venvs/reframe-py313/bin/python`，目前無 CUDA。

| 階段 | 檔案責任 | 狀態 |
| --- | --- | --- |
| B1 可靠度／快取 | perception/reliability.py、pose.py | 實作中 |
| B2 遮罩／分割排程 | contracts.py、perception/segmentation.py | 實作中 |
| B3 三級切換 | saliency/cascade.py、factory.py | 實作中 |
| B4 設定／相容遷移 | config.py、cli.py | 完成；24 項相關測試通過 |
| B5 整合／收集器 | pipeline.py、precrop.py、segments.py、scripts | 實作中 |

## 不變條件

- 原 FPS、逐來源影格輸出、音訊尾端處理、precrop 定義、dead-zone=0。
- segmentation／Pose／neural cache 各自保存真實量測時間，預測不能刷新。
- 顯存內不保存全片；沒有新增稀疏 DeepGaze 時間窗口或跨場景 ID 強接。
- CPU 測試與 CUDA skip 不當作 T4 通過；不承諾未量測的加速比。

## B4：profile 與相容遷移

| Profile | 分割候選間隔上限 | 真實分割最大年齡 | Pose 軟更新 | Pose 硬期限 |
| --- | --- | --- | --- | --- |
| `conservative` | 3 幀 | 0.1 秒 | 0.2 秒 | 0.2 秒 |
| `balanced`（預設） | 6 幀 | 0.1 秒 | 0.2 秒 | 0.35 秒 |
| `fast` | 6 幀 | 0.1 秒 | 0.25 秒 | 0.4 秒 |

全部保留 saliency interval=3。間隔與硬期限是上限，不保證每次都沿用；
`--seg-max-age`、`--cue-interval`、`--pose-max-age` 等顯式覆寫優先，與旗標前後順序無關。
`--pose-max-age 0` 禁止跨幀沿用；`--cue-interval 0` 每幀更新。
profile 管理追蹤排程，原 `--preset` 仍管理物件類別及相機設定。

舊 motion／zoom response、damping、alpha、max-step、frame hold 與 saliency-trust-repo
旗標仍可使用，但不再列於一般 help；明確使用時每次命令顯示一則合併遷移說明。
現代秒數／速度參數優先，舊數值保留原換算，`legacy_flags` 可供追查。

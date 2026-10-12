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

核心與收集器已實作並完成本機驗證；T4 延後集中測試。先查看 `git status --short`、最近提交與各項測試結果；
不要覆蓋使用者的音訊 notebook。
Python 3.13 開發環境：`/workspace/venvs/reframe-py313/bin/python`，目前無 CUDA。

| 階段 | 檔案責任 | 狀態 |
| --- | --- | --- |
| B1 可靠度／快取 | perception/reliability.py、pose.py | 完成；54 項相關測試通過，提交 `5ddcf1a`，T4 待測 |
| B2 遮罩／分割排程 | contracts.py、perception/segmentation.py | 完成；40 項分割與 11 項 primary 測試通過，提交 `5ddcf1a`，T4 待測 |
| B3 三級切換 | saliency/cascade.py、factory.py | 完成；37 項 cascade／timing 測試通過，提交 `5ddcf1a`，T4 待測 |
| B4 設定／相容遷移 | config.py、cli.py | 完成；24 項相關測試通過，提交 `7bf651a` |
| B5 整合／收集器 | pipeline.py、precrop.py、segments.py、scripts | 完成；52 項 collector 測試通過，最終整體驗證見下方 |

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

| 舊參數（仍接受） | 建議入口與換算 |
| --- | --- |
| motion-response／motion-damping | `--pan-time`；保留原 response/damping 換算，不將任意舊值直接當秒數 |
| zoom-response／zoom-damping | `--zoom-time`；保留原換算 |
| target-alpha | `--target-time`；`-(1/30) / log(1-alpha)`，原本 alpha 接近 1 的限制保持 |
| zoom-alpha | `--zoom-speed`；舊值 ×30 |
| max-step-x／max-step-y | `--pan-speed-x/y`；分別 ×30/1920、×30/1080，單位為來源寬／高比例每秒 |
| max-missed-frames／min-subject-hold-frames | `--lost-hold-seconds`／`--subject-hold-seconds`；舊值 ÷30 |
| saliency-trust-repo | 不再需要；相容接受但無作用，新收集器不再產生 |

## B1–B3：實際安全條件

- 肩髖需至少 3 點、涵蓋肩／髖兩組、幾何不塌縮，信心至少 `max(0.5, keypoint_conf)`。
  此門檻比舊 cascade 的 0.35 嚴格，同時移除對額外臉／手點的依賴；DeepGaze 次數不保證各素材都下降。
- 身體與肢體位移分開計算；不可靠手點不等於整體失效，可靠的大幅手／腳動作仍阻止延長。
- 遮罩比較使用已下載的面積／質心／頂部摘要，正規化到人物框後比較。沒有新增全尺寸遮罩下載；
  這些摘要不是完整輪廓相似度，仍需靠光流、Pose、畫面變化與短時效共同保護。
- 分割間隔 >3 需要兩次相容的真實 Pose 與近期真實遮罩證據。balanced 於穩定 0.5 秒評估 4 幀、
  1 秒評估 6 幀；fast 於 0.5 秒評估 6 幀。來源時間限制始終優先。
- 弱 Pose 但可靠遮罩可退回最多 3 幀，避免只因背面或手點弱就永遠逐幀重跑分割；
  同時弱 Pose／遮罩、主角遺失、光流失敗或畫面變化仍會補偵測。
- 遠處小型且已確認的旁人可以用低信任預測繼續推進 tracker；它不作為可靠構圖或延長快取證據，
  原始 ID 保留給風險監視。新人物、近距離、交錯、面積／顯著程度上升會重新評估。
- 出現新的風險會撤銷神經快取與降級等待，資訊不足時可越過冷卻執行 DeepGaze；
  同一持續風險不每幀重置冷卻。便宜顯著性足夠時仍可停在 L2，16 幀模型窗口必須先備妥。
- 未加入跨暗場／遮擋的 ID 強接或離線插值。光流仍主要處理平移，3D 旋轉可能合理回退偵測。

## 使用與最終 T4 安排

一般命令預設 balanced，可明確寫出：

```bash
python -m reframe input.mp4 output.mp4 --tracking-profile balanced --device 0
```

既有其他設定可照常附加，例如 precrop、鎖定主角及編碼器。只要想關閉偵測跳幀，可加
`--seg-max-gap 1`；這不會同時關閉 Pose 快取。完全逐幀 Pose 另用 `--cue-interval 0`。

Colab 的 `--profiles baseline/stage1/stage2` 是資料流 profile；`--tracking-profile` 是推論排程。
Notebook 保留 `SEG_MAX_GAP=1` 用於原本資料流比較；測方案 B 時改為 `None`，讓各追蹤 profile 採自己的預設。
請保持同一來源、模型、精度、dead-zone、編碼器與資料流路徑；不同設定會產生不同短測／續作 identity。

優化完成後，先在同一條已通過 GPU 短測的資料流路徑比較 **conservative 與 balanced**，
各一次短測，再各一次完整影片；fast 僅於 balanced 品質合格後選擇補測。
使用原收集器即可，以下是未執行的命令模板，不會自動安排多輪長片：

```bash
python scripts/colab_torchcodec.py --input /content/video.mp4 \
  --seg-model /content/yolo26n-seg.pt --pose-model /content/yolo26n-pose.pt \
  --output-dir /content/scheme-b --profiles stage2 --tracking-profile balanced --max-frames 180
# 同設定短測成功，且人工檢查後才另外執行：
# 將 --max-frames 180 改成 --full-video
# conservative 對照僅更換 --tracking-profile，其餘設定維持一致
```

保留 170 秒 3D 全片與固定鏡頭 205 秒後背面／手部片段；開頭 180 幀不能取代晚出現情境。
每個 T4 階段責任見 [待驗證清單](t4-validation-pending.md)，目前均未在 T4 執行。

## 診斷欄位與判讀

Summary 的 `tracking_settings` 是展開後的有效設定。以下為累計欄位，在 Segment 內記錄該段差值：

- `seg_mask_evidence_counts`：穩定／改變或首次／缺少或無效的真實遮罩觀測。
- `seg_measurement_age_counts`：各個回傳 track 的真實分割年齡；**數的是人物觀測，不是影片幀數**。
- `seg_interval_counts`：排程間隔分布；Segment 原 `seg_detector_gap_counts` 才是兩次真實偵測的實際間距。
- `seg_secondary_prediction_frames`：包含低信任旁人預測的影格數；`seg_pose_feedback_counts` 是姿態回饋原因。
- `pose_extension_hits`、`pose_extension_reasons`、`pose_refresh_reasons`：Pose 超過軟時間後沿用、拒絕與重推論原因。
- `pose_cache_age_bins`：所有沿用 Pose 的年齡分布，按 ROI 計次；含軟時間內及延長部分。
- `saliency_risk_escalations`、`saliency_cooldown_bypasses`、`saliency_risk_reasons`：新風險事件與實際越過冷卻次數；
  一個事件可能包含多個原因，原因加總不一定等於事件數。
- `saliency_pose_reliability_reasons`：軀幹／局部肢體與運動原因；不是互斥的影片影格分類。

新欄位出現時收集器驗證分段加總；歷史 log 缺少新欄位仍可解析。保留原始 stdout/stderr、traceback 與失敗紀錄。

## 本機證據（不等於 T4 通過）

- Python 3.13.5、torch 2.11.0+cpu、torchcodec 0.17.0+cpu；本機沒有 CUDA。
- 真實兩個 YOLO 模型＋DeepGaze 的 CPU tensor pipeline：90／90 幀、60 FPS，3 個 Segment 的舊／新累計統計守恆。
  YOLO 90 次、Pose 11 個 ROI、DeepGaze 4 次，無 fallback。此多人近距離短片因低信心／競爭人物保留密集偵測，
  不當作跳幀加速證據；Pose 延長 0 次，延長功能由受控測試涵蓋。
  測試 saliency max-side=128、180×320 輸出，只證明功能，不與 T4 正式設定比較速度。
  視訊 1.5 秒、音訊 1.6 秒，尾端差在既有 0.1 秒容許邊界；正式長片音訊仍待核對。
  證據：`/workspace/reframe-runtime/validation/scheme-b-tensor.log`、`scheme-b-tensor.mp4`、`scheme-b-cpu-evidence.json`。
- 模擬時序測試涵蓋 60 FPS 實際偵測 gap=6、30 FPS gap≤3、倒退／切鏡、遮罩被錯誤刷新、手部異常、
  新／近／突出旁人、畫面變暗與光流失敗等；不以受控模型輸出代替真實長片品質驗證。
- Notebook JSON 與全部 Python cell 語法檢查通過；沒有執行 Colab 或 GPU 安裝。
- 最終完整回歸：**366 項執行，361 通過／5 個 CUDA 專用測試跳過**，沒有失敗。
  包含現有真實 YOLO 權重整合、DeepGaze 參考架構全部 state tensors／兩組時間窗口數值比對。
  完整紀錄：`/tmp/scheme-b-final-tests-verified.log`。
  初次參考比對指定了只有 Git metadata 的目錄；改用既有 Hub 參考資料，檢查程式碼／權重 SHA-256 後重跑通過，
  沒有停用雜湊或略過權重驗證。

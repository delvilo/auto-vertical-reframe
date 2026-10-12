# 歷史驗證報表：截至 2026-10-10

整理日期：2026-10-13（Asia/Taipei）。本報表保留整理前 `MR_VALIDATION.md`、README 已完成測量與本機留存分析的結果；
不包含舊操作步驟，也不代表目前程式重新通過這些測試。
原文件可由 [71b4ed2 的 MR_VALIDATION.md](https://github.com/delvilo/auto-vertical-reframe/blob/71b4ed294100efcf81446cd3d50eb07c9f559a01/MR_VALIDATION.md)
及 [同版 README](https://github.com/delvilo/auto-vertical-reframe/blob/71b4ed294100efcf81446cd3d50eb07c9f559a01/README.md) 查核。
現行工作入口為 [AI agent 交接文件](../agent-handoff.md)，新一輪結果另見 [2026-10-12 本機報表](implementation-validation-2026-10-12.md)。

## 本機歷史測試

各列是不同開發階段的套件快照，**測試與 subtests 有重疊，不可累加成總測試數**。
這些數量保留原文記載，本次文件整理沒有重新執行。

| 階段／歷史提交 | 原記錄結果 | 主要覆蓋與限制 |
| --- | --- | --- |
| 恢復 DeepGaze MR／`5be6ff8` | 61 tests 通過；另記 5 個既有 security tests 與 `weights_only=True` AST 檢查通過 | backend 契約、16 幀觀測、reset、資源清理、診斷及真實 FFmpeg；pyflakes／whitespace／editable 安裝通過，套件 0.2.1 |
| Pose 優先三級策略／`856a1f2` | 105 tests、29 subtests 通過 | precrop 座標、Pose 優先、map 來源、三級切換、lazy temporal upload、FFmpeg／清理；這套測試本身未下載神經權重 |
| 動態分割＋光流／`6476628` | 143 tests、72 subtests 通過 | 真實 OpenCV flow、ByteTrack／Kalman 配合受控偵測、ID／frame clock、真實量測年齡、切鏡／幾何重設與失敗回退 |
| 延遲縮圖／`3fd016f` | 164 tests、84 subtests 通過 | eager／lazy 決策與框一致、相鄰 KLT、每幀最多縮圖一次、資源釋放、計時守恆與收集器失敗狀態 |
| 灰階縮圖／主角跳幀／細分計時 | 194 tests、126 subtests 通過 | 三層互斥 host 計時、primary 保護、真實 CPU 收集器與輸出驗證 |
| 完整影片／Segment 收集器，記錄於 `c79872d` | 217 tests、159 subtests 通過 | 非整數 FPS、切鏡／時間窗、跨 Segment gap、部分失敗恢復、計時對帳及 ZIP 內容 |

MR 回復階段原文記載 Python 3.13.16、torch 2.8.0+cpu、torchvision 0.23.0+cpu；
後續階段改記 Python 3.13.5、torch 2.11.0+cpu、torchvision 0.26.0+cpu。
共同記錄包含 NumPy 2.5.2、OpenCV 4.14.0.94、Ultralytics 8.4.173；三級策略驗證另記 FFmpeg 7.1.5。
這些是不同環境快照，不應併成同一套環境；舊 patch 版本此刻未重新驗證，原文件未附每次完整環境鎖定檔。

## 真實權重與影片功能證據

- 回復 MR 的真實 CPU 短測：16 張 RGB、96×64，輸出有限 float32 96×64 顯著圖；
  `model_loaded=true`、backend=deepgazemr、forward 1 次、fallback 0。
  舊 Hub loader 遇 CUDA storage 警告後，以既有 CPU 安全載入器讀相同權重；原警告／traceback 未隱藏。
- 新三級策略的獨立真實 MR 短測：前 15 幀不載入模型，第 16 幀載入並 forward 1 次，有限 128×64 map、fallback 0。
  首次 GitHub API 取得失敗後，使用官方來源的 Hub 快取；未停用 TLS、checksum 或載入保護。
- YOLO 真實模型整合使用 Ultralytics bus 靜態圖製成 640×360、12 FPS、AAC 影片；全幅／left／middle／right
  都輸出 12 幀 180×320 H.264/AAC 並成功解碼。全幅／left／middle 各 L1=9、L2=3；right 為 L1=0、L2=12；
  四組 DeepGaze 均未載入、forward=0。24 幀 middle 延伸測試 L1=21、L2=3、Pose 推論 ROI=15／省略=69，
  temporal ring 已滿 16 幀而 DeepGaze 仍未載入；post-restore 全部關閉。ROI 數不是影片幀數。
- lazy 收集器真實 CPU 短測完成六個程序（三組各暖機一次、正式一次）：每組保留 12 幀／12 FPS／音訊，
  JSON／CSV／中位數與 log 驗證通過。eager／lazy 都建 12 張縮圖，此短片不能證明節省比例。
- 灰階／主角排程階段兩個 60 幀 CPU render：收集器短片穩定後保持 pose-only、DeepGaze 標成 not_exercised；
  另一片真實 DeepGaze forward 3 次、fallback 0、YOLO 27 次／預測 33 幀。
  兩片皆 60 FPS、視訊 1.000 秒、AAC 0.981 秒、起點差 0；不代表移動素材品質或感知音畫同步已驗證。
  先轉 uint8 灰階再 area resize 的合成 precrop 測試，與原縮圖畫素差在 2 個亮度階內，並未宣稱 tracking 畫素完全相同。
- 長片收集器 CPU 整合：90 幀／60 FPS，兩組各 8 幀暖機及一次完整量測，四程序成功。
  每個完整輸出保留 90 幀與音訊、四個 0.4 秒／最後不足段 Segment；計數／時間對帳通過。
  暖機不納入中位數，ZIP 完整且只含 log／JSON／CSV／JSONL。此 fixture 未觸發 DeepGaze。

## 靜態 CPU 跳幀對照

90 幀／60 FPS 靜態 bus 素材、middle precrop、conf=0.45、180×320 libx264，權重已快取；
仍含各自程序與模型啟動。0.45 用來排除 fixture 的弱局部人物，並非當時的新預設。

| 測量 | gap 1 | gap 3 |
| --- | ---: | ---: |
| YOLO／預測幀 | 90／0 | 36／54 |
| 分割階段 | 8.498 秒 | 4.523 秒 |
| 光流（已包含於分割） | 0 | 0.296 秒 |
| 管線總時間 | 47.953 秒 | 41.793 秒 |
| Pose ROI／DeepGaze forward | 10／4 | 10／4 |
| 主體切換／最大預測年齡 | 0／0 | 0／0.0333 秒 |

兩片解碼畫素逐元素相同，皆 90 幀／60 FPS／1.500 秒；AAC 起點 0、長度 1.493 秒。
DeepGaze 各觀測完整 90 幀。YOLO 呼叫減少 60% **不是**總耗時減少 60%，且不能外推 T4 或移動影片。

## 使用者上傳的歷史 T4 測量

早期固定 DeepGaze 排程／post-restore 的 90 幀結果保留於 [2026-10-08 T4 報告](../colab-t4-results-2026-10-08.md)。
其中 legacy fallback=5 的逐次原因不足，不能斷言全為 warmup 或全為模型錯誤；它不驗證後續三級策略。

後續兩批均為前 600 幀／10 秒、4K／60 FPS、middle precrop、dead-zone=0.06、FP32 saliency、關閉 post-restore；
Python 3.13.15、torch 2.11.0+cu130／CUDA build 13.0、T4，YOLO／Pose／DeepGaze 實際 CUDA、HEVC NVENC。
每次獨立程序仍包含初始化，不能視為隔離 CUDA kernel benchmark。

| 批次／版本 | 組別 | 總秒數 | 分割秒數 | 縮圖秒數／張數 | YOLO／預測幀 |
| --- | --- | ---: | ---: | --- | --- |
| 四份 log，29 模組 hash 對應 `6476628`；第二輪 | gap1 | 117.139 | 18.100 | 舊版未細分 | 600／0 |
| 同上 | gap3 eager | 130.457 | 32.111 | 舊版未細分 | 587／13 |
| 九份 log，29 模組 hash 對應 `3fd016f`；排除三暖機、兩正式輪中位數 | gap1 lazy | 114.780 | 18.095 | 0／0 | 600／0 |
| 同上 | gap3 eager | 126.791 | 30.693 | 14.872／600 | 587／13 |
| 同上 | gap3 lazy | 120.721 | 25.014 | 8.194／329 | 587／13 |

第一批 gap1 首次下載 Hub／548 MB VGG，耗時 136.761 秒，不能拿來當暖快取基準。
第二輪 gap3 多花 13.318 秒；分割增加 14.011 秒，光流本身僅約 0.063 秒，不能把退步全歸因於光流。

第二批 lazy 較同版 eager 省總時間 4.79%，但仍比 gap1 慢 5.18%；縮圖少 271 張、耗時少 44.90%。
311 次當下延後中有 40 張後續補建，並非省 311 張；單張縮圖約 24.8–24.9 ms，改善來自減少張數。
lazy 的排程約 0.077 秒、光流約 0.065 秒，縮圖仍占 8.194 秒。saliency 約 55.54 秒占總量 46%，
另有約 26.76 秒當時未納入四階段；不能直接當成 I/O 或 GPU 純計算。

兩批 gap3 都只省 13/600 次 YOLO（2.17%），最長量測年齡 0.0333 秒；0.1 秒上限非主要限制。
第二批 eager／lazy 的偵測原因、Pose debug ROI 與模型呼叫相同：Pose ROI=328、MR forward=135、L1/L2/L3=124/216/260、
層級切換 246。各組 subject_switches=0 不等於沒有追蹤遺失；ZIP 無成片／逐幀 hash，不能宣稱畫面完全相同。
第二批 ffprobe 皆 600 幀／60 FPS、1080×1920、視訊 10 秒、AAC 9.98458 秒，起點均 0；未見 OOM／CPU fallback／fatal traceback。
只有兩次正式重複與一個片段，不宜跨批直接相減或外推全部影片。

## 長片中止與「6.8 小時」的原始背景

6.8 小時是舊收集器 **四組 ×（一次完整暖機＋兩次完整量測）＝十二遍完整影片**的估算：
170 秒 × 60 FPS × 12 ÷ 5 processing FPS＝24,480 秒，約 6.8 小時，另加啟動、下載與 I/O。
這不是單支影片耗時；單遍同假設約 34 分鐘，也不是目前建議的集中測試總時數。

2026-10-10 上傳的實際中止批次使用四次 600 幀短暖機＋八次全片，來源 10,252 幀，
總計畫量 84,416 幀；以 5 processing FPS 估約 4.69 小時。實際只完成暖機及第一個全片推論：

- gap1-baseline-run1 處理 10,252 幀／170.866667 秒，管線 1966.974 秒；YOLO=10,252、Pose ROI=4,759、DeepGaze=2,904，fallback=0。
- 輸入音訊較短（170.741995 秒），舊 mux 的 `-shortest` 使輸出只剩 10,244 幀／170.733333 秒。
  原程序 returncode=0，收集器因 frame count 不符退出 1；其餘七次正式測試未開始。
- 35 個 Segment 的 Summary 對帳通過；這是逐幀 YOLO 基準，不能證明 gap3 改善。
  2,848 次 DeepGaze 請求原因為 locked_subject_missing，但這不是遺失影格數，也無法單靠 log 判斷真人換人或同人 ID 變更。
- CPU FFmpeg 重現：2 秒／120 幀配 1.875 秒音訊，舊 mux 只得 112 幀；補音訊靜音後保留 120 幀。
  此證據解釋故障機制；後來的音訊修正在 `4e6f240`，不能把故障版本標成已修正版 T4 驗證。

## 上游來源、限制與證據保存

[DeepGaze MR 官方來源](https://github.com/mtangemann/deepgazemr/tree/7beab05d96a36eb7b343995a10d5e91d46e52ddf)：
原模型輸入為連續 16 幀、VGG19 特徵沿時間平均，光流不是額外模型輸入。
當時 adapter 對每個排程窗口呼叫 forward，未採 upstream predict 的逐幀特徵重用；新精簡載入亦不等於已完成這項重用。
原 Hub 曾跟隨預設分支，現行 pinned checkpoint 載入的證據另見 2026-10-12 報表。
舊 max-side 384 增至 768 會使像素量約四倍，當時沒有品質改善／T4 時間或顯存實測。

原始 log、MP4、JSON 及本機分析保存在 `/workspace/reframe-runtime/validation/` 的
`cascade-*`、`adaptive-*`、`thumbnail-collector-smoke-20261009/`、`three-optimizations-*`、
`long-video-collector-cpu-20261009/`、`colab-lazy-20261009/`、`colab-long-20261010/` 等路徑。
它們位於原雲端環境、未隨 Git 提交；重新 clone 不會取得，環境回收可能使原件消失。
保留的純報表是可攜摘要，不能代替已遺失的原始證據。

舊文混用「目前尚未 T4 驗證」與後來上傳成功的段落，各自指其當時版本；
本報表依版本／階段區分，未將這些時序記載視為目前版本的驗證。

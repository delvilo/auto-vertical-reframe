# Colab T4：尾幀／音訊回歸測試

使用 [直接在 Colab 開啟的 notebook](https://colab.research.google.com/github/delvilo/auto-vertical-reframe/blob/main/notebooks/colab-audio-tail.ipynb)，
或從 [GitHub notebook 頁面](../notebooks/colab-audio-tail.ipynb) 下載。
這次只修正短音訊讓 `-shortest` 提早截掉畫面的問題，保留現有主角、追蹤 ID、
Pose、三級顯著性與 DeepGaze 決策。第二人較搶鏡時仍可使用 DeepGaze。
結果表可填入短暫變暗、人物交錯或 3D 慢動作運鏡的時間區間，對照每五秒的
`locked_subject_missing` 與 DeepGaze 次數；這些統計不能單獨證明變暗是原因，
也不把追蹤 ID 缺失視為視覺焦點錯誤。

Notebook 內附 runner 及可核對 SHA-256 的最小修補；舊版 checkout 可直接套用，已修正版可重複核對。
在先前已安裝本專案的 Python 3.13+／T4 Colab 開啟，修改第一個 code cell 的影片與
兩個 YOLO 權重路徑，再依序執行。沿用現有 PyTorch／CUDA；不重新安裝套件。
來源版本不符時會停止並保留現有檔案，避免把不同構圖版本混進這次回歸。

1. 用兩秒、120 幀、60 FPS，且音訊刻意較短的合成影片測試 direct 與 lossless 輸出。
   驗證實際使用 HEVC NVENC、完整 120 幀、FPS 及最後八幀標記，不載入 AI 模型。
2. 通過後只執行一次完整 `gap1-baseline-run1.mp4`，沒有四組比較、模型暖機或重複測試。
   沿用原本 middle、lock-first-subject、bgr-area、all、FP32 saliency、關閉 post-restore。
3. 核對來源／輸出幀數、FPS、尺寸、音訊是否存在、模型實際 CUDA 裝置、每五秒分段
   與 Summary 的加總。保留原生錯誤、模型模組 SHA-256、修補差異及音畫結束時間。
4. 正常結束或可捕捉的失敗時，備份診斷 ZIP、runner log 與已產生的完整 MP4 到 Drive。
   硬性回收 runtime 仍可能失去尚未備份的資料。

先前同一影片基準約 **33 分鐘**；這次另加兩秒 mux 測試、啟動、ffprobe 和備份時間。
若模型權重尚未快取，仍需下載；不以減少 DeepGaze 呼叫數作為成功條件。

先前影片是 10,252 幀／60 FPS／170.866667 秒。若選用同一檔案，修正版應保留
**10,252 幀**，而非先前的 10,244 幀。實際驗收值由來源 ffprobe 取得，不硬編碼。
`apad` 在短音訊尾端補靜音，保留原畫面；沒有複製尾幀來湊數。
FFmpeg／AAC 緩衝可能留下短的音訊尾端，報告會列出音畫結束時間差，不宣稱逐取樣同步。
請另聽看影片最後一秒，以及人物交錯片段；自動幀數檢查不能證明視覺構圖和音畫同步。

本機 Python 3.13.5／FFmpeg 7.1.5 的真實編碼測試涵蓋兩條輸出路徑的短、等長、長、
無音訊與部分影片。GPU／NVENC 的最終證據由這份 Colab 測試收集。

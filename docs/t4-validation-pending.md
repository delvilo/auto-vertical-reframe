# T4 待驗證紀錄（優化完成後集中執行）

更新：2026-10-12（Asia/Taipei）

依使用者決定，先持續優化及本機驗證，**暫不要求每次改動都重跑 T4**。
本表保留每階段的 GPU 驗證責任；CPU 通過、CUDA 測試被 skip、沒有觸發 DeepGaze，均不能改成 T4 通過。
目前全部 T4 項目仍待測。後續改動應新增或更新列，保留歷史證據，不刪除未完成項目。

## 版本與執行原則

- 本輪優化前版本：`f9ac46a`；M1 `0c9b154`、M2 `3b476df`、M3 `51c45e0`。
- 最終測試時固定 commit、Python／torch／torchvision／torchcodec／CUDA／driver／FFmpeg 版本，
  來源影片與模型 SHA-256。不要拿不同影片、gap 或 saliency 大小的耗時宣稱加速。
- 保留 checkpoint FP32 與原三級決策、ByteTrack、全數輸出幀／FPS、推論用 precrop；
  不以降低畫質、少處理影格或改變主角判斷取代運算優化。
- 每項填寫「測試 commit、設定、證據路徑、實際結果」。沒有證據的項目保持待測。
- 若最終版本出現回歸，才針對相關 milestone 做比較；不預設把每個中間版本都跑完整長片。

## 各階段待測項目

| ID／階段 | T4 必須確認 | 需要保留的證據 | 狀態 |
| --- | --- | --- | --- |
| T4-M1 載入精簡 | 使用獨立冷快取目錄確認只取得 DeepGaze checkpoint／center bias，未取得完整 VGG；實際 CUDA forward 成功，FP32 結果與原架構無異常差異 | artifact SHA、下載檔案／大小、`saliency_model_provenance`、init／forward 計時、實際裝置與呼叫數；數值比較結果 | 待測 |
| T4-M2 解碼 | 實際 NVDEC、`cpu_fallback=false` 且 status known；1080p／4K、最後不足一批的影格完整，CFR 時戳／FPS 正確 | preflight、`video_io`、ffprobe frame count／FPS、末幀、decode／setup 時間 | 待測 |
| T4-M3A YOLO／Pose | GPU letterbox 後的框、mask、關鍵點正確回到 ROI／來源座標；三種 precrop、不同大小 Pose ROI、連續 ByteTrack ID 正常 | GPU 測試輸出、模型 device、成片／debug 抽查；與 reference 的框／關鍵點差異 | 待測 |
| T4-M3B DeepGaze tensor window | 小圖留在 GPU；完整 16 幀時間順序、ring wrap、跨切鏡 reset、久未使用後更新正確；實際模型有執行且無 fallback | CUDA window tests、`saliency_tensor_*`、16 幀來源／結果比對、真實 forward／fallback 計數 | 待測 |
| T4-M3C 輸出／顯存 | GPU crop／resize 成片無色彩或邊界異常；NVENC、原 FPS／全部尾幀、音訊終點正常；長片無 OOM／顯存持續累積 | MP4／ffprobe、音畫檢查、像素傳輸量、`cuda_allocator_peak`；必要時 nvidia-smi 外部顯存採樣 | 待測 |
| T4-M4 收集器 | 中斷時保留已 flush log／Segment，備份可用；同設定成功測試可跳過，變更程式／來源／設定會重跑；裝置降級會被拒絕 | `record.json`、`results.json`、診斷 ZIP、備份及 resume 輸出 | 待測 |
| T4-M5 大圖縮放暫存優化 | 大圖縮小的 CUDA 畫素誤差、邊緣／非整數縮放與原路徑可接受；peak temporary memory／縮放耗時改善；YOLO／Pose 結果無異常漂移 | 舊／新 resize 的 CUDA event 計時及 peak delta；畫素 max／mean error、框／關鍵點差異 | 待實作／待測 |
| T4-M6 窗口批次預處理 | pending=0／1／3／16、跨 ring 尾端的窗口仍相同；小圖批次化實際減少操作成本，沒有增加不必要的上傳或重算 | 新舊 window／saliency 比對、preprocess batch 與 frame 數、CUDA event 計時、跨裝置資料量 | 本機通過；T4 待測 |

`cuda_allocator_peak` 只涵蓋 PyTorch allocator，不能代表 NVDEC／NVENC 和其他程序的全部顯存。
host wall 計時包含不同程度的等待，不能直接視為 CUDA kernel 時間。比較 GPU 小操作時另用 CUDA events，
僅在測試邊界同步，不把逐幀同步加入正式 pipeline。

## 最終測試順序與較省時的安排

1. 固定最終程式版本，再使用 [Colab notebook](../notebooks/colab-torchcodec.ipynb)。
   先做依賴／NVDEC／NVENC 檢查與 GPU 單元測試；此時處理 M5／M6 的小操作比對。
2. `baseline`、`stage1`、`stage2` 各做一次 180 幀短測，對齊模型、gap、precrop、輸出尺寸等設定。
   短測以來源實際幀數為上限。不要用十幾幀的 AAC 成片代替正式音訊尾端驗證。
3. 若短測沒有觸發 DeepGaze，T4-M1／M3B／M6 的真實模型部分維持待測；
   另用已知需要 DeepGaze 的片段或後續長片取得證據，不放寬三級策略來湊呼叫數。
4. 完整影片先比較最終版本的 `baseline` 與 `stage2`，各一遍、一次只跑一組。
   `stage1` 長片僅於需定位解碼／前處理回歸時補跑，避免三組多輪長片全部重做。
   相同程式與設定的成功紀錄可沿用；舊 commit 的測試不能被當作新 commit 已驗證。
5. 保留 170 秒 3D 影片的全程追蹤，查看兩人交錯、暗場、慢動作／旋轉運鏡；
   另以固定鏡頭片段涵蓋背面、手遮臉／身體及手部進出畫面。若有該 242.62 秒來源，特別核對 205 秒後。
   開頭短測不涵蓋這些晚出現的情境，不能替代長片的視覺檢查。
6. 對照每五秒與切鏡 Segment：真實 YOLO／預測幀、Pose 更新、DeepGaze 呼叫與 tier reason、
   主角切換及各階段計時。DeepGaze 在第二人搶鏡時可能是正確決策，不以呼叫數越少作為單一成功條件。

本次安排不要求現在執行上述 T4 作業。完成代碼優化後，再補齊一組集中驗證紀錄。

## 本輪優化的本機證據（不等於 T4 通過）

- M6：逐幀更新改為最多兩段連續批次，直接寫入既有 float32 ring；保留原本 16 幀觀測與 32 槽鏡像。
  包含跨尾端、完整刷新、增量／零更新、reset 與 NumPy 正規化相同比對。
  新增 `saliency_tensor_preprocess_batches`、`saliency_tensor_cross_device_batches`；
  原 `saliency_tensor_cross_device_copies` 仍是影格數，不能誤讀成批次操作數。
  三個 CUDA 專用測試仍因沒有 GPU 跳過，最終集中測試時必須補跑。
  由於正規化直接寫入 ring，dtype conversion／寫入計時歸入 preprocess；評估時同時看 preprocess＋transfer，
  不能把計時項目之間的移動當作速度收益。

## 結果登記

| ID | 測試 commit／環境 | 輸入與設定 | 證據路徑 | 結果／仍待確認 |
| --- | --- | --- | --- | --- |
| 全部 | 尚未在 T4 執行 | — | — | 保持待測 |

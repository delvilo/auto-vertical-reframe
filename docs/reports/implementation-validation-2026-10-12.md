# 影像資料流與方案 B：已完成實作及本機驗證報表

證據日期：2026-10-12（Asia/Taipei）。整理日期：2026-10-13。
交接程式基準：`71b4ed294100efcf81446cd3d50eb07c9f559a01`。

TorchCodec 分階段資料流、精簡 DeepGaze、方案 B 與舊參數相容遷移已完成本機階段。
**本報表沒有新的 T4 驗證結果；不能由 CPU 測試推論 CUDA 正確性、顯存峰值或加速幅度。**
後續責任見 [AI agent 交接文件](../agent-handoff.md) 與 [T4 驗證規格](../t4-validation.md)。

## 已完成里程碑

| 提交 | 已完成內容 | 已取得的證據與界線 |
| --- | --- | --- |
| `0c9b154` | DeepGaze 僅建立實際使用的 VGG 特徵層，直接 strict 載入既有 checkpoint | 全部 47 個 state tensors、其中 32 個 VGG 特徵張量與 pinned upstream 一致；不建立 classifier、不執行遠端 Hub Python |
| `3b476df` | 可選 TorchCodec 解碼、1–8 幀有界批次、CFR 時戳與 fallback 診斷 | CPU reader／尾批與完整幀序通過；NVDEC 仍無新實測 |
| `51c45e0` | tensor YOLO／Pose／DeepGaze 輸入、precrop、輸出裁切縮放 | CPU 真實模型與影片整合通過；CUDA 座標、色彩及記憶體仍待測 |
| `f9ac46a` | 分階段 Colab 收集器、來源／設定識別、續作與備份 | CPU 三路徑完整輸出、成功紀錄重用與備份一致性通過；GPU 裝置檢查仍需實機證據 |
| `46bcbfb` | DeepGaze 16 幀窗口改以最多兩段連續批次正規化到既有 ring | 本機窗口次序、增量、環繞與 reset 通過；保留每個來源影格的觀測 |
| `7de1838` | CUDA 大幅縮小先取 uint8 鄰點，再建立小型 float 暫存 | CPU 幾何誤差至多 1/255；CPU 不啟用此 CUDA 路徑，未量得 GPU 收益 |
| `124a1da` | 所有 preset／收集器的平移 `dead-zone` 預設為 0 | 非零覆寫仍可用；未改變平滑、速度限制及獨立變焦死區 |
| `7bf651a` | conservative／balanced／fast 追蹤 profile 與舊參數相容遷移 | 明確新參數優先、舊值換算保留；舊旗標仍接受並提示遷移 |
| `5ddcf1a` | 共用肩髖可靠度、局部肢體判斷、真實遮罩時效、主角風險排程及有限 Pose 延長 | 受控測試覆蓋背面、手部、60 FPS gap=6／30 FPS gap≤3、旁人、暗場、切鏡與光流失敗 |
| `71b4ed2` | profile 診斷接入 Summary／Segment／收集器與設定識別 | 新統計分段守恆、續作設定辨識與 notebook 語法檢查通過 |

DeepGaze checkpoint 約 76.6 MiB、center bias 約 7.0 MiB；省去約 548.1 MiB 的完整 ImageNet VGG19 權重。
這主要改善冷啟動下載、載入與 CPU 記憶體，並不代表穩態 GPU forward 變快。
精簡模型保留 float64 center bias；兩個窗口 `16×3×64×96` 與 `16×3×96×64`
對 pinned upstream 的 CPU 輸出皆逐元素相同，`max_abs_error=0`。

## 最終本機回歸

機器可讀摘要：[implementation-validation-2026-10-12.json](implementation-validation-2026-10-12.json)。

- Python 3.13.5、torch 2.11.0+cpu、torchvision 0.26.0+cpu、torchcodec 0.17.0+cpu、Ultralytics 8.4.173；無 CUDA。
- **366 項執行，361 項通過、5 項 CUDA 專用測試跳過，0 失敗；耗時 53.076 秒。** 不累加先前重疊的分組測試數。
- 涵蓋真實 YOLO 權重整合、DeepGaze 全量 state tensors／兩種窗口數值比對、排程、CLI 遷移、影片音訊與收集器。
- 五個跳過項目為兩個 CUDA sampled-resize／峰值測試及三個 CUDA DeepGaze 正規化／裝置傳輸測試。
  原始 log 中測試 fixture 的 `cuda:0`／NVENC 字串是模擬資料，不能當作 GPU 執行證據。
- 最終測試執行於整理提交前、已包含 collector 修改的開發工作樹；相關程式與測試對應上述三個方案 B 提交。
  提交前後僅補文件說明，沒有另外宣稱曾在 clean checkout `71b4ed2` 重新跑過同一套測試。

## 真實 CPU tensor 短片

摘要保存於 [scheme-b-cpu-evidence.json](scheme-b-cpu-evidence.json)。
這是同日開發工作樹的功能短測，沒有另外封存當時完整工作樹雜湊；不能冒充 T4 benchmark。

| 項目 | 結果 |
| --- | --- |
| 輸入／輸出 | `adaptive-input.mp4`，640×360；輸出 180×320、`precrop=middle`、libx264 |
| 影格與 FPS | 輸入／輸出皆 90 幀、60 FPS；視訊長度 1.5 秒 |
| 路徑 | TorchCodec CPU 解碼＋tensor 處理；實際裝置 CPU，fallback=false 且狀態已知 |
| 排程設定 | balanced，seg gap 上限 6／最大年齡 0.1 秒，Pose 軟更新 0.2／硬期限 0.35 秒 |
| 模型呼叫 | YOLO 90 次、Pose 11 個 ROI、DeepGaze 真實 forward 4 次；顯著性 fallback 0 |
| 實際跳幀／快取延長 | 分割預測幀 0、Pose 延長 0；低信心與鄰近競爭人物使這段保留密集偵測 |
| 顯著性 | L1 0 幀、L2 15 幀、L3 75 幀；L3 幀數包含快取沿用，不等於 forward 次數 |
| 時間窗口 | 90 幀皆觀測；64 幀分 7 個批次預處理；CPU 同裝置無跨裝置複製 |
| 統計 | 3 個 0.5 秒 Segment，舊／新累計欄位分段守恆 |
| 耗時 | 19.175729 秒；saliency max-side=128，僅作本機功能紀錄，不與正式 T4 設定比較速度 |
| 音訊 | 視訊終點 1.5 秒、音訊終點 1.6 秒；差 0.1 秒，在既有容許邊界，不能取代長片驗收 |

上述短片沒有實際觸發 extended gap 或 Pose 延長；這兩項目前只有受控時序測試證據。
沒有任何真實長片新結果證明其品質與處理時間已達預期。

## 權重與證據來源

DeepGaze 參考版本：`7beab05d96a36eb7b343995a10d5e91d46e52ddf`。
參考測試驗證程式與權重雜湊，未停用 strict 載入或略過真實權重比較。

| 檔案 | SHA-256 |
| --- | --- |
| `deepgazemr-ledov.pt` | `db0e3c4738664a6e8026229b8d1a74edda9ea1d571432273018a13c88a2c8ec6` |
| `center-bias-ledov.pt` | `cd6519218d45a7a18fc15e8dc9d09448fee2153264ecd1e7317d877274c5ce9d` |
| `yolo26n-seg.pt` | `361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f` |
| `yolo26n-pose.pt` | `eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9` |
| `adaptive-input.mp4` | `dbdc291427018b83a02156de37c7dbae42d350cce13195e5ef5d5d2b12ad63b2` |
| `scheme-b-tensor.log` | `19ef96cb1b2e0d8fe3007640cbccc0291652d1c523b581f6dd80851f00e1ac9a` |
| `scheme-b-tensor.mp4` | `9cf8f12da9435623a8ca5b564b787c1909246744b22d8a5577f717df9a3bb085` |
| 原始 `scheme-b-cpu-evidence.json` | `3c653c095e07f39a42b83cfe606ed2ed007df0fedbd112d8607db686f29ac160` |
| `scheme-b-final-tests-verified.log` | `4f6251ad4cbc0908954ca062c33db582927ba0a8ec52e9ec366993858892e4f8` |

本報表與小型 JSON 摘要納入 Git。原始完整測試 log 位於 `/tmp/scheme-b-final-tests-verified.log`；
短片 log、成片、輸入與原摘要位於 `/workspace/reframe-runtime/validation/`，模型位於
`/workspace/reframe-runtime/models/`。這些皆為原雲端環境的本機檔案，**不會隨 clone 自動取得**，環境回收後可能消失。
沒有將原始大 log、模型或 MP4 提交至 Git；僅有雜湊不能重建遺失檔案。

## 歷史 T4 與已知限制

- [2026-10-08 T4 報告](../colab-t4-results-2026-10-08.md) 保留原版本、環境與範圍：是早期 90 幀短測，非目前版本的新驗證。
- 後續使用者長片回報的 3D 素材為 10,252 幀／60 FPS／170.8667 秒；視覺焦點全程正確，含兩人交錯、短暫暗場與慢速 3D 旋轉。
  固定鏡頭素材為 14,557 幀／60 FPS／242.6167 秒；先前 139 次 DeepGaze 中 111 次集中於 205 秒後的背面片段。
  這是較早版本的使用者資料與分析背景，不是 `71b4ed2` 的驗證；完整原始上傳包未納入此 Git 報表。
- 第二人在部分片段更搶鏡時，DeepGaze 可能是正確決策；呼叫較少本身不等於改善。
- 極短 12／32 幀測試曾暴露既有 FFmpeg AAC 音訊超尾問題；收集器會拒絕超過容許值的結果，未放寬驗證。
  全部影片尾幀與音訊修正邏輯已保留，但正式長片在新 GPU 路徑上仍無驗收結果。
- torchvision 仍為 Ultralytics 依賴；torchaudio 不是專案依賴。此次實作沒有聲稱已完全移除 torchvision。
- 光流主要處理平移、遮罩僅有面積／質心／頂部摘要；不等於 3D 旋轉估計或完整輪廓匹配。
  預測不刷新真實量測時間，沒有跨暗場／遮擋強接 ID；離線雙向插值方案 C 尚未實作。
- 原 FPS、全數影格、precrop 定義、DeepGaze 原順序 16 幀窗口保持；GPU 路徑仍為可選。
  缺少 T4 證據前，無法證明新路徑較快、無 CUDA 錯誤或長片無顯存累積。

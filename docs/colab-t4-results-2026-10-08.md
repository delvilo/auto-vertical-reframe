# Colab T4 GPU 執行紀錄（2026-10-08）

本紀錄分析使用者提供的成功 log，未在此雲端 CPU 機器重新執行 GPU 測試。附件內容視為執行證據，不作為操作指示。

## 證據與版本

- 安裝：`cropreframe-mr-install.log`。
- 短片測試：`reframe-mr-test.log`。
- 安裝 log 記錄分支：`codex/modular-saliency-stage1`。
- 安裝 log 記錄 commit：`7d1b705a291e279eb0a486865e22e1e57eca50a7`。
- 套件：`auto-vertical-reframe 0.2.1`，editable 安裝於 `/content/auto-vertical-reframe`。
- 測試記錄 package SHA256：`cffc0bca97377fdf44dda3aadd91fb63839d7c5c032a6722c9147008917b912d`。
- 使用者表示分支已更新至 master；本次 Git read-only 查詢只取得 main（`5be6ff80c343da2cb063b6d41a8108091eefdc3d`），master 無結果。上述 Colab 實測 commit 與目前遠端分支的對應尚未確認，不能把這份結果視為 master 的重新驗證。
- log 時間為 08:07:28–08:08:09，未標示時區，保留原始時間，不推定為台灣時間。

## 環境與安裝結果

| 項目 | 實測值 |
| --- | --- |
| Python | 3.13.15，/usr/bin/python3 |
| GPU | Tesla T4，compute capability 7.5 |
| torch | 2.11.0+cu130 |
| torchvision | 0.26.0+cu130 |
| CUDA build / available | 13.0 / True |
| cuDNN runtime | 92700 |
| ultralytics | 8.4.173 |
| scenedetect | 0.7.1 |
| lap | 0.5.13 |
| numpy | 2.1.3 |
| opencv-python | 4.14.0.94 |

安裝程式保留既有 torch／torchvision CUDA build；移除重疊 OpenCV distributions，最後安裝單一 opencv-python provider。MediaPipe 原本未安裝，後續不可匯入檢查通過。依賴安裝、editable wheel 建置、套件匯入與 `python3 -m reframe --help` 均成功。此 log 未記錄完整測試套件或 pip check 的執行結果。

## 測試命令

測試工作目錄為輸入影片所在目錄，安裝後可使用套件入口執行：

```bash
python3 -m reframe vv110.mp4 vv110V_mr_test.mp4 \
  --seg-model yolo26n-seg.pt --pose-model yolo26n-pose.pt --device 0 \
  --lock-first-subject --dead-zone 0.15 --post-restore \
  --video-encoder hevc_nvenc --max-frames 90 \
  --saliency-model deepgazemr --saliency-device cuda --saliency-trust-repo \
  --saliency-max-side 384 --saliency-interval 3 --no-saliency-amp \
  --native-debug --log-level DEBUG --ffmpeg-log-level info
```

此命令依 log 的實際參數重建；log 顯示入口檔為 `reframe/__main__.py`。測試使用 talking_head、person、direct encode、inline scene detection；pose imgsz=640、batch size=4、cue interval=0.2 秒。DeepGaze 使用 CUDA，AMP 明確關閉；首次執行下載 DeepGaze repository 與 VGG19 權重。兩個 YOLO 模型均實際在 cuda:0 推論，不僅是裝置選項設為 GPU。

## 短片結果

| 指標 | 結果 |
| --- | --- |
| 輸入 | vv110.mp4，3840×2160，60 fps，約 172.01 秒 |
| 處理範圍 | 前 90 幀，約 1.5 秒影片 |
| 輸出 | vv110V_mr_test.mp4，1080×1920，60 fps |
| 編碼 | HEVC Main／hevc_nvenc，yuv420p |
| 音訊 | MP3 輸入轉 AAC LC，44.1 kHz、stereo、192 kb/s 設定 |
| 後處理 | hqdn3d 降噪與 unsharp |
| FFmpeg | 2 幀 encoder probe exit=0；實際 direct encode exit=0 |
| frames_with_subject | 90/90（100%） |
| frames_with_pose | 90/90（100%，包含快取線索） |
| frames_with_head_cues | 48/90（53.3%） |
| pose_rois_inferred / matched | 19 / 19 |
| pose_cache_hits | 193（ROI／track 快取計數，不是影片幀數） |
| scene_resets / subject_switches | 1 / 0 |
| frames_with_two_person | 0；two-person framing 未啟用 |
| 輸出大小 | FFmpeg 報告 3322 KiB |

分割與姿態 log 分別確認 `actual inference device=cuda:0`，Summary 亦確認 `video_encoder_actual=hevc_nvenc`。因此此次短片成功驗證 T4 的 YOLO 分割、姿態推論與 NVENC 編碼。頭部線索比例不等同人臉偵測率，pose 覆蓋率也不表示每幀都重新跑姿態模型。

## DeepGaze MR 結果與回退

模型載入成功，實際 backend 為 deepgazemr，device=cuda，AMP=False。90 幀共 30 次更新（interval=3），其中 25 次由 backend 產生，5 次為 fallback；另有 60 幀沿用／傳播結果。更新計數比例為 83.3% backend、16.7% fallback，不能解讀成 5/90 幀推論失敗。

最終 map 為 384×216，map frame index=90、inferred frame index=88、timestamp=1.45；狀態為 predicted／propagated，最終 fallback reason=null。這只說明結束時沒有 fallback reason，不能抹除先前的 5 次回退。CLI 說明提到 temporal warmup 使用 handcrafted 輸出，但本測試 log 沒有逐次原因，無法確認這 5 次全部是 warmup，也不能認定它們全部是模型錯誤。

## 耗時、警告與驗證範圍

- 程式開始至 Summary 約 41.38 秒，包含初始化、首次下載與模型載入，90/41.38 約 2.18 fps；這不是穩態推論 benchmark。
- FFmpeg 最後報告 18.00 秒、5.0 fps、speed=0.0825x。此數值包含等待上游供應影格，不能當成 NVENC 單獨吞吐量。
- torchvision 的 pretrained／weights 參數棄用警告，以及安裝時 MoviePy invalid escape SyntaxWarning 未阻止此次執行。log 未見 ERROR 或 traceback。
- 此次驗證範圍是 90 幀短片；沒有完整 172 秒處理、暖快取 benchmark、峰值 GPU 記憶體、輸出影片独立 ffprobe／解碼檢查，亦未提供視覺構圖與音畫同步人工檢查結果。
- 成功結論依據實際 GPU inference、模型載入、FFmpeg exit=0 與最終 Summary；本紀錄不宣稱 AMP、其他 GPU、其他版本或完整影片已驗證。

# Colab T4：TorchCodec 兩階段測試

使用 [Colab notebook](https://colab.research.google.com/github/delvilo/auto-vertical-reframe/blob/main/notebooks/colab-torchcodec.ipynb)
或 [GitHub notebook 檔](../notebooks/colab-torchcodec.ipynb)。這是新 notebook，不覆蓋原本的音訊尾端測試。

| Profile | 解碼 | 模型與輸出影像處理 |
| --- | --- | --- |
| `baseline` | OpenCV／CPU | 既有 NumPy 路徑，包含精簡後 DeepGaze 載入 |
| `stage1` | TorchCodec／NVDEC | 共用小縮圖；模型與輸出保留 reference 路徑 |
| `stage2` | TorchCodec／NVDEC | GPU 裁切、縮放與 tensor 模型輸入；最後成片傳給 FFmpeg |

三組都保留原 FPS、所有輸出幀、middle precrop、主角鎖定、FP32 saliency、原有顯著性決策，以及
FFmpeg／HEVC NVENC 和音訊尾端修正。`--post-restore` 關閉。gap 預設 1；要測 gap 3 時三組一起改，
三組均保留 `gray-area` 縮圖、`primary` 跳幀策略和 lazy thumbnail；避免把偵測排程變更當成解碼加速。
GPU 色彩轉換／插值可能造成細微像素與模型結果差異，仍須看影片。

## 先短測，再逐組完整影片

沿用可執行專案的 Python 3.13+、PyTorch 2.11+／相容 torchvision 環境。Notebook 保持 torch／torchvision
版本，僅安裝相符的 TorchCodec 0.17 CUDA wheel；提供的安裝 cell 適用 CUDA 13.0 build。
TorchCodec 使用 `--no-deps --force-reinstall`，避免已裝同版本 CPU wheel 時被 pip 誤判為滿足需求，
不會因此重裝或升級 torch／torchvision。
其他 CUDA build 會停止，不自動更換 PyTorch。FFmpeg 必須同時有 CLI、可載入的 shared libraries；
只有 `ffmpeg -version` 成功不足以證明 TorchCodec 能載入。

1. 修改影片、兩個 YOLO 權重與 Drive 路徑。使用固定 `RUN_NAME`，再次執行才能找到續作紀錄。
2. 第一階段預設只跑 `baseline stage1`，各 **180 幀一次**，沒有額外 600 幀暖機或多輪完整影片。
3. 看完第一階段輸出後執行 `stage2` 的 180 幀短測；確認人物交錯、背面、遮擋、變暗等片段仍需長片比較。
4. 把最後 cell 的 `RUN_FULL` 改為 `True`，一次選一個 profile。每組完整測試前，收集器要求同一份
   影片／模型／程式／環境／設定已有成功的 180 幀短測。90 幀可供快速排錯，但不解鎖完整測試。

實際短測幀數為 `min(來源幀數, --max-frames)`；來源短於 180 幀時會處理整支短片，
不能因此宣稱已測足 180 幀。避免把上限設為幾幀或十幾幀：FFmpeg `apad -shortest` 的
音訊緩衝可能讓極短成片的 AAC 終點超過影片，收集器仍會如實判定音訊檢查失敗，
不會放寬完整影片的驗收門檻。失敗的 log 和已輸出的 MP4 都保留供排查。

```python
import subprocess, sys
from pathlib import Path

repo = Path('/content/auto-vertical-reframe-torchcodec')
common = [sys.executable, '-u', str(repo / 'scripts/colab_torchcodec.py'),
    '--input', '/content/video.mp4',
    '--seg-model', '/content/drive/MyDrive/crop/yolo26n-seg.pt',
    '--pose-model', '/content/drive/MyDrive/crop/yolo26n-pose.pt',
    '--output-dir', '/content/torchcodec-test',
    '--backup-dir', '/content/drive/MyDrive/reframe-benchmarks/torchcodec-test',
    '--seg-max-gap', '1', '--decode-batch-size', '4']
subprocess.run(common + ['--profiles', 'baseline', 'stage1', '--max-frames', '180'], cwd=repo, check=True)
# 第一階段確認後：
subprocess.run(common + ['--profiles', 'stage2', '--max-frames', '180'], cwd=repo, check=True)
# 短測成功、人工確認後另執行一次完整影片：
# subprocess.run(common + ['--profiles', 'stage2', '--full-video'], cwd=repo, check=True)
```

## 收集內容與續作

每次嘗試有自己的資料夾：`native.log` 保留原始 stdout／stderr、`record.json` 記錄命令、裝置、
Summary、ffprobe、檢查結果與程式 SHA-256，`run.segments.csv/jsonl` 記錄每五秒／鏡頭分段，
`output.mp4` 保留成片。`diagnostics.zip` 包含所有嘗試的診斷，不包含大型 MP4。

`results.json` 是最近這次呼叫的結果；先前每次的 `record.json` 仍保留。可從每個 Segment 的
`seg_detector_calls`／`seg_predicted_frames`／間隔分布及顯著性分層次數，對照片段中的動作。
Summary 的 `video_io` 記錄解碼與像素傳輸成本；pipeline／DeepGaze 計時仍由既有收集器驗證。
GPU 操作多為非同步，host wall time 與資料回 CPU 等待時間不能直接當成 CUDA kernel 時間。
`video_io.cuda_allocator_peak` 是該程序的 PyTorch allocator 峰值（allocated／reserved），
不包含 NVDEC 等外部配置的全部顯存，不能當作整張 GPU 的記憶體用量。

收集器拒絕 CPU fallback 冒充 NVDEC、模型或 encoder 裝置降級、缺尾幀、FPS 改變、分段加總錯誤，
並檢查輸出音畫終點差（容許 AAC 封包尾端誤差，預設至少 0.1 秒）。這不能取代人工音畫同步檢查。
DeepGaze 呼叫為 0 時明確記為 `not_exercised`；不能據此宣稱 DeepGaze 的 GPU 路徑已驗證。

再次執行相同命令，只有來源／權重／程式／環境／設定完全一致且輸出檔 SHA-256 未變的成功紀錄才跳過。
失敗或中斷的嘗試保留，下次建立新 attempt，從該 profile 的第一幀重跑；**沒有在影片中途恢復模型狀態**。
程式修改後重新短測是刻意的要求。若 Colab runtime 被回收，可從 Drive 備份把整個測試資料夾複製回
相同 `OUTPUT_DIR` 再執行；不要只還原 ZIP，因為續作也要驗證 MP4。

每完成一個 profile、或可捕捉的例外，備份診斷和現有輸出到 Drive。硬性刪除 runtime 時無法執行 finally；
尚未完成的 profile 可能沒有最新備份。`native.log` 中仍可讀取已 flush 的 Segment 行。

先比較 `subprocess_wall_seconds`／pipeline elapsed、解碼、render、傳輸 bytes 與模型呼叫次數，再看成片。
第一組可能承擔模型下載／cache 冷啟動；這份短測先驗證功能，不足以代表穩態加速比。
本機 CPU 測試不能證明 T4 加速或 NVENC／NVDEC 可用，這些需由實際 Colab 結果確認。

如需在沒有 GPU 的開發環境排錯，可明確加上 `--allow-cpu-diagnostic --device cpu
--saliency-device cpu --decode-device cpu --video-encoder libx264`。結果會標記 `cpu_diagnostic`；
此旗標不會自動選擇 CPU，仍會驗證指定裝置。預設 GPU 比較拒絕這些 CPU 設定。
完整影片僅接受一個 `--profiles` 值，避免不小心連跑多組數小時的測試。

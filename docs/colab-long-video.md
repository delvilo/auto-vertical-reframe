# Colab T4：170 秒長影片測試

本流程使用 Python 3.13+ 與目前已安裝的 Colab CUDA 環境，不重新安裝 PyTorch／CUDA。
程式保留來源 FPS，連續處理整支影片，每 5 秒影片時間及切鏡產生一筆分段紀錄。
統計不重置追蹤或快取，不新增 GPU 強制同步；切鏡本身仍依原本邏輯重置模型的時序狀態。

## 時間與測試範圍

以 **170 秒、60 FPS、約 5 processing FPS** 為估算前提：

`170 × 60 ÷ 5 = 2040 秒 = 34 分鐘／次`

| 模式 | 測試內容 | 估計處理時間 |
| --- | --- | --- |
| `full`（預設） | 四組各一次完整暖機、兩次完整正式測試，共 12 次 | 6.8 小時 |
| `short-warmup` | 四組各 600 幀暖機、兩次完整正式測試 | 約 4.7 小時 |
| `comparison` | 逐幀基準與完整優化各 600 幀暖機、一次完整正式測試 | 約 72 分鐘 |

另有開頭 60 幀短測與檔案探測、模型下載、程序啟動及備份時間。實際處理速度會隨動作、
切鏡與 DeepGaze 呼叫量改變；30 FPS 或其他幀率應代入實際值。模型每次獨立程序仍須初始化，
暖機主要預先走過模型路徑、下載權重及建立系統快取；短暖機若沒有觸發 DeepGaze，首次正式
呼叫仍可能承擔載入／下載成本。暖機與短測不納入正式中位數。

目前只有 CPU 回歸及實際短片流程通過；這份 script 用來收集新的 T4 證據，並不代表已確認 T4 加速。

## 1. 更新程式、掛載 Drive 並準備來源

先把 notebook 設為 T4 GPU，專案與依賴須已依 README 安裝。更改 `VIDEO_SOURCE` 為您的影片。
來源先複製到 `/content`，讓各組使用相同本機檔案；輸出也寫在本機，避免 Drive I/O 混入比較。

```python
from datetime import datetime, timezone
from pathlib import Path
import json, shutil, subprocess, sys
from google.colab import drive

assert sys.version_info >= (3, 13), "此專案需要 Python 3.13+"
drive.mount("/content/drive")
repo = Path("/content/auto-vertical-reframe")
assert (repo / ".git").is_dir(), "請先 clone 並安裝專案"
subprocess.run(["git", "pull", "--ff-only", "origin", "main"], cwd=repo, check=True)
subprocess.run(["nvidia-smi"], check=True)

VIDEO_SOURCE = Path("/content/drive/MyDrive/video/170s-test.mp4")  # 修改這裡
assert VIDEO_SOURCE.is_file(), VIDEO_SOURCE
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
workspace = Path("/content") / ("reframe-long-" + stamp)
workspace.mkdir()
video = workspace / VIDEO_SOURCE.name
shutil.copy2(VIDEO_SOURCE, video)
smoke_dir = workspace / "smoke"
run_dir = workspace / "benchmark"
backup_dir = Path("/content/drive/MyDrive/reframe-benchmarks") / workspace.name
backup_dir.mkdir(parents=True)

def checkpoint(name):
    return str(next((p for p in (VIDEO_SOURCE.parent / name, repo / name) if p.is_file()), Path(name)))

common = [sys.executable, "-u", str(repo / "scripts/benchmark_colab.py"),
          "--input", str(video),
          "--seg-model", checkpoint("yolo26n-seg.pt"),
          "--pose-model", checkpoint("yolo26n-pose.pt"),
          "--device", "0", "--saliency-device", "cuda", "--video-encoder", "hevc_nvenc",
          "--stats-interval", "5", "--log-level", "INFO", "--archive"]
```

## 2. 短測，接著執行選定模式

`MODE = "full"` 即上述約 6.8 小時的完整方案。保留 `middle` precrop、鎖定首位主角、
conf=0.3、FP32 saliency 及關閉 post-restore，四組只改偵測／縮圖條件。

```python
MODE = "full"  # full / short-warmup / comparison
modes = {
    "full": ["--full-video", "--suite", "optimizations", "--repeats", "2"],
    "short-warmup": ["--full-video", "--suite", "optimizations", "--repeats", "2",
                     "--warmup-frames", "600"],
    "comparison": ["--full-video", "--suite", "comparison", "--repeats", "1",
                   "--warmup-frames", "600"],
}
assert MODE in modes

def backup_archives():
    for directory in (smoke_dir, run_dir):
        archive = directory.with_suffix(".zip")
        if archive.is_file():
            shutil.copy2(archive, backup_dir / archive.name)

try:
    subprocess.run(common + ["--output-dir", str(smoke_dir), "--smoke"], cwd=repo, check=True)
    backup_archives()
    smoke = json.loads((smoke_dir / "results.json").read_text())
    print("短測模型裝置：", smoke["runs"][0]["device_validation"])
    subprocess.run(common + ["--output-dir", str(run_dir)] + modes[MODE], cwd=repo, check=True)
finally:
    backup_archives()  # 正常結束或已捕捉的失敗時備份；runtime 被刪除時無法保證執行
    print("本機結果：", run_dir)
    print("Drive 備份：", backup_dir)
```

已執行的模型必須使用要求的 CUDA 裝置；模型失敗 fallback 或 NVENC fallback 會令測試失敗，
保留原始錯誤及已完成資料。`not_exercised` 表示短片未呼叫該模型，不代表該模型已通過 GPU 測試；
完整影片若呼叫模型，會再檢查裝置與 fallback。INFO 保留進度、Segment、Summary 及所有原生錯誤；
需要逐筆 Pose ROI 除錯時才改為 DEBUG，以免長片產生大量 log。

請讓 Colab session 保持運行；`/content` 屬暫存空間。正常結束／失敗時的 ZIP 備份不能保證抵抗
runtime 直接被回收。輸出目錄拒絕覆寫，重新測試請重新建立時間戳目錄。

## 3. 比較整體與影片時間線

整體 `aggregates.csv` 排除暖機及失敗執行。每次 MP4 仍保留原始幀率與音訊，ffprobe 驗證
幀數、幀率、尺寸及音訊是否存在；同時記錄時間戳，不能取代目視裁切與音畫同步檢查。

```python
import pandas as pd
from google.colab import files

totals = pd.read_csv(run_dir / "aggregates.csv")
columns = ["case", "measured_runs", "median.summary.elapsed_seconds",
           "median.summary.seg_detector_calls", "median.summary.seg_predicted_frames",
           "median.summary.seg_timing_seconds.thumbnail",
           "median.summary.saliency_actual_forward_calls"]
display(totals[[name for name in columns if name in totals.columns]])
segments = pd.read_csv(run_dir / "segments.csv")
measured = segments.loc[(segments["benchmark_warmup"] == False) &
                        (segments["benchmark_partial"] == False)].copy()
measured["video_seconds"] = measured["end_seconds"] - measured["start_seconds"]
measured["yolo_calls_per_video_second"] = measured["seg_detector_calls"] / measured["video_seconds"]
measured["skip_percent"] = 100 * measured["seg_predicted_frames"] / measured["frames"]
display(measured[["benchmark_case", "benchmark_repeat", "scene_index", "start_seconds", "end_seconds",
                  "seg_detector_calls", "seg_predicted_frames", "yolo_calls_per_video_second",
                  "skip_percent", "saliency_actual_forward_calls", "processing_wall_seconds"]])
files.download(str(run_dir.with_suffix(".zip")))
```

將每一輪的 YOLO 使用頻率畫成時間線；每段高度是該段平均頻率，並非逐幀動作還原：

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(14, 4))
for (case, repeat), group in measured.groupby(["benchmark_case", "benchmark_repeat"], sort=False):
    group = group.sort_values("frame_start")
    edges = [group["start_seconds"].iloc[0], *group["end_seconds"].tolist()]
    ax.stairs(group["yolo_calls_per_video_second"].to_numpy(), edges,
              label=f"{case} / run {repeat}", baseline=None)
ax.set(xlabel="Video time (seconds)", ylabel="YOLO segmentation calls / video second")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(run_dir / "yolo-frequency.png", dpi=150)
plt.show()
```

分段 CSV／JSONL 使用影片時間，與 log 行首的執行時鐘不同。YOLO 分割頻率為
「該段真實偵測次數 ÷ 該段影片秒數」，不是 processing FPS；Pose ROI 數也不是分割模型呼叫數。
實際偵測 gap 分布會跨越一般統計窗保留上次偵測位置，但切鏡後重新計算；每段最初偵測不一定
有一筆 gap，所以 gap 筆數不必等於偵測次數。首尾不足 5 秒或切鏡段落會保留部分視窗。
CSV 的 `seg_detector_gap_counts.1`／`.2`／`.3` 為實際相鄰偵測間隔分布，
`seg_scheduled_interval_counts` 則是排程當時選用的目標間隔，兩者不能混為一談。
`seg_refresh_reasons.*`、`seg_gate_diagnostics.*` 可定位重偵測原因；`benchmark_partial=true`
表示該次失敗或中斷，已落盤的完整分段仍保留，但不納入正式統計。

不同方法可能產生不同追蹤與構圖結果，請對照快速動作、人物交錯、暫時出框、切鏡前後的輸出影片。
不要只比較跳幀率；還需查看端到端耗時、主角遺失原因、DeepGaze 次數及畫面穩定性。
ZIP 包含 log／JSON／CSV／JSONL，不包含大型輸出影片；需要保留影片時另行複製到 Drive。

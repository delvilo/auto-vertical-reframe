# T4 GPU 驗證規格與執行順序

更新：2026-10-13（Asia/Taipei）。本規格先前針對 opt-in 實作之完整真實 T4 GPU 執行已於 2026-10-13 由 `colab-operator` 驗證完成，詳細數據見
[T4 GPU 全流程真實驗證報表](reports/t4-2026-10-13-gpu-verified.md) 與 [YouTube 4K 人像驗證報表](reports/youtube-validation-2026-10-13-3079cd0d9549.md)（此為歷史回報證據，非本次預設／回退新架構之證明；使用者回報成功但未附帶新版本日誌）。
本文件保留作為後續排查、重測與量測準則之規格基準。
## 1. 固定比較條件

- 固定乾淨 checkout 的 commit，記錄 Python、torch、torchvision、torchcodec、Ultralytics、CUDA build、
  driver、GPU 型號／顯存、FFmpeg CLI 與可載入 shared libraries。修改程式後重新做相符短測。
- 保存來源影片與兩個 YOLO 權重的 SHA-256、ffprobe 資料；DeepGaze checkpoint／center bias 的雜湊另記。
  原片先複製到本機磁碟，避免 Drive 讀取波動影響比較；備份則寫另一個不重疊的目錄。
- 保持模型精度、輸出尺寸、編碼器、precrop、主角鎖定、dead-zone、相機參數及輸入一致。
  原 FPS、逐來源影格輸出、音訊尾端修正保持有效；不得以丟幀、降解析度或錯過主角換取速度。
- `--profiles` 是資料流選擇；`--tracking-profile` 是排程選擇。`conservative` 是目前程式的設定，
  **不等於歷史版本**；`--seg-max-gap 1` 只停用分割跳幀，並未停用 Pose 快取。
- 一次執行一個長片組合。不重跑歷史 `optimizations --repeats 2 --warmup-frames 600` 大矩陣。
  單次對照只支持這份素材與環境的觀察；有明顯波動或回歸才補重複測量。

| 資料流 profile | 解碼 | 模型輸入／輸出處理 |
| --- | --- | --- |
| `baseline` | OpenCV／CPU | reference NumPy；模型仍要求 CUDA |
| `stage1` | TorchCodec／NVDEC | reference，隔離解碼路徑影響 |
| `stage2` | TorchCodec／NVDEC | tensor／GPU 影像處理 |

一般 CLI 預設影像解碼器為 `torchcodec`（`--decode-device cuda:0`、`--decode-batch-size 4`），啟動階段（模組載入、初始化或首批次驗證）失敗時自動記錄堆疊並回退至 OpenCV，且若指定 `--frame-processing tensor` 則自動降級為 `reference`。`baseline` 資料流則明確傳入 `--video-decoder opencv` 略過 TorchCodec 探測。來源為變動幀率（VFR／`ValueError`）或解碼中途發生錯誤維持嚴格中斷（fatal）。時戳校驗方面：TorchCodec 會逐幀嚴格校驗顯示時戳並拒絕非 CFR 串流；OpenCV 解碼則依賴容器回報之 FPS，無法建立逐幀時戳校驗。

| 追蹤 profile | YOLO 候選 gap 上限 | 真實分割期限 | Pose 軟更新／硬期限 |
| --- | --- | --- | --- |
| `conservative` | 3 幀 | 0.1 秒 | 0.2／0.2 秒 |
| `balanced`（預設） | 6 幀 | 0.1 秒 | 0.2／0.35 秒 |
| `fast`（選測） | 6 幀 | 0.1 秒 | 0.25／0.4 秒 |

以上都是上限，不保證使用最大 gap；風險、光流、Pose／遮罩證據可能要求立即偵測。
預設 0.1 秒限制下，60 FPS 可評估 gap6，30 FPS 最多 gap3；29.97／59.94 FPS 取 floor 後為 2／5 幀，
不要先四捨五入來源 FPS。`--pose-max-age 0` 禁止跨幀沿用。

## 2. 環境檢查與安裝

需要 Python **3.13+**。目前 staged collector 要求 PyTorch **2.11+**、TorchCodec **0.17.x**；
專案基礎 requirements 的 torch 下限較低，單獨安裝專案不代表已符合此測試組合。
保留相容的 torch／torchvision build；Ultralytics 仍使用 torchvision，不能因為換 TorchCodec 就移除它。
本流程不需要安裝 torchaudio。安裝後使用新 Python 程序，避免 notebook 已匯入的舊模組留在記憶體。

以下 Bash 設定供後續各節共用；先填實際路徑。`REFRAME_PYTHON` 是已選定的 Python 3.13+。

```bash
set -euo pipefail
REFRAME_PYTHON=python3
REFRAME_REPO=/content/auto-vertical-reframe
REFRAME_INPUT=/content/video-170s.mp4
REFRAME_SEG=/content/yolo26n-seg.pt
REFRAME_POSE=/content/yolo26n-pose.pt
REFRAME_OUT=/content/reframe-t4-final
REFRAME_BACKUP=/content/drive/MyDrive/reframe-benchmarks/reframe-t4-final
cd "$REFRAME_REPO"
mkdir -p "$REFRAME_OUT/environment"
git status --short
git rev-parse HEAD | tee "$REFRAME_OUT/environment/commit.txt"
"$REFRAME_PYTHON" --version
nvidia-smi | tee "$REFRAME_OUT/environment/nvidia-smi.txt"
ffmpeg -version > "$REFRAME_OUT/environment/ffmpeg.txt"
ffprobe -version > "$REFRAME_OUT/environment/ffprobe.txt"
sha256sum "$REFRAME_INPUT" "$REFRAME_SEG" "$REFRAME_POSE" > "$REFRAME_OUT/environment/input-model.sha256"
```

若尚未安裝，先檢查既有 torch／torchvision 能匯入與 CUDA build。`install_colab.py` 會固定現有
torch／torchvision 版本、移除舊 MediaPipe 與重疊的 cv2 provider，再安裝單一 OpenCV。
不要在沒有既有相容 accelerator 環境時，假設這個工具會替你選好 CUDA wheel。

```bash
"$REFRAME_PYTHON" - <<'PY'
import sys, torch, torchvision
from packaging.version import Version
assert sys.version_info >= (3, 13), sys.version
assert Version(torch.__version__.split('+')[0]) >= Version('2.11'), torch.__version__
assert torch.cuda.is_available()
print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
assert torch.version.cuda == '13.0', '下方安裝範例僅適用既有 cu130；其他 build 先查相容 wheel'
PY
# 僅在上方確認 cu130 且需要安裝時執行；已配置妥當的環境不必重裝：
"$REFRAME_PYTHON" install_colab.py
"$REFRAME_PYTHON" -m pip install --no-deps --force-reinstall \
  torchcodec==0.17.0 --index-url https://download.pytorch.org/whl/cu130
```

其他 CUDA build 不直接套用上述 wheel、不自動替換 torch；先記錄不相容原因與可用版本再處理。
TorchCodec 必須能載入 FFmpeg shared libraries；只有 `ffmpeg -version` 成功不夠。

```bash
"$REFRAME_PYTHON" - <<'PY' | tee "$REFRAME_OUT/environment/python-cuda.json"
import json, sys, torch, torchvision, torchcodec
from torchcodec.decoders import VideoDecoder
boxes = torch.tensor([[0., 0., 2., 2.]], device='cuda')
scores = torch.tensor([1.], device='cuda')
assert torchvision.ops.nms(boxes, scores, 0.5).numel() == 1
print(json.dumps(dict(python=sys.version, torch=torch.__version__, torchvision=torchvision.__version__,
    torchcodec=torchcodec.__version__, cuda_build=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
    gpu=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0))))
PY
"$REFRAME_PYTHON" -m pip freeze > "$REFRAME_OUT/environment/pip-freeze.txt"
"$REFRAME_PYTHON" -m pip check 2>&1 | tee "$REFRAME_OUT/environment/pip-check.txt"
```

收集器還會以實際來源建立 NVDEC decoder、取得影格並檢查 fallback，另試跑指定 NVENC encoder。
NVDEC 必須回報 `cpu_fallback=false` **且** `cpu_fallback_status_known=true`；CUDA tensor 本身不證明硬解。
GPU 比較不加 `--allow-cpu-diagnostic`，也不偷偷改用 CPU decoder／模型或軟體 encoder。

**硬體驗證與日常執行的嚴格度差異**：日常 CLI 允許 TorchCodec 啟動失敗時自動回退至 OpenCV，但 T4 硬體驗證與效能收集器（`scripts/colab_torchcodec.py`、`scripts/youtube_validation.py`）之驗收標準嚴於一般執行。收集器透過 `validate_decoder` 嚴格檢驗 `video_io` 之實際欄位，要求 `stage1`／`stage2` 必須為真實 `torchcodec` 解碼且確認 `nvdec_verified`，直接拒絕任何啟動回退至 OpenCV 或 CPU 解碼。Summary `video_io` 會完整記錄 `requested_decoder`、`requested_frame_processing`、`decoder_fallback` 與 `decoder_fallback_reason` 以供審計。

## 3. 先跑既有測試，再補真實 GPU 缺口

```bash
"$REFRAME_PYTHON" -m unittest discover -s tests -p 'test_frame_pixels.py' -v \
  2>&1 | tee "$REFRAME_OUT/frame-pixels-tests.log"
"$REFRAME_PYTHON" -m unittest discover -s tests -p 'test_deepgaze_tensor.py' -v \
  2>&1 | tee "$REFRAME_OUT/deepgaze-tensor-tests.log"
"$REFRAME_PYTHON" -m unittest discover -s tests -p 'test_tracking_profiles.py' -v \
  2>&1 | tee "$REFRAME_OUT/tracking-profile-tests.log"
```

- 前兩個檔共有 **5 個 CUDA 專用測試**：2 個真實 tensor 縮放／記憶體測試，3 個 DeepGaze 窗口測試。
  後 3 個使用 `RecordingTensorModel`，只驗證資料與裝置流程，不是實際 DeepGaze 權重推論。
- `test_tensor_yolo.py` 即使提供 `REFRAME_MODEL_DIR`，真權重案例目前仍明確用 CPU。
  `test_lean_deepgaze.py` 的 `DEEPGAZE_REFERENCE_REPO` 參考比較也用 CPU；設環境變數不會改成 CUDA。
- 5 個 CUDA 測試在 T4 仍 skip 要查原因；不能把整套成功結束寫成 5 項已通過。
  真模型座標／數值比較、CUDA event 微效能測試仍須接手 agent 建立受控工具並保留程式與結果。
  目前沒有可直接呼叫的 `gpu-parity` 或 `cuda-benchmark` 指令，不要在報告中假稱已有此工具。

## 4. 最少的短測與長片矩陣

先固定版本再執行。收集器固定 middle precrop、主角鎖定、conf=0.3、lazy／gray-area／primary、
saliency max-side=384、FP32（AMP 關閉）、post-restore 關閉、每 5 秒／切鏡 Segment。
輸出尺寸為同版程式預設 1080×1920；明確保留 dead-zone=0。每次記錄的原始 command 可供核對。

```bash
REFRAME_COMMON=("$REFRAME_PYTHON" -u "$REFRAME_REPO/scripts/colab_torchcodec.py"
  --repo "$REFRAME_REPO" --input "$REFRAME_INPUT"
  --seg-model "$REFRAME_SEG" --pose-model "$REFRAME_POSE"
  --output-dir "$REFRAME_OUT" --backup-dir "$REFRAME_BACKUP"
  --device 0 --saliency-device cuda --decode-device cuda:0
  --video-encoder hevc_nvenc --decode-batch-size 4 --dead-zone 0)

# A. 隔離資料流：三條路徑設定相同，先各 180 幀一次。
"${REFRAME_COMMON[@]}" --tracking-profile conservative --seg-max-gap 1 \
  --profiles baseline stage1 --max-frames 180
"${REFRAME_COMMON[@]}" --tracking-profile conservative --seg-max-gap 1 \
  --profiles stage2 --max-frames 180

# B. 追蹤方案：用同一份完整原片做前 180 幀，不帶 gap 覆寫。
"${REFRAME_COMMON[@]}" --profiles stage2 --tracking-profile conservative --max-frames 180
"${REFRAME_COMMON[@]}" --profiles stage2 --tracking-profile balanced --max-frames 180
```

看完裝置、幀數、音訊與成片，並完成下節事件片段檢查後，再逐條執行以下長片命令。
**最低限度兩個 170 秒完整輸出**用來比較追蹤；不自動排入其他長片組合。

```bash
"${REFRAME_COMMON[@]}" --profiles stage2 --tracking-profile conservative --full-video
"${REFRAME_COMMON[@]}" --profiles stage2 --tracking-profile balanced --full-video
```

如要測量「長片資料流加速」才另跑下組，否則此項維持 `pending`；不是最低必跑項。
上面的 tracking 長片沒有 gap1，不能拿來充當這組對照。stage1 長片只在需定位回歸時補測。

```bash
# 選測，沿用 A 的相符短測；不要與上一組混算加速比：
# "${REFRAME_COMMON[@]}" --profiles baseline --tracking-profile conservative --seg-max-gap 1 --full-video
# "${REFRAME_COMMON[@]}" --profiles stage2 --tracking-profile conservative --seg-max-gap 1 --full-video
```

`fast` 只在 balanced 品質合格後選測，同樣先做相符 180 幀，再決定是否需要長片。
若 stage2 出錯，保留失敗紀錄、先定位資料流原因；切回 baseline 的診斷須另列設定與結果。

## 5. 事件素材、餘批與視覺驗收

| 素材／條件 | 必須檢查 |
| --- | --- |
| 約 170.87 秒、4K／60 FPS、10,252 幀的原 3D 影片 | 全程焦點、兩人交錯、幾秒暗場、3D 慢動作／旋轉；不憑 ID 變化認定焦點遺失 |
| 約 242.62 秒、1080p／60 FPS、14,557 幀固定鏡頭 | 尤其 205 秒後背面區段；肩髖可靠時身體構圖、手遮臉／身體、手部進出與靠近鏡頭 |
| 低運動與大動作、切鏡、短／長遮擋、光流失敗 | 快取期限、補偵測延遲、漂移與錯接身分；第二人真正搶鏡時保留合理 DeepGaze |
| 30 FPS 與 60 FPS 的受控素材 | 相同秒齡安全限制；預設 30 FPS 不超過 gap3、60 FPS 可評估 gap6 |
| 1080p／4K、decode batch 1／4／8 | 解碼、色彩與尾幀完整，最後不足一批時仍正常 |

上表幀數是歷史素材描述；手上檔案要重新 ffprobe／hash 核對，不吻合就另列來源，不能冒用舊證據。
未取得某個事件素材就標 `pending`；不能用別的影片假裝涵蓋。開頭 180 幀不涵蓋晚段事件。

事件剪輯至少保留幾秒前導讓追蹤／16 幀窗口建立；記錄原片 SHA、要求／实际起迄時間、
剪輯方法、剪輯檔 SHA／ffprobe、Segment 相對時間映射。切片會重新建立主角 ID，不能代替全片連續性。
不要只 `-c copy -ss` 後假設剪輯精確落在要求的非關鍵幀；避免以重新編碼的不同像素混入同來源比較。
兩種 tracking profile 必須讀同一份事件剪輯。可用上述 common array 改 `--input` 為事件片，
先做兩組 180 幀，通過後對這份數十秒事件片各做 `--full-video`；它們不解鎖原 170 秒影片的長測。

分開檢查「max-frames 中途停止」和「實際 EOF 不足一批」。reader 按來源總幀數取批，
長片加 `--max-frames 180 --decode-batch-size 8` 可能已解碼 184 幀而只輸出 180 幀，不能證明 EOF 餘批正確。
真正只有 180 幀的素材才可用 `180 % 8 = 4` 覆蓋 EOF 餘批；batch4 改用真正 181 幀的素材，
或直接把 reader 讀至 EOF。原 10,252 幀 `%4=0`、`%8=4`，全片 batch8 可覆蓋 EOF 餘批，batch4 則不能。
另外保留長來源在 180／181 幀中途停止的測試，核對輸出尾幀與多解碼但未輸出的幀數。
181 幀或剪輯檢查不替代原片長測要求的 180 幀 identity。避免十幾幀 AAC 成片：既有封包緩衝可能使音訊超尾，
應保留失敗並排查，不放寬音訊驗收門檻來讓測試通過。

## 6. 每階段驗證責任（目前全部 GPU 責任仍待完成）

| ID | 檢查內容與需要的證據 |
| --- | --- |
| M1 精簡載入 | 使用獨立新 `TORCH_HOME` 冷快取，只下载 pinned DeepGaze checkpoint／center bias，記檔案大小／SHA、`saliency_model_provenance`、init／forward 時間；不取得 548 MB ImageNet VGG。真實 CUDA FP32 forward 有執行、無 fallback；與 pinned 參考架構比較兩種幾何／時間窗口，保留原 float64 center bias。 |
| M2 解碼 | 真 NVDEC 與已知無 fallback；1080p／4K、batch1／4／8 與餘批、CFR 時戳偏差不超過 1 ms、原 FPS／全數幀／末幀。保存 preflight、`video_io`、ffprobe；VFR／不連續時戳應明確拒絕，不偷偷改播放速度。 |
| M3A YOLO／Pose | 真實 CUDA reference 與 tensor 模型的框、mask、關鍵點回到正確 ROI／來源座標；middle／left／right、奇數尺寸與混合大小 Pose ROI、連續 ByteTrack ID。保存可比輸入、量化差異與畫面。 |
| M3B DeepGaze 窗口 | 16 個連續來源幀按原順序觀測，不是 16 個稀疏 YOLO 關鍵幀；ring wrap、切鏡 reset、久未推論後更新；真實模型有 forward、無 fallback，與 reference 的窗口／saliency 比較。 |
| M3C 輸出／顯存 | GPU crop／resize 沒有 RGB/BGR 或邊界異常；實際 NVENC、所有尾幀／FPS、音訊終點及人工音畫同步；長片无 OOM／顯存持續累積。保存 MP4、ffprobe、傳輸 bytes、allocator 峰值與必要的外部顯存採樣。 |
| M4 收集器 | 中斷保留已 flush log／Segment、備份可用；相同成功組合跳過，改來源／模型／程式／設定則新 attempt；裝置／encoder fallback 必須拒絕。只在短測做中斷演練，勿破壞長片成功證據。 |
| M5 縮放暫存 | CUDA sampled resize 與 native 像素誤差、邊緣／非整數／非連續 precrop；GPU 微測 event 耗時与峰值、真 YOLO／Pose 輸出。既有像素測試要求最大誤差 ≤1、4K 峰值低於 native 一半；不能由此直接宣稱速度提升。 |
| M6 窗口批次化 | pending=0／1／3／16、跨 ring 尾端、reset／完整刷新；窗口／真實 saliency 相同，沒有多傳或多算。記 preprocess batch／frame 數、cross-device bytes、CUDA event 時間。 |
| D0 死區預設 | dead-zone=0 的小位移跟隨與抖動偏好；明確非零覆寫仍有效，平滑／變焦死區未消失。記有效設定與成片；此項不是推論加速。 |
| P0 CLI／profile | conservative／balanced／fast 展開值、顯式覆寫優先且不受參數順序影響、舊旗標仍相容並顯示遷移提示、新秒數／速度優先；Summary `tracking_settings` 與 record identity 一致。比較旧／新等價命令時避免同時改資料流／構圖。 |
| P1 軀幹／手部／遮罩 | 背面與遮臉但肩髖可靠、手遮身體／出入／靠近鏡頭；局部手點失準不使全 Pose 失效，可靠大動作仍示警；預測遮罩不刷新量測時間，較嚴肩髖門檻的實際影響另記。 |
| P2 主角排程 | 遠處失效旁人不長期阻擋跳幀，低信任旁人恢復不誤報新人；新人、靠近／交錯、第二人搶鏡仍重新評估；暗場與 3D 旋轉不錯接主角 ID。 |
| P3 動態更新 | 30／60 FPS 真實秒齡、低／高運動、光流失敗、短／長遮擋、切鏡；分割／Pose／neural cache 不互相偽造時間。記實際 gap、漂移、Pose 延長／拒絕原因，以及 `pose-max-age=0`。 |
| P4 三級切換 | 新風險即時評估，持續同風險不每幀越過冷卻；穩定後才降級。核對合理 DeepGaze 呼叫、快取時效與 16 幀時間窗口；不能僅以呼叫次數下降判成功。 |

**離線雙向插值 C（原 P5）未被選擇、尚未實作，不在此輪必測範圍。** 不應為了完成清單先新增 C。

M1 冷啟動用新的獨立 cache 目錄，不清空使用者 cache；保存完整下載／載入 log。
若 pipeline 短測沒觸發 DeepGaze，其 GPU 項保持 `not_exercised`。可使用已知需要 DeepGaze 的事件，
或另外寫真模型元件測試確認載入／forward；不修改三級門檻來強迫正式 benchmark 呼叫。
元件測試成功不能代替 pipeline 真實窗口與長片品質驗證。其餘效能對照使用相同暖快取條件。

收集器目前沒有 `--precrop left/right`、Pose batch、saliency size 等覆寫入口。
M3A 與部分 P0／P3 要用 `python -m reframe` 或新增受控工具；先保存其完整命令、有效設定與驗證程式。
比較工具須定義座標配對／數值容差並報實際差異，不能默默忽略未配對人物或以 CPU 結果代替。
原測試中 CPU box／keypoint 3 像素容差只適用其測試，不自動變成所有 CUDA 素材的品質驗收標準。

## 7. 計時、分段與驗收判讀

- `subprocess_wall_seconds`／pipeline `elapsed_seconds`、setup/init、解碼、render_write、
  YOLO／Pose／DeepGaze 次數與各階段時間分開列。外層 source ffprobe 全片掃描與備份可能額外耗時。
  首次下载不混入暖機穩態加速比；新組合只跑一次時清楚標註是否冷啟動。
- GPU 非同步；host wall time 包含不同等待，不能當 CUDA kernel 時間。M5／M6 使用暖機後的
  bounded loop＋CUDA events，僅於測試邊界同步；不要把逐幀 synchronize 放進正式 pipeline。
- `video_io.cuda_allocator_peak` 只涵蓋 PyTorch allocator，不包含 NVDEC／NVENC／其他程序全部顯存。
  長片另觀察 allocated／reserved 與 `nvidia-smi`；單一峰值不能證明沒有持續累積。
- 每 5 秒及切鏡 Segment 對照真實畫面。`seg_detector_calls`／`seg_predicted_frames` 是分割次數／預測幀；
  `seg_detector_gap_counts` 是實際兩次偵測的距離，`seg_interval_counts` 只是排程候選間隔分布。
- `seg_measurement_age_counts` 數人物觀測，不是影片幀數；`pose_cache_age_bins` 與 Pose 推論數按 ROI。
  `pose_extension_hits`／`pose_extension_reasons`／`pose_refresh_reasons` 配合 `seg_mask_evidence_counts` 看安全沿用。
- `saliency_risk_escalations`／`saliency_cooldown_bypasses`／`saliency_risk_reasons` 是事件與原因，
  一事件可含多原因；`saliency_pose_reliability_reasons` 也不是互斥影格分類。原因總和不必等於事件數。
- `saliency_tensor_cross_device_copies` 是影格數，`*_batches` 才是批次数；M6 同看 preprocess＋transfer，
  不把計時歸類移動當作加速。M5 另看 `video_io.pixel_metrics.bilinear_sampled_calls` 是否實際觸發。
- collector 驗證 Segment 加總、原 FPS（差 ≤1e-6）、原幀數／來源短測上限、音訊存在與編碼器。
  音畫終點容差 `max(0.1 秒, 2/FPS)`；通過不代表人工聽看已同步。沒有聲音的來源保持無聲。
- 模型需有呼叫數與實際 CUDA device；DeepGaze forward=0 記 `not_exercised`。主角切換計數不等於視覺失焦，
  原 3D 素材使用者確認焦點全程正確，第二人部分片段確實更顯著；不要為壓低計數破壞此結果。

## 8. 續作、備份與交付報告

成功重用需要來源／模型 hash、程式 fingerprint、Python／套件／GPU／FFmpeg、有效設定全部相同，
且原 output.mp4 SHA 未變。full 必須在**同一 output-dir** 找到這份原片的相符 `max_frames=180` 成功記錄。
90／181 幀、剪輯、改 batch／profile／期限／dead-zone 都不替代該短測；來源不足 180 幀只測到實際幀數。
程式 fingerprint 或套件改變會使 identity 改變。純文件提交不改變 code fingerprint，可沿用其餘條件相同的成功紀錄；
仍須另外記錄測試時 commit，不能把舊版測試當成不同實作的新驗證。

```text
reframe-t4-final/
  environment/                       # 手動保存版本、硬體與 hash
  results.json                       # 最近一次呼叫，不是完整歷史總表
  diagnostics.zip                    # 診斷，不含大型 MP4
  stage2-short180-<identity>-attempt001/
    native.log                       # 原 stdout/stderr 與 traceback
    record.json                      # command、identity、Summary、裝置、ffprobe、檢查結果
    run.segments.csv / run.segments.jsonl
    output.mp4
  stage2-full-<identity>-attempt001/...
```

每組完成或可捕捉的失敗會備份；runtime 硬性刪除時 finally 無法保證執行。
中斷的同一 profile 下次從第一幀重跑；這不是影片中途模型狀態恢復。
還原時複製整個備份含 MP4 到本機 output-dir；只還原 ZIP 無法驗證成功輸出。
不要刪失敗 attempt、編輯 record 成成功或覆寫原 log。報告應遍歷各 attempt 的 record，而非只讀最新 results。

新增 `docs/reports/t4-YYYY-MM-DD-<short-sha>.md` 與可攜 JSON／CSV；記錄：

| 欄位 | 必填內容 |
| --- | --- |
| 範圍 | M1–M6、M3A/B/C、D0、P0–P4 各自狀態與測試 commit；C 記未選／不適用 |
| 環境與來源 | GPU、driver／CUDA／套件／FFmpeg、影片／模型 SHA、ffprobe、事件來源時間映射 |
| 條件 | 完整命令、tracking_settings、資料流、batch、精度、dead-zone、冷／暖快取 |
| 執行 | 實際裝置／fallback、模型呼叫、幀數／FPS／音訊、耗時與顯存、Segment 檔 |
| 品質 | 背面／手部／交錯／暗場／3D／切鏡具體時間點、漂移／裁切／焦點、人工音畫檢查 |
| 判定 | `pass`／`fail`／`not_exercised`／`pending`、證據路徑、未完成原因与下一步 |

`pass` 必須符合該列全部驗收；只通過部分就逐項拆列。`fail` 保留原錯誤；`not_exercised` 表示
測試已跑但相關路徑未觸發；`pending` 表示尚未執行或缺素材／工具／環境。
發現回歸時只修正必要範圍、重跑相關短測再決定是否重跑長片；不要為取得綠燈停用裝置、時效或尾幀檢查。

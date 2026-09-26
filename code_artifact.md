# 虛擬攝影機阻尼運動平滑參數調校指南

在自動直式重構（Auto-Reframe）系統中，虛擬攝影機的平滑程度決定了成品的「電影感」與「觀看舒適度」。若參數設置不當，畫面容易產生**機械式抖動（Jitter）**、**嚴重滯後導致出框（Lag）**或**鏡頭晃過頭的反覆震盪（Overshoot）**。

本指南針對系統中的二階阻尼運動方程與對應 CLI 參數提供完整的調校邏輯與實戰配方。

---

## 1. 物理動力學模型與公式

系統在每影格更新攝影機中心座標 $(X, Y)$ 與縮放倍率 $Z$ 時，採用如下彈簧—阻尼物理模型（Spring-Damper Simulation）：

$$v_{t+1} = (v_t \cdot \lambda_{\text{damping}}) + ((Target_t - Current_t) \cdot \alpha_{\text{response}})$$
$$v_{t+1} = \text{clamp}(v_{t+1}, -V_{\text{max}}, V_{\text{max}})$$
$$Current_{t+1} = Current_t + v_{t+1}$$

### 核心參數對照表

| CLI 參數名稱 | 數學符號 | 物理類比 | 預設值 (Talking Head) | 調高時的影響 | 調低時的影響 |
| :--- | :---: | :--- | :---: | :--- | :--- |
| `--motion-response` | $\alpha_{\text{response}}$ | 彈簧剛度 (Stiffness) | `0.12` | 鏡頭反應極快，但易引起抽搐與超調 | 鏡頭平滑柔和，但易滯後脫焦 |
| `--motion-damping` | $\lambda_{\text{damping}}$ | 運動慣性 / 黏滯阻尼 | `0.82` | 減速滑行尾韻長，消除微小手震 | 減速戛然而止，停頓生硬 |
| `--max-step-x` / `-y` | $V_{\text{max}}$ | 最大物理速度截斷 | `7.0` / `5.0` | 允許劇烈跟鏡，但突變時可能引起暈眩 | 杜絕劇烈晃動，但極快運動下主體會被切掉 |
| `--target-alpha` | $\alpha_{\text{filter}}$ | 目標低通濾波 | `0.12` | 即時跟隨當前觀測，減少延遲 | 濾除檢測邊界框震顫，但增加鏡頭反應延遲 |
| `--zoom-response` | $\alpha_{\text{zoom}}$ | 變焦彈力係數 | `0.08` | 快速推進/拉遠鏡頭 | 變焦溫和無感 |
| `--zoom-damping` | $\lambda_{\text{zoom}}$ | 變焦阻尼係數 | `0.85` | 變焦停止時滑順過渡 | 變焦停止時帶有機械頓挫感 |

---

## 2. 常見畫面破綻與對症調校矩陣

| 症狀現象 | 根本原因 | 調校處方 |
| :--- | :--- | :--- |
| **鏡頭微幅高頻抖動 (Jitter)**<br>主體明明站著不動，鏡頭卻一直微幅震顫 | 目標邊界框在每格有微小像素雜訊，且 $\alpha_{\text{response}}$ 太高或阻尼過低 | 1. 調降 `--motion-response`（例：`0.12` $\to$ `0.08`）<br>2. 調升 `--motion-damping`（例：`0.82` $\to$ `0.88`）<br>3. 調降 `--target-alpha`（例：`0.12` $\to$ `0.08`） |
| **鏡頭嚴重拖沓、出框 (Lagging)**<br>主角起跑或大步走時，人已經走到畫面邊緣鏡頭才慢慢動 | 慣性拖拽過大，或最大步長被死鎖 | 1. 調大 `--max-step-x`（例：`7.0` $\to$ `14.0`）<br>2. 調高 `--motion-response`（例：`0.12` $\to$ `0.16`）<br>3. 適度降低 `--motion-damping`（例：`0.82` $\to$ `0.76`） |
| **鏡頭超調反彈 (Overshoot / Oscillation)**<br>主角停下腳步時，鏡頭衝過頭又往回拉一下才停住 | 典型的**欠阻尼系統**（$\alpha_{\text{response}}$ 過大而阻尼比不足） | 1. 調降 `--motion-response`（減小拉力）<br>2. 調高 `--motion-damping`（提供足夠剎車能量） |
| **縮放忽近忽遠 (Zoom Pumping)**<br>說話者手部晃動或稍微前傾時，畫面反覆 Zoom-in / Zoom-out | 偵測框高度頻繁微調，引起數位變焦神經過敏 | 1. 降低 `--zoom-alpha`（預設 `0.035` $\to$ `0.02`）<br>2. 降低 `--zoom-response`（預設 `0.08` $\to$ `0.04`）<br>3. 提高 `--zoom-damping`（預設 `0.85` $\to$ `0.90`） |

---

## 3. 標準四步驟調校流程 (Step-by-Step Calibration)

建議使用 `--save-debug-preview` 生成輔助診斷影片，並依照下列順序由底層至外層調校：

```
[步驟 1: 濾除目標雜訊] ──> [步驟 2: 尋找極限靈敏度] ──> [步驟 3: 平衡阻尼尾韻] ──> [步驟 4: 設限物理邊界]
  (target-alpha)           (motion-response)           (motion-damping)         (max-step-x/y)
```

### 步驟 1：過濾目標觀測雜訊 (`--target-alpha`)
- **目的**：消除 YOLO 遮罩或關節點在相鄰影格間產生的高頻幾何抖動。
- **做法**：先將 `--motion-damping` 設為 `0.0`（純比例控制），觀察主體靜止時鏡頭是否穩定。若靜止時目標點仍在抖動，將 `--target-alpha` 從 `0.12` 逐步降至 `0.06` ~ `0.08`，直到靜態時鏡頭十字準星完全穩定。

### 步驟 2：調校跟隨靈敏度 (`--motion-response`)
- **目的**：決定鏡頭對加速度變化的啟動時間。
- **做法**：尋找影片中主體突然轉移位置的片段（例如站起或橫向快步走）。
  - 若主體啟動時迅速偏出畫面中心，調高 `--motion-response`（每次加 `0.02`）。
  - 當開始出現「主體停止後鏡頭仍往前衝並回彈」時，表示已達臨界震盪點，需將數值調回前一個階梯。

### 步骤 3：配置運動阻尼與滑行感 (`--motion-damping`)
- **目的**：決定鏡頭減速時的柔和尾韻（Deceleration Curve）。
- **做法**：觀察主體停下腳步的瞬間。
  - 電影感運鏡通常需要鏡頭在主體停止後，仍有約 $0.3 \sim 0.5$ 秒的漸慢滑行（Ease-out）。
  - 將 `--motion-damping` 設定在 `0.80 \sim 0.88` 區間。數值越高，滑行越絲滑；若高於 `0.92` 則會造成嚴重的剎車不及。

### 步驟 4：設定防暈眩安全閥 (`--max-step-x`, `--max-step-y`)
- **目的**：避免因短暫誤檢（例如背景雜訊被判為人）導致攝影機瞬間跨屏飛移。
- **公式原則**：
  $$V_{\text{max}} \approx \frac{\text{主體最大正常瞬時移動像素}}{FPS} \times 1.25$$
  - 一般 30fps 橫向移動影片，橫搖限制 `--max-step-x` 建議設為 `6.0 ~ 12.0`。
  - 人類垂直上下位移（起立蹲下）頻率較低，`--max-step-y` 建議設定為橫向的 $60\% \sim 70\%$（例如 `4.0 ~ 7.0`）。

---

## 4. 針對典型場景的參數配方 (Production Presets)

### 配方 A：單人 Vlog / 講座 / 直播切片 (Talking Head)
*特點：主體位移小、追求鏡頭極度平穩、杜絕任何機械抖動。*
```bash
python auto_reframe.py input.mp4 output.mp4 \
    --preset talking_head \
    --target-alpha 0.08 \
    --motion-response 0.09 \
    --motion-damping 0.86 \
    --max-step-x 6.0 \
    --max-step-y 4.0 \
    --zoom-response 0.05 \
    --zoom-damping 0.90
```

### 配方 B：動態舞蹈 / 運動賽事 / 滑板 (Sports & Action)
*特點：主體速度極快、方向瞬變，必須優先保證主體不出框。*
```bash
python auto_reframe.py input.mp4 output.mp4 \
    --preset sports \
    --target-alpha 0.18 \
    --motion-response 0.17 \
    --motion-damping 0.74 \
    --max-step-x 16.0 \
    --max-step-y 10.0 \
    --min-zoom 1.0 \
    --max-zoom 1.25
```

### 配方 C：雙人訪談 / 播客 (Podcast Dialogue)
*特點：兩人可能微動，變焦需極端克制，避免因發言交替造成攝影機來回橫搖拉風箱。*
```bash
python auto_reframe.py input.mp4 output.mp4 \
    --preset talking_head \
    --two-person-framing \
    --two-person-threshold 0.72 \
    --min-subject-hold-frames 24 \
    --switch-score-threshold 1.30 \
    --motion-response 0.08 \
    --motion-damping 0.88 \
    --zoom-response 0.03 \
    --zoom-damping 0.92
```

---

## 5. 阻尼運動動態行為除錯檢驗

在執行含 `--save-debug-preview` 的測試時，可透過輸出畫面上的 HUD 檢視以下數值以驗證阻尼表現：

1. **`velocity_x` / `velocity_y` 曲線**：
   - 當主體靜止時，速度向量應在 5~10 影格內平滑衰減收斂至 $0.0$。
   - 若速度長時期在正負值之間交替跳動（如 $+1.2 \to -0.8 \to +0.9$），代表系統處於**共振震盪狀態**，必須立即調降 `motion-response`。
2. **`Zoom` 數值穩定度**：
   - 說話者正常手勢揮動時，Zoom 倍率的小數點後兩位不應有連續且顯著的抽動。若有抽動，請調降 `zoom-alpha`。
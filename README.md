# Fast Timing Model Estimation for New PVT Corners

用機器學習從**已標定的 5 個 PVT corner** 外推出**新 PVT corner** 的完整
Liberty (`.lib`) 時序／功耗查找表。題目來自 CAD contest Problem D。

一句話總結成果：以留出 cell 直接量測，pooled 分數 **96.34**（alpha 拓樸，
比賽計分公式）；換算到官方 100-cell 測試母體約 **97.3 – 97.6**。delay 四類表
（`cell_rise` / `cell_fall` / `rise_transition` / `fall_transition`）全部達
**98.9 – 99.2**，誤差幾乎全部集中在 `fall_power` 一張表。

---

## 目錄

- [問題設定](#問題設定)
- [快速開始](#快速開始)
- [演算法](#演算法)
- [研究成果](#研究成果)
- [有效與無效的方法](#有效與無效的方法)
- [專案結構](#專案結構)
- [已知限制](#已知限制)

---

## 問題設定

一個 standard cell library 在不同 **PVT corner**（Process / Voltage /
Temperature）下有不同的時序與功耗特性。完整標定每一個 corner 需要大量
SPICE 模擬，成本很高。本專案要做的是：

> 給定 5 個**已完整標定**的 corner，預測另外 10 個 corner 的所有查找表數值。

### Corner 命名

`<process><voltage>v<temp>c`，例如 `ff0p99vm40c` = fast-fast 製程、0.99 V、
−40 °C（`m` = minus、`0p99` = 0.99）。

資料集共 15 個 corner，分成三個電壓層：

| 電壓層 | ss | ff | tt |
|---|---|---|---|
| 升壓 (boost) | `ss0p9v` | `ff1p1v` | `tt1p0v` |
| 標準 (nominal) | `ss0p81v` | `ff0p99v` | `tt0p9v` |
| 降壓 (buck) | `ss0p72v` | `ff0p88v` | `tt0p8v` |

每層各有 −40 °C / 125 °C 兩個溫度（tt 只有 25 °C 一點），合計 5 × 3 = 15。

### 三個拓樸（哪些是已知、哪些要預測）

| 拓樸 | anchor（已知的 5 個） | target（要預測的 10 個） | 外推方向 |
|---|---|---|---|
| **alpha** | 標準電壓 | 升壓 5 + 降壓 5 | 上下**各一步** |
| **beta** | 升壓 | 標準（一步下）+ 降壓（**兩步**下） | 向下 |
| **final** | 降壓 | 標準（一步上）+ 升壓（**兩步**上） | 向上 |

三個拓樸的程式定義在 `src/models/phase4_features.py`（`ALPHA_TOPOLOGY` /
`BETA_TOPOLOGY` / `FINAL_TOPOLOGY`）。

### 要預測的表格

每個 cell 的每個 pin／timing arc 底下有 6 種 7×7 查找表：

`cell_rise`、`cell_fall`、`rise_transition`、`fall_transition`（delay 家族）
以及 `rise_power`、`fall_power`（internal power 家族）。

`index_1` = input transition（slew）、`index_2` = output capacitance（load）。
**這兩軸在全部 15 個 lib 中逐位元相同**，所以格點可以位置對位置直接對應，
不需要重新內插。

### 資料佈局

```
testcase/
├── training_set/base_nom_{0p8v,0p9v,1p0v}/   # 官方訓練集：400 cell × 15 corner（全有值）
├── alpha_test/full/                          # 100 cell × 5 個標準電壓 corner（有值）
├── alpha_test/partial/                       # 同 100 cell × 10 個升/降壓 corner（values 挖空）
├── beta_test/  final_test/                   # 另外兩個拓樸的對應資料
```

訓練集的 400 個 cell 與測試用的 100 個 alpha cell **零重疊**。

---

## 快速開始

### 1. 環境

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# torch 建議裝 CPU wheel（Apple Silicon 上會自動走 MPS）：
# pip install torch --index-url https://download.pytorch.org/whl/cpu
```

相依：`numpy`、`scipy`、`scikit-learn`、`torch`、`pytest`。

### 2. 跑測試

```bash
pytest                                   # 全部測試
pytest tests/test_phase4_features.py     # 單一檔
```

### 3. 量分數（日常主要用法）

這支腳本**只印分數、不寫任何檔案**：

```bash
# alpha 拓樸、預設 config、全部 10 個 target corner
python3 scripts/phase4_final_validate.py --stage alpha

# 換拓樸
python3 scripts/phase4_final_validate.py --stage beta
python3 scripts/phase4_final_validate.py --stage final

# 只跑一個 corner（快速煙霧測試）
python3 scripts/phase4_final_validate.py --stage alpha --corners tt1p0v25c

# 存下逐點誤差，供下游稽核腳本使用
python3 scripts/phase4_final_validate.py --stage alpha \
    --dump-errors output/_phase4_cache/alpha_huber_s1_errors.npz
```

主要參數：

| 參數 | 說明 |
|---|---|
| `--stage {alpha,beta,final}` | 選拓樸，預設 `alpha` |
| `--config <tag>` | 模型 config，預設 `mlp_w256_b4_huber` |
| `--corners a,b,c` \| `all` | 只跑部分 target corner |
| `--seeds N` | seed ensemble 數量，預設 1 |
| `--fold {0..4}` | 5-fold 交叉驗證；不給就用官方 320/80 切分 |
| `--anchors` / `--targets` | 自訂 corner 集合，繞過預定義的 stage |
| `--dump-errors PATH` | 把逐點誤差存成 `.npz` |

**執行時間**：一個拓樸的 10 個 corner 在 Apple Silicon（MPS）上約 **40–60 分鐘**
（每個 corner 訓練一個獨立模型）。

> ⚠️ **一次只跑一個 corner。** 批次跑整個 stage 偶爾會被系統非確定性中止
> （成因不明，八個推測都已被排除）。用內建的逐 corner 批次腳本，並關掉緩衝，
> 否則被中止時會只留下 0 bytes 的 log：
>
> ```bash
> PYTHONUNBUFFERED=1 scripts/run_corner_sweep.sh <stage> <config> <log-prefix> [dump-dir]
> # 例：PYTHONUNBUFFERED=1 scripts/run_corner_sweep.sh alpha mlp_w256_b4_huber alpha_huber
> ```
>
> 它會逐 corner 呼叫 validate、每個 corner 最多重試 3 次、log 寫到
> `logs/round_20260811/<prefix>_<corner>.log`。中止不會污染數據（分數與 dump
> 都只在結束時一次寫出）。

合併逐 corner 的誤差 dump：

```bash
python3 scripts/merge_error_dumps.py <out.npz> <in1.npz> <in2.npz> ...
```

### 4. 產出 `.lib` 交付檔

```bash
python3 scripts/phase4_final_predict.py --dry-run    # 照常計算與檢查，不寫檔
python3 scripts/phase4_final_predict.py              # 真的寫 output/*.lib
```

這是唯一會寫 `.lib` 的腳本。它用**全部 400 個訓練 cell** 重新訓練（留出驗證
已在 validate 階段做完），對 100 顆 alpha cell 推論，並以模板填空產生輸出。
路徑與拓樸寫死 alpha，沒有 `--stage`。

> 本專案目前**不產出交付檔案**（`output/*.lib` 已刪除以節省空間）。要還原：
> `git restore output/`（commit `dc2f924`）或重跑上面的指令。

### 5. 分數換算到官方測試母體

留出驗證的 80 顆 cell 與官方 100-cell 測試母體的 **drive strength 分布不同**
（中位數：train400 = 6、alpha = 2、beta = 4、final = 8），而 drive strength
與 `fall_power` 的病理現象強相關。以下腳本吃 `--dump-errors` 產生的 `.npz`，
做 drive-matched 重新加權：

```bash
# 範圍（推薦：這個換算有 ±0.2 級的方法自由度，引用範圍不要引用點估計）
python3 scripts/phase4_composition_sensitivity.py <dump.npz> --population alpha

# 單一分桶方案的點估計（--bucket-scheme 必填、無預設）
python3 scripts/phase4_composition_audit.py <dump.npz> --population alpha \
    --bucket-scheme {coarse|fine|none|occupancy|stable}
```

### 6. 其他分析腳本

| 用途 | 腳本 |
|---|---|
| 三拓樸對照與配對統計 | `scripts/phase4_topology_compare.py` |
| 產出報告用的全部數字（`--json` 可機讀） | `scripts/phase4_report_numbers.py` |
| 跨 corner 平滑的可行性閘門 | `scripts/phase4_smoothing_gate.py` |

---

## 演算法

整體是一個**逐格點回歸**問題：把每個 `(cell, pin, arc, table_type, grid_point)`
當成一筆樣本（7×7 = 49 點／表），用該 cell 在 5 個 anchor corner 上的對應數值
及其衍生量當特徵，回歸出 target corner 的數值。

**每個 target corner 訓練一個獨立模型**（10 個 corner = 10 個模型）。

### 特徵：43 維

程式碼在 `src/models/phase4_features.py`（`FEATURE_NAMES`）。

| 類別 | 特徵 | 數量 | 物理意義 |
|---|---|---|---|
| **anchor 原始值** | `log_anchor_<corner>` | 5 | 5 個 anchor corner 在**同一格點**的表格值取 `log(\|v\|+ε)` |
| **響應簽名**（response signature） | `log_ratio_ff_hot_cold`、`ss_hot_cold`、`ss_ff_hot`、`ss_ff_cold`、`tt_ss_hot`、`tt_ff_hot` | 6 | anchor **兩兩之間的 log 比值**，把「純溫度敏感度」與「純製程敏感度」分離出來 |
| **表格局部梯度** | `log_grad_row_<anchor>`、`log_grad_col_<anchor>` | 10 | 沿 7×7 表格 row／col 方向的一階差分，描述該格點附近的曲面形狀 |
| **格點座標** | `slew_idx_norm`、`load_idx_norm` | 2 | 行列索引正規化到 [−1, 1] |
| **格點刻度** | `log_slew`、`log_load` | 2 | `index_1`（input transition）、`index_2`（output cap）的實際值取 log |
| **cell 屬性** | `log_drive_strength` | 1 | 從 cell 名稱解析（`AN2AM16` → 16） |
| **cell 屬性** | `family_code` | 1 | cell function family 的整數編碼（固定詞彙表，OOV → −1） |
| **arc 屬性 one-hot** | `sense_*` | 4 | `timing_sense`（positive/negative/non_unate/na） |
| **arc 屬性 one-hot** | `ttype_*` | 6 | `timing_type`（combinational / rising_edge / …） |
| **表格類型 one-hot** | `table_*` | 6 | 這筆樣本屬於 6 種表格中的哪一種 |

**「響應簽名」是本專案最有價值的特徵設計**——它不是把 corner 的 (P, V, T)
座標直接餵給模型，而是餵「這顆 cell 在這個格點上**對製程和溫度的反應有多敏感**」。
anchor 的角色（ff_hot / ss_cold / tt_mid …）是從 corner 名稱動態解析
（`src/features/corners.py`），不是寫死的。

### 標籤：log-ratio

```
y = log( |target| / |nearest_anchor| )
```

`nearest_anchor` 是與 target corner **同製程、同溫度、只差電壓**的那個 anchor。
重建時：

```
prediction = |nearest_anchor| × exp(clip(ŷ, ±20))
```

anchor 本身為 0 的點（已知失效的 power arc）不參與訓練，輸出強制為 0。

比較實驗顯示 log-ratio 標籤比直接回歸原值（raw）勝 3.5 分以上——因為表格數值
跨越好幾個數量級，而 corner 之間的**比值**遠比絕對值穩定。

### 模型：殘差 MLP

`src/models/phase4_mlp.py`，預設 config `mlp_w256_b4_huber`：

```
Linear(43 → 256) + ReLU
  ↓
4 × ResBlock:  LayerNorm → Linear(256→512) → ReLU → Dropout(0.05) → Linear(512→256) → +skip
  ↓
LayerNorm → Linear(256 → 1)
```

| 超參數 | 值 |
|---|---|
| optimizer | Adam，lr 2e-3，weight decay 1e-5 |
| batch size | 8192 |
| max epochs | 150，early stopping patience 15 |
| LR 排程 | `ReduceLROnPlateau`（factor 0.5、patience 4、min 1e-6） |
| 損失 | **Huber**，δ = log 2 ≈ 0.693 |
| 輸入正規化 | 逐欄 mean/std（只用 dev-train 列擬合） |
| 裝置 | MPS（Apple GPU）優先，否則 CPU |

可選 config：`mlp_w256_b4_full`（同結構但用 MSE）、`mlp_w192_b3_full`（較小）、
`gbdt_full`（`HistGradientBoostingRegressor`）、以及兩個 `*_tiny_smoke`
（只供管線煙霧測試）。

### 訓練協議與零洩漏

- 400 個官方訓練 cell 用固定 seed 切 **320 train / 80 validation**。
- 早停用的 dev-train / dev-val 是從 **320 之內**再切一次，**絕不碰到那 80 顆**。
- 特徵矩陣只讀 5 個 anchor corner，**從不讀 target corner 的值**；target 真值
  只作為 label 進入，完全對應真實推論時的輸入形狀。
- 這些不變量有 runtime assertion 與單元測試把關。

### 評分公式

`src/scoring/scorer.py`：

```
e_i    = min(1, |y_i − ŷ_i| / |y_i|)        # 逐點 capped 相對誤差
Score  = 100 − 100 × sqrt( mean(e_i²) )     # RMS 匯總
```

`y_i = 0` 時，`ŷ_i = 0` 則 `e_i = 0`，否則 `e_i = 1`。

一個拓樸的 pooled 分數是把該拓樸**全部 10 個 target corner × 全部驗證 cell**
的逐點誤差攤平成一個陣列後套公式——**不是**先算各 corner 分數再平均。

### 輸出：模板填空

`src/liberty/`：

- **parser** 用結構化括號掃描逐層剖析 `library → cell → pin → arc → table`，
  並記錄每一列數值在原始檔案文字中的**字元偏移量**（`row_spans`）。
- **writer** 只用那些偏移量對原始檔案文字做**字串切片替換**，
  **絕不重新序列化整個 lib**。已有值的表格逐位元組保持原樣。
- 數值格式 `'%.6g'`（0 一律印 `0` 而非 `-0`），與參考 lib 中約 120 萬個數值
  逐一比對 0 mismatch。

這樣才能通過官方 checker 的逐字元對齊要求。

---

## 研究成果

以下全部是現行預設 config（`mlp_w256_b4_huber`、1 seed、80 顆留出 cell、
全 10 個 target corner）的實測值。

### 總分

| 口徑 | alpha | beta | final |
|---|---|---|---|
| 訓練 cell 留出（直接量測） | **96.3359** | **95.7273** | **95.7718** |
| 官方 100-cell 母體（drive 匹配換算，範圍） | **97.32 – 97.55** | **96.91 – 97.25** | **97.11 – 97.30** |

每個口徑 1,965,880 個評分點。

> **官方口徑是範圍，不是點估計。** 換算依賴 drive strength 分桶方案，20 種同樣
> 站得住腳的重建讓分數滑動 0.19 – 0.34 分，而且**沒有任何判準能選出該用哪一種**
> （兩個 prevalence proxy 把候選方案排成相反順序）。這個滑動幅度比本專案量到的
> 任何一項模型改進都大。

### 逐 corner（三個拓樸）

每個拓樸的 target 集合不同（見[問題設定](#三個拓樸哪些是已知哪些要預測)），
所以三欄的 corner 只有部分重疊；`—` 表示該 corner 在該拓樸裡是 anchor（已知），
不需要預測。

| target corner | alpha | beta | final |
|---|---|---|---|
| `tt1p0v25c` | **97.78** | — | 96.70 |
| `tt0p9v25c` | — | **97.81** | **97.44** |
| `tt0p8v25c` | 97.32 | 96.63 | — |
| `ss0p9v125c` | 96.98 | — | 95.79 |
| `ss0p9vm40c` | 96.42 | — | 95.06 |
| `ss0p81v125c` | — | 97.01 | 96.85 |
| `ss0p81vm40c` | — | 96.21 | 96.41 |
| `ss0p72v125c` | 96.95 | 95.78 | — |
| `ss0p72vm40c` | **94.96** | **93.83** | — |
| `ff1p1v125c` | 96.16 | — | 94.83 |
| `ff1p1vm40c` | 95.77 | — | **94.10** |
| `ff0p99v125c` | — | 96.20 | 96.12 |
| `ff0p99vm40c` | — | 95.89 | 95.53 |
| `ff0p88v125c` | 95.91 | 94.85 | — |
| `ff0p88vm40c` | 95.97 | 94.56 | — |
| **pooled** | **96.34** | **95.73** | **95.77** |

三個拓樸共通的樣態：

- **製程軸：tt > ss > ff**，三個拓樸一致（各拓樸的製程平均分別是
  97.55 / 96.33 / 95.95、97.22 / 95.71 / 95.38、97.07 / 96.03 / 95.15）。
  tt 只有 25 °C 一個溫度點、且位在製程分布中央，最好預測。
- **溫度軸：低溫（`m40c`）比高溫（`125c`）難**，12 組同製程同電壓的配對中
  **11 組**如此（唯一例外是 alpha 的 `ff0p88`，+0.05，在噪音內）。
  最大的一組是 `ss0p72`：alpha −1.98、beta −1.95。
- **步數軸：各拓樸的最低分都落在「離 anchor 兩步」的那半邊**。beta 是
  `ss0p72vm40c` 93.83、final 是 `ff1p1vm40c` 94.10；alpha 全部 target 都只有
  一步，最低分 `ss0p72vm40c` 也還有 94.96。同一顆 corner 從一步變兩步的代價
  很清楚：`ss0p72vm40c` 94.96（alpha，一步）→ 93.83（beta，兩步）＝ **−1.13**。
- **但步數不壓過製程軸**：tt 的兩步 corner（beta `tt0p8v25c` 96.63、final
  `tt1p0v25c` 96.70）仍然高於好幾個只差一步的 ss／ff corner。**製程選得對，
  比離得近更重要。**

### 逐表格類型（三個拓樸）

| table_type | alpha | beta | final |
|---|---|---|---|
| `cell_rise` | **99.19** | **98.80** | **99.00** |
| `cell_fall` | 99.13 | 98.77 | 98.91 |
| `rise_transition` | 99.13 | 98.73 | 98.90 |
| `fall_transition` | 98.89 | 98.40 | 98.76 |
| `rise_power` | 97.75 | 97.56 | 98.13 |
| `fall_power` ← 唯一系統性弱點 | **91.47** | **90.14** | **90.02** |

delay 四類在三個拓樸下都是 **98.4 – 99.2**，相當於 1 % 左右的相對誤差，達到
文獻中 ML library characterization 的可用水準。**拓樸換了，弱點的位置不換**——
掉分永遠集中在 `fall_power`：delay 四類的三拓樸全距只有 **0.36 – 0.49 分**，
`fall_power` 卻是 **1.45 分**。

不過「pooled 的差距全都來自 `fall_power`」只對一半成立：

- **final 對 alpha**（−0.56）整包是 `fall_power`——兩者其餘 5 張表的分數同為
  98.70，一分不差。
- **beta 對 alpha**（−0.61）是兩邊一起掉——其餘 5 張表也從 98.70 掉到 98.39。

即**向上外插只傷 `fall_power`，向下外插連 delay 一起傷**。這是三個拓樸之間
唯一一處樣態不對稱的地方（pooled 分數上的拓樸代價則是對稱的，見下）。

### 誤差結構：問題極度集中

把 e² 質量（誤差平方的總量）拆開來看，三個拓樸講的是同一個故事：

| 子群 | 佔點數 | 分數 | **佔總 e² 質量** |
|---|---|---|---|
| | alpha / beta / final | alpha / beta / final | alpha / beta / final |
| `fall_power`：符號翻轉點 | **0.085 / 0.126 / 0.128 %** | 0.00 / 0.00 / 0.00 | **63.4 / 69.2 / 71.8 %** |
| `fall_power`：近零值點 | 0.58 / 0.62 / 0.47 % | 78.3 / 78.5 / 76.0 | 20.3 / 15.7 / 15.0 % |
| `fall_power`：其餘 | 15.9 / 15.8 / 15.9 % | 97.8 / 98.1 / 97.6 | 5.8 / 3.2 / 5.2 % |
| 其他 5 種表格 | 83.5 % | 98.7 / 98.4 / 98.7 | 10.5 / 11.9 / 7.9 % |

**佔 0.085 % 的點貢獻了 63 % 的誤差**（alpha）；到了 final 是 0.128 % 的點
貢獻 71.8 %。**外推越遠，誤差質量越往這個極小的子群集中。**

- **符號翻轉點**：target 的值與 anchor 異號。模型架構是 `|anchor| × exp(·)`，
  結構上繼承 anchor 的符號，這些點**必錯**（分數 0，三個拓樸皆然）。
  診斷顯示這是**輪廓現象**——2D 曲面的零交越曲線隨 corner 移動——逐點回歸
  結構上表達不了。
- **近零值點**（|y| < 1e-4）：相對誤差的分母趨近 0，誤差天然爆炸。

這兩群加起來只佔 0.6 – 0.75 % 的點，卻扛著 **79 – 87 %** 的誤差質量。

### 拓樸代價：兩步外插約 −0.6，且對方向對稱

| 拓樸 | 相對 alpha 的代價 |
|---|---|
| beta（向下兩步） | −0.609 |
| final（向上兩步） | −0.564 |

差 0.045，小於 seed 噪音（0.048）——**在 pooled 分數上量不到方向的差別**，
儘管物理上 SS 0.72 V 當 target 與當 anchor 直覺上不該一樣貴。

拓樸層級的對稱其實是兩個約 0.16 的反向次級效應相消：一步那半邊由升壓 anchor
往下略勝、兩步那半邊由降壓 anchor 往上略勝，兩者都不顯著。第二步本身則極顯著
地要價 −1.09（向下）／−1.33（向上），10/10 corner 為負。

### 可重現性

訓練管線在 **alpha/beta 兩個拓樸 × mse/huber 兩個 config × 10 corner = 40 次
重跑全部逐位元相符**（相隔兩週的重跑亦然）。5-fold 交叉驗證下（早期 mse 口徑）
pooled 為 **96.45 ± 0.43**，折間標準差就是「任何改進宣稱必須顯著超過」的噪音水準。

---

## 有效與無效的方法

這是本專案最有參考價值的部分：**大部分聽起來合理的改進都是無效的**。

### ✅ 有效

| 方法 | 增益 | 說明 |
|---|---|---|
| **log-ratio 標籤** | **+3.5 以上** | 回歸 `log(target/anchor)` 而非原值。單項最大改進 |
| **響應簽名特徵** | GBDT **+1.7** / MLP +0.35 | anchor 兩兩 log 比值，分離溫度／製程敏感度 |
| **MLP 容量調到 256×4** | 掃描出的甜蜜點 | 更大更小都較差 |
| **Huber 損失**（δ = log 2） | **+0.06 – +0.11** | 三個拓樸都小幅為正 |
| 官方訓練集 + 多 anchor（vs 早期單 anchor） | 90.5 → 95.8 | 任務結構正確化 |

Huber 的官方協議實測（80 顆留出 cell、全 10 corner、1 seed）：
pooled 96.2742 → 96.3359 = **+0.0616**，逐 corner 平均 +0.0613 ± 0.0376、
9/10 為正、約 95 % 區間 [+0.038, +0.085]。三個拓樸分別是 alpha +0.0616、
beta +0.1114、final +0.0744——**都為正，但幅度與拓樸難度或尾巴質量無關**
（final 的翻轉質量最高卻只排第二，逐 corner 配對下與 alpha 分不開）。

### ❌ 無效或有害（實測，不要重做）

| 嘗試 | 結果 |
|---|---|
| GBDT + MLP ensemble | 95.68 < 單 MLP，淘汰 |
| `signed_ratio` 標籤（想救符號翻轉） | +0.23，增益不足 |
| 顯式的「符號翻轉」特徵 | **−2.60**，有害 |
| 近零點 5× 加權 | **−3.57**，有害 |
| 跨表格特徵（`fall_power` 加入 `cell_fall`/`fall_transition` 的 anchor 值） | +0.016 ≈ 零 |
| 3-seed ensemble | +0.077，低於預設閘門 |
| 曲面建模（整條 row/col 剖面當特徵） | 3/3 corner 為負 |
| 多拓樸資料增強（訓練列翻倍） | 3/3 corner 為負 |
| 跨 corner 平滑 | 天花板僅 +0.25，未過閘門 |

幾個陰性結果本身是有診斷價值的：

- **跨表格特徵近乎零增益**證明 `cell_fall` / `fall_transition`（同一轉換事件的
  delay 資訊）**不攜帶** `fall_power` 近零點所需的「成分抵消精度」——誤差來自
  **資訊不足**，不是特徵集不夠豐富。
- **跨 corner 平滑**的天花板只有 +0.25，根因是 corner 幾何：同組內最多
  2 電壓 × 2 溫度 = 4 點（tt 組只有 2 點），「不傷真值的平滑基底」幾乎就是
  恆等映射。
- **曲面建模與資料增強都 3/3 為負**，說明瓶頸既不是模型表達力也不是訓練資料量。

四輪獨立診斷一致指向同一結論：**在現有 anchor 特徵集下，誤差已接近不可縮減**。
要再往上，必須同時解決符號翻轉（需要能表達零交越輪廓的結構，不是逐點回歸）
與近零值點的相對誤差爆炸。

---

## 專案結構

```
src/
├── liberty/
│   ├── parser.py        # .lib 剖析（含 row_spans 字元偏移），不含估算邏輯
│   └── writer.py        # 模板填空，不重新序列化
├── features/
│   ├── corners.py       # corner 檔名 → (process, voltage, temperature)
│   ├── cellinfo.py      # cell 名稱 → (base, family, drive_strength)
│   └── align.py         # index_2 格點對齊（本資料集下是 no-op）
├── models/
│   ├── phase4_features.py     # 拓樸定義、特徵工程、label/重建、資料集組裝
│   ├── phase4_mlp.py          # 殘差 MLP + Huber/MSE 損失
│   ├── phase4_gbdt.py         # GBDT 分支
│   ├── phase4_final_config.py # config registry、DEFAULT_CONFIG_TAG
│   └── phase2_*, phase3_*     # 歷史版本（物理模型、早期 ML）
├── scoring/
│   ├── scorer.py        # 比賽計分公式
│   ├── audits.py        # drive-matched 母體換算
│   ├── ensemble.py, loco.py
└── paths.py             # 集中資料佈局常數

scripts/                 # 全部可執行的驗證／預測／稽核腳本
tests/                   # 15 個測試檔（切分決定性、零洩漏、parser 往返、mutation 覆蓋）
docs/                    # 完整研究歷程記錄
logs/                    # 逐 corner 執行 log
```

**分層原則**：Liberty parser 不含估算邏輯；估算模型不碰檔案 I/O；輸出一律模板
填空。數值處理用 numpy 向量化，不對表格寫 Python 逐元素迴圈。

### 文件導覽

| 文件 | 內容 |
|---|---|
| `docs/current_status.md` | **單頁入口**——「現在什麼是真的」。接手前先讀這頁 |
| `docs/round_20260810.md` | 最新一輪：禁止重試清單、beta/final 拓樸實測、機制推翻 |
| `docs/phase4_results.md` | Phase 4 定案報告、5-fold CV、改進輪全記錄 |
| `docs/model_comparison.md` | Phase 3 的 GBDT vs MLP vs 物理模型對照 |
| `docs/algorithm_report.html` | 圖表版報告（逐 corner、子群表、熱力圖） |
| `docs/程式架構.html` | 程式架構導覽 |

---

## 已知限制

1. **符號翻轉點結構上無解**。模型是 `|anchor| × exp(·)`，繼承 anchor 符號。
   要解需要能表達 2D 零交越輪廓的模型，不是逐點回歸。
2. **官方母體分數只能給範圍**。drive 分桶方案有 ±0.2 級的方法自由度，且沒有
   判準能挑出唯一方案。`--bucket-scheme` 刻意設為**必填無預設**，就是為了讓人
   無法在不知情的狀況下引用點估計。
3. **`phase4_final_predict.py` 只支援 alpha**，路徑與拓樸寫死，beta/final
   產不出交付檔。
4. **訓練偶發非確定性中止**，死點固定在第一次碰 MPS 的瞬間，成因不明（八個
   推測全被排除）。中止不污染數據，但「重試會過」不可靠——曾有連續 8 次失敗
   跨越數小時。逐 corner 執行 + 關閉緩衝是唯一可靠的作法。
5. **本專案目前只產出數字，不產出交付檔案**（2026-08-10 決定不參賽）。寫檔的
   程式碼保留但暫不使用。

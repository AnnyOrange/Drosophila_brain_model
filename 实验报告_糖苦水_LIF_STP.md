# 实验报告：糖/苦/水刺激（LIF + STP）

说明：按“先 LIF，后 STP”的顺序逐步执行。每完成一个实验块就追加结果，便于断点续跑。

## 实验设置（固定项）
- 数据：630 版（`2023_03_23_completeness_630_final.csv` + `2023_03_23_connectivity_630_final.parquet`）
- 运行：`conda run -n flybrain`
- 并行：`n_proc=1`
- 试次：`n_run=2`（快速扫描；后续可加大）

---

## LIF-糖：频率扫描（10–200 Hz，步长 10）
**状态**：已完成  
**命令**：
```bash
conda run -n flybrain python my_code/sugar_circuit.py --n-run 2 --n-proc 1 --path-res results/my_code_sugar freq --freqs 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200
```
**结果**：  
- MN9：10–30 Hz 未检测到放电；40 Hz 起出现放电，随频率增加上升，200 Hz 约 90 Hz。  
- 响应神经元数（rate > 0）：10 Hz 时 29 个，200 Hz 时 422 个。  
- Top5（200 Hz）：  
  - 720575940637568838  
  - 720575940629176663  
  - 720575940628853239  
  - 720575940638202345  
  - 720575940621502051  
- MN9 频率-响应（Hz）：  
  - 40: 4.0  
  - 50: 26.5  
  - 60: 38.0  
  - 80: 58.0  
  - 100: 65.5  
  - 150: 85.5  
  - 200: 90.0  

---

## LIF-苦：频率扫描（10–200 Hz，步长 10）
**状态**：已完成  
**命令**：
```bash
conda run -n flybrain python my_code/bitter_circuit.py --n-run 2 --n-proc 1 --path-res results/my_code_bitter --freqs 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200
```
**结果**：  
- 可用 bitter GRN：完整性表中仅 10 个（其余 16 个缺失）。  
- MN9：10–20 Hz 未检测到放电；30 Hz 起出现低频放电，100 Hz 附近约 18 Hz，整体显著低于糖路。  
- 响应神经元数（rate > 0）：10 Hz 时 13 个，100 Hz 时 330 个，200 Hz 时 285 个。  
- Top5（200 Hz）：  
  - 720575940627821896  
  - 720575940614273292  
  - 720575940622371037  
  - 720575940628832256  
  - 720575940614734186  
- MN9 频率-响应（Hz）：  
  - 30: 1.5  
  - 50: 7.0  
  - 70: 11.5  
  - 100: 18.0  
  - 150: 8.5  
  - 200: 7.5  

---

## LIF-水：频率扫描（20–260 Hz，步长 20）
**状态**：已完成  
**命令**：
```bash
conda run -n flybrain python my_code/water_circuit.py --n-run 2 --n-proc 1 --path-res results/my_code_water freq --freqs 20 40 60 80 100 120 140 160 180 200 220 240 260
```
**结果**：  
- MN9：20–120 Hz 未检测到放电；140 Hz 起出现放电并随频率上升（260 Hz 约 48.5 Hz）。  
- 响应神经元数（rate > 0）：20 Hz 时 33 个，260 Hz 时 349 个。  
- Top5（260 Hz）：  
  - 720575940622486922  
  - 720575940625861168  
  - 720575940612579053  
  - 720575940613996959  
  - 720575940606002609  
- MN9 频率-响应（Hz）：  
  - 140: 2.5  
  - 160: 10.0  
  - 180: 20.5  
  - 200: 27.5  
  - 240: 52.0  
  - 260: 48.5  

---

## STP：方案 B（序列刺激）
**状态**：已完成  
**目标**：验证糖/苦/水在 STP 下是否保持“糖高、苦低、水无/弱”的性质  
**设置**：phase1=150 Hz，phase2=47 Hz，1 秒/phase，n_run=10  
**结果**：  
- 糖（sugar→sugar）：  
  - phase1 平均 77.1，phase2 平均 18.5，phase2/phase1 ≈ 0.24  
  - 汇总：`results/my_code_mixed/stp_seq_sugar150_47_summary.csv`  
- 苦（bitter→bitter）：  
  - phase1 平均 10.4，phase2 平均 5.9，phase2/phase1 ≈ 0.61  
  - 汇总：`results/my_code_mixed/stp_seq_bitter150_47_summary.csv`  
- 水（water→water）：  
  - phase1 平均 0.0，phase2 平均 0.0  
  - 汇总：`results/my_code_mixed/stp_seq_water150_47_summary.csv`  

---

## LIF-糖水混合刺激（同时 100 Hz）
**状态**：已完成  
**命令**：
```bash
conda run -n flybrain --no-capture-output python - <<'PY'
from copy import deepcopy
from brian2 import Hz
from model import default_params, run_exp
from my_code.sugar_circuit import SUGAR_NEURONS
from my_code.water_circuit import WATER_NEURONS

params = deepcopy(default_params)
params['n_run'] = 2
params['r_poi'] = 100 * Hz
params['r_poi2'] = 100 * Hz

run_exp(
    exp_name='sugar_water_100_100',
    neu_exc=list(SUGAR_NEURONS),
    neu_exc2=list(WATER_NEURONS),
    params=params,
    path_res='results/my_code_mixed',
    path_comp='2023_03_23_completeness_630_final.csv',
    path_con='2023_03_23_connectivity_630_final.parquet',
    n_proc=1,
    force_overwrite=False,
)
PY
```
**结果**：  
- MN9：87.5 Hz  
- 响应神经元数（rate > 0）：450  
- Top5：  
  - 720575940629888530  
  - 720575940622695448  
  - 720575940618165019  
  - 720575940616103218  
  - 720575940617937543  

### 对照：低频糖与糖水混合（MN9/响应数）
| 条件 | MN9 频率 (Hz) | 响应神经元数 |
| --- | --- | --- |
| sugar 70 Hz | 44.00 | 355 |
| sugar 80 Hz | 58.00 | 358 |
| sugar 90 Hz | 62.50 | 370 |
| sugar 100 Hz | 65.50 | 365 |
| sugar+water 100/100 | 87.50 | 450 |

---

## LIF-论文缺失项：稀疏网格复现（MN9，n_run=2）
**说明**：按论文 Figure 3/4 的联合刺激，使用稀疏频率网格复现（0/50/100/150/200；水含 260）。  
**结果 CSV**：  
- `results/my_code_paper/mn9_sugar_bitter_sparse.csv`  
- `results/my_code_paper/mn9_sugar_ir94e_sparse.csv`  
- `results/my_code_paper/mn9_sugar_water_sparse.csv`  

### Sugar + Bitter（MN9，Hz）
| sugar\bitter | 0 | 50 | 100 | 150 | 200 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 22.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| 100 | 66.00 | 34.00 | 5.00 | 0.00 | 0.00 |
| 150 | 90.50 | 61.00 | 34.50 | 10.00 | 0.00 |
| 200 | 92.50 | 74.00 | 54.00 | 27.50 | 11.50 |

### Sugar + Ir94e（MN9，Hz）
| sugar\ir94e | 0 | 50 | 100 | 150 | 200 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 21.50 | 3.00 | 0.00 | 0.00 | 1.00 |
| 100 | 56.00 | 40.50 | 19.50 | 5.50 | 4.00 |
| 150 | 84.00 | 80.00 | 60.00 | 45.00 | 39.00 |
| 200 | 91.50 | 81.50 | 79.00 | 77.00 | 64.00 |

### Sugar + Water（MN9，Hz）
| sugar\water | 0 | 50 | 100 | 150 | 200 | 260 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.00 | 0.00 | 0.00 | 5.50 | 30.00 | 46.50 |
| 50 | 17.50 | 45.50 | 61.50 | 65.00 | 67.00 | 73.50 |
| 100 | 69.00 | 79.00 | 80.50 | 88.50 | 91.00 | 85.50 |
| 150 | 83.50 | 91.50 | 97.00 | 97.00 | 96.50 | 97.00 |
| 200 | 91.00 | 101.50 | 99.50 | 112.50 | 106.50 | 101.00 |

### 结论（MN9）
- 糖本身驱动 MN9：糖频率升高时 MN9 放电增加，与单路糖扫描一致。
- 糖 + 苦：显著抑制糖通路。示例：糖 100 Hz 为 66.0 Hz，叠加苦 100 Hz 降至 5.0 Hz；苦 150/200 Hz 时几乎归零。
- 糖 + Ir94e：同样抑制，但弱于苦。示例：糖 100 Hz 为 56.0 Hz，叠加 Ir94e 100 Hz 降至 19.5 Hz；Ir94e 200 Hz 仍残留 4.0 Hz。
- 糖 + 水：叠加/增强效应。示例：糖 100 Hz 为 69.0 Hz，糖+水 100/100 为 80.5 Hz；糖 200 + 水 150 达 112.5 Hz。

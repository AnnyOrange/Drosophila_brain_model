# 实验设计（先 LIF，后 STP）：糖/苦/水刺激对比

本文件仅包含实验设计与执行计划，暂不运行实验。等你确认后，再按计划逐步执行并回填报告。

## 目标
- 用**纯 LIF 模型**分别对糖/苦/水 GRN 施加多频率刺激，比较 MN9 及整体响应。
- 用**带 STP 模型**在相同或可比条件下重复关键刺激，评估 STP 对 MN9 和网络响应的影响。

## 数据与模型
- 数据版本：默认 630（`2023_03_23_completeness_630_final.csv` + `2023_03_23_connectivity_630_final.parquet`）。
- LIF：使用 `model.run_exp`（已封装在 `my_code/sugar_circuit.py` / `my_code/bitter_circuit.py` / `my_code/water_circuit.py`）。
- STP：使用 `my_code/model_stp.py` + `my_code/run_stp_experiment.py`（需扩展以支持水 GRN 列表）。

## 关键观测指标
- MN9 放电率（Hz），以及是否发生放电（非零率）。
- Top 响应神经元（Top5/Top10）及其放电率。
- 响应神经元数（rate > 0）。
- （可选）CB0248/CB0192 等已知中枢节点放电率，用于糖路一致性对照。

## 实验 1：LIF 频率扫描（糖/苦/水）
**统一设置**  
- `n_run`: 2（快跑）→ 确认后可提升到 20/30  
- `n_proc`: 1  
- `t_run`: 默认（来自 `model.default_params`）  
- 输出目录：  
  - 糖：`results/my_code_sugar/`  
  - 苦：`results/my_code_bitter/`  
  - 水：`results/my_code_water/`

**频率范围（按论文 Figure 设置）**  
- 糖：10–200 Hz（步长 10）  
- 苦：0–200 Hz（步长 20，或与糖一致 10）  
- 水：20–260 Hz（步长 20）  
（若时间紧，可先用 40/60/80/100/150/200 的稀疏采样）

**产物**  
- `.parquet`：每频率实验输出  
- `.rates.csv` / `.rate_std.csv`：汇总表  
- `MN9` 曲线图（可选）

**拟执行命令（示例）**  
```bash
# 糖
conda run -n flybrain python my_code/sugar_circuit.py --n-run 2 --n-proc 1 --path-res results/my_code_sugar freq --freqs 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200

# 苦（需确认 bitter_circuit.py 是否支持 freq 子命令/范围）
conda run -n flybrain python my_code/bitter_circuit.py --n-run 2 --n-proc 1 --path-res results/my_code_bitter --freqs 20 40 60 80 100 120 140 160 180 200 --summary

# 水
conda run -n flybrain python my_code/water_circuit.py --n-run 2 --n-proc 1 --path-res results/my_code_water freq --freqs 20 40 60 80 100 120 140 160 180 200 220 240 260
```

## 实验 2：STP 条件下的糖/苦/水
**目标**：在相同刺激强度下观察 MN9 是否出现衰减/增强，以及与 LIF 的差异。

**方案 A（单频率对照）**  
对糖/苦/水各跑 100 Hz（或 150 Hz）单一刺激：  
- 糖：`sugar_rate=100`  
- 苦：`bitter_rate=100`  
- 水：需要在 STP 脚本中新增 `water_rate` 与 `WATER_NEURONS` 支持  

**方案 B（序列刺激/混合刺激）**  
复用 `run_stp_sequence.py` 或 `mixed_stim.py` 的结构：  
- 糖 150 → 糖 47（已有）  
- 苦 150 → 糖 47（已有）  
- 水 150 → 水 47（需新增）  

**产物**  
- `.parquet`：每条件输出  
- `*_summary.csv`：MN9 phase1/phase2 汇总（若为序列）  

## 需要你确认/补充的点
- 苦味频率扫描的具体步长（10 还是 20）。  
- LIF 全频率扫描是否先做“稀疏采样”，还是直接全量。  
- STP 是否先做单频率对照（方案 A），还是直接做序列（方案 B）。  
- STP 是否允许我先扩展脚本以支持水 GRN 输入（需要新增 `WATER_NEURONS` 和 `water_rate` 参数）。

---
确认后我将：
1) 按 LIF 频率扫描执行（糖→苦→水），逐步更新报告；  
2) 扩展 STP 脚本支持水刺激；  
3) 执行 STP 对照/序列实验并回填结论。  

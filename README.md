[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philshiu/Drosophila_brain_model/blob/main/example.ipynb)

# Model for the _Drosophila_ brain

Activate and silence neurons in a computational model based on the fruit fly connectome

## 论文关联

This repository accompanies the paper [A leaky integrate-and-fire computational model based on the connectome of the entire adult Drosophila brain reveals insights into sensorimotor processing](https://www.biorxiv.org/content/10.1101/2023.05.02.539144v1).

It includes all code necessary to reproduce the data presented there. The raw output of the model is several GB and therefore not posted on github. Instead, it can be found in this [online archive](https://doi.org/10.17617/3.CZODIW).

---

## 目录

- [核心功能](#核心功能)
- [安装](#安装)
- [基础使用](#基础使用)
- [实验脚本使用](#实验脚本使用)
  - [糖回路实验](#糖回路实验)
  - [苦回路实验](#苦回路实验)
  - [水回路实验](#水回路实验)
  - [混合刺激实验](#混合刺激实验)
  - [STP（短时程可塑性）实验](#stp短时程可塑性实验)
- [版本说明](#版本说明)
- [数据结构](#数据结构)
- [额外提示](#额外提示)

---

## 核心功能

With this computational model, one can manipulate the neural activity of a set of _Drosophila_ neurons, which can be addressed via their [Flywire](https://flywire.ai/) ID. The output of the model is the spike times and rates of all affected neurons.

Two types of manipulations are currently implemented:

- **Activation（激活）**: Neurons can be activated at a fixed frequency to model optogenetic activation. This triggers Poisson spiking in the target neurons. Two sets of neurons with distinct frequencies can be defined.
- **Silencing（沉默）**: In addition to activation, a different set of neurons can be silenced to model optogenetic silencing. This sets all synaptic connections to and from those neurons to zero.
- **Background Noise（背景噪声）**: The model includes a background Poisson noise mechanism to maintain baseline firing rates (1-5 Hz) across all neurons, making the simulation more biologically realistic. This is a key enhancement compared to the original model.

See [example.ipynb](example.ipynb) for a tutorial and a more detailed explanation.

---

## 安装

The Drosophila brain model can be run on Mac, Windows or Unix, and installation should take 10 minutes.

### Quick Start

To begin using the model without a local install, click on the _Open In Colab_ badge at the top of this README. Note that simulations can take substantially longer to run on Google Colab than on a local installation, depending on the number of CPU cores you have access to.

### via conda

Install in a [Anaconda](https://www.anaconda.com/) environment:

```bash
conda env create -f environment.yml
conda activate brian2  # 或您的环境名称
```

For GPU acceleration support:

```bash
conda env create -f environment_cuda.yml
```

For exact package versions matching the original paper:

```bash
conda env create -f environment_full.yml
```

### Brian 2 performance

The model is written in python built using the *Brian 2* simulator. See the official [Brian 2 documentation](https://brian2.readthedocs.io/en/stable/introduction/install.html) for detailed installation instructions for your system. Specifically, follow the instructions for [C++ code generation](https://brian2.readthedocs.io/en/stable/introduction/install.html#requirements-for-c-code-generation) to install *Brian 2* with increased performance.

### Dependencies

See [environment_full.yml](environment_full.yml) for specific package versions used in the original work.

---

## 基础使用

### 在 Jupyter Notebook 中使用

See [example.ipynb](example.ipynb) for a tutorial showing how to configure neuron activation/silencing and run simulations.

Basic usage in Python:

```python
from model import default_params, run_exp

# 运行糖刺激实验
run_exp(
    exp_name='sugarR_100Hz',
    neu_exc=[720575940637568838, ...],  # Flywire IDs of sugar neurons
    params=default_params,
    path_res='./results',
    path_comp='./2023_03_23_completeness_630_final.csv',
    path_con='./2023_03_23_connectivity_630_final.parquet',
    n_proc=-1,  # 使用所有可用CPU核心
    force_overwrite=True
)
```

### 版本切换：630 vs 783

默认使用 Flywire 630 版本。要切换到 783 版本，在脚本中修改路径：

```python
config = {
    'path_res'  : './results/new',
    'path_comp' : './Completeness_783.csv',        # 切换到 783 版
    'path_con'  : './Connectivity_783.parquet',    # 切换到 783 版
    'n_proc'    : -1,
}
```

---

## 实验脚本使用

本仓库包含 `my_code/` 目录，提供了一系列命令行实验脚本，用于糖/苦/水回路研究以及 STP 实验。

### 糖回路实验

**重要说明**：所有实验脚本**默认启用背景噪声**（`r_bg=2000 Hz`，`w_bg=0.5 mV`），以维持神经元 1-5 Hz 的基线放电。

#### 频率扫描

对糖感受神经元进行不同频率的激活扫描：

```bash
# 基本用法：扫描多个频率（背景噪声默认启用）
python my_code/sugar_circuit.py freq --freqs 25 50 75 100 125 150 175 200 --path-res results/my_code_sugar

# 禁用背景噪声
python my_code/sugar_circuit.py freq --freqs 100 --no-bg

# 指定数据路径和并行核心数
python my_code/sugar_circuit.py freq \
    --freqs 10 20 30 40 50 60 70 80 90 100 \
    --path-res results/my_code_sugar \
    --path-comp 2023_03_23_completeness_630_final.csv \
    --path-con 2023_03_23_connectivity_630_final.parquet \
    --n-proc 4 \
    --n-run 30

# 强制覆盖已存在的结果
python my_code/sugar_circuit.py freq --freqs 100 --force-overwrite
```

#### 神经元沉默实验

在激活糖回路的同时，逐个沉默特定神经元以观察其影响：

```bash
# 在 100 Hz 激活时，沉默指定神经元
python my_code/sugar_circuit.py silence \
    --freq 100 \
    --silence-ids 720575940617937543 720575940621754367 720575940622695448 \
    --path-res results/my_code_sugar

# 指定参数
python my_code/sugar_circuit.py silence \
    --freq 150 \
    --silence-ids 720575940617937543 \
    --path-res results/my_code_sugar \
    --n-run 20 \
    --n-proc 8
```

#### 结果可视化

从已生成的 parquet 文件生成图表：

```bash
# 绘制热图和单神经元响应
python my_code/sugar_circuit.py plot \
    --glob "sugarR*.parquet" \
    --path-res results/my_code_sugar \
    --mn-target 720575940660219265 \
    --out-prefix results/my_code_sugar/plots/sugar
```

这会生成：
- `sugar.heatmap.png`：神经元×实验的响应热图
- `sugar.mn_target.png`：指定神经元（如 MN9）在不同实验中的响应柱状图
- `sugar.rates.csv` 和 `sugar.rate_std.csv`：放电率和标准差数据表

---

### 苦回路实验

#### 频率扫描

```bash
# 基本扫描
python my_code/bitter_circuit.py --freqs 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200

# 完整参数
python my_code/bitter_circuit.py \
    --freqs 25 50 75 100 125 150 175 200 \
    --path-res results/my_code_bitter \
    --path-comp 2023_03_23_completeness_630_final.csv \
    --path-con 2023_03_23_connectivity_630_final.parquet \
    --n-proc 4 \
    --n-run 30 \
    --summary

# 显示摘要信息（Top 响应神经元和 MN9 放电率）
python my_code/bitter_circuit.py --freqs 100 --summary
```

**注意**：苦味神经元在完整性表中可能只有部分可用。脚本会自动过滤并报告缺失的神经元。

---

### 水回路实验

#### 频率扫描

```bash
# 基本扫描（水回路通常需要更高频率才能起效）
python my_code/water_circuit.py freq --freqs 20 40 60 80 100 120 140 160 180 200 220 240 260

# 完整参数
python my_code/water_circuit.py freq \
    --freqs 100 150 200 260 \
    --path-res results/my_code_water \
    --path-comp 2023_03_23_completeness_630_final.csv \
    --path-con 2023_03_23_connectivity_630_final.parquet \
    --n-proc 4 \
    --n-run 30

# 绘图
python my_code/water_circuit.py plot \
    --glob "waterR*.parquet" \
    --path-res results/my_code_water \
    --mn-target 720575940660219265
```

---

### 混合刺激实验

#### 单一序列（两阶段刺激）

在单个仿真中运行两阶段刺激序列，观察网络记忆效应：

```bash
# 糖序列：150 Hz → 47 Hz
python my_code/mixed_stim.py \
    --sugar-rates "150,47" \
    --phase-ms 1000 \
    --path-res results/my_code_mixed \
    --exp-name mixed_seq_sugar

# 苦序列：150 Hz → 47 Hz
python my_code/mixed_stim.py \
    --bitter-rates "150,47" \
    --phase-ms 1000

# 糖+苦组合
python my_code/mixed_stim.py \
    --sugar-rates "100,50" \
    --bitter-rates "0,100" \
    --phase-ms 1000
```

#### 无 STP 序列实验

运行多个试次的无 STP 序列实验：

```bash
# 运行多个试次的序列实验
python my_code/run_nostp_sequence.py \
    --n-run 20 \
    --phase-ms 1000 \
    --path-res results/my_code_mixed
```

这会生成：
- `nostp_sugar47_run*.parquet`：47 Hz 单独刺激
- `nostp_sugar150_47_run*.parquet`：150 Hz → 47 Hz 糖序列
- `nostp_bitter150_sugar47_run*.parquet`：苦 150 Hz → 糖 47 Hz 序列

#### 定时混合刺激

```bash
python my_code/mixed_stim_timed.py \
    --freq1 150 \
    --freq2 47 \
    --phase-ms 1000 \
    --n-run 10 \
    --path-res results/my_code_mixed
```

---

### STP（短时程可塑性）实验

#### STP 单次实验

在指定突触上应用 STP，测试糖/苦刺激对 MN9 的影响：

```bash
# 基本用法
python my_code/run_stp_experiment.py \
    --sugar-rate 150 \
    --bitter-rate 0 \
    --n-run 10 \
    --path-res results/my_code_stp

# 完整参数（自定义 STP 参数）
python my_code/run_stp_experiment.py \
    --sugar-rate 150 \
    --bitter-rate 100 \
    --n-run 20 \
    --t-run-ms 2000 \
    --path-res results/my_code_stp \
    --exp-name stp_sugar150_bitter100 \
    --stp-U 0.5 \
    --stp-tau-d-ms 200 \
    --stp-tau-f-ms 500 \
    --path-comp 2023_03_23_completeness_630_final.csv \
    --path-con 2023_03_23_connectivity_630_final.parquet
```

**STP 参数说明**：
- `--stp-U`：释放概率 U（默认 0.5）
- `--stp-tau-d-ms`：抑制性恢复时间常数（毫秒，默认 200）
- `--stp-tau-f-ms`：易化性恢复时间常数（毫秒，默认 500）

#### STP 序列实验

运行带 STP 的两阶段序列刺激：

```bash
# 糖序列：150 Hz → 47 Hz
python my_code/run_stp_sequence.py \
    --freq1 150 \
    --freq2 47 \
    --phase-ms 1000 \
    --phase1-input sugar \
    --phase2-input sugar \
    --n-run 10 \
    --path-res results/my_code_mixed \
    --exp-name stp_seq_sugar150_47

# 苦 → 糖序列
python my_code/run_stp_sequence.py \
    --freq1 150 \
    --freq2 47 \
    --phase-ms 1000 \
    --phase1-input bitter \
    --phase2-input sugar \
    --n-run 10 \
    --exp-name stp_seq_bitter150_sugar47

# 水序列
python my_code/run_stp_sequence.py \
    --freq1 150 \
    --freq2 47 \
    --phase-ms 1000 \
    --phase1-input water \
    --phase2-input water \
    --n-run 10 \
    --exp-name stp_seq_water150_47

# 自定义 STP 参数
python my_code/run_stp_sequence.py \
    --freq1 150 \
    --freq2 47 \
    --phase-ms 1000 \
    --n-run 10 \
    --stp-U 0.3 \
    --stp-tau-d-ms 150 \
    --stp-tau-f-ms 600
```

实验会自动生成汇总 CSV 文件（`*_summary.csv`），包含每个试次的 phase1 和 phase2 的 MN9 放电数。

---

## 版本说明

### 核心模型增强

本仓库的 `model.py` 相比原始版本增加了**背景噪声（Background Noise）**机制：

- **目的**：维持神经元的基线放电率（1-5 Hz），使仿真更接近真实的生物大脑状态
- **实现原理**：背景噪声通过泊松脉冲（Poisson Input）注入到每个神经元的电导 $g$ 中。虽然输入频率高达 2000 Hz（`r_bg=2000 * Hz`），但由于每个脉冲的权重较小（`w_bg=0.5 * mV`）且存在时间常数衰减，最终在神经元层面产生 1-5 Hz 的随机自发放电。这模拟了生物大脑中持续存在的背景活动。
- **参数控制**：在 `default_params` 中通过 `use_bg`（开关）、`r_bg`（频率）和 `w_bg`（权重）控制。

**影响**：
- 即使在无外部刺激时，神经元也会有微弱的基线放电（1-5 Hz）。
- 当施加定向刺激（如糖/苦/水）时，响应会叠加在基线之上。
- 这使模型更真实，但可能使信号稍微模糊（需要适当调整参数）。

### 数据文件

- **630 版本**（论文使用）：
  - `2023_03_23_completeness_630_final.csv`
  - `2023_03_23_connectivity_630_final.parquet`
- **783 版本**（公共版本）：
  - `Completeness_783.csv`
  - `Connectivity_783.parquet`
- **其他数据**：
  - `Supplemental_file1_neuron_annotations.tsv`：神经元注释信息（139,244 行，31 列）
  - `sez_neurons.pickle`：SEZ（下食管区）神经元子集字典

---

## 数据结构

### 仿真输出格式

所有实验脚本生成的 `.parquet` 文件具有统一结构：

| 列名 | 说明 | 类型 |
|------|------|------|
| `t` | Spike 时间（秒） | float |
| `trial` | 试次编号（0 到 n_run-1） | int |
| `flywire_id` | 发放神经元的 Flywire root ID | int |
| `exp_name` | 实验名称 | string |

### 完整性文件

`Completeness_*.csv` 包含两列：
- `Unnamed: 0`：Flywire root ID（作为索引）
- `Completed`：是否标记为“完整”（通常全为 True）

### 连接性文件

`Connectivity_*.parquet` 包含以下列：
- `Presynaptic_ID` / `Postsynaptic_ID`：突触前/后神经元的 Flywire ID
- `Presynaptic_Index` / `Postsynaptic_Index`：对应的 0-based 索引
- `Connectivity`：突触数量
- `Excitatory`：突触类型（+1 兴奋性，-1 抑制性）
- `Excitatory x Connectivity`：权重基数（用于计算突触权重）

---

## 额外提示

### 性能优化

1. **CPU 并行**：大多数脚本支持 `--n-proc` 参数。使用 `-1` 自动使用所有可用核心。
2. **Brian2 C++ 加速**：按照 [Brian2 文档](https://brian2.readthedocs.io/en/stable/introduction/install.html#requirements-for-c-code-generation) 配置 C++ 代码生成可显著提升性能。
3. **GPU 支持**：使用 `environment_cuda.yml` 环境配置可启用 GPU 加速。

### 参数调整

#### 背景噪声控制

**默认行为**：所有实验脚本（包括 `sugar_circuit.py`、`bitter_circuit.py`、`water_circuit.py` 等）**默认启用背景噪声**（`r_bg=2000 Hz`，`w_bg=0.5 mV`），无需额外指定。背景噪声会在所有神经元上产生 1-5 Hz 的基线放电。

**实现原理**：背景噪声通过泊松脉冲（Poisson Input）注入到每个神经元的电导 $g$ 中。虽然输入频率高达 2000 Hz，但由于每个脉冲的权重较小（0.5 mV）且存在时间常数衰减，最终在神经元层面产生 1-5 Hz 的随机自发放电。这模拟了生物大脑中持续存在的背景活动。

**通过命令行禁用**：大多数命令行实验脚本现在支持 `--no-bg` 参数。

```bash
python my_code/sugar_circuit.py freq --freqs 100 --no-bg
```

**禁用背景噪声（Python API）**：如需在自定义代码中禁用，可在参数字典中设置 `use_bg=False`：

```python
from model import default_params, run_exp
from copy import deepcopy
from brian2 import Hz

# 创建参数副本并禁用背景噪声
params = deepcopy(default_params)
params['r_bg'] = 0 * Hz  # 设置为 0 禁用背景噪声

# 运行实验
run_exp(
    exp_name='sugarR_100Hz_no_bg',
    neu_exc=[...],
    params=params,  # 使用修改后的参数
    ...
)
```

**调整背景噪声强度**：

```python
params = deepcopy(default_params)
params['r_bg'] = 1500 * Hz  # 降低背景噪声频率
params['w_bg'] = 0.3 * mV   # 降低背景噪声权重
```

**注意**：目前命令行脚本不支持直接禁用背景噪声。如需此功能，需要修改脚本或使用 Python API。

#### 其他参数调整

- **试次数量**：`--n-run` 参数控制重复试次数。更多试次可提高统计可靠性，但会增加计算时间。
- **刺激频率**：糖回路通常在 40-200 Hz 有效，水回路需要更高频率（140+ Hz），苦回路响应较弱。

### 结果分析

使用 `utils.py` 中的函数进行后处理：

```python
import utils as utl

# 加载多个实验的 spike 数据
df_spike = utl.load_exps(['results/my_code_sugar/sugarR_100Hz.parquet', ...])

# 计算放电率
df_rate, df_rate_std = utl.get_rate(
    df_spike,
    t_run=default_params['t_run'],
    n_run=30,
    flyid2name={720575940660219265: 'MN9'}  # 可选：为神经元添加名称
)
```

### 常见问题

1. **"No module named 'model'"**：确保在项目根目录运行脚本，或使用 `conda activate brian2` 激活环境。
2. **内存不足**：减少 `--n-proc` 或 `--n-run` 参数。
3. **结果文件已存在**：使用 `--force-overwrite` 强制覆盖，或删除旧文件。

### 实验报告

详细的实验设计和结果报告请参考：
- `实验设计_糖苦水_LIF_STP.md`：实验设计文档
- `实验报告_糖苦水_LIF_STP.md`：完整实验结果
- `工作笔记_结论汇总.md`：核心结论总结
- `项目介绍.md`：项目结构详细说明

---

## License

See [LICENSE](LICENSE) file for details.

---

## 引用

If you use this model in your research, please cite:

```
A leaky integrate-and-fire computational model based on the connectome 
of the entire adult Drosophila brain reveals insights into sensorimotor processing
Phil Shiu et al., bioRxiv (2023)
DOI: 10.1101/2023.05.02.539144
```


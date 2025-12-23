# my_code（用户自定义脚本）

此目录存放用户自定义的糖路复现实验代码，与上游仓库代码区分开。脚本在运行时会自动将仓库根目录加入 `sys.path`，无需额外安装本地包。

## 运行示例
```bash
# 频率扫描（默认 25–200 Hz）
python my_code/sugar_circuit.py freq --path-res results/my_code_sugar

# 可选：减少试次以加快调试（默认 n_run 来自 model.default_params）
python my_code/sugar_circuit.py --n-run 2 freq --freqs 100 --path-res results/my_code_sugar

# 单频率 + 逐个沉默特定神经元
python my_code/sugar_circuit.py silence --freq 100 --silence-ids 720575940617937543 720575940621754367 --path-res results/my_code_sugar

# 汇总并绘图（需 matplotlib/seaborn）
python my_code/sugar_circuit.py plot --glob "sugarR*.parquet" --path-res results/my_code_sugar

# 苦味回路测试（默认激活 LB2/LB4 苦相关 GRN，100 Hz，支持 --n-run 缩短试次）
python my_code/bitter_circuit.py --freqs 100 --n-run 2 --path-res results/my_code_bitter --summary
```

## 依赖
- 复用仓库现有的 `model.py` 与 `utils.py`，默认使用 `2023_03_23_*`（630 版）数据；通过 `--path-comp/--path-con` 可切换到 783 版。

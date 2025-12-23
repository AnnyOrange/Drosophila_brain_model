import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# MN9 ID (来自 figures.ipynb)
ID_MN9 = 720575940660219265

# 结果路径
DIRS = {
    "Control (No Noise)": "./results/verification_control",
    "Innovation (With Noise)": "./results/verification_noise"
}

FREQS = [0, 50, 100, 150, 200]

def get_firing_rate(folder, filename_pattern, neuron_id):
    path = Path(folder) / f"{filename_pattern}.parquet"
    if not path.exists():
        return 0.0
    try:
        df = pd.read_parquet(path)
        # 筛选特定神经元
        spikes = df[df['flywire_id'] == neuron_id]
        # 计算: 总脉冲 / (30个trial * 1.0秒)
        return len(spikes) / 30.0
    except:
        return 0.0

# 存储数据的字典
results = {}

# ================= 1. 读取数据 =================
for label, folder in DIRS.items():
    matrix = np.zeros((len(FREQS), len(FREQS)))
    
    for i, f_bitter in enumerate(FREQS):
        for j, f_sugar in enumerate(FREQS):
            # 注意文件名格式与运行脚本一致
            fname = f"S{f_sugar}_B{f_bitter}" 
            rate = get_firing_rate(folder, fname, ID_MN9)
            matrix[i, j] = rate # 行是苦，列是糖
            
    results[label] = matrix

# ================= 2. 绘图验证 =================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- A. 绘制 Heatmaps ---
for idx, (label, matrix) in enumerate(results.items()):
    ax = axes[0, idx]
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="viridis", ax=ax,
                xticklabels=FREQS, yticklabels=FREQS)
    ax.set_title(f"{label}: MN9 Firing Rate")
    ax.set_xlabel("Sugar Input (Hz)")
    ax.set_ylabel("Bitter Input (Hz)")
    ax.invert_yaxis() # 让0在左下角

# --- B. 绘制交互曲线 (Interaction Curves) ---
# 我们想看苦味是如何抑制糖味反应的
# X轴 = 苦味强度, Y轴 = MN9放电率, 每条线 = 不同的糖味强度

markers = ['o', 's', '^', 'D', 'v']
colors = sns.color_palette("flare", len(FREQS))

for idx, (label, matrix) in enumerate(results.items()):
    ax = axes[1, idx]
    # 遍历每一列（即固定的糖强度）
    for j, f_sugar in enumerate(FREQS):
        # 取出这一列数据：不同苦味下的反应
        y_values = matrix[:, j] 
        ax.plot(FREQS, y_values, marker=markers[j], label=f"Sugar={f_sugar}Hz", color=colors[j], linewidth=2)
    
    ax.set_title(f"{label}: Inhibition Curves")
    ax.set_xlabel("Bitter Input (Hz)")
    ax.set_ylabel("MN9 Firing Rate (Hz)")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("verification_result.png")
plt.show()
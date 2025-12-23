import pandas as pd
import numpy as np

# 路径设置
PATH_RES_FILE = "./results/cali/baseline_calibration.parquet"
PATH_COMP = "./Completeness_783.csv" # 需要读取这个来获知神经元总数

def calculate_baseline_rate():
    # 1. 读取实验结果
    print("正在读取结果文件...")
    df_spikes = pd.read_parquet(PATH_RES_FILE)
    
    # 2. 读取神经元总表以获取 N_total
    df_neurons = pd.read_csv(PATH_COMP, index_col=0)
    n_total_neurons = len(df_neurons)
    
    # 3. 获取实验参数 (假设 simulation 时间为 1秒，trial 数为 30)
    # 这些值需要与您 default_params 中的设置一致
    duration_sec = 1.0 
    n_trials = 30
    
    # 4. 计算方法 A: 全局平均放电率 (Global Average Firing Rate)
    # 公式：总脉冲数 / (神经元总数 * 试验次数 * 持续时间)
    total_spikes = len(df_spikes)
    global_avg_hz = total_spikes / (n_total_neurons * n_trials * duration_sec)
    
    print("-" * 30)
    print(f"【校准结果】")
    print(f"神经元总数: {n_total_neurons}")
    print(f"总脉冲数 (30次试验): {total_spikes}")
    print(f"全局平均基线放电率: {global_avg_hz:.4f} Hz")
    
    # 5. 计算方法 B: 活跃神经元比例
    # 统计有多少个唯一的神经元ID发过至少 1 个脉冲
    active_neuron_ids = df_spikes['flywire_id'].unique()
    n_active = len(active_neuron_ids)
    active_ratio = (n_active / n_total_neurons) * 100
    
    print(f"活跃神经元数量: {n_active}")
    print(f"活跃比例: {active_ratio:.2f}%")
    
    # 6. 判断是否符合目标
    target_min, target_max = 1.0, 5.0
    if target_min <= global_avg_hz <= target_max:
        print(f"✅ 成功：基线处于目标区间 ({target_min}-{target_max} Hz)")
    elif global_avg_hz < target_min:
        print(f"⚠️ 偏低：建议增加 r_bg 或 w_bg")
    else:
        print(f"⚠️ 偏高：建议降低 r_bg 或 w_bg")
    print("-" * 30)

if __name__ == "__main__":
    calculate_baseline_rate()
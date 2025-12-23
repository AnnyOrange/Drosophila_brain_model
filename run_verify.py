import numpy as np
import pandas as pd
import os
from copy import deepcopy
from brian2 import Hz, mV

# Standard Brian2 import
from brian2 import *

# Enable GPU usage via Brian2CUDA
import brian2cuda
set_device("cuda_standalone")

# 引入您修改后的模型代码
from model import run_exp, default_params

# ================= 配置区域 =================
# 数据路径 (请修改为您实际的路径)
PATH_COMP = "./2023_03_23_completeness_630_final.csv"
PATH_CON = "./2023_03_23_connectivity_630_final.parquet"


# 定义两组实验的输出路径
DIR_CONTROL = "./results/verification_control" # 无噪音组
DIR_NOISE = "./results/verification_noise"     # 有噪音组

# 确保目录存在
os.makedirs(DIR_CONTROL, exist_ok=True)
os.makedirs(DIR_NOISE, exist_ok=True)

# ================= 1. 提取神经元 ID (来自 figures.ipynb Fig 3A) =================
# Sugar GRNs (Excitatory Input)
neu_sugar = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768,
    720575940630797113, 720575940632889389, 720575940621754367, 720575940621502051, 720575940640649691,
    720575940639332736, 720575940616885538, 720575940639198653, 720575940620900446, 720575940617937543,
    720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]

# Bitter GRNs (Inhibitory Input)
neu_bitter = [
    720575940621778381, 720575940602353632, 720575940617094208, 720575940619197093, 720575940626287336,
    720575940618600651, 720575940627692048, 720575940630195909, 720575940646212996, 720575940610483162,
    720575940645743412, 720575940627578156, 720575940622298631, 720575940621008895, 720575940629146711,
    720575940610259370, 720575940610481370, 720575940619028208, 720575940614281266, 720575940613061118,
    720575940604027168
]

# 定义刺激频率网格 (0 - 200 Hz)
freqs = [0, 50, 100, 150, 200]

# ================= 2. 定义运行函数 =================
def run_grid_search(condition_name, bg_rate, output_dir):
    print(f"\n========== 开始运行组别: {condition_name} (r_bg={bg_rate}) ==========")
    
    # 设置当前组的参数
    current_params = deepcopy(default_params)
    current_params['r_bg'] = bg_rate  # 设置背景噪音频率
    
    # 遍历糖浓度
    for f_sugar in freqs:
        # 遍历苦浓度
        for f_bitter in freqs:
            # 实验名称
            exp_name = f"S{f_sugar}_B{f_bitter}"
            
            # 设置输入频率
            # r_poi 对应 neu_exc (糖)
            # r_poi2 对应 neu_exc2 (苦)
            current_params['r_poi'] = f_sugar * Hz
            current_params['r_poi2'] = f_bitter * Hz
            
            print(f">>> Running {condition_name}: Sugar={f_sugar}Hz, Bitter={f_bitter}Hz")
            
            run_exp(
                exp_name=exp_name,
                neu_exc=neu_sugar,    # 糖神经元
                neu_exc2=neu_bitter,  # 苦神经元
                path_res=output_dir,
                path_comp=PATH_COMP,
                path_con=PATH_CON,
                params=current_params,
                n_proc=-1,             # 并行数
                force_overwrite=True # 避免重复运行
            )

# ================= 3. 执行两组实验 =================

# 实验组 A: Control (复现原论文，无噪音)
# 注意：原论文假设基线为 0Hz，所以设置 r_bg = 0
run_grid_search("Control_Quiet", 0 * Hz, DIR_CONTROL)

# 实验组 B: Experimental (有噪音)
run_grid_search("Innovation_Noise", 2000 * Hz, DIR_NOISE)

print("\n所有模拟完成！请运行分析脚本。")
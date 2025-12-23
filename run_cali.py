import os

# Standard Brian2 import
from brian2 import *

# Enable GPU usage via Brian2CUDA
import brian2cuda
set_device("cuda_standalone")

# 引入您修改后的模型代码文件中定义的 run_exp
from model import run_exp, default_params 

# 1. 设置数据路径 (请修改为您实际存放数据的路径)
PATH_COMP = "./Completeness_783.csv"
PATH_CON = "./Connectivity_783.parquet"
PATH_RES = "./results/cali"  # 结果输出文件夹

# 确保输出目录存在
os.makedirs(PATH_RES, exist_ok=True)

# 2. 调整背景噪音参数 (在这里微调，或者直接修改 model.py 中的 default_params)
# 初始尝试建议使用您修改后的默认值 (2200 Hz)
# 如果跑出来结果不对，可以在这里覆盖 default_params
# default_params['r_bg'] = 2000 * Hz 

# 3. 运行空实验
print("开始运行基线校准实验...")
run_exp(
    exp_name="baseline_calibration", # 实验名称
    neu_exc=[],                      # 【关键】空列表，代表没有外部特定刺激
    path_res=PATH_RES,
    path_comp=PATH_COMP,
    path_con=PATH_CON,
    params=default_params,           # 传入包含背景噪音参数的配置
    neu_slnc=[],                     # 没有神经元被沉默
    n_proc=-1,                        # 并行核心数，根据您的电脑配置调整 (-1 为使用所有核心)
    force_overwrite=True
)
print("实验结束。")
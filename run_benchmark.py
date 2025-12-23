import numpy as np
import pandas as pd
import os
import time
import argparse
from multiprocessing import Pool, current_process
from copy import deepcopy
from brian2 import Hz, second, prefs

# ================= 1. 实验配置 =================
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['cpu', 'gpu'], help='测试模式')
args = parser.parse_args()

# 基础路径配置
PATH_COMP = "./2023_03_23_completeness_630_final.csv"
PATH_CON = "./2023_03_23_connectivity_630_final.parquet"
RESULT_DIR = f"./results/benchmark_{args.mode}"
os.makedirs(RESULT_DIR, exist_ok=True)

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

# 频率网格 (5x5 = 25 任务)
freqs = [0, 50, 100, 150, 200]
task_configs = [(s, b) for s in freqs for b in freqs]

# ================= 2. 工作函数 =================
def run_benchmark_task(config):
    f_sugar, f_bitter = config
    exp_name = f"S{f_sugar}_B{f_bitter}"
    
    # 1. 彻底清除之前的 Brian2 状态 (至关重要)
    from brian2 import device, set_device
    device.reinit()
    device.activate()
    
    from model import run_exp, default_params
    n_run = default_params['t_run']
    
    p_id = current_process()._identity[0]
    
    start_time = time.time()
    
    if args.mode == 'gpu':
        # --- 全量 GPU 逻辑 (4卡并行) ---
        gpu_ids = ['0', '1', '2', '3']
        gpu_id = gpu_ids[(p_id - 1) % len(gpu_ids)]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        build_dir = f"./output/output_gpu_{exp_name}"
        
        import brian2cuda
        set_device("cuda_standalone", directory=build_dir)
        backend_info = f"GPU {gpu_id}"
    else:
        # --- 全量 CPU 逻辑 ---
        build_dir = f"./output/output_cpu_{exp_name}"
        set_device("cpp_standalone", directory=build_dir)
        # 每进程1核，不开启内层OpenMP，避免资源竞争
        prefs.devices.cpp_standalone.openmp_threads = 1
        backend_info = f"CPU Core {p_id}"

    print(f"🚀 [{args.mode.upper()}] {backend_info} 正在处理: {exp_name}...")

    params = deepcopy(default_params)
    params['r_bg'] = 2000 * Hz
    params['r_poi'] = f_sugar * Hz
    params['r_poi2'] = f_bitter * Hz
    params['n_run'] = 1

    try:
        run_exp(
            exp_name=exp_name,
            neu_exc=neu_sugar, 
            neu_exc2=neu_bitter,
            path_res=RESULT_DIR,
            path_comp=PATH_COMP,
            path_con=PATH_CON,
            params=params,
            n_proc=1,
            force_overwrite=True
        )
        duration = time.time() - start_time
        return {'sugar': f_sugar, 'bitter': f_bitter, 'time': duration}
    except Exception as e:
        print(f"❌ {exp_name} 失败: {e}")
        return None

# ================= 3. 主程序 =================
if __name__ == '__main__':
    # 根据模式决定并行进程数
    pool_size = 4 if args.mode == 'gpu' else 25
    
    start_all = time.time()
    
    with Pool(processes=pool_size, maxtasksperchild=1) as pool:
        results = pool.map(run_benchmark_task, task_configs)
    
    print(f"\n" + "="*40)
    print(f"开始 {args.mode.upper()} 性能测试")
    print(f"并行规模: {pool_size} 核心/显卡")
    print(f"任务总数: {len(task_configs)}")
    print("="*40 + "\n")
    
    total_time = time.time() - start_all
    
    # 结果保存
    results = [r for r in results if r is not None]
    df = pd.DataFrame(results)
    df['total_wall_time'] = total_time  # 记录整组实验总耗时
    df['parallel_degree'] = pool_size   # 记录当时用了多少核心/显卡
    df.to_csv(f"benchmark_results_{args.mode}.csv", index=False)
    
    print(f"\n✅ {args.mode.upper()} 测试完成！")
    print(f"总耗时: {total_time:.2f} 秒 (约 {total_time/60:.2f} 分钟)")
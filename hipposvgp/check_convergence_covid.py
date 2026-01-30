import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os
import argparse

# 导入项目中的模块
sys.path.append(os.getcwd())

from hipposvgp.hsvgp import HIPPOOSVGP as NumericalOSVGP
from hipposvgp.hsvgp_analytic import HIPPOOSVGP as AnalyticOSVGP
from hipposvgp.kernel import SE_kernel
from hipposvgp.hippo import HiPPO_LegS
from hipposvgp.likelihood import GaussianLikelihood, NegativeBinomial
from scripts.onedim.covid.data import COVIDDataset

def setup():
    parser = argparse.ArgumentParser('Argument Parser')
    parser.add_argument('--seed_num', type=int, default=0)
    args, unknown = parser.parse_known_args()
    return args

def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_covid_convergence(device='cuda'):
    log_file_path = 'convergence_covid_results.txt'
    
    with open(log_file_path, 'w') as log_file:
        def log_print(text):
            print(text)
            log_file.write(text + '\n')
            log_file.flush()

        log_print(f"Running COVID-19 Task 1 Convergence Check on {device}...")

        # ==========================================
        # 1. 加载数据
        # ==========================================
        try:
            dataset = COVIDDataset("data/covid/processed_timeseries_5tasks/task_1")
        except FileNotFoundError:
            log_print("Error: 找不到数据文件。请确保在项目根目录下运行。")
            return

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        for batch in dataloader:
            deaths, days, _, _, _, num_days, _, _, _, _, _, _, _ = batch
            deaths, days = deaths.to(device).double(), days.to(device).double()
            N = num_days.double()
            break
        
        # 数据切片
        days = days.permute(1,0)[:int(N[0].item())]
        
        # --- [关键修复] 时间平移 ---
        # 原始数据的 Start=15.0，会导致数值解(初始化为0)与解析解(积分至15)状态不一致。
        # 我们将时间强行平移，使第一天变为 Day 0.0。
        time_shift = torch.min(days).item()
        days = days - time_shift
        
        start = 0.0
        end_data = torch.max(days).item()
        maxT_data = end_data - start
        
        log_print(f"Data Loaded & Shifted: Start={start:.2f}, End={end_data:.2f}, Duration={maxT_data:.2f}")

        # ==========================================
        # 2. 初始化 Kernel
        # ==========================================
        inducing_size = 15     
        rff_sample_size = 1000 
        
        seed_everything(42)
        
        kernel_base = SE_kernel(1).to(device)
        
        # 使用你提供的真实训练值
        real_trained_log_ls = 3.1289
        kernel_base.log_ls.data.fill_(real_trained_log_ls)
        
        real_lengthscale_days = math.exp(real_trained_log_ls)
        log_print(f"Using REAL trained parameters: log_ls={real_trained_log_ls:.4f} (Lengthscale ≈ {real_lengthscale_days:.2f} days)")
        
        # 生成频率 w
        w_fixed = (1.0 / torch.exp(kernel_base.log_ls)) * torch.randn(1, rff_sample_size, device=device)
        
        likelihood = NegativeBinomial().to(device)

        # ==========================================
        # 3. 运行收敛性测试
        # ==========================================
        step_sizes = [2.0, 1.0, 0.5, 0.1, 0.05, 0.01] 
        errors = []

        header = f"{'Step Size':<10} | {'Num Steps':<10} | {'Rel Error':<20} | {'Max Diff':<15}"
        divider = "-" * 65
        log_print(header)
        log_print(divider)

        for dt in step_sizes:
            num_discrete_steps = math.ceil(maxT_data / dt)
            current_end = start + num_discrete_steps * dt
            
            # 初始化 HiPPO (给足缓冲区)
            hippo = HiPPO_LegS(inducing_size, device, max_length=num_discrete_steps + 200)

            # A. 数值版模型
            model_num = NumericalOSVGP(
                kernel=SE_kernel(1),
                likelihood=likelihood,
                start=start,
                end=current_end,
                rff_sample_size=rff_sample_size,
                num_discrete_steps=num_discrete_steps,
                hippo=hippo,
                inducing_size=inducing_size,
                device=device,
                flag_update_kernel=False
            ).to(device)

            # B. 解析版模型
            model_ana = AnalyticOSVGP(
                kernel=SE_kernel(1),
                likelihood=likelihood,
                start=start,
                end=current_end,
                rff_sample_size=rff_sample_size,
                num_discrete_steps=num_discrete_steps,
                hippo=hippo,
                inducing_size=inducing_size,
                device=device,
                flag_update_kernel=False,
                use_analytic_Z=True
            ).to(device)

            # C. 计算 Z
            with torch.no_grad():
                Z_num, _, _ = model_num.get_Z(w_fixed)
                Z_ana, _, _ = model_ana.get_Z_analytic(w_fixed)
                
                # 计算误差
                diff_norm = (Z_num - Z_ana).norm().item()
                base_norm = Z_ana.norm().item()
                
                if base_norm < 1e-12:
                    rel_error = diff_norm
                else:
                    rel_error = diff_norm / base_norm
                    
                max_diff = (Z_num - Z_ana).abs().max().item()
                errors.append(rel_error)
                
                row_str = f"{dt:<10.4f} | {num_discrete_steps:<10} | {rel_error:<20.6f} | {max_diff:<15.6f}"
                log_print(row_str)

        log_print(f"\nResults saved to {log_file_path}")

    # D. 绘图
    plt.figure(figsize=(8, 6))
    plt.loglog(step_sizes, errors, 'o-', linewidth=2, label='Actual Error')
    plt.xlabel('Step Size ($\Delta t$)')
    plt.ylabel('Relative Error (Z Matrix)')
    plt.title(f'Convergence Check (Trained LS={real_trained_log_ls:.2f}, Time Shifted)')
    plt.grid(True, which="both", ls="-")
    
    if len(errors) > 1:
        ref_x = np.array(step_sizes)
        ref_y = errors[-1] * (ref_x / ref_x[-1]) 
        plt.loglog(ref_x, ref_y, 'r--', label='O($\Delta t$) Reference')
    
    plt.legend()
    plt.gca().invert_xaxis() 
    plt.savefig('convergence_check_covid.png')
    print("Convergence plot saved to 'convergence_check_covid.png'")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    check_covid_convergence(device)
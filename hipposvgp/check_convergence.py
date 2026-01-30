import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os

# 确保能导入项目中的模块
sys.path.append(os.getcwd())

# 导入两个版本的模型
from hipposvgp.hsvgp import HIPPOOSVGP as NumericalOSVGP
from hipposvgp.hsvgp_analytic import HIPPOOSVGP as AnalyticOSVGP
from hipposvgp.kernel import SE_kernel
from hipposvgp.hippo import HiPPO_LegS
from hipposvgp.likelihood import GaussianLikelihood

def check_accuracy(device='cuda'):
    log_file_path = 'convergence_results.txt'  # 保存表格的文件名
    
    # 打开文件准备写入
    with open(log_file_path, 'w') as log_file:
        
        # 定义一个辅助函数：既打印到屏幕，又写入文件
        def log_print(text):
            print(text)
            log_file.write(text + '\n')
            log_file.flush() # 确保实时写入

        log_print(f"Running convergence check on {device}...")
        
        # --- 1. 固定实验参数 (保持不变) ---
        inducing_size = 150   # 勒让德多项式阶数
        rff_sample_size = 1000 
        start = 0.0
        end = 10.0            # 固定总时长 T=10
        maxT = end - start
        
        # --- 2. 定义要测试的步长列表 (逐渐减小) ---
        # 我们希望看到随着步长减小，误差直线下降
        step_sizes = [2.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.005] 
        
        errors = []
        
        # 固定随机种子，确保 w 在所有测试中是一样的
        torch.manual_seed(42)
        
        # [关键] 预先生成固定的频率 w
        # 我们需要在 GPU 上生成它，确保所有模型用的是同一组随机频率
        kernel_base = SE_kernel(1).to(device)
        # 手动生成 w 以规避潜在的 device 问题
        w_fixed = (1.0 / torch.exp(kernel_base.log_ls)) * torch.randn(1, rff_sample_size, device=device)

        # HiPPO 模块 (可复用)
        hippo = HiPPO_LegS(inducing_size, device)
        likelihood = GaussianLikelihood(0.1).to(device)

        # 打印表头
        header = f"{'Step Size':<10} | {'Num Steps':<10} | {'Rel Error':<20} | {'Max Diff':<15}"
        divider = "-" * 65
        log_print(header)
        log_print(divider)

        for dt in step_sizes:
            # 根据步长计算需要的离散步数
            num_discrete_steps = math.ceil(maxT / dt)
            # 微调 end 以匹配整数步数
            current_end = start + num_discrete_steps * dt
            
            # --- A. 初始化数值版模型 (ODE) ---
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

            # --- B. 初始化解析版模型 (Bessel) ---
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

            # --- C. 计算 Z 矩阵并对比 ---
            with torch.no_grad():
                # 传入相同的 w
                Z_num, _, _ = model_num.get_Z(w_fixed)
                Z_ana, _, _ = model_ana.get_Z_analytic(w_fixed)
                
                # 计算相对误差: ||Z_num - Z_ana|| / ||Z_ana||
                diff_norm = (Z_num - Z_ana).norm().item()
                base_norm = Z_ana.norm().item()
                
                if base_norm < 1e-12:
                    rel_error = diff_norm
                else:
                    rel_error = diff_norm / base_norm
                    
                max_diff = (Z_num - Z_ana).abs().max().item()
                
                errors.append(rel_error)
                
                # 打印并保存这一行结果
                row_str = f"{dt:<10.4f} | {num_discrete_steps:<10} | {rel_error:<20.6f} | {max_diff:<15.6f}"
                log_print(row_str)

        log_print(f"\nResults saved to {log_file_path}")

    # --- D. 绘制收敛曲线图 ---
    plt.figure(figsize=(8, 6))
    # 双对数坐标轴 (Log-Log Plot) 是展示收敛率的标准方式
    plt.loglog(step_sizes, errors, 'o-', linewidth=2, label='Actual Error')
    
    plt.xlabel('Step Size ($\Delta t$)')
    plt.ylabel('Relative Error $||Z_{num} - Z_{ana}|| / ||Z_{ana}||$')
    plt.title('Convergence of Numerical HiPPO to Analytic Bessel Solution')
    plt.grid(True, which="both", ls="-")
    
    # 添加参考线 (O(dt) 线性收敛)
    # 如果两条线平行，说明收敛速率符合预期
    if len(errors) > 1:
        ref_x = np.array(step_sizes)
        # 让参考线通过最后一个点
        ref_y = errors[-1] * (ref_x / ref_x[-1]) 
        plt.loglog(ref_x, ref_y, 'r--', label='O($\Delta t$) Reference')
    
    plt.legend()
    # X轴反转：通常习惯把步长小的放在右边，或者按数值大小放
    # 这里保持 matplotlib 默认 (大在右)，或者反转看个人习惯
    plt.gca().invert_xaxis() 
    
    plt.savefig('convergence_check.png')
    print("Convergence plot saved to 'convergence_check.png'")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    check_accuracy(device)
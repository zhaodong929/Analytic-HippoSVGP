import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os

# Ensure project modules can be imported
sys.path.append(os.getcwd())

# Import both model implementations
from hipposvgp.hsvgp import HIPPOOSVGP as NumericalOSVGP
from hipposvgp.hsvgp_analytic import HIPPOOSVGP as AnalyticOSVGP
from hipposvgp.kernel import SE_kernel
from hipposvgp.hippo import HiPPO_LegS
from hipposvgp.likelihood import GaussianLikelihood

def check_accuracy(device='cuda'):
    log_file_path = 'convergence_results.txt'  # Output file for tabular results
    
    # Open log file for writing
    with open(log_file_path, 'w') as log_file:
        
        # Helper function: print to console and write to file
        def log_print(text):
            print(text)
            log_file.write(text + '\n')
            log_file.flush() # Ensure immediate writes

        log_print(f"Running convergence check on {device}...")
        
        # --- 1. Fixed experiment settings ---
        inducing_size = 150   # Legendre polynomial order
        rff_sample_size = 1000 
        start = 0.0
        end = 10.0            # Fixed total duration T=10
        maxT = end - start
        
        # --- 2. Step sizes to test (decreasing) ---
        # We expect error to decrease as the step size shrinks
        step_sizes = [2.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.005] 
        
        errors = []
        
        # Fix random seed so w is identical across runs
        torch.manual_seed(42)
        
        # [Key] Pre-generate a fixed frequency tensor w
        # Generate on GPU so all models share the same random frequencies
        kernel_base = SE_kernel(1).to(device)
        # Generate w manually to avoid potential device mismatches
        w_fixed = (1.0 / torch.exp(kernel_base.log_ls)) * torch.randn(1, rff_sample_size, device=device)

        # Reusable HiPPO module
        hippo = HiPPO_LegS(inducing_size, device)
        likelihood = GaussianLikelihood(0.1).to(device)

        # Print table header
        header = f"{'Step Size':<10} | {'Num Steps':<10} | {'Rel Error':<20} | {'Max Diff':<15}"
        divider = "-" * 65
        log_print(header)
        log_print(divider)

        for dt in step_sizes:
            # Number of discrete steps for this step size
            num_discrete_steps = math.ceil(maxT / dt)
            # Slightly adjust end so steps are an integer count
            current_end = start + num_discrete_steps * dt
            
            # --- A. Initialize numerical model (ODE) ---
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

            # --- B. Initialize analytic model (Bessel) ---
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

            # --- C. Compute and compare Z matrices ---
            with torch.no_grad():
                # Use the same w for both models
                Z_num, _, _ = model_num.get_Z(w_fixed)
                Z_ana, _, _ = model_ana.get_Z_analytic(w_fixed)
                
                # Relative error: ||Z_num - Z_ana|| / ||Z_ana||
                diff_norm = (Z_num - Z_ana).norm().item()
                base_norm = Z_ana.norm().item()
                
                if base_norm < 1e-12:
                    rel_error = diff_norm
                else:
                    rel_error = diff_norm / base_norm
                    
                max_diff = (Z_num - Z_ana).abs().max().item()
                
                errors.append(rel_error)
                
                # Print and persist this row
                row_str = f"{dt:<10.4f} | {num_discrete_steps:<10} | {rel_error:<20.6f} | {max_diff:<15.6f}"
                log_print(row_str)

        log_print(f"\nResults saved to {log_file_path}")

    # --- D. Plot convergence curve ---
    plt.figure(figsize=(8, 6))
    # Log-log axis is standard for convergence-rate visualization
    plt.loglog(step_sizes, errors, 'o-', linewidth=2, label='Actual Error')
    
    plt.xlabel('Step Size ($\Delta t$)')
    plt.ylabel('Relative Error $||Z_{num} - Z_{ana}|| / ||Z_{ana}||$')
    plt.title('Convergence of Numerical HiPPO to Analytic Bessel Solution')
    plt.grid(True, which="both", ls="-")
    
    # Add reference line (O(dt) linear convergence)
    # Parallel lines indicate the expected convergence rate
    if len(errors) > 1:
        ref_x = np.array(step_sizes)
        # Anchor the reference line at the last point
        ref_y = errors[-1] * (ref_x / ref_x[-1]) 
        plt.loglog(ref_x, ref_y, 'r--', label='O($\Delta t$) Reference')
    
    plt.legend()
    # Invert x-axis (often used to place smaller step sizes on the right)
    # Keeping or inverting is mostly stylistic
    plt.gca().invert_xaxis() 
    
    plt.savefig('convergence_check.png')
    print("Convergence plot saved to 'convergence_check.png'")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    check_accuracy(device)

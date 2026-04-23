import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter

# Ensure project modules can be imported
sys.path.append(os.getcwd())

ICML_PLOT_STYLE = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

MAIN_COLOR = "#1f77b4"
REFERENCE_COLOR = "#c44e52"


def setup():
    parser = argparse.ArgumentParser(
        description="Check convergence of the numerical HiPPO features to the analytic Bessel construction."
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to use, e.g. cpu or cuda. Defaults to cuda when available.",
    )
    parser.add_argument(
        "--results-file",
        default="convergence_results.txt",
        help="Path to the tabular convergence results file.",
    )
    parser.add_argument(
        "--output-stem",
        default="convergence_check",
        help="Output path stem for the generated figure files.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip the numerical experiment and redraw the figure from an existing results file.",
    )
    return parser.parse_args()


def configure_icml_style():
    plt.rcParams.update(ICML_PLOT_STYLE)


def load_logged_results(log_file_path):
    step_sizes = []
    errors = []

    with open(log_file_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            if "|" not in raw_line:
                continue

            parts = [part.strip() for part in raw_line.split("|")]
            if len(parts) != 4:
                continue

            try:
                step_size = float(parts[0])
                rel_error = float(parts[2])
            except ValueError:
                continue

            step_sizes.append(step_size)
            errors.append(rel_error)

    if not step_sizes:
        raise ValueError(f"No convergence rows found in {log_file_path}.")

    return np.asarray(step_sizes, dtype=float), np.asarray(errors, dtype=float)


def plot_convergence(step_sizes, errors, output_stem="convergence_check"):
    configure_icml_style()

    sort_idx = np.argsort(step_sizes)[::-1]
    step_sizes = np.asarray(step_sizes, dtype=float)[sort_idx]
    errors = np.asarray(errors, dtype=float)[sort_idx]

    fig, ax = plt.subplots(figsize=(3.35, 2.45), constrained_layout=True)

    ax.loglog(
        step_sizes,
        errors,
        color=MAIN_COLOR,
        marker="o",
        markersize=4.6,
        markerfacecolor="white",
        markeredgewidth=1.15,
        linewidth=2.0,
        label="ODE error",
        zorder=3,
    )

    if len(errors) > 1:
        anchor_idx = len(step_sizes) // 2
        ref_x = step_sizes
        ref_y = errors[anchor_idx] * (ref_x / step_sizes[anchor_idx])
        ax.loglog(
            ref_x,
            ref_y,
            linestyle="--",
            color=REFERENCE_COLOR,
            linewidth=1.6,
            label=r"$\mathcal{O}(\Delta t)$",
            zorder=2,
        )

    ax.set_xlabel(r"Step size $\Delta t$")
    ax.set_ylabel("Relative feature error")

    ax.grid(which="major", color="#d6d6d6", linewidth=0.65)
    ax.grid(which="minor", color="#eeeeee", linewidth=0.45)
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="major", direction="out", length=3.4, width=0.8)
    ax.tick_params(which="minor", direction="out", length=2.1, width=0.6)

    for spine in ax.spines.values():
        spine.set_color("#4a4a4a")
        spine.set_linewidth(0.8)

    ax.legend(frameon=False, loc="upper right", handlelength=2.2)
    ax.invert_xaxis()

    png_path = f"{output_stem}.png"
    pdf_path = f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Convergence plot saved to '{png_path}' and '{pdf_path}'")


def check_accuracy(device='cuda', log_file_path='convergence_results.txt'):
    import torch

    from hipposvgp.hsvgp import HIPPOOSVGP as NumericalOSVGP
    from hipposvgp.hsvgp_analytic import HIPPOOSVGP as AnalyticOSVGP
    from hipposvgp.hippo import HiPPO_LegS
    from hipposvgp.kernel import SE_kernel
    from hipposvgp.likelihood import GaussianLikelihood

    # Open log file for writing
    with open(log_file_path, 'w', encoding='utf-8') as log_file:

        # Helper function: print to console and write to file
        def log_print(text):
            print(text)
            log_file.write(text + '\n')
            log_file.flush()  # Ensure immediate writes

        log_print(f"Running convergence check on {device}...")

        # --- 1. Fixed experiment settings ---
        inducing_size = 150  # Legendre polynomial order
        rff_sample_size = 1000
        start = 0.0
        end = 10.0  # Fixed total duration T=10
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

    return np.asarray(step_sizes, dtype=float), np.asarray(errors, dtype=float)

if __name__ == '__main__':
    args = setup()

    if args.plot_only:
        step_sizes, errors = load_logged_results(args.results_file)
    else:
        import torch

        device_name = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(device_name)
        step_sizes, errors = check_accuracy(device=device, log_file_path=args.results_file)

    plot_convergence(step_sizes, errors, output_stem=args.output_stem)

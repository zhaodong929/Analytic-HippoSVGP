import argparse
import importlib
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


def _parse_freqs(freqs_str):
    return [int(x) for x in freqs_str.split(",") if x.strip()]


def _evaluate_model(mod, model, days_test_list, deaths_test_list, w, mc_samples, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for idx, days_test in enumerate(days_test_list):
            mean, var = model.pred_f(days_test, w)
            pred_mean = mod._predict_mean(model, mean)
            rmse, r2 = mod._rmse_r2(deaths_test_list[idx], pred_mean)

            f_samples = mean + torch.sqrt(var) * torch.randn(
                mc_samples, var.shape[0], var.shape[1], device=device
            )
            nlpd = -torch.log(
                model.likelihood.log_prob(
                    f_samples.view(-1, 1),
                    deaths_test_list[idx]
                    .unsqueeze(0)
                    .tile(mc_samples, 1, 1)
                    .view(-1, 1),
                )
                .view(mc_samples, -1)
                .exp()
                .mean(dim=0)
            ).mean()

            rows.append(
                {
                    "eval_task": idx + 1,
                    "rmse": float(rmse),
                    "r2": float(r2),
                    "nlpd": float(nlpd.item()),
                }
            )
    return rows


def _run_one(module_name, freq, seed, num_epochs, mc_samples, device, module_overrides=None):
    mod = importlib.import_module(module_name)
    mod = importlib.reload(mod)

    # Set module globals expected by the original scripts.
    mod.args = SimpleNamespace(seed_num=seed)
    mod.seed_everything(seed)
    mod.rff_sample_size = freq
    if module_overrides is not None:
        for key, value in module_overrides.items():
            setattr(mod, key, value)

    inducing_size = 15
    lr = 1e-2
    step_size = 2
    mod.hippo = mod.HiPPO_LegS(inducing_size, device)

    prev_model, days_test_list, deaths_test_list, days_train, deaths_train, w, prev_discrete_steps, prev_end = (
        mod.train_task1(device, inducing_size, num_epochs, lr, step_size)
    )

    for task_id in range(2, 6):
        prev_model, days_test_list, deaths_test_list, days_train, deaths_train, w, prev_discrete_steps, prev_end = (
            mod.train_nexttask(
                prev_model,
                days_test_list,
                deaths_test_list,
                days_train,
                deaths_train,
                device,
                task_id,
                inducing_size,
                num_epochs,
                lr,
                step_size,
                w,
                prev_discrete_steps,
                prev_end,
            )
        )

    rows = _evaluate_model(mod, prev_model, days_test_list, deaths_test_list, w, mc_samples, device)
    # Mean across tasks for a single summary value per freq.
    mean_rmse = float(np.mean([r["rmse"] for r in rows]))
    mean_nlpd = float(np.mean([r["nlpd"] for r in rows]))
    mean_r2 = float(np.mean([r["r2"] for r in rows]))

    # Cleanup to free memory between runs.
    del prev_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rows, mean_rmse, mean_nlpd, mean_r2


def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row[h]) for h in header) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare frequency-space model vs RFF baseline across frequency counts."
    )
    parser.add_argument(
        "--freqs",
        type=str,
        default="10,20,50,100,200,500,1000",
        help="Comma-separated list of frequency counts.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for data split.")
    parser.add_argument("--epochs", type=int, default=5000, help="Training epochs per task.")
    parser.add_argument(
        "--freqspace_train_spectral_task1",
        action="store_true",
        help=(
            "If set, freqspace Task1 will train spectral frequencies/weights. "
            "Default keeps them fixed for parity checking against RFF baseline."
        ),
    )
    parser.add_argument(
        "--freqspace_freeze_log_sf_task1",
        action="store_true",
        help=(
            "If set, freeze freqspace Task1 kernel log_sf. "
            "Default keeps log_sf trainable (only frequencies/weights are frozen)."
        ),
    )
    parser.add_argument(
        "--mc_samples",
        type=int,
        default=1000,
        help="Monte Carlo samples for NLPD.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="ohsgp_freq_sweep",
        help="Output directory for results.",
    )
    args = parser.parse_args()

    freqs = _parse_freqs(args.freqs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    if args.epochs != 5000:
        print(
            "[warning] analytic baseline Task1 is hardcoded to 5000 epochs; "
            "forcing comparison epochs to 5000 for fairness."
        )
    effective_epochs = 5000

    summary_rows = []
    per_task_rows = []

    for freq in freqs:
        # RFF baseline (analytic kernel)
        rows, mean_rmse, mean_nlpd, mean_r2 = _run_one(
            "scripts.onedim.covid.test_ohsgp_analytic",
            freq,
            args.seed,
            effective_epochs,
            args.mc_samples,
            device,
        )
        summary_rows.append(
            {
                "model": "rff_baseline",
                "freq": freq,
                "mean_rmse": mean_rmse,
                "mean_nlpd": mean_nlpd,
                "mean_r2": mean_r2,
            }
        )
        for r in rows:
            per_task_rows.append(
                {
                    "model": "rff_baseline",
                    "freq": freq,
                    "eval_task": r["eval_task"],
                    "rmse": r["rmse"],
                    "r2": r["r2"],
                    "nlpd": r["nlpd"],
                }
            )

        # Frequency-space model (learnable frequencies)
        rows, mean_rmse, mean_nlpd, mean_r2 = _run_one(
            "scripts.onedim.covid.test_ohsgp_freqspace",
            freq,
            args.seed,
            effective_epochs,
            args.mc_samples,
            device,
            module_overrides={
                "freqspace_train_spectral_task1": args.freqspace_train_spectral_task1,
                "freqspace_train_log_sf_task1": (not args.freqspace_freeze_log_sf_task1),
            },
        )
        summary_rows.append(
            {
                "model": "freqspace_rsf",
                "freq": freq,
                "mean_rmse": mean_rmse,
                "mean_nlpd": mean_nlpd,
                "mean_r2": mean_r2,
            }
        )
        for r in rows:
            per_task_rows.append(
                {
                    "model": "freqspace_rsf",
                    "freq": freq,
                    "eval_task": r["eval_task"],
                    "rmse": r["rmse"],
                    "r2": r["r2"],
                    "nlpd": r["nlpd"],
                }
            )

    _write_csv(
        out_dir / "summary.csv",
        ["model", "freq", "mean_rmse", "mean_nlpd", "mean_r2"],
        summary_rows,
    )
    _write_csv(
        out_dir / "per_task.csv",
        ["model", "freq", "eval_task", "rmse", "r2", "nlpd"],
        per_task_rows,
    )

    # Plot summary curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model in ["rff_baseline", "freqspace_rsf"]:
        xs = [r["freq"] for r in summary_rows if r["model"] == model]
        ys_rmse = [r["mean_rmse"] for r in summary_rows if r["model"] == model]
        ys_nlpd = [r["mean_nlpd"] for r in summary_rows if r["model"] == model]
        axes[0].plot(xs, ys_rmse, marker="o", label=model)
        axes[1].plot(xs, ys_nlpd, marker="o", label=model)

    axes[0].set_title("Mean RMSE vs #frequencies")
    axes[0].set_xlabel("#frequencies")
    axes[0].set_ylabel("mean RMSE")
    axes[0].legend()

    axes[1].set_title("Mean NLPD vs #frequencies")
    axes[1].set_xlabel("#frequencies")
    axes[1].set_ylabel("mean NLPD")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "freq_vs_perf.png", dpi=150)

    print(f"Saved summary to {out_dir / 'summary.csv'}")
    print(f"Saved per-task to {out_dir / 'per_task.csv'}")
    print(f"Saved plot to {out_dir / 'freq_vs_perf.png'}")


if __name__ == "__main__":
    main()

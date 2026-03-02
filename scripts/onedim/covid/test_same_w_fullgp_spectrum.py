import argparse
import importlib
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch


def _parse_freqs(freqs_str):
    return [int(x) for x in freqs_str.split(",") if x.strip()]


def _extract_spectrum_tensor(obj, min_numel):
    if isinstance(obj, torch.Tensor):
        if obj.numel() < min_numel:
            raise ValueError(
                f"Spectrum tensor has {obj.numel()} values, but at least {min_numel} are required."
            )
        return obj

    if isinstance(obj, dict):
        preferred_keys = [
            "omega",
            "w",
            "shared_w",
            "frequencies",
            "freqs",
            "spectral_w",
            "spectral_samples",
        ]
        for key in preferred_keys:
            if key in obj and isinstance(obj[key], torch.Tensor) and obj[key].numel() >= min_numel:
                return obj[key]

        candidates = [v for v in obj.values() if isinstance(v, torch.Tensor) and v.numel() >= min_numel]
        if candidates:
            return max(candidates, key=lambda t: t.numel())

    raise ValueError("Could not find a usable spectrum tensor in the checkpoint.")


def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row[h]) for h in header) + "\n")


def _plot_per_task(out_dir, per_task_rows):
    freqs = sorted({r["freq"] for r in per_task_rows})
    models = sorted({r["model"] for r in per_task_rows})
    label_map = {
        "rff_baseline": "RFF baseline",
        "freqspace_rsf": "Freq-space (RSF)",
    }
    metrics = [
        ("rmse", "RMSE (Lower is Better)"),
        ("r2", "R2 Score (Higher is Better)"),
        ("nlpd", "NLPD (Lower is Better)"),
    ]

    for freq in freqs:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
        for metric_idx, (metric_key, metric_title) in enumerate(metrics):
            ax = axes[metric_idx]
            for model in models:
                series = [r for r in per_task_rows if r["freq"] == freq and r["model"] == model]
                series = sorted(series, key=lambda x: x["eval_task"])
                x = [f"T{r['eval_task']}" for r in series]
                y = [r[metric_key] for r in series]
                ax.plot(x, y, marker="o", label=label_map.get(model, model))
            ax.set_title(metric_title)
            ax.tick_params(axis="x", rotation=45)

        axes[0].set_ylabel("value")
        fig.suptitle(f"Per-task comparison (freq={freq})")
        axes[-1].legend(loc="best")
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(out_dir / f"compare_freq_{freq}.png", dpi=150)
        plt.close(fig)


def _plot_summary(out_dir, summary_rows):
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
    plt.close(fig)


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
                    deaths_test_list[idx].unsqueeze(0).tile(mc_samples, 1, 1).view(-1, 1),
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


@contextmanager
def _patch_fixed_w_sampler(fixed_w):
    # Patch kernel spectral sampling globally so both scripts always use the same w.
    from hipposvgp import kernel as kernel_mod

    orig_se = kernel_mod.SE_kernel.sample_from_spectral
    orig_fs = kernel_mod.FreqSpaceKernel.sample_from_spectral
    fixed_w_cpu = fixed_w.detach().cpu().flatten()

    def _fixed_sample(self, num_rff=None):
        n = fixed_w_cpu.numel() if num_rff is None else int(num_rff)
        if n > fixed_w_cpu.numel():
            raise ValueError(f"Requested {n} spectral components, but fixed spectrum has {fixed_w_cpu.numel()}.")
        target_device = getattr(self, "device", "cpu")
        target_dtype = getattr(self, "dtype", torch.float64)
        return fixed_w_cpu[:n].to(device=target_device, dtype=target_dtype).clone()

    kernel_mod.SE_kernel.sample_from_spectral = _fixed_sample
    kernel_mod.FreqSpaceKernel.sample_from_spectral = _fixed_sample
    try:
        yield
    finally:
        kernel_mod.SE_kernel.sample_from_spectral = orig_se
        kernel_mod.FreqSpaceKernel.sample_from_spectral = orig_fs


def _prepare_module(module_name, freq, seed, device, out_dir, analytic_phase_origin_mode):
    mod = importlib.import_module(module_name)
    mod = importlib.reload(mod)

    if module_name.endswith("test_ohsgp_analytic"):
        analytic_out_root = out_dir / "analytic_runs" / f"freq_{freq}"
        mod.args = SimpleNamespace(
            seed_num=seed,
            out_root=str(analytic_out_root),
            phase_origin_mode=analytic_phase_origin_mode,
        )
        if hasattr(mod, "OUTPUT_DIR"):
            mod.OUTPUT_DIR = os.path.join(
                mod.args.out_root, f"models{seed}", mod.args.phase_origin_mode
            )
            Path(mod.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    elif module_name.endswith("test_ohsgp_freqspace"):
        mod.args = SimpleNamespace(seed_num=seed, num_components=freq)
        Path(f"./ohsgp_freqspace/models{seed}").mkdir(parents=True, exist_ok=True)
    else:
        mod.args = SimpleNamespace(seed_num=seed)

    mod.seed_everything(seed)
    mod.rff_sample_size = freq
    mod.hippo = mod.HiPPO_LegS(15, device)
    return mod


def _run_one(module_name, freq, seed, num_epochs, mc_samples, device, out_dir, analytic_phase_origin_mode, fixed_w):
    with _patch_fixed_w_sampler(fixed_w):
        mod = _prepare_module(
            module_name=module_name,
            freq=freq,
            seed=seed,
            device=device,
            out_dir=out_dir,
            analytic_phase_origin_mode=analytic_phase_origin_mode,
        )

        inducing_size = 15
        lr = 1e-2
        step_size = 2
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
        mean_rmse = float(np.mean([r["rmse"] for r in rows]))
        mean_nlpd = float(np.mean([r["nlpd"] for r in rows]))
        mean_r2 = float(np.mean([r["r2"] for r in rows]))

        del prev_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return rows, mean_rmse, mean_nlpd, mean_r2


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train-and-compare analytic vs freqspace across multiple frequencies, "
            "while forcing both versions to use the same full-GP learned spectrum w."
        )
    )
    parser.add_argument("--freqs", type=str, default="10,20,50,100,200,500,1000")
    parser.add_argument("--seed_num", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--mc_samples", type=int, default=1000)
    parser.add_argument(
        "--spectrum_path",
        type=str,
        default="ohsgp_shared/omega_seed0_rff1000.pt",
        help="Path to learned full-GP spectrum checkpoint.",
    )
    parser.add_argument(
        "--analytic_phase_origin_mode",
        type=str,
        default="global_start",
        choices=["zero", "global_start"],
    )
    parser.add_argument("--out_dir", type=str, default="ohsgp_freq_sweep_sharedw")
    args = parser.parse_args()

    freqs = _parse_freqs(args.freqs)
    if not freqs:
        raise ValueError("No valid frequencies provided.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.spectrum_path):
        raise FileNotFoundError(f"Missing spectrum checkpoint: {args.spectrum_path}")

    max_freq = max(freqs)
    spectrum_obj = torch.load(args.spectrum_path, map_location="cpu")
    spectrum = _extract_spectrum_tensor(spectrum_obj, min_numel=max_freq).detach().flatten().to(torch.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summary_rows = []
    per_task_rows = []

    print(f"[info] device={device}")
    print(f"[info] spectrum={args.spectrum_path}")
    print(f"[info] freqs={freqs}")
    print(f"[info] out_dir={out_dir}")

    for freq in freqs:
        fixed_w = spectrum[:freq].clone()
        print(f"[run] freq={freq} (shared full-GP w)")

        rows, mean_rmse, mean_nlpd, mean_r2 = _run_one(
            module_name="scripts.onedim.covid.test_ohsgp_analytic",
            freq=freq,
            seed=args.seed_num,
            num_epochs=args.epochs,
            mc_samples=args.mc_samples,
            device=device,
            out_dir=out_dir,
            analytic_phase_origin_mode=args.analytic_phase_origin_mode,
            fixed_w=fixed_w,
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

        rows, mean_rmse, mean_nlpd, mean_r2 = _run_one(
            module_name="scripts.onedim.covid.test_ohsgp_freqspace",
            freq=freq,
            seed=args.seed_num,
            num_epochs=args.epochs,
            mc_samples=args.mc_samples,
            device=device,
            out_dir=out_dir,
            analytic_phase_origin_mode=args.analytic_phase_origin_mode,
            fixed_w=fixed_w,
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
    _plot_summary(out_dir, summary_rows)
    _plot_per_task(out_dir, per_task_rows)

    print(f"Saved summary to {out_dir / 'summary.csv'}")
    print(f"Saved per-task to {out_dir / 'per_task.csv'}")
    print(f"Saved plot to {out_dir / 'freq_vs_perf.png'}")
    print(f"Saved per-frequency plots to {out_dir / 'compare_freq_*.png'}")


if __name__ == "__main__":
    main()

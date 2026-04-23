#!/usr/bin/env python
"""Solar-only runtime benchmarks for ODE vs analytic HiPPO-SVGP scaling."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import NullFormatter, ScalarFormatter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipposvgp.hippo import HiPPO_LegS
from hipposvgp.hsvgp import HIPPOOSVGP as ODEHIPPOOSVGP
from hipposvgp.hsvgp_analytic import HIPPOOSVGP as AnalyticHIPPOOSVGP
from hipposvgp.kernel import SE_kernel
from hipposvgp.likelihood import GaussianLikelihood

DEFAULT_T_SWEEP = [100, 300, 1000, 3000, 10000]
DEFAULT_M_SWEEP = [8, 16, 32, 64, 128]
DEFAULT_FIXED_M = 64
DEFAULT_FIXED_T = 1000

SOLAR_LOG_LS = 1.4108
SOLAR_LOG_SF = -0.5334
SOLAR_LOG_NOISE = -4.5

METHOD_SPECS = {
    "ode": {
        "label": "ODE baseline",
        "class": ODEHIPPOOSVGP,
        "color": "#1f77b4",
        "linestyle": "-",
        "marker": "o",
    },
    "analytic": {
        "label": "Analytic HiPPO-SVGP",
        "class": AnalyticHIPPOOSVGP,
        "color": "#c44e52",
        "linestyle": "--",
        "marker": "s",
    },
}

ICML_STYLE = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "legend.fontsize": 7.2,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark runtime scaling for the ODE and analytic HiPPO-SVGP implementations on the solar stream."
    )
    parser.add_argument(
        "--experiment",
        default="all",
        choices=["T", "M", "all"],
        help="Which scaling experiment to run.",
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=["kernel_only", "full_update", "all"],
        help="Which timing mode to run.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cpu.")
    parser.add_argument(
        "--dtype",
        default="float64",
        choices=["float32", "float64"],
        help="Computation dtype shared by both methods.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Global benchmark seed.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per configuration.")
    parser.add_argument("--repeats", type=int, default=20, help="Measured repeats per configuration.")
    parser.add_argument(
        "--t-values",
        type=int,
        nargs="+",
        default=DEFAULT_T_SWEEP,
        help="Sweep values for runtime-vs-T.",
    )
    parser.add_argument(
        "--m-values",
        type=int,
        nargs="+",
        default=DEFAULT_M_SWEEP,
        help="Sweep values for runtime-vs-M.",
    )
    parser.add_argument(
        "--fixed-m-for-t",
        type=int,
        default=DEFAULT_FIXED_M,
        help="Basis size M used in the runtime-vs-T experiment.",
    )
    parser.add_argument(
        "--fixed-t-for-m",
        type=int,
        default=DEFAULT_FIXED_T,
        help="Discrete stream length T used in the runtime-vs-M experiment.",
    )
    parser.add_argument(
        "--num-observations",
        type=int,
        default=256,
        help="Number of solar observations used inside each timed call.",
    )
    parser.add_argument(
        "--rff-sample-size",
        type=int,
        default=1000,
        help="Number of random Fourier features R shared by all runs.",
    )
    parser.add_argument(
        "--optimizer-lr",
        type=float,
        default=1e-2,
        help="Learning rate for the single full_update optimizer step.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/benchmarks",
        help="Root directory for CSV outputs and figures.",
    )
    parser.add_argument(
        "--figure-dir",
        default="outputs/benchmarks/figures",
        help="Directory for publication-style figures.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Run timings and save CSVs without generating figures.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[dtype_name]


def configure_plot_style() -> None:
    plt.rcParams.update(ICML_STYLE)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_base_solar_stream(repo_root: Path) -> np.ndarray:
    train = np.load(repo_root / "data" / "solar_data_train.npy")
    test = np.load(repo_root / "data" / "solar_data_test.npy")
    data = np.vstack([train, test])
    order = np.argsort(data[:, 0])
    return data[order]


def resample_solar_observations(
    base_stream: np.ndarray,
    num_observations: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    x_base = base_stream[:, 0]
    y_base = base_stream[:, 1]
    x_obs = np.linspace(float(x_base.min()), float(x_base.max()), num_observations, dtype=np.float64)
    y_obs = np.interp(x_obs, x_base, y_base)
    x_tensor = torch.as_tensor(x_obs.reshape(-1, 1), dtype=dtype, device=device)
    y_tensor = torch.as_tensor(y_obs.reshape(-1, 1), dtype=dtype, device=device)
    return x_tensor, y_tensor, float(x_obs.min()), float(x_obs.max())


def sample_fixed_w(
    num_rff: int,
    log_ls: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    eps = torch.randn(num_rff, generator=generator, dtype=dtype, device="cpu")
    return (math.exp(-log_ls) * eps).to(device=device, dtype=dtype).unsqueeze(0)


def build_model(
    method: str,
    inducing_size: int,
    num_discrete_steps: int,
    start: float,
    end: float,
    device: torch.device,
    dtype: torch.dtype,
    rff_sample_size: int,
) -> torch.nn.Module:
    kernel = SE_kernel(1, device=device)
    likelihood = GaussianLikelihood(variance=math.exp(SOLAR_LOG_NOISE))
    hippo = HiPPO_LegS(inducing_size, device)

    kwargs = dict(
        kernel=kernel,
        likelihood=likelihood,
        start=start,
        end=end,
        rff_sample_size=rff_sample_size,
        num_discrete_steps=num_discrete_steps,
        hippo=hippo,
        inducing_size=inducing_size,
        device=device,
        dtype=dtype,
        flag_fast=False,
        flag_update_kernel=False,
    )
    if method == "analytic":
        kwargs["use_analytic_Z"] = True
        kwargs["phase_origin_mode"] = "zero"

    model = METHOD_SPECS[method]["class"](**kwargs).to(device=device)
    model = model.to(dtype=dtype)

    with torch.no_grad():
        model.kernel.log_ls.fill_(SOLAR_LOG_LS)
        model.kernel.log_sf.fill_(SOLAR_LOG_SF)
        model.likelihood.log_variance.fill_(SOLAR_LOG_NOISE)

    model.kernel.log_ls.requires_grad_(False)
    model.kernel.log_sf.requires_grad_(False)
    model.likelihood.log_variance.requires_grad_(False)
    return model


def run_kernel_only(model: torch.nn.Module, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        kff, kfu, kuu, *_ = model.KuuKfuKff_rff_se(w, x)
        return kff.sum() + kfu.sum() + kuu.sum()


def run_full_update(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    optimizer_lr: float,
) -> torch.Tensor:
    optimizer = torch.optim.Adam([model.mv, model.Lv], lr=optimizer_lr)
    optimizer.zero_grad(set_to_none=True)
    elbo, _, _ = model.ELBO(x, y, w, recompute_k=True, cache_k=False)
    loss = -elbo
    loss.backward()
    optimizer.step()
    return loss.detach()


def time_callable(device: torch.device, fn) -> tuple[float, float]:
    sync_device(device)
    start = time.perf_counter()
    out = fn()
    sync_device(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if torch.is_tensor(out):
        checksum = float(out.detach().abs().sum().item())
    else:
        checksum = float(out)
    return elapsed_ms, checksum


def benchmark_configuration(
    *,
    method: str,
    mode: str,
    inducing_size: int,
    num_discrete_steps: int,
    x: torch.Tensor,
    y: torch.Tensor,
    start: float,
    end: float,
    device: torch.device,
    dtype: torch.dtype,
    rff_sample_size: int,
    seed: int,
    warmup: int,
    repeats: int,
    optimizer_lr: float,
) -> tuple[List[float], float]:
    w = sample_fixed_w(
        num_rff=rff_sample_size,
        log_ls=SOLAR_LOG_LS,
        seed=seed,
        device=device,
        dtype=dtype,
    )
    timings_ms: List[float] = []
    checksum_total = 0.0

    total_iters = warmup + repeats
    for iteration in range(total_iters):
        set_global_seed(seed + iteration)
        model = build_model(
            method=method,
            inducing_size=inducing_size,
            num_discrete_steps=num_discrete_steps,
            start=start,
            end=end,
            device=device,
            dtype=dtype,
            rff_sample_size=rff_sample_size,
        )
        model.train(mode == "full_update")

        if mode == "kernel_only":
            timed_fn = lambda: run_kernel_only(model, x, w)
        elif mode == "full_update":
            timed_fn = lambda: run_full_update(model, x, y, w, optimizer_lr)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        elapsed_ms, checksum = time_callable(device, timed_fn)
        checksum_total += checksum
        if iteration >= warmup:
            timings_ms.append(elapsed_ms)
    return timings_ms, checksum_total


def summarize_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[float]] = defaultdict(list)
    for row in rows:
        key = (
            row["experiment"],
            row["mode"],
            row["method"],
            int(row["sweep_value"]),
        )
        grouped[key].append(float(row["time_ms"]))

    summary_rows: List[Dict[str, object]] = []
    for (experiment, mode, method, sweep_value), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=np.float64)
        summary_rows.append(
            {
                "experiment": experiment,
                "mode": mode,
                "method": method,
                "sweep_value": sweep_value,
                "repeats": len(values),
                "median_ms": float(np.median(arr)),
                "mean_ms": float(np.mean(arr)),
                "std_ms": float(np.std(arr, ddof=0)),
                "min_ms": float(np.min(arr)),
                "max_ms": float(np.max(arr)),
            }
        )
    return summary_rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_experiment(
    *,
    experiment: str,
    summary_rows: Sequence[Dict[str, object]],
    figure_dir: Path,
) -> None:
    configure_plot_style()
    experiment_rows = [row for row in summary_rows if row["experiment"] == experiment]
    if not experiment_rows:
        return

    modes = [mode for mode in ("kernel_only", "full_update") if any(r["mode"] == mode for r in experiment_rows)]
    if not modes:
        return

    fig, axes = plt.subplots(
        len(modes),
        1,
        figsize=(3.35, 2.0 * len(modes) + 0.2),
        constrained_layout=True,
        squeeze=False,
    )

    x_label = r"Stream length $T$" if experiment == "T" else r"Basis size $M$"
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 3))

    for row_idx, mode in enumerate(modes):
        ax = axes[row_idx, 0]
        mode_rows = [row for row in experiment_rows if row["mode"] == mode]

        for method in ("ode", "analytic"):
            method_rows = sorted(
                [row for row in mode_rows if row["method"] == method],
                key=lambda item: int(item["sweep_value"]),
            )
            x_vals = np.asarray([int(row["sweep_value"]) for row in method_rows], dtype=np.float64)
            median = np.asarray([float(row["median_ms"]) for row in method_rows], dtype=np.float64)
            std = np.asarray([float(row["std_ms"]) for row in method_rows], dtype=np.float64)
            lower = np.maximum(median - std, 1e-6)
            upper = median + std

            spec = METHOD_SPECS[method]
            ax.plot(
                x_vals,
                median,
                color=spec["color"],
                linestyle=spec["linestyle"],
                marker=spec["marker"],
                linewidth=1.8,
                markersize=4.0,
                markerfacecolor="white",
                markeredgewidth=1.0,
                label=spec["label"],
            )
            ax.fill_between(x_vals, lower, upper, color=spec["color"], alpha=0.12)

        ax.set_xscale("log", base=10 if experiment == "T" else 2)
        ax.set_yscale("log")
        ax.set_ylabel("Median time (ms)")
        ax.set_title("Kernel construction" if mode == "kernel_only" else "One ELBO update step")
        ax.grid(which="major", color="#d8d8d8", linewidth=0.65)
        ax.grid(which="minor", color="#efefef", linewidth=0.45)
        ax.set_xticks(sorted({int(row["sweep_value"]) for row in mode_rows}))
        ax.get_xaxis().set_major_formatter(formatter)
        ax.get_xaxis().set_minor_formatter(NullFormatter())
        ax.get_yaxis().set_major_formatter(formatter)
        ax.tick_params(which="major", direction="out", length=3.2, width=0.8)
        ax.tick_params(which="minor", direction="out", length=2.0, width=0.6)

        if row_idx == len(modes) - 1:
            ax.set_xlabel(x_label)
        if row_idx == 0:
            ax.legend(frameon=False, loc="upper left", handlelength=2.4)

    ensure_dir(figure_dir)
    stem = figure_dir / f"runtime_vs_{experiment}"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def run_experiment(
    *,
    experiment: str,
    sweep_values: Iterable[int],
    modes: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
    base_stream: np.ndarray,
    num_observations: int,
    fixed_m_for_t: int,
    fixed_t_for_m: int,
    rff_sample_size: int,
    seed: int,
    warmup: int,
    repeats: int,
    optimizer_lr: float,
) -> List[Dict[str, object]]:
    x, y, start, end = resample_solar_observations(
        base_stream=base_stream,
        num_observations=num_observations,
        device=device,
        dtype=dtype,
    )

    raw_rows: List[Dict[str, object]] = []
    for sweep_value in sweep_values:
        if experiment == "T":
            inducing_size = fixed_m_for_t
            num_discrete_steps = int(sweep_value)
        elif experiment == "M":
            inducing_size = int(sweep_value)
            num_discrete_steps = fixed_t_for_m
        else:
            raise ValueError(f"Unsupported experiment: {experiment}")

        for mode in modes:
            for method in ("ode", "analytic"):
                timings_ms, checksum_total = benchmark_configuration(
                    method=method,
                    mode=mode,
                    inducing_size=inducing_size,
                    num_discrete_steps=num_discrete_steps,
                    x=x,
                    y=y,
                    start=start,
                    end=end,
                    device=device,
                    dtype=dtype,
                    rff_sample_size=rff_sample_size,
                    seed=seed,
                    warmup=warmup,
                    repeats=repeats,
                    optimizer_lr=optimizer_lr,
                )
                for repeat_idx, elapsed_ms in enumerate(timings_ms):
                    raw_rows.append(
                        {
                            "experiment": experiment,
                            "mode": mode,
                            "method": method,
                            "sweep_value": int(sweep_value),
                            "inducing_size": int(inducing_size),
                            "num_discrete_steps": int(num_discrete_steps),
                            "num_observations": int(num_observations),
                            "rff_sample_size": int(rff_sample_size),
                            "repeat_idx": repeat_idx,
                            "time_ms": float(elapsed_ms),
                            "checksum": float(checksum_total),
                        }
                    )
    return raw_rows


def main() -> None:
    args = parse_args()
    repo_root = REPO_ROOT
    output_root = (repo_root / args.output_root).resolve()
    figure_dir = (repo_root / args.figure_dir).resolve()
    raw_dir = output_root / "raw"
    aggregated_dir = output_root / "aggregated"

    ensure_dir(raw_dir)
    ensure_dir(aggregated_dir)
    ensure_dir(figure_dir)

    set_global_seed(args.seed)
    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")
    dtype = resolve_dtype(args.dtype)
    base_stream = load_base_solar_stream(repo_root)

    experiments = ["T", "M"] if args.experiment == "all" else [args.experiment]
    modes = ["kernel_only", "full_update"] if args.mode == "all" else [args.mode]

    config_payload = {
        "seed": args.seed,
        "device": str(requested_device),
        "dtype": args.dtype,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "t_values": args.t_values,
        "m_values": args.m_values,
        "fixed_m_for_t": args.fixed_m_for_t,
        "fixed_t_for_m": args.fixed_t_for_m,
        "num_observations": args.num_observations,
        "rff_sample_size": args.rff_sample_size,
        "optimizer_lr": args.optimizer_lr,
        "solar_data_note": (
            "Observations are resampled deterministically from the repository's solar train/test arrays "
            "onto a uniform grid over the original solar time range. The runtime-vs-T sweep fixes the "
            "observation set and varies num_discrete_steps=T to isolate stream-length scaling."
        ),
    }
    with (output_root / "benchmark_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2)

    all_raw_rows: List[Dict[str, object]] = []
    for experiment in experiments:
        sweep_values = args.t_values if experiment == "T" else args.m_values
        raw_rows = run_experiment(
            experiment=experiment,
            sweep_values=sweep_values,
            modes=modes,
            device=requested_device,
            dtype=dtype,
            base_stream=base_stream,
            num_observations=args.num_observations,
            fixed_m_for_t=args.fixed_m_for_t,
            fixed_t_for_m=args.fixed_t_for_m,
            rff_sample_size=args.rff_sample_size,
            seed=args.seed,
            warmup=args.warmup,
            repeats=args.repeats,
            optimizer_lr=args.optimizer_lr,
        )
        all_raw_rows.extend(raw_rows)

        summary_rows = summarize_rows(raw_rows)
        write_csv(
            raw_dir / f"runtime_vs_{experiment}_raw.csv",
            raw_rows,
            fieldnames=[
                "experiment",
                "mode",
                "method",
                "sweep_value",
                "inducing_size",
                "num_discrete_steps",
                "num_observations",
                "rff_sample_size",
                "repeat_idx",
                "time_ms",
                "checksum",
            ],
        )
        write_csv(
            aggregated_dir / f"runtime_vs_{experiment}_summary.csv",
            summary_rows,
            fieldnames=[
                "experiment",
                "mode",
                "method",
                "sweep_value",
                "repeats",
                "median_ms",
                "mean_ms",
                "std_ms",
                "min_ms",
                "max_ms",
            ],
        )
        if not args.skip_plots:
            plot_experiment(
                experiment=experiment,
                summary_rows=summary_rows,
                figure_dir=figure_dir,
            )

    print(f"Saved benchmark artifacts under: {output_root}")
    if not args.skip_plots:
        print(f"Saved figures under: {figure_dir}")
    print(f"Recorded {len(all_raw_rows)} measured timing rows.")


if __name__ == "__main__":
    main()

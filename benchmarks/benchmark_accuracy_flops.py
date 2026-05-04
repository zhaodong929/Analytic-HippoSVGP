#!/usr/bin/env python
"""Solar RMSE and FLOP benchmarks for ODE vs analytic HiPPO-SVGP."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_scaling import (  # type: ignore
    DEFAULT_FIXED_M,
    DEFAULT_FIXED_T,
    DEFAULT_M_SWEEP,
    DEFAULT_T_SWEEP,
    SOLAR_LOG_LS,
    ensure_dir,
    load_base_solar_stream,
    load_base_solar_stream as load_runtime_stream,
    resample_solar_observations,
    resolve_dtype,
    run_full_update,
    run_kernel_only,
    sample_fixed_w,
    set_global_seed,
    sync_device,
    build_model,
)

try:
    from torch.profiler import ProfilerActivity, profile, record_function
except ImportError:  # pragma: no cover
    ProfilerActivity = None
    profile = None
    record_function = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark solar RMSE and profiler-estimated FLOPs for ODE and analytic HiPPO-SVGP."
    )
    parser.add_argument(
        "--experiment",
        default="all",
        choices=["T", "M", "all"],
        help="Which sweep to run.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cpu.")
    parser.add_argument(
        "--dtype",
        default="float64",
        choices=["float32", "float64"],
        help="Shared computation dtype.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Global random seed.")
    parser.add_argument(
        "--t-values",
        type=int,
        nargs="+",
        default=DEFAULT_T_SWEEP,
        help="Discrete-step sweep for the delta-t experiment.",
    )
    parser.add_argument(
        "--m-values",
        type=int,
        nargs="+",
        default=DEFAULT_M_SWEEP,
        help="Basis-size sweep.",
    )
    parser.add_argument(
        "--fixed-m-for-t",
        type=int,
        default=DEFAULT_FIXED_M,
        help="Basis size M used in the delta-t sweep.",
    )
    parser.add_argument(
        "--fixed-t-for-m",
        type=int,
        default=DEFAULT_FIXED_T,
        help="Discrete-step count T used in the M sweep.",
    )
    parser.add_argument(
        "--rff-sample-size",
        type=int,
        default=1000,
        help="Number of random Fourier features R shared by all runs.",
    )
    parser.add_argument(
        "--flops-rff-sample-size",
        type=int,
        default=None,
        help="Optional profiler-only RFF count. Defaults to --rff-sample-size.",
    )
    parser.add_argument(
        "--runtime-num-observations",
        type=int,
        default=256,
        help="Number of observations used for FLOP profiling, matching the runtime benchmark.",
    )
    parser.add_argument(
        "--profile-cuda-activities",
        action="store_true",
        help="Include CUDA activities in torch.profiler. Off by default to reduce profiler memory overhead.",
    )
    parser.add_argument(
        "--train-iters",
        type=int,
        default=300,
        help="Number of variational optimization steps before RMSE evaluation.",
    )
    parser.add_argument(
        "--optimizer-lr",
        type=float,
        default=1e-2,
        help="Learning rate for variational optimization and one-step FLOP profiling.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/benchmarks",
        help="Root directory for benchmark CSV outputs.",
    )
    return parser.parse_args()


def load_solar_train_test(
    repo_root: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    train = np.load(repo_root / "data" / "solar_data_train.npy")
    test = np.load(repo_root / "data" / "solar_data_test.npy")
    train = train[np.argsort(train[:, 0])]
    test = test[np.argsort(test[:, 0])]

    x_train = torch.as_tensor(train[:, 0:1], dtype=dtype, device=device)
    y_train = torch.as_tensor(train[:, 1:2], dtype=dtype, device=device)
    x_test = torch.as_tensor(test[:, 0:1], dtype=dtype, device=device)
    y_test = torch.as_tensor(test[:, 1:2], dtype=dtype, device=device)
    start = float(train[:, 0].min())
    end = float(train[:, 0].max())
    return x_train, y_train, x_test, y_test, start, end


def delta_t_from_steps(start: float, end: float, num_discrete_steps: int) -> float:
    return float(end - start) / float(num_discrete_steps)


def release_memory(device: torch.device, *objs) -> None:
    for obj in objs:
        if obj is not None:
            del obj
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def train_variational_posterior(
    model: torch.nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    w: torch.Tensor,
    train_iters: int,
    optimizer_lr: float,
    seed: int,
) -> float:
    optimizer = torch.optim.Adam([model.mv, model.Lv], lr=optimizer_lr)
    start_time = time.perf_counter()
    for step in range(train_iters):
        set_global_seed(seed + step)
        optimizer.zero_grad(set_to_none=True)
        recompute_k = step == 0
        cache_k = step == 0
        elbo, _, _ = model.ELBO(
            x_train,
            y_train,
            w,
            recompute_k=recompute_k,
            cache_k=cache_k,
        )
        loss = -elbo
        loss.backward()
        optimizer.step()
    sync_device(torch.device(model.device) if not isinstance(model.device, torch.device) else model.device)
    return time.perf_counter() - start_time


def evaluate_rmse_nll(
    model: torch.nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    w: torch.Tensor,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        mean, var = model.pred_f(x_test, w, full_cov=False)
        rmse = torch.sqrt(torch.mean((y_test - mean) ** 2))
        var = torch.clamp(var.squeeze(-1), min=1e-12)
        resid = (y_test.squeeze(-1) - mean.squeeze(-1)) ** 2
        nll = 0.5 * torch.mean(torch.log(2.0 * torch.tensor(math.pi, device=var.device, dtype=var.dtype) * var) + resid / var)
    return float(rmse.item()), float(nll.item())


def profile_flops(device: torch.device, fn, include_cuda_activities: bool) -> float:
    if profile is None or ProfilerActivity is None or record_function is None:
        return float("nan")

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda" and include_cuda_activities:
        activities.append(ProfilerActivity.CUDA)

    sync_device(device)
    try:
        with profile(
            activities=activities,
            with_flops=True,
            profile_memory=False,
            record_shapes=False,
        ) as prof:
            with record_function("benchmark_profile"):
                out = fn()
                if torch.is_tensor(out):
                    _ = out.detach()
                sync_device(device)
    except torch.cuda.OutOfMemoryError as exc:
        release_memory(device)
        raise RuntimeError(
            "FLOP profiling ran out of memory. Retry with a smaller profiler-only "
            "`--flops-rff-sample-size` such as 512. Keep `--profile-cuda-activities` off unless you specifically need it."
        ) from exc

    total_flops = 0.0
    for evt in prof.key_averages():
        total_flops += float(getattr(evt, "flops", 0.0) or 0.0)
    return total_flops if total_flops > 0.0 else float("nan")


def benchmark_accuracy_and_flops(
    *,
    experiment: str,
    sweep_values: Iterable[int],
    device: torch.device,
    dtype: torch.dtype,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    acc_start: float,
    acc_end: float,
    runtime_x: torch.Tensor,
    runtime_y: torch.Tensor,
    runtime_start: float,
    runtime_end: float,
    fixed_m_for_t: int,
    fixed_t_for_m: int,
    rff_sample_size: int,
    flops_rff_sample_size: int,
    seed: int,
    train_iters: int,
    optimizer_lr: float,
    profile_cuda_activities: bool,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    accuracy_rows: List[Dict[str, object]] = []
    flops_rows: List[Dict[str, object]] = []

    for sweep_value in sweep_values:
        if experiment == "T":
            inducing_size = fixed_m_for_t
            num_discrete_steps = int(sweep_value)
        elif experiment == "M":
            inducing_size = int(sweep_value)
            num_discrete_steps = fixed_t_for_m
        else:
            raise ValueError(f"Unsupported experiment: {experiment}")

        delta_t = delta_t_from_steps(acc_start, acc_end, num_discrete_steps)

        for method in ("ode", "analytic"):
            set_global_seed(seed)
            w = sample_fixed_w(
                num_rff=rff_sample_size,
                log_ls=SOLAR_LOG_LS,
                seed=seed,
                device=device,
                dtype=dtype,
            )

            acc_model = build_model(
                method=method,
                inducing_size=inducing_size,
                num_discrete_steps=num_discrete_steps,
                start=acc_start,
                end=acc_end,
                device=device,
                dtype=dtype,
                rff_sample_size=rff_sample_size,
            )
            train_time_sec = train_variational_posterior(
                model=acc_model,
                x_train=x_train,
                y_train=y_train,
                w=w,
                train_iters=train_iters,
                optimizer_lr=optimizer_lr,
                seed=seed,
            )
            rmse, nll = evaluate_rmse_nll(
                model=acc_model,
                x_test=x_test,
                y_test=y_test,
                w=w,
            )
            accuracy_rows.append(
                {
                    "experiment": experiment,
                    "method": method,
                    "sweep_value": int(sweep_value),
                    "inducing_size": int(inducing_size),
                    "num_discrete_steps": int(num_discrete_steps),
                    "delta_t": float(delta_t),
                    "train_iters": int(train_iters),
                    "train_time_sec": float(train_time_sec),
                    "rmse": float(rmse),
                    "nll": float(nll),
                }
            )
            release_memory(device, acc_model)

            w_flops = sample_fixed_w(
                num_rff=flops_rff_sample_size,
                log_ls=SOLAR_LOG_LS,
                seed=seed,
                device=device,
                dtype=dtype,
            )

            kernel_model = build_model(
                method=method,
                inducing_size=inducing_size,
                num_discrete_steps=num_discrete_steps,
                start=runtime_start,
                end=runtime_end,
                device=device,
                dtype=dtype,
                rff_sample_size=flops_rff_sample_size,
            )
            kernel_flops = profile_flops(
                device,
                lambda: run_kernel_only(kernel_model, runtime_x, w_flops),
                include_cuda_activities=profile_cuda_activities,
            )
            release_memory(device, kernel_model)

            update_model = build_model(
                method=method,
                inducing_size=inducing_size,
                num_discrete_steps=num_discrete_steps,
                start=runtime_start,
                end=runtime_end,
                device=device,
                dtype=dtype,
                rff_sample_size=flops_rff_sample_size,
            )
            full_update_flops = profile_flops(
                device,
                lambda: run_full_update(update_model, runtime_x, runtime_y, w_flops, optimizer_lr),
                include_cuda_activities=profile_cuda_activities,
            )
            release_memory(device, update_model, w, w_flops)

            flops_rows.extend(
                [
                    {
                        "experiment": experiment,
                        "mode": "kernel_only",
                        "method": method,
                        "sweep_value": int(sweep_value),
                        "inducing_size": int(inducing_size),
                        "num_discrete_steps": int(num_discrete_steps),
                        "delta_t": float(delta_t),
                        "num_observations": int(runtime_x.shape[0]),
                        "rff_sample_size": int(flops_rff_sample_size),
                        "estimated_flops": float(kernel_flops),
                    },
                    {
                        "experiment": experiment,
                        "mode": "full_update",
                        "method": method,
                        "sweep_value": int(sweep_value),
                        "inducing_size": int(inducing_size),
                        "num_discrete_steps": int(num_discrete_steps),
                        "delta_t": float(delta_t),
                        "num_observations": int(runtime_x.shape[0]),
                        "rff_sample_size": int(flops_rff_sample_size),
                        "estimated_flops": float(full_update_flops),
                    },
                ]
            )

    return accuracy_rows, flops_rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    output_root = (REPO_ROOT / args.output_root).resolve()
    aggregated_dir = output_root / "aggregated"
    ensure_dir(aggregated_dir)

    set_global_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")
    dtype = resolve_dtype(args.dtype)

    x_train, y_train, x_test, y_test, acc_start, acc_end = load_solar_train_test(
        REPO_ROOT,
        device=device,
        dtype=dtype,
    )

    runtime_stream = load_runtime_stream(REPO_ROOT)
    runtime_x, runtime_y, runtime_start, runtime_end = resample_solar_observations(
        runtime_stream,
        num_observations=args.runtime_num_observations,
        device=device,
        dtype=dtype,
    )

    experiments = ["T", "M"] if args.experiment == "all" else [args.experiment]
    all_accuracy_rows: List[Dict[str, object]] = []
    all_flops_rows: List[Dict[str, object]] = []
    flops_rff_sample_size = args.flops_rff_sample_size or args.rff_sample_size

    config_payload = {
        "seed": args.seed,
        "device": str(device),
        "dtype": args.dtype,
        "t_values": args.t_values,
        "m_values": args.m_values,
        "fixed_m_for_t": args.fixed_m_for_t,
        "fixed_t_for_m": args.fixed_t_for_m,
        "rff_sample_size": args.rff_sample_size,
        "flops_rff_sample_size": flops_rff_sample_size,
        "runtime_num_observations": args.runtime_num_observations,
        "train_iters": args.train_iters,
        "optimizer_lr": args.optimizer_lr,
        "profile_cuda_activities": bool(args.profile_cuda_activities),
        "delta_t_note": (
            "For the T sweep, T denotes the number of HiPPO/ODE discretization steps over a fixed solar time span; "
            "the plotted quantity is delta_t = (t_end - t_start) / T, not the number of solar observations."
        ),
        "flops_note": (
            "estimated_flops comes from torch.profiler(with_flops=True) on a single forward or forward-backward-step call. "
            "These are profiler estimates rather than hardware peak FLOPS."
        ),
    }
    with (output_root / "accuracy_flops_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2)

    for experiment in experiments:
        sweep_values = args.t_values if experiment == "T" else args.m_values
        accuracy_rows, flops_rows = benchmark_accuracy_and_flops(
            experiment=experiment,
            sweep_values=sweep_values,
            device=device,
            dtype=dtype,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            acc_start=acc_start,
            acc_end=acc_end,
            runtime_x=runtime_x,
            runtime_y=runtime_y,
            runtime_start=runtime_start,
            runtime_end=runtime_end,
            fixed_m_for_t=args.fixed_m_for_t,
            fixed_t_for_m=args.fixed_t_for_m,
            rff_sample_size=args.rff_sample_size,
            flops_rff_sample_size=flops_rff_sample_size,
            seed=args.seed,
            train_iters=args.train_iters,
            optimizer_lr=args.optimizer_lr,
            profile_cuda_activities=bool(args.profile_cuda_activities),
        )
        all_accuracy_rows.extend(accuracy_rows)
        all_flops_rows.extend(flops_rows)

        write_csv(
            aggregated_dir / f"accuracy_vs_{experiment}_summary.csv",
            accuracy_rows,
            fieldnames=[
                "experiment",
                "method",
                "sweep_value",
                "inducing_size",
                "num_discrete_steps",
                "delta_t",
                "train_iters",
                "train_time_sec",
                "rmse",
                "nll",
            ],
        )
        write_csv(
            aggregated_dir / f"flops_vs_{experiment}_summary.csv",
            flops_rows,
            fieldnames=[
                "experiment",
                "mode",
                "method",
                "sweep_value",
                "inducing_size",
                "num_discrete_steps",
                "delta_t",
                "num_observations",
                "rff_sample_size",
                "estimated_flops",
            ],
        )

    print(f"Saved accuracy/FLOP artifacts under: {output_root}")
    print(f"Recorded {len(all_accuracy_rows)} RMSE rows and {len(all_flops_rows)} FLOP rows.")


if __name__ == "__main__":
    main()

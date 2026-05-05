#!/usr/bin/env python
"""Solar RMSE and FLOP benchmarks built on the repository's solar evaluation path."""

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
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_scaling import (  # type: ignore
    DEFAULT_FIXED_M,
    DEFAULT_FIXED_T,
    DEFAULT_M_SWEEP,
    DEFAULT_T_SWEEP,
    SOLAR_LOG_LS,
    build_model,
    ensure_dir,
    load_base_solar_stream as load_runtime_stream,
    resample_solar_observations,
    resolve_dtype,
    run_full_update,
    run_kernel_only,
    sample_fixed_w,
    set_global_seed,
    sync_device,
)
from hipposvgp.hippo import HiPPO_LegS
from hipposvgp.sgpr import HIPPO_OSGPR_VFE, HIPPO_SVGP
from scripts.onedim.solar.baseline import FullGP
from scripts.onedim.solar.test_ohsgp_analytic_solar import (
    ANALYTIC_FIXED_CFG,
    HIPPO_OSGPR_VFE_ANALYTIC,
    HIPPO_SVGP_ANALYTIC,
    _configure_analytic_runtime,
    _sample_w,
    resolve_fixed_hypers,
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
        help="Equivalent global discretization counts used to derive delta_t for the T sweep.",
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
        help="Equivalent global discretization count used to derive delta_t for the M sweep.",
    )
    parser.add_argument(
        "--rff-sample-size",
        type=int,
        default=1000,
        help="Model-side random Fourier feature count R.",
    )
    parser.add_argument(
        "--rff-sample-size-qb",
        type=int,
        default=5000,
        help="Prediction/qb spectral sample count used by the solar scripts.",
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
        "--optimizer-lr",
        type=float,
        default=1e-2,
        help="Learning rate for the one-step full_update FLOP profile.",
    )
    parser.add_argument(
        "--split-replace",
        type=int,
        default=1,
        choices=[0, 1],
        help="Match the original solar scripts: 1 samples test indices with replacement.",
    )
    parser.add_argument("--kernel-type", default="se", choices=["se"], help="Kernel family.")
    parser.add_argument("--model-full-dir", default="./models", help="Directory containing the FullGP checkpoint.")
    parser.add_argument("--fullgp-ckpt", default="steps_nll2000", help="FullGP checkpoint filename.")
    parser.add_argument(
        "--spectral-sample-device",
        default="cpu",
        choices=["cpu", "match"],
        help="Device used to sample w in the analytic script path.",
    )
    parser.add_argument(
        "--phase-origin-mode",
        default="global_start",
        choices=["global_start", "zero"],
        help="Phase origin used in the analytic spherical-Bessel construction.",
    )
    parser.add_argument(
        "--analytic-compile",
        type=int,
        default=0,
        choices=[0, 1],
        help="Forwarded to the analytic script runtime configuration.",
    )
    parser.add_argument(
        "--analytic-compile-mode",
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        help="Forwarded to the analytic script runtime configuration.",
    )
    parser.add_argument(
        "--analytic-loop-backend",
        default="auto",
        choices=["auto", "python", "scan"],
        help="Forwarded to the analytic script runtime configuration.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/benchmarks",
        help="Root directory for benchmark CSV outputs.",
    )
    return parser.parse_args()


def release_memory(device: torch.device, *objs) -> None:
    for obj in objs:
        if obj is not None:
            del obj
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_solar_task_split(
    *,
    seed: int,
    split_replace: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    data_train = np.load(REPO_ROOT / "data" / "solar_data_train.npy")
    data_test = np.load(REPO_ROOT / "data" / "solar_data_test.npy")
    data = np.vstack([data_train, data_test])
    data = data[np.argsort(data[:, 0])]
    full_span = float(data[:, 0].max() - data[:, 0].min())

    rng = np.random.RandomState(seed)
    ind_test = rng.choice(np.arange(data.shape[0]), size=91, replace=bool(split_replace))
    ind_train: List[int] = []
    for i in range(data.shape[0]):
        if i in ind_test:
            continue
        ind_train.append(i)

    # Match the original solar scripts exactly:
    # `data_test = data[ind_test]` preserves duplicates when split_replace=1.
    data_test = data[ind_test]
    data_train = data[ind_train]
    data_train = data_train[np.argsort(data_train[:, 0])]

    x = torch.tensor(np.expand_dims(data_train[:, 0], 1), dtype=dtype, device=device)
    y = torch.tensor(np.expand_dims(data_train[:, 1], 1), dtype=dtype, device=device)
    x_t = np.expand_dims(data_test[:, 0], 1)
    y_t = np.expand_dims(data_test[:, 1], 1)
    x_test = torch.tensor(np.arange(-200.0, 200.0, 2.0).reshape((-1, 1)), dtype=dtype, device=device)

    x_list: List[torch.Tensor] = []
    y_list: List[torch.Tensor] = []
    split_size = x.shape[0] // 10
    for i in range(9):
        x_list.append(x[i * split_size : (i + 1) * split_size])
        y_list.append(y[i * split_size : (i + 1) * split_size])
    x_list.append(x[9 * split_size :])
    y_list.append(y[9 * split_size :])

    return {
        "x": x,
        "y": y,
        "x_t": x_t,
        "y_t": y_t,
        "x_test": x_test,
        "x_list": x_list,
        "y_list": y_list,
        "full_span": full_span,
    }


def build_arg_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        seed_num=args.seed,
        device=str(args.device),
        data="solar",
        kernel_type=args.kernel_type,
        inducing_size=0,
        rff_sample_size=args.rff_sample_size,
        rff_sample_size_qb=args.rff_sample_size_qb,
        rff_sample_size_pred=args.rff_sample_size_qb,
        step_size=0.0,
        model_full_dir=args.model_full_dir,
        fullgp_ckpt=args.fullgp_ckpt,
        hyper_source="load_fullgp",
        log_ls=1.4108,
        log_sf=-0.5334,
        log_noise=-4.5,
        split_replace=args.split_replace,
        out_dir="",
        analytic_z_type="bessel",
        phase_origin_mode=args.phase_origin_mode,
        spectral_sample_device=args.spectral_sample_device,
        analytic_compile=args.analytic_compile,
        analytic_compile_mode=args.analytic_compile_mode,
        analytic_loop_backend=args.analytic_loop_backend,
    )


def assign_fixed_hypers(model, log_ls: float, log_sf: float, log_noise: float, device: torch.device) -> None:
    model.log_noise = nn.Parameter(
        torch.tensor(np.array([log_noise]), dtype=torch.float64, device=device),
        requires_grad=False,
    )
    model.Kernel.log_ls = nn.Parameter(
        torch.tensor(np.array([log_ls]), dtype=torch.float64, device=device),
        requires_grad=False,
    )
    model.Kernel.log_sf = nn.Parameter(
        torch.tensor(np.array([log_sf]), dtype=torch.float64, device=device),
        requires_grad=False,
    )


def maybe_save_prediction_plot(
    *,
    out_dir: Path,
    filename: str,
    x_test: torch.Tensor,
    y_mean: np.ndarray,
    y_cov: np.ndarray,
    x: torch.Tensor,
    y: torch.Tensor,
    x_t: np.ndarray,
    y_t: np.ndarray,
    x_list: Sequence[torch.Tensor],
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 3))
    plt.plot(x_test.detach().cpu().numpy().flatten(), y_mean.flatten(), label=filename.split(".")[0], color="orange")
    plt.fill_between(
        x_test.detach().cpu().numpy().flatten(),
        (y_mean.flatten() - 1.96 * np.sqrt(np.abs(y_cov).flatten())),
        (y_mean.flatten() + 1.96 * np.sqrt(np.abs(y_cov).flatten())),
        alpha=0.3,
        color="orange",
    )
    plt.scatter(
        x.detach().cpu().numpy().flatten(),
        y.detach().cpu().numpy().flatten(),
        color="blue",
        label="train",
        marker=".",
    )
    plt.scatter(x_t.flatten(), y_t.flatten(), color="black", label="test", marker=".")
    plt.axvline(x=torch.max(x_list[0]).item(), color="grey", linestyle="--", label="splits")
    for i in range(1, 10):
        plt.axvline(x=torch.max(x_list[i]).item(), color="grey", linestyle="--")
    plt.legend(bbox_to_anchor=(1.01, 0.75), loc="upper left")
    plt.savefig(out_dir / filename, format="pdf", bbox_inches="tight")
    plt.close()


def task_rmse_nll(model, x_t: np.ndarray, y_t: np.ndarray, threshold: float, device: torch.device, w) -> tuple[float, float]:
    mask = x_t.flatten() < threshold
    x_eval = torch.tensor(x_t[mask], dtype=torch.float64, device=device)
    y_eval = torch.tensor(y_t[mask], dtype=torch.float64, device=device)
    nll, rmse = model.nll_rmse(x_eval, y_eval, w=w)
    return float(rmse.item()), float(nll.item())


def run_ode_sequence(
    *,
    args_ns: argparse.Namespace,
    split_data: dict[str, object],
    inducing_size: int,
    step_size: float,
    device: torch.device,
    save_dir: Path | None = None,
) -> tuple[List[Dict[str, object]], float]:
    x = split_data["x"]
    y = split_data["y"]
    x_t = split_data["x_t"]
    y_t = split_data["y_t"]
    x_test = split_data["x_test"]
    x_list = split_data["x_list"]
    y_list = split_data["y_list"]

    log_ls, log_sf, log_noise, _ = resolve_fixed_hypers(args_ns, x, y, device)
    hippo = HiPPO_LegS(inducing_size, device)
    prev_discrete_steps = 0.0
    rows: List[Dict[str, object]] = []

    start_t = time.time()
    maxT = torch.max(x_list[0]).item() - torch.min(x_list[0]).item()
    start = torch.min(x_list[0]).item()
    num_discrete_steps = math.ceil(maxT / step_size)
    end = start + num_discrete_steps * step_size
    maxT = end - start

    model = HIPPO_SVGP(
        X=x_list[0],
        y=y_list[0],
        maxT=maxT,
        inducing_size=inducing_size,
        rff_sample_size=args_ns.rff_sample_size,
        num_discrete_steps=num_discrete_steps,
        start=start,
        device=device,
        hippo=hippo,
        kernel_type=args_ns.kernel_type,
    ).to(device)
    assign_fixed_hypers(model, log_ls, log_sf, log_noise, device)
    prev_discrete_steps += num_discrete_steps

    w = model.Kernel.sample_from_spectral(args_ns.rff_sample_size_qb).unsqueeze(0)
    if save_dir is not None:
        y_mean, y_cov = model.pred(x_test, w=w)
        maybe_save_prediction_plot(
            out_dir=save_dir,
            filename=f"hsgp1_{args_ns.kernel_type}_{inducing_size}z.pdf",
            x_test=x_test,
            y_mean=y_mean.detach().cpu().numpy(),
            y_cov=y_cov.detach().cpu().numpy(),
            x=x,
            y=y,
            x_t=x_t,
            y_t=y_t,
            x_list=x_list,
        )

    mu_old, Su_old, Kaa_old = model.qb_opt(w)
    Z_old, _ = model.get_Z(w)
    rmse, nll = task_rmse_nll(model, x_t, y_t, torch.max(x_list[0]).item(), device, w)
    rows.append(
        {
            "task": 1,
            "rmse": rmse,
            "nll": nll,
            "num_discrete_steps": int(num_discrete_steps),
            "task_start": float(start),
            "task_end": float(end),
        }
    )

    Kaa_old = Kaa_old.detach()
    mu_old = mu_old.detach()
    Su_old = Su_old.detach()
    Z_old = Z_old.detach()

    for i in range(1, 10):
        start = end
        end = torch.max(x_list[i]).item()
        num_discrete_steps = math.ceil((end - start) / step_size)
        end = start + num_discrete_steps * step_size

        cur_model = HIPPO_OSGPR_VFE(
            X=x_list[i],
            y=y_list[i],
            prev_discrete_steps=prev_discrete_steps,
            mu_old=mu_old,
            Su_old=Su_old,
            Kaa_old=Kaa_old,
            Z_old=Z_old,
            end=end,
            start=start,
            rff_sample_size=args_ns.rff_sample_size,
            hippo=hippo,
            num_discrete_steps=num_discrete_steps,
            inducing_size=inducing_size,
            device=device,
            kernel_type=args_ns.kernel_type,
        ).to(device)
        assign_fixed_hypers(cur_model, log_ls, log_sf, log_noise, device)
        prev_discrete_steps += num_discrete_steps

        if save_dir is not None:
            y_mean, y_cov = cur_model.pred(x_test, w=w)
            maybe_save_prediction_plot(
                out_dir=save_dir,
                filename=f"hsgp{i + 1}_{args_ns.kernel_type}_{inducing_size}z.pdf",
                x_test=x_test,
                y_mean=y_mean.detach().cpu().numpy(),
                y_cov=y_cov.detach().cpu().numpy(),
                x=x,
                y=y,
                x_t=x_t,
                y_t=y_t,
                x_list=x_list,
            )

        mu_old, Su_old, Kaa_old = cur_model.qb_opt(w)
        Z_old, _ = cur_model.get_Z(w)
        rmse, nll = task_rmse_nll(cur_model, x_t, y_t, torch.max(x_list[i]).item(), device, w)
        rows.append(
            {
                "task": i + 1,
                "rmse": rmse,
                "nll": nll,
                "num_discrete_steps": int(num_discrete_steps),
                "task_start": float(start),
                "task_end": float(end),
            }
        )

        Kaa_old = Kaa_old.detach()
        mu_old = mu_old.detach()
        Su_old = Su_old.detach()
        Z_old = Z_old.detach()
        release_memory(device, cur_model)

    release_memory(device, model, w, mu_old, Su_old, Kaa_old, Z_old)
    return rows, time.time() - start_t


def run_analytic_sequence(
    *,
    args_ns: argparse.Namespace,
    split_data: dict[str, object],
    inducing_size: int,
    step_size: float,
    device: torch.device,
    save_dir: Path | None = None,
) -> tuple[List[Dict[str, object]], float]:
    x = split_data["x"]
    y = split_data["y"]
    x_t = split_data["x_t"]
    y_t = split_data["y_t"]
    x_test = split_data["x_test"]
    x_list = split_data["x_list"]
    y_list = split_data["y_list"]

    log_ls, log_sf, log_noise, _ = resolve_fixed_hypers(args_ns, x, y, device)
    hippo = HiPPO_LegS(inducing_size, device)
    prev_discrete_steps = 0.0
    rows: List[Dict[str, object]] = []
    phase_origin = torch.min(x_list[0]).item() if args_ns.phase_origin_mode == "global_start" else 0.0

    start_t = time.time()
    maxT = torch.max(x_list[0]).item() - torch.min(x_list[0]).item()
    start = torch.min(x_list[0]).item()
    num_discrete_steps = math.ceil(maxT / step_size)
    end = start + num_discrete_steps * step_size
    maxT = end - start

    model = HIPPO_SVGP_ANALYTIC(
        X=x_list[0],
        y=y_list[0],
        maxT=maxT,
        inducing_size=inducing_size,
        rff_sample_size=args_ns.rff_sample_size,
        num_discrete_steps=num_discrete_steps,
        start=start,
        device=device,
        hippo=hippo,
        kernel_type=args_ns.kernel_type,
    ).to(device)
    _configure_analytic_runtime(model, args_ns, phase_origin)
    assign_fixed_hypers(model, log_ls, log_sf, log_noise, device)
    prev_discrete_steps += num_discrete_steps

    w = _sample_w(model, args_ns.rff_sample_size_qb, sample_device_mode=args_ns.spectral_sample_device)
    if save_dir is not None:
        y_mean, y_cov = model.pred(x_test, w=w)
        maybe_save_prediction_plot(
            out_dir=save_dir,
            filename=f"hsgp1_analytic_{args_ns.kernel_type}_{inducing_size}z.pdf",
            x_test=x_test,
            y_mean=y_mean.detach().cpu().numpy(),
            y_cov=y_cov.detach().cpu().numpy(),
            x=x,
            y=y,
            x_t=x_t,
            y_t=y_t,
            x_list=x_list,
        )

    mu_old, Su_old, Kaa_old = model.qb_opt(w)
    Z_old, _ = model.get_Z(w)
    rmse, nll = task_rmse_nll(model, x_t, y_t, torch.max(x_list[0]).item(), device, w)
    rows.append(
        {
            "task": 1,
            "rmse": rmse,
            "nll": nll,
            "num_discrete_steps": int(num_discrete_steps),
            "task_start": float(start),
            "task_end": float(end),
        }
    )

    Kaa_old = Kaa_old.detach()
    mu_old = mu_old.detach()
    Su_old = Su_old.detach()
    Z_old = Z_old.detach()

    for i in range(1, 10):
        start = end
        end = torch.max(x_list[i]).item()
        num_discrete_steps = math.ceil((end - start) / step_size)
        end = start + num_discrete_steps * step_size

        cur_model = HIPPO_OSGPR_VFE_ANALYTIC(
            X=x_list[i],
            y=y_list[i],
            prev_discrete_steps=prev_discrete_steps,
            mu_old=mu_old,
            Su_old=Su_old,
            Kaa_old=Kaa_old,
            Z_old=Z_old,
            end=end,
            start=start,
            rff_sample_size=args_ns.rff_sample_size,
            hippo=hippo,
            num_discrete_steps=num_discrete_steps,
            inducing_size=inducing_size,
            device=device,
            kernel_type=args_ns.kernel_type,
        ).to(device)
        _configure_analytic_runtime(cur_model, args_ns, phase_origin)
        cur_model.analytic_psd_project = ANALYTIC_FIXED_CFG["analytic_psd_project"]
        cur_model.analytic_robust_cholesky = ANALYTIC_FIXED_CFG["analytic_robust_cholesky"]
        cur_model.analytic_symmetrize = ANALYTIC_FIXED_CFG["analytic_symmetrize"]
        cur_model.analytic_base_jitter = ANALYTIC_FIXED_CFG["analytic_base_jitter"]
        assign_fixed_hypers(cur_model, log_ls, log_sf, log_noise, device)
        prev_discrete_steps += num_discrete_steps

        if save_dir is not None:
            y_mean, y_cov = cur_model.pred(x_test, w=w)
            maybe_save_prediction_plot(
                out_dir=save_dir,
                filename=f"hsgp{i + 1}_analytic_{args_ns.kernel_type}_{inducing_size}z.pdf",
                x_test=x_test,
                y_mean=y_mean.detach().cpu().numpy(),
                y_cov=y_cov.detach().cpu().numpy(),
                x=x,
                y=y,
                x_t=x_t,
                y_t=y_t,
                x_list=x_list,
            )

        mu_old, Su_old, Kaa_old = cur_model.qb_opt(w)
        Z_old, _ = cur_model.get_Z(w)
        rmse, nll = task_rmse_nll(cur_model, x_t, y_t, torch.max(x_list[i]).item(), device, w)
        rows.append(
            {
                "task": i + 1,
                "rmse": rmse,
                "nll": nll,
                "num_discrete_steps": int(num_discrete_steps),
                "task_start": float(start),
                "task_end": float(end),
            }
        )

        Kaa_old = Kaa_old.detach()
        mu_old = mu_old.detach()
        Su_old = Su_old.detach()
        Z_old = Z_old.detach()
        release_memory(device, cur_model)

    release_memory(device, model, w, mu_old, Su_old, Kaa_old, Z_old)
    return rows, time.time() - start_t


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


def summarize_accuracy_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Dict[str, object]]] = {}
    for row in rows:
        key = (row["experiment"], row["method"], int(row["sweep_value"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for (experiment, method, sweep_value), group in sorted(grouped.items()):
        rmse_vals = np.asarray([float(r["rmse"]) for r in group], dtype=np.float64)
        nll_vals = np.asarray([float(r["nll"]) for r in group], dtype=np.float64)
        total_time_sec = float(group[0]["total_time_sec"])
        summary_rows.append(
            {
                "experiment": experiment,
                "method": method,
                "sweep_value": sweep_value,
                "inducing_size": int(group[0]["inducing_size"]),
                "num_discrete_steps_equiv": int(group[0]["num_discrete_steps_equiv"]),
                "delta_t": float(group[0]["delta_t"]),
                "num_tasks": len(group),
                "rmse_mean": float(np.mean(rmse_vals)),
                "rmse_std": float(np.std(rmse_vals, ddof=0)),
                "rmse_min": float(np.min(rmse_vals)),
                "rmse_max": float(np.max(rmse_vals)),
                "nll_mean": float(np.mean(nll_vals)),
                "nll_std": float(np.std(nll_vals, ddof=0)),
                "total_time_sec": total_time_sec,
            }
        )
    return summary_rows


def benchmark_accuracy_and_flops(
    *,
    args: argparse.Namespace,
    experiment: str,
    sweep_values: Iterable[int],
    device: torch.device,
    dtype: torch.dtype,
    runtime_x: torch.Tensor,
    runtime_y: torch.Tensor,
    runtime_start: float,
    runtime_end: float,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    args_ns = build_arg_namespace(args)
    split_data = load_solar_task_split(
        seed=args.seed,
        split_replace=args.split_replace,
        device=device,
        dtype=dtype,
    )
    full_span = float(split_data["full_span"])
    accuracy_rows: List[Dict[str, object]] = []
    flops_rows: List[Dict[str, object]] = []
    flops_rff_sample_size = args.flops_rff_sample_size or args.rff_sample_size

    for sweep_value in sweep_values:
        if experiment == "T":
            inducing_size = args.fixed_m_for_t
            num_discrete_steps_equiv = int(sweep_value)
            delta_t = full_span / float(num_discrete_steps_equiv)
        elif experiment == "M":
            inducing_size = int(sweep_value)
            num_discrete_steps_equiv = int(args.fixed_t_for_m)
            delta_t = full_span / float(num_discrete_steps_equiv)
        else:
            raise ValueError(f"Unsupported experiment: {experiment}")

        ode_rows, ode_total_time = run_ode_sequence(
            args_ns=args_ns,
            split_data=split_data,
            inducing_size=inducing_size,
            step_size=delta_t,
            device=device,
            save_dir=None,
        )
        analytic_rows, analytic_total_time = run_analytic_sequence(
            args_ns=args_ns,
            split_data=split_data,
            inducing_size=inducing_size,
            step_size=delta_t,
            device=device,
            save_dir=None,
        )

        for method, rows, total_time_sec in (
            ("ode", ode_rows, ode_total_time),
            ("analytic", analytic_rows, analytic_total_time),
        ):
            for row in rows:
                accuracy_rows.append(
                    {
                        "experiment": experiment,
                        "method": method,
                        "sweep_value": int(sweep_value),
                        "inducing_size": int(inducing_size),
                        "num_discrete_steps_equiv": int(num_discrete_steps_equiv),
                        "delta_t": float(delta_t),
                        "task": int(row["task"]),
                        "rmse": float(row["rmse"]),
                        "nll": float(row["nll"]),
                        "task_num_discrete_steps": int(row["num_discrete_steps"]),
                        "task_start": float(row["task_start"]),
                        "task_end": float(row["task_end"]),
                        "total_time_sec": float(total_time_sec),
                    }
                )

        for method in ("ode", "analytic"):
            w_flops = sample_fixed_w(
                num_rff=flops_rff_sample_size,
                log_ls=SOLAR_LOG_LS,
                seed=args.seed,
                device=device,
                dtype=dtype,
            )
            kernel_model = build_model(
                method=method,
                inducing_size=inducing_size,
                num_discrete_steps=num_discrete_steps_equiv,
                start=runtime_start,
                end=runtime_end,
                device=device,
                dtype=dtype,
                rff_sample_size=flops_rff_sample_size,
                analytic_compile=bool(args.analytic_compile),
                analytic_compile_mode=args.analytic_compile_mode,
            )
            kernel_flops = profile_flops(
                device,
                lambda: run_kernel_only(kernel_model, runtime_x, w_flops),
                include_cuda_activities=bool(args.profile_cuda_activities),
            )
            release_memory(device, kernel_model)

            update_model = build_model(
                method=method,
                inducing_size=inducing_size,
                num_discrete_steps=num_discrete_steps_equiv,
                start=runtime_start,
                end=runtime_end,
                device=device,
                dtype=dtype,
                rff_sample_size=flops_rff_sample_size,
                analytic_compile=bool(args.analytic_compile),
                analytic_compile_mode=args.analytic_compile_mode,
            )
            full_update_flops = profile_flops(
                device,
                lambda: run_full_update(update_model, runtime_x, runtime_y, w_flops, args.optimizer_lr),
                include_cuda_activities=bool(args.profile_cuda_activities),
            )
            release_memory(device, update_model, w_flops)

            flops_rows.extend(
                [
                    {
                        "experiment": experiment,
                        "mode": "kernel_only",
                        "method": method,
                        "sweep_value": int(sweep_value),
                        "inducing_size": int(inducing_size),
                        "num_discrete_steps_equiv": int(num_discrete_steps_equiv),
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
                        "num_discrete_steps_equiv": int(num_discrete_steps_equiv),
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
    raw_dir = output_root / "raw"
    ensure_dir(aggregated_dir)
    ensure_dir(raw_dir)

    set_global_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")
    dtype = resolve_dtype(args.dtype)

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
        "rff_sample_size_qb": args.rff_sample_size_qb,
        "flops_rff_sample_size": flops_rff_sample_size,
        "runtime_num_observations": args.runtime_num_observations,
        "optimizer_lr": args.optimizer_lr,
        "split_replace": args.split_replace,
        "spectral_sample_device": args.spectral_sample_device,
        "profile_cuda_activities": bool(args.profile_cuda_activities),
        "analytic_runtime_impl": "pure_torch_miller_recurrence",
        "delta_t_note": (
            "For the T sweep, delta_t is derived from the full solar time span divided by an equivalent global step count. "
            "Per-task num_discrete_steps is then computed inside the original 10-task solar scripts using that step_size."
        ),
        "accuracy_note": (
            "RMSE/NLL are produced by the repository's existing scripts/onedim/solar/test_hsgp_online_fixedkernel.py "
            "and scripts/onedim/solar/test_ohsgp_analytic_solar.py evaluation logic, aggregated over the 10 solar tasks."
        ),
        "flops_note": (
            "estimated_flops comes from torch.profiler(with_flops=True) on a single forward or forward-backward-step call. "
            "These are profiler estimates rather than hardware peak FLOPS. The analytic runtime/FLOP path uses the pure-torch "
            "Miller recurrence implementation rather than the older scipy-based spherical-Bessel path."
        ),
    }
    with (output_root / "accuracy_flops_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2)

    for experiment in experiments:
        sweep_values = args.t_values if experiment == "T" else args.m_values
        accuracy_rows, flops_rows = benchmark_accuracy_and_flops(
            args=args,
            experiment=experiment,
            sweep_values=sweep_values,
            device=device,
            dtype=dtype,
            runtime_x=runtime_x,
            runtime_y=runtime_y,
            runtime_start=runtime_start,
            runtime_end=runtime_end,
        )
        summary_rows = summarize_accuracy_rows(accuracy_rows)
        all_accuracy_rows.extend(accuracy_rows)
        all_flops_rows.extend(flops_rows)

        write_csv(
            raw_dir / f"accuracy_vs_{experiment}_raw.csv",
            accuracy_rows,
            fieldnames=[
                "experiment",
                "method",
                "sweep_value",
                "inducing_size",
                "num_discrete_steps_equiv",
                "delta_t",
                "task",
                "rmse",
                "nll",
                "task_num_discrete_steps",
                "task_start",
                "task_end",
                "total_time_sec",
            ],
        )
        write_csv(
            aggregated_dir / f"accuracy_vs_{experiment}_summary.csv",
            summary_rows,
            fieldnames=[
                "experiment",
                "method",
                "sweep_value",
                "inducing_size",
                "num_discrete_steps_equiv",
                "delta_t",
                "num_tasks",
                "rmse_mean",
                "rmse_std",
                "rmse_min",
                "rmse_max",
                "nll_mean",
                "nll_std",
                "total_time_sec",
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
                "num_discrete_steps_equiv",
                "delta_t",
                "num_observations",
                "rff_sample_size",
                "estimated_flops",
            ],
        )

    print(f"Saved accuracy/FLOP artifacts under: {output_root}")
    print(f"Recorded {len(all_accuracy_rows)} task-level RMSE rows and {len(all_flops_rows)} FLOP rows.")


if __name__ == "__main__":
    main()

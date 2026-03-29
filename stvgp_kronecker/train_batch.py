"""Synthetic and ERA5 batch training entrypoint for Stage 1."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

try:
    from .era5_dataset import (
        ERA5_LAND_VARIABLES,
        build_temporal_blocks,
        count_processed_era5_locations,
        discover_processed_era5_task_dirs,
        era5_variable_name,
        load_era5_grid,
        load_processed_era5_task,
        load_processed_era5_tasks,
    )
    from .spatial_kernel import SpatialKernelConfig
    from .st_model_batch import BatchKroneckerSTHiPPOSVGP
    from .temporal_analytic import TemporalAnalyticConfig
    from .visualization import choose_snapshot_indices, parse_snapshot_indices, save_era5_prediction_maps
except ImportError:  # pragma: no cover - allows direct `python path/to/train_batch.py`
    from stvgp_kronecker.era5_dataset import (
        ERA5_LAND_VARIABLES,
        build_temporal_blocks,
        count_processed_era5_locations,
        discover_processed_era5_task_dirs,
        era5_variable_name,
        load_era5_grid,
        load_processed_era5_task,
        load_processed_era5_tasks,
    )
    from stvgp_kronecker.spatial_kernel import SpatialKernelConfig
    from stvgp_kronecker.st_model_batch import BatchKroneckerSTHiPPOSVGP
    from stvgp_kronecker.temporal_analytic import TemporalAnalyticConfig
    from stvgp_kronecker.visualization import choose_snapshot_indices, parse_snapshot_indices, save_era5_prediction_maps


def make_synthetic_dataset(
    num_times: int = 24,
    spatial_grid_size: int = 5,
    noise_std: float = 0.05,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a small separable synthetic dataset."""

    times = torch.linspace(0.0, 6.0, num_times, dtype=dtype)
    coords_1d = torch.linspace(-1.0, 1.0, spatial_grid_size, dtype=dtype)
    x1, x2 = torch.meshgrid(coords_1d, coords_1d, indexing="ij")
    spatial = torch.stack([x1.reshape(-1), x2.reshape(-1)], dim=-1)

    temporal_signal = torch.sin(1.3 * times) + 0.25 * torch.cos(0.7 * times)
    spatial_signal = torch.exp(-2.0 * ((spatial[:, 0] - 0.25) ** 2 + (spatial[:, 1] + 0.2) ** 2))
    spatial_signal = spatial_signal - 0.7 * torch.exp(
        -3.5 * ((spatial[:, 0] + 0.4) ** 2 + (spatial[:, 1] - 0.35) ** 2)
    )

    clean = temporal_signal[:, None] * spatial_signal[None, :]
    noise = noise_std * torch.randn_like(clean)
    observations = clean + noise
    return times, spatial, observations


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))


def predictive_nll(y_true: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(torch.log(2.0 * torch.pi * var) + (y_true - mean) ** 2 / var)


def current_gpu_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024.0**2)


def resolve_device(device_arg: str) -> torch.device:
    """Resolve the requested runtime device, preferring CUDA when available."""

    normalized = device_arg.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        print("[device] requested cuda but CUDA is unavailable; falling back to cpu")
        normalized = "cpu"
    device = torch.device(normalized)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    return device


def move_tensor_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move one tensor to the requested runtime device."""

    return torch.as_tensor(tensor).to(device=device)


def clone_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Create a detached clone of a module state dict."""

    return {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}


def maybe_project_spatial_inducing(
    model: BatchKroneckerSTHiPPOSVGP,
    spatial: torch.Tensor,
) -> None:
    """Keep learned spatial inducing points inside the observed spatial bounds."""

    z_param = getattr(model, "z_s", None)
    if not isinstance(z_param, torch.nn.Parameter):
        return

    lower = spatial.min(dim=0).values.to(dtype=z_param.dtype, device=z_param.device)
    upper = spatial.max(dim=0).values.to(dtype=z_param.dtype, device=z_param.device)
    with torch.no_grad():
        z_param.data.clamp_(min=lower, max=upper)


def format_model_hyperparameters(model: BatchKroneckerSTHiPPOSVGP) -> str:
    """Summarize the currently active kernel/noise hyperparameters."""

    temporal_lengthscale = float(model.temporal_builder.lengthscale.detach().cpu())
    temporal_variance = float(model.temporal_builder.variance.detach().cpu())
    spatial_lengthscale = model.spatial_kernel.lengthscale.detach().cpu().reshape(-1).tolist()
    spatial_variance = float(model.spatial_kernel.variance.detach().cpu())
    noise_std = float(torch.exp(model.log_noise_std).detach().cpu())
    spatial_lengthscale_str = ",".join(f"{value:.4f}" for value in spatial_lengthscale)
    return (
        f"temporal_lengthscale={temporal_lengthscale:.4f} "
        f"temporal_variance={temporal_variance:.4f} "
        f"spatial_lengthscale=[{spatial_lengthscale_str}] "
        f"spatial_variance={spatial_variance:.4f} "
        f"noise_std={noise_std:.4f}"
    )


def resolve_era5_task_dirs(args: argparse.Namespace) -> list[str]:
    """Resolve ERA5 task directories from explicit paths or auto-discovery."""

    if getattr(args, "era5_full_task12", False):
        return [str(path) for path in discover_processed_era5_task_dirs(args.era5_task_root, ["task_1", "task_2"])]

    explicit_dirs = [item for item in args.era5_task_dirs if item]
    if args.era5_task_dir:
        explicit_dirs = [args.era5_task_dir]
    if explicit_dirs:
        return explicit_dirs

    if not args.era5_all_tasks:
        return []

    return [str(path) for path in discover_processed_era5_task_dirs(args.era5_task_root)]


def resolve_era5_max_locations(args: argparse.Namespace, task_dirs: list[str]) -> int | None:
    """Resolve how many processed ERA5 locations to keep.

    `era5_full_task12` and `era5_use_all_locations` both mean "do not truncate".
    """

    if getattr(args, "era5_full_task12", False) or getattr(args, "era5_use_all_locations", False):
        return None
    if getattr(args, "era5_max_locations", None) is None:
        return None
    if args.era5_max_locations <= 0:
        return None
    return int(args.era5_max_locations)


def infer_processed_era5_location_count(
    task_dirs: list[str],
    scaled: bool,
    location_stride: int,
) -> int | None:
    """Infer the shared location count across processed ERA5 tasks."""

    if not task_dirs:
        return None
    return min(
        count_processed_era5_locations(task_dir, scaled=scaled, location_stride=location_stride)
        for task_dir in task_dirs
    )


def validate_era5_spatial_coords(spatial_coords: torch.Tensor) -> None:
    """ERA5 processed tasks must use exactly two spatial coordinates: `(lon, lat)`."""

    if spatial_coords.ndim != 2 or spatial_coords.shape[1] != 2:
        raise ValueError(
            "Processed ERA5 inputs must have shape [N_s, 2] with spatial coordinates ordered as (lon, lat)."
        )


def set_hyperparameter_trainability(
    model: BatchKroneckerSTHiPPOSVGP,
    trainable: bool,
) -> None:
    """Freeze or unfreeze kernel/noise hyperparameters while leaving `Z_s` untouched."""

    for parameter in [
        model.log_noise_std,
        model.temporal_builder.log_variance,
        model.temporal_builder.log_lengthscale,
        model.spatial_kernel.log_variance,
        model.spatial_kernel.log_lengthscale,
    ]:
        parameter.requires_grad_(trainable)


def resolve_era5_covariate_indices(args: argparse.Namespace) -> list[int]:
    requested = getattr(args, "era5_covariate_indices", None)
    if requested is None:
        return []
    if len(requested) == 0:
        return [index for index in range(len(ERA5_LAND_VARIABLES)) if index != args.era5_variable_index]
    if requested:
        selected: list[int] = []
        for item in requested:
            index = int(item)
            if index != args.era5_variable_index and index not in selected:
                selected.append(index)
        return selected
    return []


def maybe_save_era5_maps(
    *,
    args: argparse.Namespace,
    mode: str,
    task_name: str,
    variable_name: str,
    split_name: str,
    split_tensor: Any,
    pred_output: dict[str, torch.Tensor],
) -> None:
    """Optionally save ERA5 ground-truth / prediction / error maps."""

    if not getattr(args, "save_era5_maps", False):
        return

    requested = parse_snapshot_indices(args.map_time_indices, split_tensor.times.shape[0])
    snapshot_indices = choose_snapshot_indices(
        split_tensor.times.shape[0],
        requested=requested,
        max_snapshots=args.map_max_snapshots,
    )
    base_dir = (
        Path(args.map_output_dir)
        if args.map_output_dir
        else Path("outputs") / "stvgp_kronecker_maps" / mode / task_name / split_name
    )
    saved = save_era5_prediction_maps(
        spatial_coords=split_tensor.spatial_coords,
        times=split_tensor.times,
        observations=split_tensor.observations,
        predicted_mean=pred_output["mean"],
        output_dir=base_dir,
        filename_prefix=f"{mode}_{split_name}",
        variable_name=variable_name,
        snapshot_indices=snapshot_indices,
        point_size=args.map_point_size,
    )
    if saved:
        print(f"[maps] saved {len(saved)} {mode} {split_name} map(s) under {base_dir}")


def summarize_final_metrics(
    observations: torch.Tensor,
    pred_output: dict[str, torch.Tensor],
    runtime_s: float,
    mean_block_time_s: float | None = None,
) -> dict[str, float]:
    metrics = {
        "rmse": float(rmse(observations, pred_output["mean"]).item()),
        "pred_nll": float(predictive_nll(observations, pred_output["mean"], pred_output["obs_var"]).item()),
        "runtime_s": float(runtime_s),
        "gpu_memory_mb": float(current_gpu_memory_mb()),
    }
    if mean_block_time_s is not None:
        metrics["time_per_block_s"] = float(mean_block_time_s)
    return metrics


def train_epoch(
    model: BatchKroneckerSTHiPPOSVGP,
    optimizer: torch.optim.Optimizer,
    times: torch.Tensor,
    spatial: torch.Tensor,
    observations: torch.Tensor,
    covariates: torch.Tensor | None = None,
    block_size: int | None = None,
    block_overlap: int = 0,
) -> dict[str, float]:
    """Run one full epoch, optionally splitting across temporal blocks."""

    start = time.perf_counter()
    optimizer.zero_grad()

    if block_size is None or block_size >= times.shape[0]:
        output = model(
            times,
            spatial,
            observations,
            covariates=covariates,
            cache_posterior=False,
            materialize_posterior_cov=False,
        )
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        maybe_project_spatial_inducing(model, spatial)
        return {
            "loss": float(loss.detach().cpu()),
            "rmse": float(output["rmse"].detach().cpu()),
            "time_s": time.perf_counter() - start,
        }

    blockwise = model.forward_blockwise(
        times=times,
        x_s=spatial,
        y=observations,
        block_size=block_size,
        overlap=block_overlap,
        num_discrete_steps=block_size,
        cache_last_posterior=False,
        covariates=covariates,
    )
    mean_loss = blockwise.mean_loss
    mean_loss.backward()
    optimizer.step()
    maybe_project_spatial_inducing(model, spatial)
    return {
        "loss": float(mean_loss.detach().cpu()),
        "rmse": float(blockwise.mean_rmse.detach().cpu()),
        "time_s": time.perf_counter() - start,
        "time_per_block_s": blockwise.mean_block_runtime_s,
    }


def fit_batch_model(
    model: BatchKroneckerSTHiPPOSVGP,
    optimizer: torch.optim.Optimizer | None,
    train_times: torch.Tensor,
    train_spatial: torch.Tensor,
    train_observations: torch.Tensor,
    train_covariates: torch.Tensor | None,
    train_steps: int,
    log_every: int,
    block_size: int | None = None,
    block_overlap: int = 0,
    val_times: torch.Tensor | None = None,
    val_spatial: torch.Tensor | None = None,
    val_observations: torch.Tensor | None = None,
    val_covariates: torch.Tensor | None = None,
    selection_metric: str = "pred_nll",
    eval_every: int = 1,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
) -> dict[str, Any]:
    """Optimize the batch model and optionally keep the best validation checkpoint."""

    if selection_metric not in {"pred_nll", "rmse"}:
        raise ValueError(f"Unsupported selection metric: {selection_metric}")
    if eval_every <= 0:
        raise ValueError(f"eval_every must be positive, got {eval_every}")
    if early_stopping_patience is not None and early_stopping_patience <= 0:
        raise ValueError(
            f"early_stopping_patience must be positive when provided, got {early_stopping_patience}"
        )
    if early_stopping_min_delta < 0:
        raise ValueError(f"early_stopping_min_delta must be non-negative, got {early_stopping_min_delta}")

    def evaluate_current_model() -> tuple[dict[str, float], dict[str, float] | None]:
        with torch.no_grad():
            if block_size is not None and block_size < train_times.shape[0]:
                train_summary = model.forward_blockwise(
                    times=train_times,
                    x_s=train_spatial,
                    y=train_observations,
                    block_size=block_size,
                    overlap=block_overlap,
                    num_discrete_steps=block_size,
                    cache_last_posterior=True,
                    covariates=train_covariates,
                )
                cached_train_runtime = train_summary.total_runtime_s
                cached_train_block_time = train_summary.mean_block_runtime_s
            else:
                train_output = model(
                    train_times,
                    train_spatial,
                    train_observations,
                    covariates=train_covariates,
                    cache_posterior=True,
                    materialize_posterior_cov=False,
                )
                cached_train_runtime = train_output["runtime_s"]
                cached_train_block_time = None

            train_pred = model.predict(train_times, train_spatial, covariates=train_covariates)
            train_eval = {
                "rmse": float(rmse(train_observations, train_pred["mean"]).item()),
                "pred_nll": float(
                    predictive_nll(train_observations, train_pred["mean"], train_pred["obs_var"]).item()
                ),
                "runtime_s": cached_train_runtime,
            }
            if cached_train_block_time is not None:
                train_eval["time_per_block_s"] = float(cached_train_block_time)

            val_eval: dict[str, float] | None = None
            if val_times is not None and val_spatial is not None and val_observations is not None:
                val_pred = model.predict(val_times, val_spatial, covariates=val_covariates)
                val_eval = {
                    "rmse": float(rmse(val_observations, val_pred["mean"]).item()),
                    "pred_nll": float(
                        predictive_nll(val_observations, val_pred["mean"], val_pred["obs_var"]).item()
                    ),
                }
        return train_eval, val_eval

    history: list[dict[str, float]] = []
    best_state = clone_state_dict(model)
    initial_train_eval, initial_val_eval = evaluate_current_model()
    best_metric = (initial_val_eval or initial_train_eval)[selection_metric]
    best_step = -1
    evaluations_since_improvement = 0
    stopped_early = False

    total_start = time.perf_counter()
    if train_steps <= 0 or optimizer is None:
        model.load_state_dict(best_state)
        return {
            "history": history,
            "best_metric": best_metric,
            "best_step": best_step,
            "training_runtime_s": time.perf_counter() - total_start,
            "stopped_early": stopped_early,
        }

    for step in range(train_steps):
        train_metrics = train_epoch(
            model,
            optimizer,
            times=train_times,
            spatial=train_spatial,
            observations=train_observations,
            covariates=train_covariates,
            block_size=block_size,
            block_overlap=block_overlap,
        )

        record = {
            "step": float(step),
            "train_loss": train_metrics["loss"],
            "step_time_s": train_metrics["time_s"],
            "evaluated": 0.0,
        }
        if "time_per_block_s" in train_metrics:
            record["step_block_time_s"] = train_metrics["time_per_block_s"]

        # Align evaluation cadence with the zero-based step number shown in logs,
        # so `--eval-every 10` evaluates at steps 0, 10, 20, ...
        should_evaluate = step % eval_every == 0 or step == train_steps - 1
        train_eval: dict[str, float] | None = None
        val_eval: dict[str, float] | None = None
        if should_evaluate:
            train_eval, val_eval = evaluate_current_model()
            monitor_value = (val_eval or train_eval)[selection_metric]

            if monitor_value < best_metric - early_stopping_min_delta:
                best_metric = monitor_value
                best_step = step
                best_state = clone_state_dict(model)
                evaluations_since_improvement = 0
            else:
                evaluations_since_improvement += 1

            record["evaluated"] = 1.0
            record["train_rmse"] = train_eval["rmse"]
            record["train_pred_nll"] = train_eval["pred_nll"]
            if val_eval is not None:
                record["val_rmse"] = val_eval["rmse"]
                record["val_pred_nll"] = val_eval["pred_nll"]
        history.append(record)

        if step % log_every == 0 or step == train_steps - 1:
            log_line = f"[train] step={step:04d} loss={train_metrics['loss']:.4f} step_time={train_metrics['time_s']:.3f}s"
            if train_eval is not None:
                log_line += f" train_rmse={train_eval['rmse']:.4f} train_nll={train_eval['pred_nll']:.4f}"
            else:
                log_line += " eval=skipped"
            if val_eval is not None:
                log_line += f" val_rmse={val_eval['rmse']:.4f} val_nll={val_eval['pred_nll']:.4f}"
            if "time_per_block_s" in train_metrics:
                log_line += f" time_per_block={train_metrics['time_per_block_s']:.3f}s"
            print(log_line)
            print(f"[hyperparams] {format_model_hyperparameters(model)}")

        if (
            should_evaluate
            and early_stopping_patience is not None
            and evaluations_since_improvement >= early_stopping_patience
        ):
            print(
                "[early-stop] step={} selection_metric={} best_step={} best_value={:.4f} "
                "patience={} min_delta={:.4g}".format(
                    step,
                    selection_metric,
                    best_step,
                    best_metric,
                    early_stopping_patience,
                    early_stopping_min_delta,
                )
            )
            stopped_early = True
            break

    model.load_state_dict(best_state)
    return {
        "history": history,
        "best_metric": best_metric,
        "best_step": best_step,
        "training_runtime_s": time.perf_counter() - total_start,
        "stopped_early": stopped_early,
    }


def build_stage1_model(
    args: argparse.Namespace,
    z_s: torch.Tensor,
    input_dim: int,
    covariate_dim: int = 0,
) -> BatchKroneckerSTHiPPOSVGP:
    """Build the Stage 1 batch model from CLI args."""

    model = BatchKroneckerSTHiPPOSVGP(
        temporal_config=TemporalAnalyticConfig(
            inducing_size=args.temporal_inducing,
            rff_sample_size=args.rff_sample_size,
            lengthscale=args.temporal_lengthscale,
            variance=1.0,
            num_discrete_steps=args.reference_steps,
            seed=args.seed,
            device=str(args.runtime_device),
        ),
        spatial_kernel_config=SpatialKernelConfig(
            input_dim=input_dim,
            kernel_type=args.spatial_kernel,
            variance=args.spatial_variance,
            lengthscale=args.spatial_lengthscale,
        ),
        z_s=z_s,
        noise_std=args.likelihood_noise,
        covariate_dim=covariate_dim,
        learn_spatial_inducing=args.learn_spatial_inducing,
    )
    return model.to(args.runtime_device)


def build_optimizer(
    model: BatchKroneckerSTHiPPOSVGP,
    learning_rate: float,
) -> torch.optim.Optimizer | None:
    """Build an optimizer over the currently trainable parameters."""

    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        return None
    return torch.optim.Adam(params, lr=learning_rate)


def select_spatial_inducing_points(
    spatial_coords: torch.Tensor,
    inducing_side: int | None = None,
    spatial_inducing_count: int | None = None,
    selection_method: str = "grid",
) -> torch.Tensor:
    """Select spatial inducing locations from observed coordinates.

    Supported methods:
    - `fps`: farthest-point sampling on normalized `(lon, lat)` coordinates
    - `grid`: snap a regular grid over the spatial extent onto observed points
    - `first`: keep the previous behavior for debugging/backward compatibility
    """

    coords = torch.as_tensor(spatial_coords)
    if coords.ndim != 2:
        raise ValueError("spatial_coords must have shape [N_s, D_s].")
    if selection_method not in {"fps", "grid", "first"}:
        raise ValueError(f"Unsupported spatial inducing selection method: {selection_method}")

    num_points = coords.shape[0]
    if spatial_inducing_count is not None and spatial_inducing_count > 0:
        target_count = min(num_points, spatial_inducing_count)
    elif inducing_side is not None and inducing_side > 0:
        target_count = min(num_points, inducing_side**2)
    else:
        return coords.clone()

    if target_count >= num_points:
        return coords.clone()
    if selection_method == "first":
        return coords[:target_count].clone()

    lower = coords.min(dim=0).values
    upper = coords.max(dim=0).values
    scale = torch.clamp(upper - lower, min=1e-12)
    normalized = (coords - lower) / scale

    def _fps_indices(points: torch.Tensor, k: int) -> torch.Tensor:
        selected = torch.empty(k, dtype=torch.long, device=points.device)
        centroid = points.mean(dim=0)
        selected[0] = torch.argmin(torch.sum((points - centroid) ** 2, dim=1))
        min_dist2 = torch.sum((points - points[selected[0]]) ** 2, dim=1)
        for idx in range(1, k):
            selected[idx] = torch.argmax(min_dist2)
            candidate_dist2 = torch.sum((points - points[selected[idx]]) ** 2, dim=1)
            min_dist2 = torch.minimum(min_dist2, candidate_dist2)
        return selected

    def _grid_indices(points: torch.Tensor, k: int) -> torch.Tensor:
        side = max(int(round(k**0.5)), 1)
        while side * side < k:
            side += 1
        xs = torch.linspace(0.0, 1.0, side, dtype=points.dtype, device=points.device)
        ys = torch.linspace(0.0, 1.0, side, dtype=points.dtype, device=points.device)
        anchors = torch.cartesian_prod(xs, ys)
        if anchors.shape[0] > k:
            anchors = anchors[:k]

        chosen: list[int] = []
        used = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
        for anchor in anchors:
            dist2 = torch.sum((points - anchor) ** 2, dim=1)
            dist2 = torch.where(used, torch.full_like(dist2, float("inf")), dist2)
            nearest = int(torch.argmin(dist2).item())
            if not used[nearest]:
                used[nearest] = True
                chosen.append(nearest)

        if len(chosen) < k:
            remaining = torch.nonzero(~used, as_tuple=False).reshape(-1)
            if remaining.numel() > 0:
                extra = _fps_indices(points[remaining], min(k - len(chosen), remaining.numel()))
                chosen.extend(remaining[extra].tolist())
        return torch.as_tensor(chosen[:k], dtype=torch.long, device=points.device)

    if selection_method == "grid":
        indices = _grid_indices(normalized, target_count)
    else:
        indices = _fps_indices(normalized, target_count)
    return coords[indices].clone()


def run_synthetic_experiment(args: argparse.Namespace) -> None:
    times, spatial, observations = make_synthetic_dataset(
        num_times=args.num_times,
        spatial_grid_size=args.spatial_grid_size,
        noise_std=args.synthetic_noise,
    )
    times = move_tensor_to_device(times, args.runtime_device)
    spatial = move_tensor_to_device(spatial, args.runtime_device)
    observations = move_tensor_to_device(observations, args.runtime_device)
    inducing_side = max(2, args.inducing_side)
    z1, z2 = torch.meshgrid(
        torch.linspace(-1.0, 1.0, inducing_side, dtype=times.dtype, device=args.runtime_device),
        torch.linspace(-1.0, 1.0, inducing_side, dtype=times.dtype, device=args.runtime_device),
        indexing="ij",
    )
    z_s = torch.stack([z1.reshape(-1), z2.reshape(-1)], dim=-1)

    args.reference_steps = args.num_times
    model = build_stage1_model(args, z_s=z_s, input_dim=2)

    set_hyperparameter_trainability(model, trainable=not args.freeze_hyperparameters)
    optimizer = build_optimizer(model, learning_rate=args.learning_rate)
    fit_summary = fit_batch_model(
        model,
        optimizer,
        train_times=times,
        train_spatial=spatial,
        train_observations=observations,
        train_covariates=None,
        train_steps=args.train_steps,
        log_every=args.log_every,
        eval_every=args.eval_every,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        block_size=args.block_size,
        block_overlap=args.block_overlap,
        selection_metric=args.selection_metric,
    )
    train_runtime = fit_summary["training_runtime_s"]
    if args.block_size is not None and args.block_size < times.shape[0]:
        blockwise = model.forward_blockwise(
            times=times,
            x_s=spatial,
            y=observations,
            block_size=args.block_size,
            overlap=args.block_overlap,
            num_discrete_steps=args.block_size,
            cache_last_posterior=True,
            covariates=None,
        )
        train_output: dict[str, Any] = blockwise.block_outputs[-1]
        final_runtime = blockwise.total_runtime_s
        final_block_time = blockwise.mean_block_runtime_s
    else:
        train_output = model(
            times,
            spatial,
            observations,
            covariates=None,
            cache_posterior=True,
            materialize_posterior_cov=False,
        )
        final_runtime = train_output["runtime_s"]
        final_block_time = None

    pred_output = model.predict(times, spatial, covariates=None)
    final_metrics = summarize_final_metrics(
        observations,
        pred_output,
        runtime_s=train_runtime + final_runtime,
        mean_block_time_s=final_block_time,
    )
    eval_line = (
        "[eval] rmse={rmse:.4f} pred_nll={pred_nll:.4f} runtime={runtime_s:.3f}s "
        "gpu_mem={gpu_memory_mb:.1f}MB"
    ).format(**final_metrics)
    if "time_per_block_s" in final_metrics:
        eval_line += f" time_per_block={final_metrics['time_per_block_s']:.3f}s"
    print(eval_line)
    print(
        "[posterior] inducing_dim={} sigma={:.4f} {}".format(
            train_output["posterior_mean_u"].numel(),
            torch.exp(model.log_noise_std).item(),
            format_model_hyperparameters(model),
        )
    )


def run_era5_probe(args: argparse.Namespace) -> None:
    task_dirs = resolve_era5_task_dirs(args)
    if task_dirs:
        covariate_indices = resolve_era5_covariate_indices(args)
        max_locations = resolve_era5_max_locations(args, task_dirs)
        if len(task_dirs) > 1:
            task = load_processed_era5_tasks(
                task_dirs=task_dirs,
                variable_index=args.era5_variable_index,
                covariate_indices=covariate_indices,
                max_locations=max_locations,
                scaled=not args.era5_unscaled,
                location_stride=args.era5_location_stride,
                resplit=args.era5_resplit,
                train_fraction=args.era5_train_fraction,
                val_fraction=args.era5_val_fraction,
            )
            task_name = "+".join(Path(path).name for path in task_dirs)
        else:
            task = load_processed_era5_task(
                task_dir=task_dirs[0] if task_dirs else args.era5_task_dir,
                variable_index=args.era5_variable_index,
                covariate_indices=covariate_indices,
                max_locations=max_locations,
                scaled=not args.era5_unscaled,
                location_stride=args.era5_location_stride,
                resplit=args.era5_resplit,
                train_fraction=args.era5_train_fraction,
                val_fraction=args.era5_val_fraction,
            )
            task_name = Path(task.task_dir).name
        validate_era5_spatial_coords(task.train.spatial_coords)
        task.train.times = move_tensor_to_device(task.train.times, args.runtime_device)
        task.train.spatial_coords = move_tensor_to_device(task.train.spatial_coords, args.runtime_device)
        task.train.observations = move_tensor_to_device(task.train.observations, args.runtime_device)
        if task.train.covariates is not None:
            task.train.covariates = move_tensor_to_device(task.train.covariates, args.runtime_device)
        task.val.times = move_tensor_to_device(task.val.times, args.runtime_device)
        task.val.spatial_coords = move_tensor_to_device(task.val.spatial_coords, args.runtime_device)
        task.val.observations = move_tensor_to_device(task.val.observations, args.runtime_device)
        if task.val.covariates is not None:
            task.val.covariates = move_tensor_to_device(task.val.covariates, args.runtime_device)
        task.test.times = move_tensor_to_device(task.test.times, args.runtime_device)
        task.test.spatial_coords = move_tensor_to_device(task.test.spatial_coords, args.runtime_device)
        task.test.observations = move_tensor_to_device(task.test.observations, args.runtime_device)
        if task.test.covariates is not None:
            task.test.covariates = move_tensor_to_device(task.test.covariates, args.runtime_device)
        z_s = select_spatial_inducing_points(
            task.train.spatial_coords,
            inducing_side=args.inducing_side,
            spatial_inducing_count=args.spatial_inducing_count,
            selection_method=args.spatial_inducing_selection,
        )

        args.reference_steps = args.reference_steps or task.train.times.shape[0]
        covariate_dim = 0 if task.train.covariates is None else task.train.covariates.shape[-1]
        model = build_stage1_model(
            args,
            z_s=z_s,
            input_dim=task.train.spatial_coords.shape[1],
            covariate_dim=covariate_dim,
        )
        set_hyperparameter_trainability(model, trainable=not args.freeze_hyperparameters)
        optimizer = build_optimizer(model, learning_rate=args.learning_rate)
        fit_summary = fit_batch_model(
            model,
            optimizer,
            train_times=task.train.times,
            train_spatial=task.train.spatial_coords,
            train_observations=task.train.observations,
            train_covariates=task.train.covariates,
            train_steps=args.train_steps,
            log_every=args.log_every,
            eval_every=args.eval_every,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            block_size=args.block_size,
            block_overlap=args.block_overlap,
            val_times=task.val.times,
            val_spatial=task.val.spatial_coords,
            val_observations=task.val.observations,
            val_covariates=task.val.covariates,
            selection_metric=args.selection_metric,
        )

        if args.block_size is not None and args.block_size < task.train.times.shape[0]:
            train_summary = model.forward_blockwise(
                times=task.train.times,
                x_s=task.train.spatial_coords,
                y=task.train.observations,
                block_size=args.block_size,
                overlap=args.block_overlap,
                num_discrete_steps=args.block_size,
                cache_last_posterior=True,
                covariates=task.train.covariates,
            )
            train_runtime = fit_summary["training_runtime_s"] + train_summary.total_runtime_s
            mean_block_time = train_summary.mean_block_runtime_s
        else:
            train_output = model(
                task.train.times,
                task.train.spatial_coords,
                task.train.observations,
                covariates=task.train.covariates,
                cache_posterior=True,
                materialize_posterior_cov=False,
            )
            train_runtime = fit_summary["training_runtime_s"] + train_output["runtime_s"]
            mean_block_time = None

        train_pred = model.predict(task.train.times, task.train.spatial_coords, covariates=task.train.covariates)
        val_pred = model.predict(task.val.times, task.val.spatial_coords, covariates=task.val.covariates)
        test_pred = model.predict(task.test.times, task.test.spatial_coords, covariates=task.test.covariates)
        full_location_count = infer_processed_era5_location_count(
            task_dirs,
            scaled=not args.era5_unscaled,
            location_stride=args.era5_location_stride,
        )
        print(
            "[era5-batch] task={} variable_index={} variable_name={} train_shape={} val_shape={} test_shape={}".format(
                task_name,
                task.variable_index,
                task.variable_name,
                tuple(task.train.observations.shape),
                tuple(task.val.observations.shape),
                tuple(task.test.observations.shape),
            )
        )
        print(
            "[era5-data] temporal_method=analytic spatial_method=svgp spatial_coord_order=(lon,lat) "
            "spatial_input_dim={} covariate_dim={} selected_locations={} available_locations={} full_task12={}".format(
                task.train.spatial_coords.shape[1],
                covariate_dim,
                task.train.spatial_coords.shape[0],
                full_location_count if full_location_count is not None else task.train.spatial_coords.shape[0],
                bool(getattr(args, "era5_full_task12", False)),
            )
        )
        if covariate_indices:
            covariate_names = [era5_variable_name(index) for index in covariate_indices]
            print(f"[era5-covariates] indices={covariate_indices} names={covariate_names}")
        print(
            "[model] best_step={} selection_metric={} best_value={:.4f} inducing_points={} {}".format(
                fit_summary["best_step"],
                args.selection_metric,
                fit_summary["best_metric"],
                model.z_s.shape[0],
                format_model_hyperparameters(model),
            )
        )
        print(
            "[train-eval] rmse={:.4f} pred_nll={:.4f}".format(
                rmse(task.train.observations, train_pred["mean"]).item(),
                predictive_nll(task.train.observations, train_pred["mean"], train_pred["obs_var"]).item(),
            )
        )
        print(
            "[val-eval] rmse={:.4f} pred_nll={:.4f}".format(
                rmse(task.val.observations, val_pred["mean"]).item(),
                predictive_nll(task.val.observations, val_pred["mean"], val_pred["obs_var"]).item(),
            )
        )
        test_line = "[test-eval] rmse={:.4f} pred_nll={:.4f} runtime={:.3f}s gpu_mem={:.1f}MB".format(
            rmse(task.test.observations, test_pred["mean"]).item(),
            predictive_nll(task.test.observations, test_pred["mean"], test_pred["obs_var"]).item(),
            train_runtime,
            current_gpu_memory_mb(),
        )
        if mean_block_time is not None:
            test_line += f" time_per_block={mean_block_time:.3f}s"
        print(test_line)
        split_name = args.map_split
        split_tensor = getattr(task, split_name)
        split_pred = {"train": train_pred, "val": val_pred, "test": test_pred}[split_name]
        maybe_save_era5_maps(
            args=args,
            mode="batch",
            task_name=task_name,
            variable_name=task.variable_name,
            split_name=split_name,
            split_tensor=split_tensor,
            pred_output=split_pred,
        )
        return

    batch = load_era5_grid(
        path=args.era5_path,
        variable=args.era5_variable,
        time_range=(args.era5_start, args.era5_end) if args.era5_start and args.era5_end else None,
        lat_range=(args.lat_min, args.lat_max) if args.lat_min is not None and args.lat_max is not None else None,
        lon_range=(args.lon_min, args.lon_max) if args.lon_min is not None and args.lon_max is not None else None,
        spatial_stride=args.spatial_stride,
        time_stride=args.time_stride,
    )
    batch.times = move_tensor_to_device(batch.times, args.runtime_device)
    batch.spatial_coords = move_tensor_to_device(batch.spatial_coords, args.runtime_device)
    batch.observations = move_tensor_to_device(batch.observations, args.runtime_device)
    print(
        f"[era5] variable={batch.variable} times={batch.times.shape[0]} "
        f"space={batch.spatial_coords.shape[0]} observations={tuple(batch.observations.shape)}"
    )
    if args.block_size is not None and args.block_size < batch.times.shape[0]:
        blocks = build_temporal_blocks(
            batch.times,
            block_size=args.block_size,
            overlap=args.block_overlap,
            num_discrete_steps=args.block_size,
        )
        print(
            f"[era5] temporal_blocks={len(blocks)} "
            f"block_size={args.block_size} overlap={args.block_overlap}"
        )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Stage 1 Kronecker spatio-temporal HiPPO-SVGP")
    parser.add_argument("--dataset", choices=["synthetic", "era5"], default="synthetic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-times", type=int, default=24)
    parser.add_argument("--spatial-grid-size", type=int, default=5)
    parser.add_argument("--synthetic-noise", type=float, default=0.05)
    parser.add_argument("--inducing-side", type=int, default=3)
    parser.add_argument("--spatial-inducing-count", type=int, default=None)
    parser.add_argument(
        "--spatial-inducing-selection",
        choices=["fps", "grid", "first"],
        default="grid",
    )
    parser.add_argument("--temporal-inducing", type=int, default=6)
    parser.add_argument("--rff-sample-size", type=int, default=256)
    parser.add_argument("--temporal-lengthscale", type=float, default=1.0)
    parser.add_argument("--spatial-kernel", choices=["rbf", "matern"], default="rbf")
    parser.add_argument("--spatial-variance", type=float, default=1.0)
    parser.add_argument("--spatial-lengthscale", type=float, default=0.6)
    parser.add_argument("--likelihood-noise", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--train-steps", type=int, default=150)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--selection-metric", choices=["pred_nll", "rmse"], default="pred_nll")
    parser.add_argument("--learn-spatial-inducing", action="store_true")
    parser.add_argument("--freeze-hyperparameters", action="store_true")
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--block-overlap", type=int, default=0)
    parser.add_argument("--reference-steps", type=int, default=None)

    parser.add_argument("--era5-path", type=str, default="")
    parser.add_argument("--era5-task-dir", type=str, default="")
    parser.add_argument("--era5-task-dirs", nargs="*", default=[])
    parser.add_argument("--era5-all-tasks", action="store_true")
    parser.add_argument("--era5-full-task12", action="store_true")
    parser.add_argument("--era5-task-root", type=str, default="data/era5/processed_timeseries_4")
    parser.add_argument("--era5-variable", type=str, default="t2m")
    parser.add_argument("--era5-variable-index", type=int, default=0)
    parser.add_argument(
        "--era5-covariate-indices",
        nargs="*",
        type=int,
        default=None,
        help="Enable ERA5 covariates. Omit the flag to disable covariates; pass the flag with no indices to use all non-target ERA5 variables.",
    )
    parser.add_argument("--era5-max-locations", type=int, default=32)
    parser.add_argument("--era5-use-all-locations", action="store_true")
    parser.add_argument("--era5-location-stride", type=int, default=1)
    parser.add_argument("--era5-unscaled", action="store_true")
    parser.add_argument("--era5-resplit", action="store_true")
    parser.add_argument("--era5-train-fraction", type=float, default=0.7)
    parser.add_argument("--era5-val-fraction", type=float, default=0.15)
    parser.add_argument("--era5-start", type=str, default="")
    parser.add_argument("--era5-end", type=str, default="")
    parser.add_argument("--lat-min", type=float, default=None)
    parser.add_argument("--lat-max", type=float, default=None)
    parser.add_argument("--lon-min", type=float, default=None)
    parser.add_argument("--lon-max", type=float, default=None)
    parser.add_argument("--spatial-stride", type=int, default=2)
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument("--save-era5-maps", action="store_true")
    parser.add_argument("--map-split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--map-time-indices", type=str, default="")
    parser.add_argument("--map-max-snapshots", type=int, default=3)
    parser.add_argument("--map-output-dir", type=str, default="")
    parser.add_argument("--map-point-size", type=float, default=18.0)
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    args.runtime_device = resolve_device(args.device)
    print(f"[device] using {args.runtime_device}")
    if args.dataset == "synthetic":
        run_synthetic_experiment(args)
        return
    run_era5_probe(args)


if __name__ == "__main__":
    main()

"""Synthetic and ERA5-oriented Stage 2 online training entrypoint."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

try:
    from .era5_dataset import (
        build_temporal_blocks,
        load_era5_grid,
        load_processed_era5_task,
        load_processed_era5_tasks,
    )
    from .spatial_kernel import SpatialKernelConfig
    from .st_model_batch import BatchKroneckerSTHiPPOSVGP
    from .st_model_online import OnlinePosteriorSummarySTGP
    from .temporal_analytic import TemporalAnalyticConfig, TemporalBlockSpec
    from .train_batch import (
        build_stage1_model,
        build_optimizer,
        clone_state_dict,
        current_gpu_memory_mb,
        fit_batch_model,
        format_model_hyperparameters,
        make_synthetic_dataset,
        move_tensor_to_device,
        predictive_nll,
        rmse,
        resolve_era5_max_locations,
        resolve_device,
        resolve_era5_task_dirs,
        select_spatial_inducing_points,
        set_hyperparameter_trainability,
        validate_era5_spatial_coords,
        infer_processed_era5_location_count,
    )
except ImportError:  # pragma: no cover - allows direct script execution
    from stvgp_kronecker.era5_dataset import (
        build_temporal_blocks,
        load_era5_grid,
        load_processed_era5_task,
        load_processed_era5_tasks,
    )
    from stvgp_kronecker.spatial_kernel import SpatialKernelConfig
    from stvgp_kronecker.st_model_batch import BatchKroneckerSTHiPPOSVGP
    from stvgp_kronecker.st_model_online import OnlinePosteriorSummarySTGP
    from stvgp_kronecker.temporal_analytic import TemporalAnalyticConfig, TemporalBlockSpec
    from stvgp_kronecker.train_batch import (
        build_stage1_model,
        build_optimizer,
        clone_state_dict,
        current_gpu_memory_mb,
        fit_batch_model,
        format_model_hyperparameters,
        make_synthetic_dataset,
        move_tensor_to_device,
        predictive_nll,
        rmse,
        resolve_era5_max_locations,
        resolve_device,
        resolve_era5_task_dirs,
        select_spatial_inducing_points,
        set_hyperparameter_trainability,
        validate_era5_spatial_coords,
        infer_processed_era5_location_count,
    )


def _make_spatial_inducing(side: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    z1, z2 = torch.meshgrid(
        torch.linspace(-1.0, 1.0, side, dtype=dtype, device=device),
        torch.linspace(-1.0, 1.0, side, dtype=dtype, device=device),
        indexing="ij",
    )
    return torch.stack([z1.reshape(-1), z2.reshape(-1)], dim=-1)


def _build_online_model(args: argparse.Namespace, z_s: torch.Tensor, input_dim: int) -> OnlinePosteriorSummarySTGP:
    model = OnlinePosteriorSummarySTGP.from_configs(
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
        enforce_shared_horizon=not args.allow_local_horizon,
    )
    return model.to(args.runtime_device)


def _maybe_pretrain_stage1(
    args: argparse.Namespace,
    train_times: torch.Tensor,
    train_spatial: torch.Tensor,
    train_observations: torch.Tensor,
    val_times: torch.Tensor | None,
    val_spatial: torch.Tensor | None,
    val_observations: torch.Tensor | None,
    z_s: torch.Tensor,
) -> tuple[BatchKroneckerSTHiPPOSVGP | None, dict[str, Any] | None]:
    if args.pretrain_steps <= 0:
        return None, None

    batch_model = build_stage1_model(args, z_s=z_s, input_dim=train_spatial.shape[1])
    set_hyperparameter_trainability(batch_model, trainable=not args.freeze_hyperparameters)
    optimizer = build_optimizer(batch_model, learning_rate=args.pretrain_learning_rate)
    fit_summary = fit_batch_model(
        batch_model,
        optimizer,
        train_times=train_times,
        train_spatial=train_spatial,
        train_observations=train_observations,
        train_steps=args.pretrain_steps,
        log_every=args.pretrain_log_every,
        block_size=args.pretrain_block_size,
        block_overlap=args.block_overlap,
        val_times=val_times,
        val_spatial=val_spatial,
        val_observations=val_observations,
        selection_metric=args.selection_metric,
    )
    print(
        "[pretrain] best_step={} selection_metric={} best_value={:.4f} inducing_points={} {}".format(
            fit_summary["best_step"],
            args.selection_metric,
            fit_summary["best_metric"],
            batch_model.z_s.shape[0],
            format_model_hyperparameters(batch_model),
        )
    )
    return batch_model, fit_summary


def run_synthetic_online(args: argparse.Namespace) -> None:
    times, spatial, observations = make_synthetic_dataset(
        num_times=args.num_times,
        spatial_grid_size=args.spatial_grid_size,
        noise_std=args.synthetic_noise,
    )
    times = move_tensor_to_device(times, args.runtime_device)
    spatial = move_tensor_to_device(spatial, args.runtime_device)
    observations = move_tensor_to_device(observations, args.runtime_device)
    z_s = _make_spatial_inducing(max(2, args.inducing_side), dtype=times.dtype, device=args.runtime_device)
    reference_horizon = TemporalBlockSpec.from_times(
        times,
        num_discrete_steps=args.reference_steps or times.shape[0],
        prev_discrete_steps=0,
    )

    args.reference_steps = args.reference_steps or times.shape[0]
    pretrained_batch, _ = _maybe_pretrain_stage1(
        args,
        train_times=times,
        train_spatial=spatial,
        train_observations=observations,
        val_times=None,
        val_spatial=None,
        val_observations=None,
        z_s=z_s,
    )
    if pretrained_batch is not None:
        z_s = pretrained_batch.z_s.detach().clone()

    model = _build_online_model(args, z_s=z_s, input_dim=2)
    if pretrained_batch is not None:
        model.load_pretrained_batch_state(clone_state_dict(pretrained_batch))
        model.freeze_model_hyperparameters()
    model.initialize(reference_horizon=reference_horizon, x_s=spatial)

    block_specs = build_temporal_blocks(
        times,
        block_size=args.block_size,
        overlap=args.block_overlap,
        num_discrete_steps=args.block_size,
    )
    total_start = time.perf_counter()
    block_times: list[float] = []
    for block_slice, block_horizon in block_specs:
        output = model.update_block(
            times=times[block_slice],
            x_s=spatial,
            y=observations[block_slice],
            horizon=None if not args.allow_local_horizon else block_horizon,
        )
        block_times.append(output["runtime_s"])
        print(
            f"[online] block={output['block_index']:02d} rmse={output['rmse'].item():.4f} "
            f"pred_nll={output['pred_nll'].item():.4f} block_time={output['runtime_s']:.3f}s"
        )

    total_runtime = time.perf_counter() - total_start
    pred_output = model.predict(times, spatial)
    mean_block_time = sum(block_times) / len(block_times)
    print(
        "[eval] rmse={:.4f} pred_nll={:.4f} runtime={:.3f}s gpu_mem={:.1f}MB time_per_block={:.3f}s".format(
            rmse(observations, pred_output["mean"]).item(),
            predictive_nll(observations, pred_output["mean"], pred_output["obs_var"]).item(),
            total_runtime,
            current_gpu_memory_mb(),
            mean_block_time,
        )
    )

    if args.compare_batch:
        batch_model = pretrained_batch or build_stage1_model(args, z_s=z_s, input_dim=2)
        batch_output = batch_model(
            times,
            spatial,
            observations,
            horizon=reference_horizon,
            cache_posterior=True,
            materialize_posterior_cov=True,
        )
        mean_diff = torch.norm(batch_output["posterior_mean_u"] - model.state.m) / (
            torch.norm(batch_output["posterior_mean_u"]) + 1e-12
        )
        online_cov = model.materialize_posterior_covariance()
        cov_diff = torch.norm(batch_output["posterior_cov_u"] - online_cov) / (
            torch.norm(batch_output["posterior_cov_u"]) + 1e-12
        )
        print(f"[compare-batch] rel_mean_diff={mean_diff.item():.6e} rel_cov_diff={cov_diff.item():.6e}")


def run_era5_online_probe(args: argparse.Namespace) -> None:
    task_dirs = resolve_era5_task_dirs(args)
    if task_dirs:
        max_locations = resolve_era5_max_locations(args, task_dirs)
        if len(task_dirs) > 1:
            task = load_processed_era5_tasks(
                task_dirs=task_dirs,
                variable_index=args.era5_variable_index,
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
        task.val.times = move_tensor_to_device(task.val.times, args.runtime_device)
        task.val.spatial_coords = move_tensor_to_device(task.val.spatial_coords, args.runtime_device)
        task.val.observations = move_tensor_to_device(task.val.observations, args.runtime_device)
        task.test.times = move_tensor_to_device(task.test.times, args.runtime_device)
        task.test.spatial_coords = move_tensor_to_device(task.test.spatial_coords, args.runtime_device)
        task.test.observations = move_tensor_to_device(task.test.observations, args.runtime_device)
        spatial = task.train.spatial_coords
        reference_horizon = TemporalBlockSpec.from_times(
            task.train.times,
            num_discrete_steps=args.reference_steps or task.train.times.shape[0],
            prev_discrete_steps=0,
        )
        z_s = select_spatial_inducing_points(
            spatial,
            inducing_side=args.inducing_side,
            spatial_inducing_count=args.spatial_inducing_count,
        )

        args.reference_steps = args.reference_steps or task.train.times.shape[0]
        pretrained_batch, pretrain_summary = _maybe_pretrain_stage1(
            args,
            train_times=task.train.times,
            train_spatial=spatial,
            train_observations=task.train.observations,
            val_times=task.val.times,
            val_spatial=task.val.spatial_coords,
            val_observations=task.val.observations,
            z_s=z_s,
        )
        if pretrained_batch is not None:
            z_s = pretrained_batch.z_s.detach().clone()

        model = _build_online_model(args, z_s=z_s, input_dim=spatial.shape[1])
        if pretrained_batch is not None:
            model.load_pretrained_batch_state(clone_state_dict(pretrained_batch))
            model.freeze_model_hyperparameters()
        model.initialize(reference_horizon=reference_horizon, x_s=spatial)

        train_blocks = build_temporal_blocks(
            task.train.times,
            block_size=args.block_size,
            overlap=args.block_overlap,
            num_discrete_steps=args.block_size,
        )
        train_start = time.perf_counter()
        block_times: list[float] = []
        for block_slice, block_horizon in train_blocks:
            output = model.update_block(
                times=task.train.times[block_slice],
                x_s=spatial,
                y=task.train.observations[block_slice],
                horizon=None if not args.allow_local_horizon else block_horizon,
            )
            block_times.append(output["runtime_s"])
            if output["block_index"] % max(args.log_every_blocks, 1) == 0:
                print(
                    f"[era5-train] block={output['block_index']:03d} rmse={output['rmse'].item():.4f} "
                    f"pred_nll={output['pred_nll'].item():.4f} block_time={output['runtime_s']:.3f}s"
                )
        train_runtime = time.perf_counter() - train_start
        if pretrain_summary is not None:
            train_runtime += pretrain_summary["training_runtime_s"]

        train_pred = model.predict(task.train.times, task.train.spatial_coords)
        val_pred = model.predict(task.val.times, task.val.spatial_coords)
        test_pred = model.predict(task.test.times, task.test.spatial_coords)
        mean_block_time = sum(block_times) / len(block_times)
        full_location_count = infer_processed_era5_location_count(
            task_dirs,
            scaled=not args.era5_unscaled,
            location_stride=args.era5_location_stride,
        )

        print(
            "[era5-online] task={} variable_index={} variable_name={} train_shape={} val_shape={} test_shape={}".format(
                task_name,
                args.era5_variable_index,
                task.variable_name,
                tuple(task.train.observations.shape),
                tuple(task.val.observations.shape),
                tuple(task.test.observations.shape),
            )
        )
        print(
            "[era5-data] temporal_method=analytic spatial_method=svgp spatial_coord_order=(lon,lat) "
            "spatial_input_dim={} selected_locations={} available_locations={} full_task12={}".format(
                task.train.spatial_coords.shape[1],
                task.train.spatial_coords.shape[0],
                full_location_count if full_location_count is not None else task.train.spatial_coords.shape[0],
                bool(getattr(args, "era5_full_task12", False)),
            )
        )
        print(
            "[model] inducing_points={} {}".format(
                model.z_s.shape[0],
                format_model_hyperparameters(pretrained_batch or model),
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
        print(
            "[test-eval] rmse={:.4f} pred_nll={:.4f} runtime={:.3f}s gpu_mem={:.1f}MB time_per_block={:.3f}s".format(
                rmse(task.test.observations, test_pred["mean"]).item(),
                predictive_nll(task.test.observations, test_pred["mean"], test_pred["obs_var"]).item(),
                train_runtime,
                current_gpu_memory_mb(),
                mean_block_time,
            )
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
    blocks = build_temporal_blocks(
        batch.times,
        block_size=args.block_size,
        overlap=args.block_overlap,
        num_discrete_steps=args.block_size,
    )
    print(
        f"[era5-online] variable={batch.variable} times={batch.times.shape[0]} "
        f"space={batch.spatial_coords.shape[0]} blocks={len(blocks)}"
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Stage 2 online Kronecker spatio-temporal HiPPO-SVGP")
    parser.add_argument("--dataset", choices=["synthetic", "era5"], default="synthetic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-times", type=int, default=24)
    parser.add_argument("--spatial-grid-size", type=int, default=5)
    parser.add_argument("--synthetic-noise", type=float, default=0.05)
    parser.add_argument("--inducing-side", type=int, default=3)
    parser.add_argument("--spatial-inducing-count", type=int, default=None)
    parser.add_argument("--temporal-inducing", type=int, default=6)
    parser.add_argument("--rff-sample-size", type=int, default=256)
    parser.add_argument("--temporal-lengthscale", type=float, default=1.0)
    parser.add_argument("--spatial-kernel", choices=["rbf", "matern"], default="rbf")
    parser.add_argument("--spatial-variance", type=float, default=1.0)
    parser.add_argument("--spatial-lengthscale", type=float, default=0.6)
    parser.add_argument("--likelihood-noise", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--block-size", type=int, default=6)
    parser.add_argument("--block-overlap", type=int, default=0)
    parser.add_argument("--reference-steps", type=int, default=None)
    parser.add_argument("--allow-local-horizon", action="store_true")
    parser.add_argument("--compare-batch", action="store_true")
    parser.add_argument("--selection-metric", choices=["pred_nll", "rmse"], default="pred_nll")
    parser.add_argument("--learn-spatial-inducing", action="store_true")
    parser.add_argument("--freeze-hyperparameters", action="store_true")
    parser.add_argument("--pretrain-steps", type=int, default=0)
    parser.add_argument("--pretrain-learning-rate", type=float, default=0.03)
    parser.add_argument("--pretrain-log-every", type=int, default=10)
    parser.add_argument("--pretrain-block-size", type=int, default=None)

    parser.add_argument("--era5-path", type=str, default="")
    parser.add_argument("--era5-task-dir", type=str, default="")
    parser.add_argument("--era5-task-dirs", nargs="*", default=[])
    parser.add_argument("--era5-all-tasks", action="store_true")
    parser.add_argument("--era5-full-task12", action="store_true")
    parser.add_argument("--era5-task-root", type=str, default="data/era5/processed_timeseries_4")
    parser.add_argument("--era5-variable", type=str, default="t2m")
    parser.add_argument("--era5-variable-index", type=int, default=0)
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
    parser.add_argument("--log-every-blocks", type=int, default=5)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    args.runtime_device = resolve_device(args.device)
    print(f"[device] using {args.runtime_device}")
    if args.dataset == "synthetic":
        run_synthetic_online(args)
        return
    run_era5_online_probe(args)


if __name__ == "__main__":
    main()

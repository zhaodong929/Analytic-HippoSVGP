from __future__ import annotations

import numpy as np
import pytest
import torch

from stvgp_kronecker.kron_ops import (
    kron_rowwise_prior_diag,
    rowwise_quadratic_form_from_precision_cholesky,
)
from stvgp_kronecker.era5_dataset import (
    count_processed_era5_locations,
    discover_processed_era5_task_dirs,
    load_processed_era5_task,
    load_processed_era5_tasks,
)
from stvgp_kronecker.spatial_kernel import SpatialKernelConfig
from stvgp_kronecker.st_model_batch import BatchKroneckerSTHiPPOSVGP
from stvgp_kronecker.st_model_online import OnlinePosteriorSummarySTGP
from stvgp_kronecker.temporal_analytic import TemporalAnalyticConfig
from stvgp_kronecker.temporal_analytic import TemporalBlockSpec
from stvgp_kronecker.train_batch import select_spatial_inducing_points


def make_model() -> BatchKroneckerSTHiPPOSVGP:
    z_s = torch.tensor(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    return BatchKroneckerSTHiPPOSVGP(
        temporal_config=TemporalAnalyticConfig(
            inducing_size=4,
            rff_sample_size=64,
            variance=1.0,
            lengthscale=1.1,
            num_discrete_steps=5,
            seed=123,
        ),
        spatial_kernel_config=SpatialKernelConfig(
            input_dim=2,
            kernel_type="rbf",
            variance=1.0,
            lengthscale=0.9,
        ),
        z_s=z_s,
        noise_std=0.1,
    )


def make_toy_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    times = torch.linspace(0.0, 2.0, 5, dtype=torch.float64)
    spatial = torch.tensor(
        [
            [-1.0, -1.0],
            [-0.5, 0.5],
            [0.3, -0.3],
            [1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    temporal_part = torch.sin(times) + 0.3 * torch.cos(0.5 * times)
    spatial_part = torch.exp(-0.8 * torch.sum((spatial - 0.2) ** 2, dim=-1))
    y = temporal_part[:, None] * spatial_part[None, :]
    return times, spatial, y


def test_temporal_and_spatial_shape_consistency() -> None:
    model = make_model()
    times, spatial, _ = make_toy_data()

    temporal = model.build_temporal_covariances(times)
    spatial_cov = model.build_spatial_covariances(spatial)

    assert temporal.kuu_t.shape == (4, 4)
    assert temporal.kfu_t.shape == (5, 4)
    assert spatial_cov.kzz_s.shape == (4, 4)
    assert spatial_cov.kxz_s.shape == (4, 4)


def test_kronecker_projection_shapes() -> None:
    model = make_model()
    times, spatial, _ = make_toy_data()
    projection = model.build_projection(times, spatial)

    assert projection.a_t.shape == (5, 4)
    assert projection.a_s.shape == (4, 4)
    assert torch.kron(projection.a_t, projection.a_s).shape == (20, 16)


def test_small_synthetic_batch_matches_dense_solution() -> None:
    model = make_model()
    times, spatial, y = make_toy_data()
    output = model(times, spatial, y)

    dense = model.materialize_full_kronecker_matrices(times, spatial)
    kuu = dense["Kuu"]
    kfu = dense["Kfu"]
    chol_kuu = torch.linalg.cholesky(
        kuu + 1e-6 * torch.mean(torch.diag(kuu)) * torch.eye(kuu.shape[0], dtype=kuu.dtype)
    )
    a_dense = torch.cholesky_solve(kfu.transpose(-1, -2), chol_kuu).transpose(-1, -2)
    sigma2 = model.sigma2
    precision = torch.cholesky_solve(
        torch.eye(kuu.shape[0], dtype=kuu.dtype),
        chol_kuu,
    ) + torch.reciprocal(sigma2) * (a_dense.transpose(-1, -2) @ a_dense)
    info = torch.reciprocal(sigma2) * (a_dense.transpose(-1, -2) @ y.reshape(-1))
    mean = torch.linalg.solve(precision, info)

    assert torch.allclose(output["posterior_mean_u"], mean, atol=1e-5, rtol=1e-5)


def test_small_synthetic_training_reduces_loss() -> None:
    model = make_model()
    times, spatial, y = make_toy_data()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        output = model(times, spatial, y, cache_posterior=False)
        output["loss"].backward()
        optimizer.step()
        losses.append(float(output["loss"].detach().cpu()))

    assert losses[-1] < losses[0]


def test_blockwise_forward_returns_consistent_shapes() -> None:
    model = make_model()
    times, spatial, y = make_toy_data()

    blockwise = model.forward_blockwise(
        times=times,
        x_s=spatial,
        y=y,
        block_size=2,
        overlap=0,
        num_discrete_steps=2,
        cache_last_posterior=True,
    )

    assert len(blockwise.block_outputs) == 3
    assert blockwise.mean_loss.ndim == 0
    assert blockwise.mean_rmse.ndim == 0
    assert blockwise.total_runtime_s >= 0.0
    assert model.posterior_state is not None
    assert blockwise.block_outputs[-1]["train_mean"].shape[1] == spatial.shape[0]


def test_online_recursion_matches_batch_solution() -> None:
    times, spatial, y = make_toy_data()
    z_s = torch.tensor(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    temporal_config = TemporalAnalyticConfig(
        inducing_size=4,
        rff_sample_size=64,
        variance=1.0,
        lengthscale=1.1,
        num_discrete_steps=times.shape[0],
        seed=123,
    )
    spatial_config = SpatialKernelConfig(
        input_dim=2,
        kernel_type="rbf",
        variance=1.0,
        lengthscale=0.9,
    )
    horizon = TemporalBlockSpec.from_times(times, num_discrete_steps=times.shape[0], prev_discrete_steps=0)

    batch_model = BatchKroneckerSTHiPPOSVGP(
        temporal_config=temporal_config,
        spatial_kernel_config=spatial_config,
        z_s=z_s,
        noise_std=0.1,
    )
    batch_output = batch_model(times, spatial, y, horizon=horizon, cache_posterior=True, materialize_posterior_cov=True)

    online_model = OnlinePosteriorSummarySTGP.from_configs(
        temporal_config=temporal_config,
        spatial_kernel_config=spatial_config,
        z_s=z_s,
        noise_std=0.1,
        enforce_shared_horizon=True,
    )
    online_model.initialize(reference_horizon=horizon, x_s=spatial)
    for start in range(0, times.shape[0], 2):
        stop = min(start + 2, times.shape[0])
        online_model.update_block(times[start:stop], spatial, y[start:stop])

    assert torch.allclose(batch_output["posterior_mean_u"], online_model.state.m, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        batch_output["posterior_cov_u"],
        online_model.materialize_posterior_covariance(),
        atol=1e-5,
        rtol=1e-5,
    )


def test_temporal_cross_covariance_is_consistent() -> None:
    temporal_config = TemporalAnalyticConfig(
        inducing_size=4,
        rff_sample_size=64,
        variance=1.0,
        lengthscale=1.1,
        num_discrete_steps=5,
        seed=123,
    )
    builder = OnlinePosteriorSummarySTGP.from_configs(
        temporal_config=temporal_config,
        spatial_kernel_config=SpatialKernelConfig(input_dim=2, kernel_type="rbf", variance=1.0, lengthscale=0.9),
        z_s=torch.tensor([[-1.0, -1.0], [1.0, 1.0]], dtype=torch.float64),
        noise_std=0.1,
    ).temporal_builder
    horizon_a = TemporalBlockSpec(start=0.0, end=2.0, num_discrete_steps=5, prev_discrete_steps=0, phase_origin=0.0)
    horizon_b = TemporalBlockSpec(start=2.0, end=4.0, num_discrete_steps=5, prev_discrete_steps=5, phase_origin=2.0)

    cross_ab = builder.compute_kuu_t_cross(horizon_a, horizon_b)
    cross_ba = builder.compute_kuu_t_cross(horizon_b, horizon_a)
    assert cross_ab.shape == (4, 4)
    assert torch.allclose(cross_ab, cross_ba.transpose(-1, -2), atol=1e-8, rtol=1e-8)


def test_online_local_horizon_transfer_updates_state() -> None:
    times, spatial, y = make_toy_data()
    z_s = torch.tensor(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    temporal_config = TemporalAnalyticConfig(
        inducing_size=4,
        rff_sample_size=64,
        variance=1.0,
        lengthscale=1.1,
        num_discrete_steps=times.shape[0],
        seed=123,
    )
    spatial_config = SpatialKernelConfig(
        input_dim=2,
        kernel_type="rbf",
        variance=1.0,
        lengthscale=0.9,
    )
    reference_horizon = TemporalBlockSpec.from_times(times, num_discrete_steps=times.shape[0], prev_discrete_steps=0)

    online_model = OnlinePosteriorSummarySTGP.from_configs(
        temporal_config=temporal_config,
        spatial_kernel_config=spatial_config,
        z_s=z_s,
        noise_std=0.1,
        enforce_shared_horizon=False,
    )
    online_model.initialize(reference_horizon=reference_horizon, x_s=spatial)

    first_block_horizon = TemporalBlockSpec.from_times(times[:2], num_discrete_steps=2, prev_discrete_steps=0)
    second_block_horizon = TemporalBlockSpec.from_times(times[2:], num_discrete_steps=3, prev_discrete_steps=2)
    first = online_model.update_block(times[:2], spatial, y[:2], horizon=first_block_horizon)
    second = online_model.update_block(times[2:], spatial, y[2:], horizon=second_block_horizon)

    assert first["temporal_transfer"].shape == (4, 4)
    assert second["temporal_transfer"].shape == (4, 4)
    assert torch.isfinite(first["temporal_transfer"]).all()
    assert torch.isfinite(second["temporal_transfer"]).all()
    assert online_model.state.num_blocks_processed == 2
    pred = online_model.predict(times, spatial)
    assert pred["mean"].shape == y.shape
    assert torch.isfinite(pred["mean"]).all()


def test_online_predictive_variance_matches_dense_precision_solver() -> None:
    times, spatial, y = make_toy_data()
    z_s = torch.tensor(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    temporal_config = TemporalAnalyticConfig(
        inducing_size=4,
        rff_sample_size=64,
        variance=1.0,
        lengthscale=1.1,
        num_discrete_steps=times.shape[0],
        seed=123,
    )
    spatial_config = SpatialKernelConfig(
        input_dim=2,
        kernel_type="rbf",
        variance=1.0,
        lengthscale=0.9,
    )
    horizon = TemporalBlockSpec.from_times(times, num_discrete_steps=times.shape[0], prev_discrete_steps=0)

    model = OnlinePosteriorSummarySTGP.from_configs(
        temporal_config=temporal_config,
        spatial_kernel_config=spatial_config,
        z_s=z_s,
        noise_std=0.1,
        enforce_shared_horizon=True,
    )
    model.initialize(reference_horizon=horizon, x_s=spatial)
    for start in range(0, times.shape[0], 2):
        stop = min(start + 2, times.shape[0])
        model.update_block(times[start:stop], spatial, y[start:stop])

    pred = model.predict(times, spatial)
    assert pred["variance_solver"] == "sylvester"

    state = model.state
    assert state.chol_lambda is not None
    kfu_t_star = model.temporal_builder.compute_kfu_t(times, state.reference_horizon)
    kxz_s_star = model.spatial_kernel.compute_kxz_s(spatial, model.z_s)
    a_t_star = kfu_t_star @ state.inv_kuu_t
    a_s_star = kxz_s_star @ state.inv_kzz_s

    prior_diag = torch.outer(
        model.temporal_builder.compute_ktt_diag(times),
        model.spatial_kernel.diag(spatial),
    )
    projected_prior_diag = kron_rowwise_prior_diag(
        a_t_star,
        state.kuu_t,
        a_s_star,
        state.kzz_s,
    )
    dense_posterior_correction = rowwise_quadratic_form_from_precision_cholesky(
        torch.kron(a_t_star, a_s_star),
        state.chol_lambda,
    ).reshape(pred["latent_var"].shape)
    dense_latent_var = torch.clamp(prior_diag - projected_prior_diag + dense_posterior_correction, min=1e-9)

    assert torch.allclose(pred["latent_var"], dense_latent_var, atol=1e-8, rtol=1e-6)


def test_load_processed_era5_task_aligns_locations(tmp_path) -> None:
    task_dir = tmp_path / "task_1"
    seq_dir = task_dir / "sequences"
    seq_dir.mkdir(parents=True)

    for lat, lon, offset in [(50.0, -1.0, 0.0), (50.1, -1.1, 1.0)]:
        path = seq_dir / f"lat_{lat:.4f}_lon_{lon:.4f}_scaled.npz"
        np.savez(
            path,
            data_train=np.array([[10 + offset, 20 + offset, 30 + offset]], dtype=np.float32),
            time_train=np.array([2, 0, 1], dtype=np.int64),
            data_val=np.array([[40 + offset, 50 + offset]], dtype=np.float32),
            time_val=np.array([4, 3], dtype=np.int64),
            data_test=np.array([[60 + offset]], dtype=np.float32),
            time_test=np.array([5], dtype=np.int64),
        )

    task = load_processed_era5_task(task_dir, variable_index=0, max_locations=2, scaled=True)
    assert task.train.times.tolist() == [0.0, 1.0, 2.0]
    assert task.train.observations.shape == (3, 2)
    assert task.val.observations.shape == (2, 2)
    assert task.test.observations.shape == (1, 2)
    assert task.train.spatial_coords.tolist() == [[-1.0, 50.0], [-1.1, 50.1]]


def test_load_processed_era5_task_resplit_rebuilds_longer_validation(tmp_path) -> None:
    task_dir = tmp_path / "task_1"
    seq_dir = task_dir / "sequences"
    seq_dir.mkdir(parents=True)

    for lat, lon, offset in [(50.0, -1.0, 0.0), (50.1, -1.1, 1.0)]:
        path = seq_dir / f"lat_{lat:.4f}_lon_{lon:.4f}_scaled.npz"
        np.savez(
            path,
            data_train=np.array([[10 + offset, 20 + offset, 30 + offset]], dtype=np.float32),
            time_train=np.array([0, 1, 2], dtype=np.int64),
            data_val=np.array([[40 + offset, 50 + offset]], dtype=np.float32),
            time_val=np.array([3, 4], dtype=np.int64),
            data_test=np.array([[60 + offset]], dtype=np.float32),
            time_test=np.array([5], dtype=np.int64),
        )

    task = load_processed_era5_task(
        task_dir,
        variable_index=0,
        max_locations=2,
        scaled=True,
        resplit=True,
        train_fraction=0.5,
        val_fraction=1.0 / 3.0,
    )
    assert task.train.times.tolist() == [0.0, 1.0, 2.0]
    assert task.val.times.tolist() == [3.0, 4.0]
    assert task.test.times.tolist() == [5.0]
    assert task.val.observations.shape == (2, 2)


def test_load_processed_era5_tasks_concatenates_multiple_tasks(tmp_path) -> None:
    root = tmp_path / "processed_timeseries_4"
    for task_idx, start in [(1, 0.0), (2, 6.0)]:
        seq_dir = root / f"task_{task_idx}" / "sequences"
        seq_dir.mkdir(parents=True)
        for lat, lon, offset in [(50.0, -1.0, 0.0), (50.1, -1.1, 1.0)]:
            path = seq_dir / f"lat_{lat:.4f}_lon_{lon:.4f}_scaled.npz"
            np.savez(
                path,
                data_train=np.array([[10 + offset, 20 + offset, 30 + offset]], dtype=np.float32),
                time_train=np.array([start + 0, start + 1, start + 2], dtype=np.float64),
                data_val=np.array([[40 + offset]], dtype=np.float32),
                time_val=np.array([start + 3], dtype=np.float64),
                data_test=np.array([[50 + offset, 60 + offset]], dtype=np.float32),
                time_test=np.array([start + 4, start + 5], dtype=np.float64),
            )

    task = load_processed_era5_tasks(
        task_dirs=[root / "task_1", root / "task_2"],
        variable_index=0,
        max_locations=2,
        scaled=True,
        resplit=True,
        train_fraction=0.5,
        val_fraction=0.25,
    )
    assert task.train.observations.shape == (6, 2)
    assert task.val.observations.shape == (3, 2)
    assert task.test.observations.shape == (3, 2)
    assert task.train.times.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert task.test.times.tolist() == [9.0, 10.0, 11.0]


def test_discover_and_count_processed_era5_tasks(tmp_path) -> None:
    root = tmp_path / "processed_timeseries_4"
    for task_idx in [1, 2]:
        seq_dir = root / f"task_{task_idx}" / "sequences"
        seq_dir.mkdir(parents=True)
        for lat, lon in [(50.0, -1.0), (50.1, -1.1)]:
            path = seq_dir / f"lat_{lat:.4f}_lon_{lon:.4f}_scaled.npz"
            np.savez(
                path,
                data_train=np.array([[1.0, 2.0]], dtype=np.float32),
                time_train=np.array([0, 1], dtype=np.int64),
                data_val=np.array([[3.0]], dtype=np.float32),
                time_val=np.array([2], dtype=np.int64),
                data_test=np.array([[4.0]], dtype=np.float32),
                time_test=np.array([3], dtype=np.int64),
            )

    task_dirs = discover_processed_era5_task_dirs(root, ["task_1", "task_2"])
    assert [path.name for path in task_dirs] == ["task_1", "task_2"]
    assert count_processed_era5_locations(task_dirs[0], scaled=True, location_stride=1) == 2


def test_spatial_inducing_fps_spreads_across_domain() -> None:
    left_strip = torch.stack(
        [
            torch.full((50,), -10.0, dtype=torch.float64),
            torch.linspace(49.0, 54.0, 50, dtype=torch.float64),
        ],
        dim=-1,
    )
    right_strip = torch.stack(
        [
            torch.full((50,), 2.0, dtype=torch.float64),
            torch.linspace(54.0, 59.0, 50, dtype=torch.float64),
        ],
        dim=-1,
    )
    spatial = torch.cat([left_strip, right_strip], dim=0)

    first = select_spatial_inducing_points(spatial, spatial_inducing_count=8, selection_method="first")
    fps = select_spatial_inducing_points(spatial, spatial_inducing_count=8, selection_method="fps")

    assert float(first[:, 0].max()) < 2.0
    assert float(fps[:, 0].min()) == -10.0
    assert float(fps[:, 0].max()) == 2.0

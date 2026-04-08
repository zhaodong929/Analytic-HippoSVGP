"""Joint online Bayesian updates for mean coefficients and GP inducing variables."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

try:
    from .kron_ops import (
        cholesky_inverse,
        flatten_grid,
        kron_rowwise_prior_diag,
        rowwise_quadratic_form_from_precision_cholesky,
        safe_cholesky,
    )
    from .spatial_kernel import BaseSpatialKernel, SpatialKernelConfig, build_spatial_kernel
    from .temporal_analytic import AnalyticTemporalBuilder, TemporalAnalyticConfig, TemporalBlockSpec
    from .train_batch import (
        clone_state_dict,
        current_gpu_memory_mb,
        format_model_hyperparameters,
        infer_processed_era5_location_count,
        maybe_save_era5_maps,
        move_tensor_to_device,
        predictive_nll,
        resolve_device,
        resolve_era5_covariate_indices,
        resolve_era5_max_locations,
        resolve_era5_task_dirs,
        rmse,
        select_spatial_inducing_points,
        validate_era5_spatial_coords,
    )
    from .train_online import _make_spatial_inducing, _maybe_pretrain_stage1, build_argparser as build_base_argparser
    from .era5_dataset import era5_variable_name, build_temporal_blocks, load_processed_era5_task, load_processed_era5_tasks
except ImportError:  # pragma: no cover
    from stvgp_kronecker.kron_ops import (
        cholesky_inverse,
        flatten_grid,
        kron_rowwise_prior_diag,
        rowwise_quadratic_form_from_precision_cholesky,
        safe_cholesky,
    )
    from stvgp_kronecker.spatial_kernel import BaseSpatialKernel, SpatialKernelConfig, build_spatial_kernel
    from stvgp_kronecker.temporal_analytic import AnalyticTemporalBuilder, TemporalAnalyticConfig, TemporalBlockSpec
    from stvgp_kronecker.train_batch import (
        clone_state_dict,
        current_gpu_memory_mb,
        format_model_hyperparameters,
        infer_processed_era5_location_count,
        maybe_save_era5_maps,
        move_tensor_to_device,
        predictive_nll,
        resolve_device,
        resolve_era5_covariate_indices,
        resolve_era5_max_locations,
        resolve_era5_task_dirs,
        rmse,
        select_spatial_inducing_points,
        validate_era5_spatial_coords,
    )
    from stvgp_kronecker.train_online import _make_spatial_inducing, _maybe_pretrain_stage1, build_argparser as build_base_argparser
    from stvgp_kronecker.era5_dataset import era5_variable_name, build_temporal_blocks, load_processed_era5_task, load_processed_era5_tasks


@dataclass
class JointOnlineState:
    lambda_precision: Optional[torch.Tensor] = None
    chol_lambda: Optional[torch.Tensor] = None
    h: Optional[torch.Tensor] = None
    m: Optional[torch.Tensor] = None
    reference_horizon: Optional[TemporalBlockSpec] = None
    kuu_t: Optional[torch.Tensor] = None
    inv_kuu_t: Optional[torch.Tensor] = None
    kzz_s: Optional[torch.Tensor] = None
    inv_kzz_s: Optional[torch.Tensor] = None
    fixed_x_s: Optional[torch.Tensor] = None
    fixed_kxz_s: Optional[torch.Tensor] = None
    beta_dim: int = 0
    gp_dim: int = 0
    num_blocks_processed: int = 0


class JointOnlineMeanGPSTGP(nn.Module):
    def __init__(
        self,
        temporal_builder: AnalyticTemporalBuilder,
        spatial_kernel: BaseSpatialKernel,
        z_s: torch.Tensor,
        noise_std: float = 0.1,
        covariate_dim: int = 0,
        jitter: float = 1e-6,
        enforce_shared_horizon: bool = True,
        prediction_time_chunk_size: int = 4,
    ) -> None:
        super().__init__()
        self.temporal_builder = temporal_builder
        self.spatial_kernel = spatial_kernel
        self.register_buffer('z_s', torch.as_tensor(z_s, dtype=z_s.dtype))
        self.log_noise_std = nn.Parameter(torch.log(torch.as_tensor(noise_std, dtype=z_s.dtype)))
        self.covariate_dim = int(covariate_dim)
        self.jitter = float(jitter)
        self.enforce_shared_horizon = bool(enforce_shared_horizon)
        self.prediction_time_chunk_size = max(int(prediction_time_chunk_size), 1)
        self.state = JointOnlineState()

    @classmethod
    def from_configs(
        cls,
        temporal_config: TemporalAnalyticConfig,
        spatial_kernel_config: SpatialKernelConfig,
        z_s: torch.Tensor,
        noise_std: float = 0.1,
        covariate_dim: int = 0,
        jitter: float = 1e-6,
        enforce_shared_horizon: bool = True,
        prediction_time_chunk_size: int = 4,
    ) -> 'JointOnlineMeanGPSTGP':
        return cls(
            temporal_builder=AnalyticTemporalBuilder(temporal_config),
            spatial_kernel=build_spatial_kernel(spatial_kernel_config),
            z_s=z_s,
            noise_std=noise_std,
            covariate_dim=covariate_dim,
            jitter=jitter,
            enforce_shared_horizon=enforce_shared_horizon,
            prediction_time_chunk_size=prediction_time_chunk_size,
        )

    @property
    def sigma2(self) -> torch.Tensor:
        return torch.exp(2.0 * self.log_noise_std)

    @property
    def beta_dim(self) -> int:
        return self.covariate_dim + 1 if self.covariate_dim > 0 else 0

    def load_pretrained_batch_hyperparameters(self, batch_state_dict: dict[str, torch.Tensor]) -> None:
        own_state = self.state_dict()
        filtered = {
            name: tensor
            for name, tensor in batch_state_dict.items()
            if name in own_state and not name.startswith('covariate_linear.')
        }
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        if unexpected:
            raise RuntimeError(f'Unexpected keys when loading pretrained batch state: {unexpected}')
        if set(missing) - {'z_s'}:
            raise RuntimeError(f'Missing keys when loading pretrained batch state: {missing}')

    def freeze_model_hyperparameters(self) -> None:
        self.log_noise_std.requires_grad_(False)
        self.temporal_builder.log_variance.requires_grad_(False)
        self.temporal_builder.log_lengthscale.requires_grad_(False)
        self.spatial_kernel.log_variance.requires_grad_(False)
        self.spatial_kernel.log_lengthscale.requires_grad_(False)

    def _as_times(self, times: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(times, dtype=self.z_s.dtype, device=self.z_s.device).reshape(-1)

    def _as_spatial(self, x_s: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(x_s, dtype=self.z_s.dtype, device=self.z_s.device)
        if tensor.ndim != 2:
            raise ValueError('Spatial inputs must have shape [N_s, D_s].')
        return tensor

    def _as_observations(self, y: torch.Tensor, num_times: int, num_space: int) -> torch.Tensor:
        tensor = torch.as_tensor(y, dtype=self.z_s.dtype, device=self.z_s.device)
        if tensor.shape == (num_times, num_space):
            return tensor
        if tensor.numel() == num_times * num_space:
            return tensor.reshape(num_times, num_space)
        raise ValueError('Observations must match the time-space grid shape.')

    def _as_covariates(self, covariates: Optional[torch.Tensor], num_times: int, num_space: int) -> Optional[torch.Tensor]:
        if self.covariate_dim <= 0:
            return None
        if covariates is None:
            raise ValueError('Covariates are required because the model was built with covariate_dim > 0.')
        tensor = torch.as_tensor(covariates, dtype=self.z_s.dtype, device=self.z_s.device)
        if tensor.shape == (num_times, num_space, self.covariate_dim):
            return tensor
        if tensor.numel() == num_times * num_space * self.covariate_dim:
            return tensor.reshape(num_times, num_space, self.covariate_dim)
        raise ValueError('Covariates must have shape [N_t, N_s, C] or flatten to N_t * N_s * C elements.')

    def _covariate_design(self, covariates: Optional[torch.Tensor], num_times: int, num_space: int) -> torch.Tensor:
        if self.beta_dim == 0:
            return torch.zeros((num_times, num_space, 0), dtype=self.z_s.dtype, device=self.z_s.device)
        covariate_tensor = self._as_covariates(covariates, num_times=num_times, num_space=num_space)
        ones = torch.ones((num_times, num_space, 1), dtype=self.z_s.dtype, device=self.z_s.device)
        return torch.cat([covariate_tensor, ones], dim=-1)

    def _beta_mean_grid(self, beta: torch.Tensor, covariates: Optional[torch.Tensor], num_times: int, num_space: int) -> torch.Tensor:
        if self.beta_dim == 0:
            return torch.zeros((num_times, num_space), dtype=self.z_s.dtype, device=self.z_s.device)
        design = self._covariate_design(covariates, num_times=num_times, num_space=num_space)
        return torch.einsum('tsc,c->ts', design, beta)

    def _same_spatial_grid(self, x1: torch.Tensor, x2: torch.Tensor) -> bool:
        return x1.shape == x2.shape and torch.allclose(x1, x2)

    def _resolve_reference_horizon(
        self,
        reference_horizon: torch.Tensor | TemporalBlockSpec | None,
        init_times: Optional[torch.Tensor] = None,
        num_discrete_steps: Optional[int] = None,
    ) -> TemporalBlockSpec:
        if reference_horizon is not None:
            return self.temporal_builder.resolve_horizon(reference_horizon)
        if init_times is None:
            raise ValueError('Either reference_horizon or init_times must be provided.')
        return TemporalBlockSpec.from_times(
            init_times,
            num_discrete_steps=num_discrete_steps or init_times.shape[0],
            prev_discrete_steps=0,
        )

    def _spatial_cross_covariance(self, x_s: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if x_s is None:
            if self.state.fixed_x_s is None or self.state.fixed_kxz_s is None:
                raise ValueError('No fixed spatial grid cached; pass x_s explicitly.')
            return self.state.fixed_x_s, self.state.fixed_kxz_s
        x_s = self._as_spatial(x_s)
        if self.state.fixed_x_s is not None and self._same_spatial_grid(x_s, self.state.fixed_x_s):
            return self.state.fixed_x_s, self.state.fixed_kxz_s
        return x_s, self.spatial_kernel.compute_kxz_s(x_s, self.z_s)

    def initialize(
        self,
        reference_horizon: torch.Tensor | TemporalBlockSpec | None = None,
        x_s: Optional[torch.Tensor] = None,
        beta_prior_precision: float = 1e-4,
        beta_prior_mean: Optional[torch.Tensor] = None,
        init_times: Optional[torch.Tensor] = None,
        num_discrete_steps: Optional[int] = None,
    ) -> JointOnlineState:
        horizon = self._resolve_reference_horizon(
            reference_horizon=reference_horizon,
            init_times=self._as_times(init_times) if init_times is not None else None,
            num_discrete_steps=num_discrete_steps,
        )
        kuu_t = self.temporal_builder.compute_kuu_t(horizon)
        inv_kuu_t = cholesky_inverse(kuu_t, jitter=self.jitter)
        kzz_s = self.spatial_kernel.compute_kzz_s(self.z_s)
        inv_kzz_s = cholesky_inverse(kzz_s, jitter=self.jitter)
        gp_precision = torch.kron(inv_kuu_t, inv_kzz_s)
        gp_dim = gp_precision.shape[0]
        beta_dim = self.beta_dim
        fixed_x_s = None
        fixed_kxz_s = None
        if x_s is not None:
            fixed_x_s = self._as_spatial(x_s)
            fixed_kxz_s = self.spatial_kernel.compute_kxz_s(fixed_x_s, self.z_s)

        if beta_dim > 0:
            prior_precision = torch.eye(beta_dim, dtype=self.z_s.dtype, device=self.z_s.device)
            prior_precision = prior_precision * float(beta_prior_precision)
            if beta_prior_mean is None:
                beta_prior_mean = torch.zeros(beta_dim, dtype=self.z_s.dtype, device=self.z_s.device)
            else:
                beta_prior_mean = torch.as_tensor(beta_prior_mean, dtype=self.z_s.dtype, device=self.z_s.device).reshape(beta_dim)
            lambda_precision = torch.block_diag(prior_precision, gp_precision)
            h = torch.cat([prior_precision @ beta_prior_mean, torch.zeros(gp_dim, dtype=self.z_s.dtype, device=self.z_s.device)])
        else:
            lambda_precision = gp_precision
            h = torch.zeros(gp_dim, dtype=self.z_s.dtype, device=self.z_s.device)

        chol_lambda = safe_cholesky(lambda_precision, jitter=self.jitter)
        m = torch.cholesky_solve(h.unsqueeze(-1), chol_lambda).squeeze(-1)
        self.state = JointOnlineState(
            lambda_precision=lambda_precision,
            chol_lambda=chol_lambda,
            h=h,
            m=m,
            reference_horizon=horizon,
            kuu_t=kuu_t,
            inv_kuu_t=inv_kuu_t,
            kzz_s=kzz_s,
            inv_kzz_s=inv_kzz_s,
            fixed_x_s=fixed_x_s,
            fixed_kxz_s=fixed_kxz_s,
            beta_dim=beta_dim,
            gp_dim=gp_dim,
            num_blocks_processed=0,
        )
        return self.state

    def _require_initialized(self) -> JointOnlineState:
        if self.state.lambda_precision is None or self.state.reference_horizon is None:
            raise RuntimeError('Call initialize(...) before update_block(...) or predict(...).')
        return self.state

    def _resolve_block_horizon(self, times: torch.Tensor, horizon: Optional[TemporalBlockSpec]) -> TemporalBlockSpec:
        state = self._require_initialized()
        if horizon is None:
            return state.reference_horizon
        block_horizon = self.temporal_builder.resolve_horizon(horizon)
        if self.enforce_shared_horizon and block_horizon != state.reference_horizon:
            raise ValueError('This joint-online implementation currently keeps one shared inducing basis.')
        return block_horizon if not self.enforce_shared_horizon else state.reference_horizon

    def _temporal_transfer_matrix(self, local_horizon: TemporalBlockSpec) -> torch.Tensor:
        state = self._require_initialized()
        if local_horizon == state.reference_horizon:
            return torch.eye(state.kuu_t.shape[0], dtype=state.kuu_t.dtype, device=state.kuu_t.device)
        cross_kuu_t = self.temporal_builder.compute_kuu_t_cross(local_horizon, state.reference_horizon)
        return cross_kuu_t @ state.inv_kuu_t

    def _recover_posterior(self) -> tuple[torch.Tensor, torch.Tensor]:
        state = self._require_initialized()
        chol_lambda = safe_cholesky(state.lambda_precision, jitter=self.jitter)
        m = torch.cholesky_solve(state.h.unsqueeze(-1), chol_lambda).squeeze(-1)
        return m, chol_lambda

    def _split_posterior_mean(self, posterior_mean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = self._require_initialized()
        beta = posterior_mean[: state.beta_dim]
        u = posterior_mean[state.beta_dim :]
        return beta, u

    def _joint_design_chunk(
        self,
        a_t_chunk: torch.Tensor,
        a_s: torch.Tensor,
        covariates_chunk: Optional[torch.Tensor],
        num_space: int,
    ) -> torch.Tensor:
        gp_chunk = torch.kron(a_t_chunk, a_s)
        if self.beta_dim == 0:
            return gp_chunk
        design_chunk = self._covariate_design(covariates_chunk, num_times=a_t_chunk.shape[0], num_space=num_space)
        design_chunk = design_chunk.reshape(-1, self.beta_dim)
        return torch.cat([design_chunk, gp_chunk], dim=1)

    def _posterior_rowwise_correction(
        self,
        a_t_star: torch.Tensor,
        a_s_star: torch.Tensor,
        covariates: Optional[torch.Tensor],
        num_space: int,
    ) -> torch.Tensor:
        state = self._require_initialized()
        chunks: list[torch.Tensor] = []
        for start in range(0, a_t_star.shape[0], self.prediction_time_chunk_size):
            end = min(start + self.prediction_time_chunk_size, a_t_star.shape[0])
            covariate_chunk = None if covariates is None else covariates[start:end]
            design_chunk = self._joint_design_chunk(
                a_t_chunk=a_t_star[start:end],
                a_s=a_s_star,
                covariates_chunk=covariate_chunk,
                num_space=num_space,
            )
            correction = rowwise_quadratic_form_from_precision_cholesky(design_chunk, state.chol_lambda)
            chunks.append(correction.reshape(end - start, num_space))
        return torch.cat(chunks, dim=0)

    def update_block(
        self,
        times: torch.Tensor,
        x_s: torch.Tensor | None,
        y: torch.Tensor,
        covariates: Optional[torch.Tensor] = None,
        horizon: Optional[TemporalBlockSpec] = None,
    ) -> dict[str, Any]:
        start_time = time.perf_counter()
        state = self._require_initialized()
        times = self._as_times(times)
        x_s, kxz_s = self._spatial_cross_covariance(x_s)
        y_grid = self._as_observations(y, num_times=times.shape[0], num_space=x_s.shape[0])
        design = self._covariate_design(covariates, num_times=times.shape[0], num_space=x_s.shape[0])
        block_horizon = self._resolve_block_horizon(times, horizon)

        kfu_t_local = self.temporal_builder.compute_kfu_t(times, block_horizon)
        kuu_t_local = self.temporal_builder.compute_kuu_t(block_horizon)
        inv_kuu_t_local = cholesky_inverse(kuu_t_local, jitter=self.jitter)
        a_t_local = kfu_t_local @ inv_kuu_t_local
        temporal_transfer = self._temporal_transfer_matrix(block_horizon)
        a_t = a_t_local @ temporal_transfer
        a_s = kxz_s @ state.inv_kzz_s

        sigma2_inv = torch.reciprocal(self.sigma2)
        y_vec = flatten_grid(y_grid)
        delta_lambda_uu = sigma2_inv * torch.kron(a_t.transpose(-1, -2) @ a_t, a_s.transpose(-1, -2) @ a_s)
        delta_h_u = sigma2_inv * flatten_grid(a_t.transpose(-1, -2) @ y_grid @ a_s)

        if state.beta_dim > 0:
            design_flat = design.reshape(-1, state.beta_dim)
            delta_lambda_bb = sigma2_inv * (design_flat.transpose(-1, -2) @ design_flat)
            delta_h_b = sigma2_inv * (design_flat.transpose(-1, -2) @ y_vec)
            delta_lambda_bu = torch.zeros(
                (state.beta_dim, state.gp_dim),
                dtype=self.z_s.dtype,
                device=self.z_s.device,
            )
            for feature_idx in range(state.beta_dim):
                feature_grid = design[:, :, feature_idx]
                delta_lambda_bu[feature_idx] = sigma2_inv * flatten_grid(
                    a_t.transpose(-1, -2) @ feature_grid @ a_s
                )
            delta_lambda = torch.cat(
                [
                    torch.cat([delta_lambda_bb, delta_lambda_bu], dim=1),
                    torch.cat([delta_lambda_bu.transpose(-1, -2), delta_lambda_uu], dim=1),
                ],
                dim=0,
            )
            delta_h = torch.cat([delta_h_b, delta_h_u], dim=0)
        else:
            delta_lambda = delta_lambda_uu
            delta_h = delta_h_u

        state.lambda_precision = state.lambda_precision + delta_lambda
        state.h = state.h + delta_h
        state.m, state.chol_lambda = self._recover_posterior()
        state.num_blocks_processed += 1

        beta, u = self._split_posterior_mean(state.m)
        mean_only = self._beta_mean_grid(beta, covariates, num_times=times.shape[0], num_space=x_s.shape[0])
        u_matrix = u.reshape(state.kuu_t.shape[0], state.kzz_s.shape[0])
        gp_mean = a_t @ u_matrix @ a_s.transpose(-1, -2)
        block_mean = mean_only + gp_mean
        block_rmse = torch.sqrt(torch.mean((block_mean - y_grid) ** 2))
        pred = self.predict(times, x_s, covariates=covariates, include_noise=True)
        block_nll = 0.5 * torch.mean(
            torch.log(2.0 * torch.pi * pred['obs_var']) + (y_grid - pred['mean']) ** 2 / pred['obs_var']
        )
        return {
            'Lambda': state.lambda_precision,
            'h': state.h,
            'm': state.m,
            'chol_lambda': state.chol_lambda,
            'A_t': a_t,
            'A_s': a_s,
            'rmse': block_rmse,
            'pred_nll': block_nll,
            'runtime_s': time.perf_counter() - start_time,
            'block_index': state.num_blocks_processed - 1,
            'block_horizon': block_horizon,
        }
    def predict(
        self,
        t_star: torch.Tensor,
        s_star: torch.Tensor,
        covariates: Optional[torch.Tensor] = None,
        include_noise: bool = True,
        horizon: Optional[TemporalBlockSpec] = None,
    ) -> dict[str, torch.Tensor]:
        state = self._require_initialized()
        if state.m is None or state.chol_lambda is None:
            raise RuntimeError('Posterior state has not been recovered yet.')

        t_star = self._as_times(t_star)
        s_star = self._as_spatial(s_star)
        pred_horizon = state.reference_horizon
        if horizon is not None and not self.enforce_shared_horizon:
            pred_horizon = self.temporal_builder.resolve_horizon(horizon)
        kfu_t_star = self.temporal_builder.compute_kfu_t(t_star, pred_horizon)
        kxz_s_star = self.spatial_kernel.compute_kxz_s(s_star, self.z_s)
        a_t_star = kfu_t_star @ state.inv_kuu_t
        a_s_star = kxz_s_star @ state.inv_kzz_s

        beta, u = self._split_posterior_mean(state.m)
        mean_only = self._beta_mean_grid(beta, covariates, num_times=t_star.shape[0], num_space=s_star.shape[0])
        u_matrix = u.reshape(state.kuu_t.shape[0], state.kzz_s.shape[0])
        gp_mean = a_t_star @ u_matrix @ a_s_star.transpose(-1, -2)
        mean = mean_only + gp_mean

        conditional_gp_diag = torch.clamp(
            torch.outer(self.temporal_builder.compute_ktt_diag(t_star), self.spatial_kernel.diag(s_star))
            - kron_rowwise_prior_diag(a_t_star, state.kuu_t, a_s_star, state.kzz_s),
            min=0.0,
        )
        posterior_correction = self._posterior_rowwise_correction(
            a_t_star=a_t_star,
            a_s_star=a_s_star,
            covariates=covariates,
            num_space=s_star.shape[0],
        )
        latent_var = torch.clamp(conditional_gp_diag + posterior_correction, min=1e-9)
        obs_var = latent_var + (self.sigma2 if include_noise else 0.0)
        return {
            'mean': mean,
            'mean_only': mean_only,
            'latent_var': latent_var,
            'obs_var': obs_var,
            'beta_mean': beta,
            'u_mean': u,
        }


def _build_joint_model(args: Any, z_s: torch.Tensor, input_dim: int, covariate_dim: int = 0) -> JointOnlineMeanGPSTGP:
    model = JointOnlineMeanGPSTGP.from_configs(
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
        enforce_shared_horizon=not args.allow_local_horizon,
        prediction_time_chunk_size=args.prediction_time_chunk_size,
    )
    return model.to(args.runtime_device)


def _extract_beta_prior_mean(batch_model: Any) -> Optional[torch.Tensor]:
    linear = getattr(batch_model, 'covariate_linear', None)
    if linear is None:
        return None
    weight = linear.weight.detach().reshape(-1)
    bias = linear.bias.detach().reshape(-1)
    return torch.cat([weight, bias], dim=0)


def _joint_compare_diagnostics(observations: torch.Tensor, pred_output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        'mean_only_rmse': rmse(observations, pred_output['mean_only']),
        'full_rmse': rmse(observations, pred_output['mean']),
        'full_minus_mean_l2': torch.norm(pred_output['mean'] - pred_output['mean_only']),
    }


def run_synthetic_joint(args: Any) -> None:
    from stvgp_kronecker.train_batch import make_synthetic_dataset

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
    pretrained_batch, pretrain_summary = _maybe_pretrain_stage1(
        args,
        train_times=times,
        train_spatial=spatial,
        train_observations=observations,
        train_covariates=None,
        val_times=None,
        val_spatial=None,
        val_observations=None,
        val_covariates=None,
        z_s=z_s,
    )
    if pretrained_batch is not None:
        z_s = pretrained_batch.z_s.detach().clone()

    model = _build_joint_model(args, z_s=z_s, input_dim=2, covariate_dim=0)
    if pretrained_batch is not None:
        model.load_pretrained_batch_hyperparameters(clone_state_dict(pretrained_batch))
        model.freeze_model_hyperparameters()
    model.initialize(reference_horizon=reference_horizon, x_s=spatial, beta_prior_precision=args.beta_prior_precision)

    block_specs = build_temporal_blocks(times, block_size=args.block_size, overlap=args.block_overlap, num_discrete_steps=args.block_size)
    total_start = time.perf_counter()
    block_times: list[float] = []
    for block_slice, block_horizon in block_specs:
        output = model.update_block(
            times=times[block_slice],
            x_s=spatial,
            y=observations[block_slice],
            covariates=None,
            horizon=None if not args.allow_local_horizon else block_horizon,
        )
        block_times.append(output['runtime_s'])
        print(
            f"[joint-online] block={output['block_index']:02d} rmse={output['rmse'].item():.4f} "
            f"pred_nll={output['pred_nll'].item():.4f} block_time={output['runtime_s']:.3f}s"
        )

    total_runtime = time.perf_counter() - total_start
    if pretrain_summary is not None:
        total_runtime += pretrain_summary['training_runtime_s']
    pred_output = model.predict(times, spatial, covariates=None)
    mean_block_time = sum(block_times) / len(block_times)
    print(
        '[eval] rmse={:.4f} pred_nll={:.4f} runtime={:.3f}s gpu_mem={:.1f}MB time_per_block={:.3f}s'.format(
            rmse(observations, pred_output['mean']).item(),
            predictive_nll(observations, pred_output['mean'], pred_output['obs_var']).item(),
            total_runtime,
            current_gpu_memory_mb(),
            mean_block_time,
        )
    )


def run_era5_joint_probe(args: Any) -> None:
    task_dirs = resolve_era5_task_dirs(args)
    if not task_dirs:
        raise ValueError('Joint-online ERA5 mode currently expects processed ERA5 task directories.')

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
        task_name = '+'.join(Path(path).name for path in task_dirs)
    else:
        task = load_processed_era5_task(
            task_dir=task_dirs[0],
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
    for split in [task.train, task.val, task.test]:
        split.times = move_tensor_to_device(split.times, args.runtime_device)
        split.spatial_coords = move_tensor_to_device(split.spatial_coords, args.runtime_device)
        split.observations = move_tensor_to_device(split.observations, args.runtime_device)
        if split.covariates is not None:
            split.covariates = move_tensor_to_device(split.covariates, args.runtime_device)

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
        selection_method=args.spatial_inducing_selection,
    )

    args.reference_steps = args.reference_steps or task.train.times.shape[0]
    pretrained_batch, pretrain_summary = _maybe_pretrain_stage1(
        args,
        train_times=task.train.times,
        train_spatial=spatial,
        train_observations=task.train.observations,
        train_covariates=task.train.covariates,
        val_times=task.val.times,
        val_spatial=task.val.spatial_coords,
        val_observations=task.val.observations,
        val_covariates=task.val.covariates,
        z_s=z_s,
    )
    beta_prior_mean = None
    if pretrained_batch is not None:
        z_s = pretrained_batch.z_s.detach().clone()
        beta_prior_mean = _extract_beta_prior_mean(pretrained_batch)

    covariate_dim = 0 if task.train.covariates is None else task.train.covariates.shape[-1]
    model = _build_joint_model(args, z_s=z_s, input_dim=spatial.shape[1], covariate_dim=covariate_dim)
    if pretrained_batch is not None:
        model.load_pretrained_batch_hyperparameters(clone_state_dict(pretrained_batch))
        model.freeze_model_hyperparameters()
    model.initialize(
        reference_horizon=reference_horizon,
        x_s=spatial,
        beta_prior_precision=args.beta_prior_precision,
        beta_prior_mean=beta_prior_mean,
    )

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
            covariates=task.train.covariates[block_slice] if task.train.covariates is not None else None,
            horizon=None if not args.allow_local_horizon else block_horizon,
        )
        block_times.append(output['runtime_s'])
        if output['block_index'] % max(args.log_every_blocks, 1) == 0:
            print(
                f"[joint-train] block={output['block_index']:03d} rmse={output['rmse'].item():.4f} "
                f"pred_nll={output['pred_nll'].item():.4f} block_time={output['runtime_s']:.3f}s"
            )
    train_runtime = time.perf_counter() - train_start
    if pretrain_summary is not None:
        train_runtime += pretrain_summary['training_runtime_s']

    train_pred = model.predict(task.train.times, task.train.spatial_coords, covariates=task.train.covariates)
    val_pred = model.predict(task.val.times, task.val.spatial_coords, covariates=task.val.covariates)
    test_pred = model.predict(task.test.times, task.test.spatial_coords, covariates=task.test.covariates)
    mean_block_time = sum(block_times) / len(block_times)
    full_location_count = infer_processed_era5_location_count(
        task_dirs,
        scaled=not args.era5_unscaled,
        location_stride=args.era5_location_stride,
    )
    print(
        '[era5-joint-online] task={} variable_index={} variable_name={} train_shape={} val_shape={} test_shape={}'.format(
            task_name,
            args.era5_variable_index,
            task.variable_name,
            tuple(task.train.observations.shape),
            tuple(task.val.observations.shape),
            tuple(task.test.observations.shape),
        )
    )
    print(
        '[era5-data] temporal_method=analytic spatial_method=svgp spatial_coord_order=(lon,lat) spatial_input_dim={} covariate_dim={} selected_locations={} available_locations={} full_task12={}'.format(
            task.train.spatial_coords.shape[1],
            covariate_dim,
            task.train.spatial_coords.shape[0],
            full_location_count if full_location_count is not None else task.train.spatial_coords.shape[0],
            bool(getattr(args, 'era5_full_task12', False)),
        )
    )
    if covariate_indices:
        covariate_names = [era5_variable_name(index) for index in covariate_indices]
        print(f'[era5-covariates] indices={covariate_indices} names={covariate_names}')
    print(
        '[model] inducing_points={} beta_dim={} beta_prior_precision={:.2e} beta_prior_mean_source={} {}'.format(
            model.z_s.shape[0],
            model.beta_dim,
            args.beta_prior_precision,
            'pretrain' if beta_prior_mean is not None else 'zero',
            format_model_hyperparameters(pretrained_batch or model),
        )
    )
    for label, split_tensor, pred in [('train', task.train, train_pred), ('val', task.val, val_pred), ('test', task.test, test_pred)]:
        compare = _joint_compare_diagnostics(split_tensor.observations, pred)
        if label == 'test':
            print(
                '[test-eval] rmse={:.4f} pred_nll={:.4f} runtime={:.3f}s gpu_mem={:.1f}MB time_per_block={:.3f}s'.format(
                    rmse(split_tensor.observations, pred['mean']).item(),
                    predictive_nll(split_tensor.observations, pred['mean'], pred['obs_var']).item(),
                    train_runtime,
                    current_gpu_memory_mb(),
                    mean_block_time,
                )
            )
        else:
            print(
                f'[{label}-eval] rmse={{:.4f}} pred_nll={{:.4f}}'.format(
                    rmse(split_tensor.observations, pred['mean']).item(),
                    predictive_nll(split_tensor.observations, pred['mean'], pred['obs_var']).item(),
                )
            )
        print(
            f'[{label}-compare] mean_only_rmse={{:.4f}} full_rmse={{:.4f}} full_minus_mean_l2={{:.4f}}'.format(
                compare['mean_only_rmse'].item(),
                compare['full_rmse'].item(),
                compare['full_minus_mean_l2'].item(),
            )
        )

    split_name = args.map_split
    split_tensor = getattr(task, split_name)
    split_pred = {'train': train_pred, 'val': val_pred, 'test': test_pred}[split_name]
    maybe_save_era5_maps(
        args=args,
        mode='joint_online',
        task_name=task_name,
        variable_name=task.variable_name,
        split_name=split_name,
        split_tensor=split_tensor,
        pred_output=split_pred,
        mean_only=split_pred['mean_only'],
    )


def build_argparser():
    parser = build_base_argparser()
    parser.prog = 'Stage 2 joint online Kronecker spatio-temporal HiPPO-SVGP'
    parser.add_argument('--beta-prior-precision', type=float, default=1e-4)
    parser.add_argument('--prediction-time-chunk-size', type=int, default=4)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    args.runtime_device = resolve_device(args.device)
    print(f'[device] using {args.runtime_device}')
    if args.dataset == 'synthetic':
        run_synthetic_joint(args)
        return
    run_era5_joint_probe(args)


if __name__ == '__main__':
    main()

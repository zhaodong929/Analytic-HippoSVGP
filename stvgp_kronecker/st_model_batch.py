"""Stage 1 batch Gaussian Kronecker spatio-temporal HiPPO-SVGP."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

from .era5_dataset import build_temporal_blocks
from .kron_ops import (
    cholesky_inverse,
    flatten_grid,
    gaussian_nll,
    kron_gram,
    kron_logdet,
    kron_rowwise_prior_diag,
    logdet_from_cholesky,
    rowwise_quadratic_form,
    rowwise_quadratic_form_from_precision_cholesky,
    safe_cholesky,
)
from .spatial_kernel import BaseSpatialKernel, SpatialKernelConfig, build_spatial_kernel
from .temporal_analytic import AnalyticTemporalBuilder, TemporalAnalyticConfig, TemporalBlockSpec


@dataclass
class TemporalCovariances:
    """Temporal covariance bundle."""

    horizon: TemporalBlockSpec
    kuu_t: torch.Tensor
    kfu_t: torch.Tensor


@dataclass
class SpatialCovariances:
    """Spatial covariance bundle."""

    kzz_s: torch.Tensor
    kxz_s: torch.Tensor


@dataclass
class ProjectionFactors:
    """Kronecker projection factors."""

    a_t: torch.Tensor
    a_s: torch.Tensor


@dataclass
class BlockwiseForwardSummary:
    """Summary of Stage 1.5 blockwise batch processing."""

    block_outputs: list[dict[str, Any]]
    mean_loss: torch.Tensor
    mean_rmse: torch.Tensor
    total_runtime_s: float
    mean_block_runtime_s: float


@dataclass
class BatchPosteriorState:
    """Cached Stage 1 posterior state."""

    horizon: TemporalBlockSpec
    kuu_t: torch.Tensor
    kzz_s: torch.Tensor
    inv_kuu_t: torch.Tensor
    inv_kzz_s: torch.Tensor
    a_t: torch.Tensor
    a_s: torch.Tensor
    m: torch.Tensor
    chol_precision_u: torch.Tensor
    sigma2: torch.Tensor
    train_times: torch.Tensor
    train_spatial: torch.Tensor


class BatchKroneckerSTHiPPOSVGP(nn.Module):
    """Static batch Gaussian model with Kronecker-separable temporal/spatial factors.

    Stage 1 uses the closed-form Gaussian posterior update described in the
    reference document:

    `S^{-1} = Kuu^{-1} + (1 / sigma^2) A^T A`
    `m = S (1 / sigma^2) A^T y`

    The implementation keeps the inducing posterior covariance `S` dense for
    simplicity, while avoiding explicit dense data-space Kronecker matrices in
    the common projection and summary computations.
    """

    def __init__(
        self,
        temporal_config: TemporalAnalyticConfig,
        spatial_kernel_config: SpatialKernelConfig,
        z_s: torch.Tensor,
        noise_std: float = 0.1,
        covariate_dim: int = 0,
        learn_spatial_inducing: bool = False,
        jitter: float = 1e-6,
    ) -> None:
        super().__init__()
        self.temporal_builder = AnalyticTemporalBuilder(temporal_config)
        self.spatial_kernel: BaseSpatialKernel = build_spatial_kernel(spatial_kernel_config)
        z_s = torch.as_tensor(z_s, dtype=temporal_config.dtype)
        if learn_spatial_inducing:
            self.z_s = nn.Parameter(z_s.clone())
        else:
            self.register_buffer("z_s", z_s.clone())
        self.log_noise_std = nn.Parameter(torch.log(torch.as_tensor(noise_std, dtype=temporal_config.dtype)))
        self.covariate_dim = int(covariate_dim)
        if self.covariate_dim > 0:
            self.covariate_linear = nn.Linear(self.covariate_dim, 1, bias=True, dtype=temporal_config.dtype)
            nn.init.zeros_(self.covariate_linear.weight)
            nn.init.zeros_(self.covariate_linear.bias)
        else:
            self.covariate_linear = None
        self.jitter = float(jitter)
        self.posterior_state: Optional[BatchPosteriorState] = None

    @property
    def sigma2(self) -> torch.Tensor:
        return torch.exp(2.0 * self.log_noise_std)

    def _as_times(self, times: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(times, dtype=self.z_s.dtype, device=self.z_s.device).reshape(-1)

    def _as_spatial(self, x_s: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(x_s, dtype=self.z_s.dtype, device=self.z_s.device)
        if tensor.ndim != 2:
            raise ValueError("Spatial inputs `X_s` must have shape [N_s, D_s].")
        return tensor

    def _as_observations(self, y: torch.Tensor, num_times: int, num_space: int) -> torch.Tensor:
        tensor = torch.as_tensor(y, dtype=self.z_s.dtype, device=self.z_s.device)
        if tensor.shape == (num_times, num_space):
            return tensor
        if tensor.numel() == num_times * num_space:
            return tensor.reshape(num_times, num_space)
        raise ValueError(
            "Observations `y` must have shape [N_t, N_s] or flatten to N_t * N_s elements."
        )

    def _as_covariates(
        self,
        covariates: Optional[torch.Tensor],
        num_times: int,
        num_space: int,
    ) -> Optional[torch.Tensor]:
        if self.covariate_linear is None:
            return None
        if covariates is None:
            raise ValueError("Covariates are required because the model was built with covariate_dim > 0.")
        tensor = torch.as_tensor(covariates, dtype=self.z_s.dtype, device=self.z_s.device)
        if tensor.shape == (num_times, num_space, self.covariate_dim):
            return tensor
        if tensor.numel() == num_times * num_space * self.covariate_dim:
            return tensor.reshape(num_times, num_space, self.covariate_dim)
        raise ValueError(
            "Covariates must have shape [N_t, N_s, C] or flatten to N_t * N_s * C elements."
        )

    def _covariate_mean(
        self,
        covariates: Optional[torch.Tensor],
        num_times: int,
        num_space: int,
    ) -> torch.Tensor:
        if self.covariate_linear is None:
            return torch.zeros((num_times, num_space), dtype=self.z_s.dtype, device=self.z_s.device)
        covariate_tensor = self._as_covariates(covariates, num_times=num_times, num_space=num_space)
        return self.covariate_linear(covariate_tensor).squeeze(-1)

    def build_temporal_covariances(
        self,
        times: torch.Tensor,
        horizon: Optional[TemporalBlockSpec] = None,
    ) -> TemporalCovariances:
        times = self._as_times(times)
        horizon = horizon or self.temporal_builder.resolve_horizon(times)
        kuu_t = self.temporal_builder.compute_kuu_t(horizon)
        kfu_t = self.temporal_builder.compute_kfu_t(times, horizon)
        return TemporalCovariances(horizon=horizon, kuu_t=kuu_t, kfu_t=kfu_t)

    def build_spatial_covariances(self, x_s: torch.Tensor) -> SpatialCovariances:
        x_s = self._as_spatial(x_s)
        kzz_s = self.spatial_kernel.compute_kzz_s(self.z_s)
        kxz_s = self.spatial_kernel.compute_kxz_s(x_s, self.z_s)
        return SpatialCovariances(kzz_s=kzz_s, kxz_s=kxz_s)

    def build_projection(
        self,
        times: torch.Tensor,
        x_s: torch.Tensor,
        temporal_covariances: Optional[TemporalCovariances] = None,
        spatial_covariances: Optional[SpatialCovariances] = None,
    ) -> ProjectionFactors:
        temporal_covariances = temporal_covariances or self.build_temporal_covariances(times)
        spatial_covariances = spatial_covariances or self.build_spatial_covariances(x_s)

        chol_t = safe_cholesky(temporal_covariances.kuu_t, jitter=self.jitter)
        chol_s = safe_cholesky(spatial_covariances.kzz_s, jitter=self.jitter)

        a_t = torch.cholesky_solve(temporal_covariances.kfu_t.transpose(-1, -2), chol_t).transpose(-1, -2)
        a_s = torch.cholesky_solve(spatial_covariances.kxz_s.transpose(-1, -2), chol_s).transpose(-1, -2)
        return ProjectionFactors(a_t=a_t, a_s=a_s)

    def forward_block(
        self,
        times: torch.Tensor,
        x_s: torch.Tensor,
        y: torch.Tensor,
        covariates: Optional[torch.Tensor] = None,
        horizon: Optional[TemporalBlockSpec] = None,
        spatial_covariances: Optional[SpatialCovariances] = None,
        cache_posterior: bool = False,
    ) -> dict[str, Any]:
        """Process one temporal block with fixed spatial inducing locations."""
        return self.forward(
            times=times,
            x_s=x_s,
            y=y,
            covariates=covariates,
            horizon=horizon,
            spatial_covariances=spatial_covariances,
            cache_posterior=cache_posterior,
        )

    def _compute_posterior(
        self,
        temporal_covariances: TemporalCovariances,
        spatial_covariances: SpatialCovariances,
        projections: ProjectionFactors,
        y_grid: torch.Tensor,
        materialize_cov: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        inv_kuu_t = cholesky_inverse(temporal_covariances.kuu_t, jitter=self.jitter)
        inv_kzz_s = cholesky_inverse(spatial_covariances.kzz_s, jitter=self.jitter)
        precision = torch.kron(inv_kuu_t, inv_kzz_s) + torch.reciprocal(self.sigma2) * kron_gram(
            projections.a_t,
            projections.a_s,
        )
        info_matrix = projections.a_t.transpose(-1, -2) @ y_grid @ projections.a_s
        info = torch.reciprocal(self.sigma2) * flatten_grid(info_matrix)
        chol_precision = safe_cholesky(precision, jitter=self.jitter)
        mean = torch.cholesky_solve(info.unsqueeze(-1), chol_precision).squeeze(-1)
        cov = None
        if materialize_cov:
            eye = torch.eye(precision.shape[0], dtype=precision.dtype, device=precision.device)
            cov = torch.cholesky_solve(eye, chol_precision)
        logdet_precision = logdet_from_cholesky(chol_precision)
        return mean, cov, chol_precision, info, logdet_precision

    def _cache_state(
        self,
        horizon: TemporalBlockSpec,
        temporal_covariances: TemporalCovariances,
        spatial_covariances: SpatialCovariances,
        projections: ProjectionFactors,
        mean: torch.Tensor,
        chol_precision_u: torch.Tensor,
        train_times: torch.Tensor,
        train_spatial: torch.Tensor,
    ) -> None:
        self.posterior_state = BatchPosteriorState(
            horizon=horizon,
            kuu_t=temporal_covariances.kuu_t,
            kzz_s=spatial_covariances.kzz_s,
            inv_kuu_t=cholesky_inverse(temporal_covariances.kuu_t, jitter=self.jitter),
            inv_kzz_s=cholesky_inverse(spatial_covariances.kzz_s, jitter=self.jitter),
            a_t=projections.a_t,
            a_s=projections.a_s,
            m=mean,
            chol_precision_u=chol_precision_u,
            sigma2=self.sigma2,
            train_times=train_times.detach().clone(),
            train_spatial=train_spatial.detach().clone(),
        )

    def materialize_posterior_covariance(self) -> torch.Tensor:
        if self.posterior_state is None:
            raise RuntimeError("Call `forward(..., cache_posterior=True)` before materializing covariance.")
        dim = self.posterior_state.chol_precision_u.shape[0]
        eye = torch.eye(
            dim,
            dtype=self.posterior_state.chol_precision_u.dtype,
            device=self.posterior_state.chol_precision_u.device,
        )
        return torch.cholesky_solve(eye, self.posterior_state.chol_precision_u)

    def forward(
        self,
        times: torch.Tensor,
        x_s: torch.Tensor,
        y: torch.Tensor,
        covariates: Optional[torch.Tensor] = None,
        horizon: Optional[TemporalBlockSpec] = None,
        spatial_covariances: Optional[SpatialCovariances] = None,
        cache_posterior: bool = True,
        materialize_posterior_cov: bool = True,
    ) -> dict[str, Any]:
        start_time = time.perf_counter()
        times = self._as_times(times)
        x_s = self._as_spatial(x_s)
        y_grid = self._as_observations(y, num_times=times.shape[0], num_space=x_s.shape[0])
        covariate_mean = self._covariate_mean(covariates, num_times=times.shape[0], num_space=x_s.shape[0])
        residual_grid = y_grid - covariate_mean

        temporal_covariances = self.build_temporal_covariances(times, horizon=horizon)
        spatial_covariances = spatial_covariances or self.build_spatial_covariances(x_s)
        projections = self.build_projection(
            times,
            x_s,
            temporal_covariances=temporal_covariances,
            spatial_covariances=spatial_covariances,
        )

        mean_u, cov_u, chol_precision_u, info_u, logdet_precision = self._compute_posterior(
            temporal_covariances,
            spatial_covariances,
            projections,
            residual_grid,
            materialize_cov=materialize_posterior_cov,
        )
        inducing_mean_matrix = mean_u.reshape(
            temporal_covariances.kuu_t.shape[0],
            spatial_covariances.kzz_s.shape[0],
        )
        train_mean = projections.a_t @ inducing_mean_matrix @ projections.a_s.transpose(-1, -2) + covariate_mean

        logdet_kuu = kron_logdet(
            temporal_covariances.kuu_t,
            spatial_covariances.kzz_s,
            jitter=self.jitter,
        )
        loss = gaussian_nll(
            flatten_grid(residual_grid if self.covariate_linear is not None else y_grid),
            self.sigma2,
            logdet_kuu=logdet_kuu,
            logdet_precision=logdet_precision,
            info_dot_mean=torch.dot(info_u, mean_u),
        )
        rmse = torch.sqrt(torch.mean((train_mean - y_grid) ** 2))

        if cache_posterior:
            self._cache_state(
                temporal_covariances.horizon,
                temporal_covariances,
                spatial_covariances,
                projections,
                mean_u,
                chol_precision_u,
                train_times=times,
                train_spatial=x_s,
            )

        runtime_s = time.perf_counter() - start_time
        return {
            "loss": loss,
            "rmse": rmse,
            "runtime_s": runtime_s,
            "train_mean": train_mean,
            "covariate_mean": covariate_mean,
            "posterior_mean_u": mean_u,
            "posterior_cov_u": cov_u,
            "posterior_precision_cholesky_u": chol_precision_u,
            "info_u": info_u,
            "Kuu_t": temporal_covariances.kuu_t,
            "Kfu_t": temporal_covariances.kfu_t,
            "Kzz_s": spatial_covariances.kzz_s,
            "Kxz_s": spatial_covariances.kxz_s,
            "A_t": projections.a_t,
            "A_s": projections.a_s,
        }

    def forward_blockwise(
        self,
        times: torch.Tensor,
        x_s: torch.Tensor,
        y: torch.Tensor,
        block_size: int,
        overlap: int = 0,
        num_discrete_steps: Optional[int] = None,
        cache_last_posterior: bool = True,
        covariates: Optional[torch.Tensor] = None,
    ) -> BlockwiseForwardSummary:
        """Run Stage 1.5 blockwise batch processing over temporal blocks.

        Each block gets its own temporal `Kfu_t` and `Kuu_t`, while the spatial
        inducing locations and `Kzz_s` / `Kxz_s` remain fixed.
        """

        start_time = time.perf_counter()
        times = self._as_times(times)
        x_s = self._as_spatial(x_s)
        y_grid = self._as_observations(y, num_times=times.shape[0], num_space=x_s.shape[0])
        covariate_grid = self._as_covariates(covariates, num_times=times.shape[0], num_space=x_s.shape[0])
        spatial_covariances = self.build_spatial_covariances(x_s)
        block_specs = build_temporal_blocks(
            times,
            block_size=block_size,
            overlap=overlap,
            num_discrete_steps=num_discrete_steps,
        )

        block_outputs: list[dict[str, Any]] = []
        for block_index, (block_slice, block_horizon) in enumerate(block_specs):
            block_output = self.forward(
                times=times[block_slice],
                x_s=x_s,
                y=y_grid[block_slice],
                covariates=covariate_grid[block_slice] if covariate_grid is not None else None,
                horizon=block_horizon,
                spatial_covariances=spatial_covariances,
                cache_posterior=cache_last_posterior and block_index == len(block_specs) - 1,
                materialize_posterior_cov=False,
            )
            block_output["block_index"] = block_index
            block_output["block_slice"] = block_slice
            block_output["block_horizon"] = block_horizon
            block_outputs.append(block_output)

        mean_loss = torch.stack([item["loss"] for item in block_outputs]).mean()
        mean_rmse = torch.stack([item["rmse"] for item in block_outputs]).mean()
        total_runtime_s = time.perf_counter() - start_time
        return BlockwiseForwardSummary(
            block_outputs=block_outputs,
            mean_loss=mean_loss,
            mean_rmse=mean_rmse,
            total_runtime_s=total_runtime_s,
            mean_block_runtime_s=total_runtime_s / max(len(block_outputs), 1),
        )

    def predict(
        self,
        t_star: torch.Tensor,
        s_star: torch.Tensor,
        covariates: Optional[torch.Tensor] = None,
        include_noise: bool = True,
    ) -> dict[str, torch.Tensor]:
        if self.posterior_state is None:
            raise RuntimeError("Call `forward(..., cache_posterior=True)` before `predict`.")

        t_star = self._as_times(t_star)
        s_star = self._as_spatial(s_star)
        state = self.posterior_state

        kfu_t_star = self.temporal_builder.compute_kfu_t(t_star, state.horizon)
        kxz_s_star = self.spatial_kernel.compute_kxz_s(s_star, self.z_s)

        a_t_star = kfu_t_star @ state.inv_kuu_t
        a_s_star = kxz_s_star @ state.inv_kzz_s

        inducing_mean_matrix = state.m.reshape(state.kuu_t.shape[0], state.kzz_s.shape[0])
        mean = a_t_star @ inducing_mean_matrix @ a_s_star.transpose(-1, -2)
        mean = mean + self._covariate_mean(covariates, num_times=t_star.shape[0], num_space=s_star.shape[0])

        prior_diag = torch.outer(
            self.temporal_builder.compute_ktt_diag(t_star),
            self.spatial_kernel.diag(s_star),
        )
        projected_prior_diag = kron_rowwise_prior_diag(
            a_t_star,
            state.kuu_t,
            a_s_star,
            state.kzz_s,
        )

        # TODO: for larger prediction grids, chunk `torch.kron(a_t_star, a_s_star)`
        # instead of materializing it densely.
        a_star_dense = torch.kron(a_t_star, a_s_star)
        posterior_correction = rowwise_quadratic_form_from_precision_cholesky(
            a_star_dense,
            state.chol_precision_u,
        ).reshape(mean.shape)
        latent_var = torch.clamp(prior_diag - projected_prior_diag + posterior_correction, min=1e-9)
        obs_var = latent_var + (state.sigma2 if include_noise else 0.0)

        return {
            "mean": mean,
            "latent_var": latent_var,
            "obs_var": obs_var,
        }

    def materialize_full_kronecker_matrices(
        self,
        times: torch.Tensor,
        x_s: torch.Tensor,
        horizon: Optional[TemporalBlockSpec] = None,
    ) -> dict[str, torch.Tensor]:
        """Debug helper for tiny problems only."""
        temporal_covariances = self.build_temporal_covariances(times, horizon=horizon)
        spatial_covariances = self.build_spatial_covariances(x_s)
        kuu = torch.kron(temporal_covariances.kuu_t, spatial_covariances.kzz_s)
        kfu = torch.kron(temporal_covariances.kfu_t, spatial_covariances.kxz_s)
        return {"Kuu": kuu, "Kfu": kfu}

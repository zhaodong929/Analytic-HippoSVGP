"""Stage 2 online posterior-summary update for the Kronecker STGP."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

from .kron_ops import (
    cholesky_inverse,
    flatten_grid,
    kron_gram,
    kron_rowwise_prior_diag,
    rowwise_quadratic_form_from_precision_cholesky,
    safe_cholesky,
)
from .spatial_kernel import BaseSpatialKernel, SpatialKernelConfig, build_spatial_kernel
from .temporal_analytic import AnalyticTemporalBuilder, TemporalAnalyticConfig, TemporalBlockSpec


@dataclass
class OnlinePosteriorState:
    """State variables for the Gaussian-conjugate online recursion."""

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
    num_blocks_processed: int = 0


class OnlinePosteriorSummarySTGP(nn.Module):
    """Gaussian online recursion with Kronecker-aware sufficient statistics.

    Current implementation detail:
    - The online summary lives in one shared inducing coordinate system.
    - By default that coordinate system is set by a fixed `reference_horizon`.
    - This makes the recursion exactly comparable to a full Gaussian batch
      solution when all blocks use the same inducing basis.

    Current block-local horizon support:
    - When `enforce_shared_horizon=False`, each block may use its own temporal
      inducing basis.
    - The implementation transfers that local basis back into the fixed
      reference inducing coordinates via the GP conditional mean
      `E[u_local | u_ref]`.

    TODO:
    - The current transfer uses the conditional-mean map only. A fuller
      treatment would also propagate the residual covariance of
      `u_local | u_ref` into the online summary rather than folding it into an
      implicit approximation.
    """

    def __init__(
        self,
        temporal_builder: AnalyticTemporalBuilder,
        spatial_kernel: BaseSpatialKernel,
        z_s: torch.Tensor,
        noise_std: float = 0.1,
        jitter: float = 1e-6,
        enforce_shared_horizon: bool = True,
    ) -> None:
        super().__init__()
        self.temporal_builder = temporal_builder
        self.spatial_kernel = spatial_kernel
        self.register_buffer("z_s", torch.as_tensor(z_s, dtype=z_s.dtype))
        self.log_noise_std = nn.Parameter(torch.log(torch.as_tensor(noise_std, dtype=z_s.dtype)))
        self.jitter = float(jitter)
        self.enforce_shared_horizon = bool(enforce_shared_horizon)
        self.state = OnlinePosteriorState()

    @classmethod
    def from_configs(
        cls,
        temporal_config: TemporalAnalyticConfig,
        spatial_kernel_config: SpatialKernelConfig,
        z_s: torch.Tensor,
        noise_std: float = 0.1,
        jitter: float = 1e-6,
        enforce_shared_horizon: bool = True,
    ) -> "OnlinePosteriorSummarySTGP":
        """Convenience constructor mirroring the Stage 1 batch model."""
        return cls(
            temporal_builder=AnalyticTemporalBuilder(temporal_config),
            spatial_kernel=build_spatial_kernel(spatial_kernel_config),
            z_s=z_s,
            noise_std=noise_std,
            jitter=jitter,
            enforce_shared_horizon=enforce_shared_horizon,
        )

    @property
    def sigma2(self) -> torch.Tensor:
        return torch.exp(2.0 * self.log_noise_std)

    def load_pretrained_batch_state(self, batch_state_dict: dict[str, torch.Tensor]) -> None:
        """Copy compatible Stage 1 parameters into the online model."""

        missing, unexpected = self.load_state_dict(batch_state_dict, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys when loading pretrained batch state: {unexpected}")
        allowed_missing = set()
        if set(missing) != allowed_missing:
            raise RuntimeError(f"Missing keys when loading pretrained batch state: {missing}")

    def freeze_model_hyperparameters(self) -> None:
        """Freeze noise and kernel hyperparameters after Stage 1 pretraining."""

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
            raise ValueError("Spatial inputs `X_s` must have shape [N_s, D_s].")
        return tensor

    def _as_observations(self, y: torch.Tensor, num_times: int, num_space: int) -> torch.Tensor:
        tensor = torch.as_tensor(y, dtype=self.z_s.dtype, device=self.z_s.device)
        if tensor.shape == (num_times, num_space):
            return tensor
        if tensor.numel() == num_times * num_space:
            return tensor.reshape(num_times, num_space)
        raise ValueError("Observations `y` must match the time-space grid shape.")

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
            raise ValueError("Either `reference_horizon` or `init_times` must be provided.")
        return TemporalBlockSpec.from_times(
            init_times,
            num_discrete_steps=num_discrete_steps or init_times.shape[0],
            prev_discrete_steps=0,
        )

    def _spatial_cross_covariance(self, x_s: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if x_s is None:
            if self.state.fixed_x_s is None or self.state.fixed_kxz_s is None:
                raise ValueError("No fixed spatial grid cached; pass `x_s` explicitly.")
            return self.state.fixed_x_s, self.state.fixed_kxz_s

        x_s = self._as_spatial(x_s)
        if self.state.fixed_x_s is not None and self._same_spatial_grid(x_s, self.state.fixed_x_s):
            return self.state.fixed_x_s, self.state.fixed_kxz_s
        return x_s, self.spatial_kernel.compute_kxz_s(x_s, self.z_s)

    def initialize(
        self,
        reference_horizon: torch.Tensor | TemporalBlockSpec | None = None,
        x_s: Optional[torch.Tensor] = None,
        init_times: Optional[torch.Tensor] = None,
        num_discrete_steps: Optional[int] = None,
    ) -> OnlinePosteriorState:
        """Initialize `Lambda`, `h`, and the prior summary."""

        horizon = self._resolve_reference_horizon(
            reference_horizon=reference_horizon,
            init_times=self._as_times(init_times) if init_times is not None else None,
            num_discrete_steps=num_discrete_steps,
        )
        kuu_t = self.temporal_builder.compute_kuu_t(horizon)
        inv_kuu_t = cholesky_inverse(kuu_t, jitter=self.jitter)
        kzz_s = self.spatial_kernel.compute_kzz_s(self.z_s)
        inv_kzz_s = cholesky_inverse(kzz_s, jitter=self.jitter)

        fixed_x_s = None
        fixed_kxz_s = None
        if x_s is not None:
            fixed_x_s = self._as_spatial(x_s)
            fixed_kxz_s = self.spatial_kernel.compute_kxz_s(fixed_x_s, self.z_s)

        lambda_precision = torch.kron(inv_kuu_t, inv_kzz_s)
        chol_lambda = safe_cholesky(lambda_precision, jitter=self.jitter)
        h = torch.zeros(lambda_precision.shape[0], dtype=lambda_precision.dtype, device=lambda_precision.device)
        m = torch.zeros_like(h)

        self.state = OnlinePosteriorState(
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
            num_blocks_processed=0,
        )
        return self.state

    def _require_initialized(self) -> OnlinePosteriorState:
        if self.state.lambda_precision is None or self.state.reference_horizon is None:
            raise RuntimeError("Call `initialize(...)` before `update_block(...)` or `predict(...)`.")
        return self.state

    def _resolve_block_horizon(
        self,
        times: torch.Tensor,
        horizon: Optional[TemporalBlockSpec],
    ) -> TemporalBlockSpec:
        state = self._require_initialized()
        if horizon is None:
            return state.reference_horizon
        block_horizon = self.temporal_builder.resolve_horizon(horizon)
        if self.enforce_shared_horizon and block_horizon != state.reference_horizon:
            raise ValueError(
                "This Stage 2 implementation keeps one shared inducing basis. "
                "Pass the reference horizon or initialize with `enforce_shared_horizon=False`."
            )
        return block_horizon if not self.enforce_shared_horizon else state.reference_horizon

    def _temporal_transfer_matrix(
        self,
        local_horizon: TemporalBlockSpec,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map local temporal inducing coordinates into the reference basis.

        Returns:
        - `transfer_local_from_reference`: matrix `T` such that
          `E[u_local | u_ref] = T u_ref`
        - `cross_kuu_t`: temporal inducing cross-covariance
        - `residual_kuu_t`: covariance of `u_local | u_ref`
        """

        state = self._require_initialized()
        if local_horizon == state.reference_horizon:
            identity = torch.eye(
                state.kuu_t.shape[0],
                dtype=state.kuu_t.dtype,
                device=state.kuu_t.device,
            )
            zeros = torch.zeros_like(state.kuu_t)
            return identity, state.kuu_t, zeros

        cross_kuu_t = self.temporal_builder.compute_kuu_t_cross(local_horizon, state.reference_horizon)
        transfer = cross_kuu_t @ state.inv_kuu_t
        local_kuu_t = self.temporal_builder.compute_kuu_t(local_horizon)
        residual_kuu_t = 0.5 * (
            local_kuu_t - cross_kuu_t @ state.inv_kuu_t @ cross_kuu_t.transpose(-1, -2)
            + (local_kuu_t - cross_kuu_t @ state.inv_kuu_t @ cross_kuu_t.transpose(-1, -2)).transpose(-1, -2)
        )
        return transfer, cross_kuu_t, residual_kuu_t

    def _residual_corrected_summary_terms(
        self,
        a_t_reference: torch.Tensor,
        a_t_local: torch.Tensor,
        residual_kuu_t: torch.Tensor,
        a_s: torch.Tensor,
        y_grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute `A^T C^{-1} A` and `A^T C^{-1} y` with residual correction.

        Here
        `C = sigma^2 I + (A_t_local R_t A_t_local^T) ⊗ (A_s Kzz_s A_s^T)`,
        where `R_t` is the conditional residual covariance of the local temporal
        inducing system given the reference inducing system.
        """

        state = self._require_initialized()
        if torch.max(residual_kuu_t.abs()) <= 1e-12:
            delta_lambda = torch.reciprocal(self.sigma2) * kron_gram(a_t_reference, a_s)
            delta_h = torch.reciprocal(self.sigma2) * flatten_grid(a_t_reference.transpose(-1, -2) @ y_grid @ a_s)
            return delta_lambda, delta_h

        residual_obs_t = 0.5 * (
            a_t_local @ residual_kuu_t @ a_t_local.transpose(-1, -2)
            + (a_t_local @ residual_kuu_t @ a_t_local.transpose(-1, -2)).transpose(-1, -2)
        )
        projected_obs_s = 0.5 * (
            a_s @ state.kzz_s @ a_s.transpose(-1, -2)
            + (a_s @ state.kzz_s @ a_s.transpose(-1, -2)).transpose(-1, -2)
        )

        eigvals_t, eigvecs_t = torch.linalg.eigh(residual_obs_t)
        eigvals_s, eigvecs_s = torch.linalg.eigh(projected_obs_s)
        eigvals_t = torch.clamp(eigvals_t, min=0.0)
        eigvals_s = torch.clamp(eigvals_s, min=0.0)

        b_t = eigvecs_t.transpose(-1, -2) @ a_t_reference
        b_s = eigvecs_s.transpose(-1, -2) @ a_s
        transformed_y = eigvecs_t.transpose(-1, -2) @ y_grid @ eigvecs_s
        weights = torch.reciprocal(self.sigma2 + eigvals_t.unsqueeze(-1) * eigvals_s.unsqueeze(0))

        info_matrix = b_t.transpose(-1, -2) @ (weights * transformed_y) @ b_s
        delta_h = flatten_grid(info_matrix)

        delta_lambda = torch.zeros(
            state.lambda_precision.shape,
            dtype=state.lambda_precision.dtype,
            device=state.lambda_precision.device,
        )
        for row_idx in range(b_t.shape[0]):
            gram_t = torch.outer(b_t[row_idx], b_t[row_idx])
            weighted_b_s = weights[row_idx].unsqueeze(-1) * b_s
            gram_s = weighted_b_s.transpose(-1, -2) @ b_s
            delta_lambda = delta_lambda + torch.kron(gram_t, gram_s)
        return delta_lambda, delta_h

    def _recover_posterior(self) -> tuple[torch.Tensor, torch.Tensor]:
        state = self._require_initialized()
        chol_lambda = safe_cholesky(state.lambda_precision, jitter=self.jitter)
        m = torch.cholesky_solve(state.h.unsqueeze(-1), chol_lambda).squeeze(-1)
        return m, chol_lambda

    def materialize_posterior_covariance(self) -> torch.Tensor:
        state = self._require_initialized()
        if state.chol_lambda is None:
            raise RuntimeError("Posterior precision Cholesky is not available.")
        dim = state.chol_lambda.shape[0]
        eye = torch.eye(dim, dtype=state.chol_lambda.dtype, device=state.chol_lambda.device)
        return torch.cholesky_solve(eye, state.chol_lambda)

    def update_block(
        self,
        times: torch.Tensor,
        x_s: torch.Tensor | None,
        y: torch.Tensor,
        horizon: Optional[TemporalBlockSpec] = None,
    ) -> dict[str, Any]:
        """Update `Lambda` and `h` with one temporal block."""

        start_time = time.perf_counter()
        state = self._require_initialized()
        times = self._as_times(times)
        x_s, kxz_s = self._spatial_cross_covariance(x_s)
        y_grid = self._as_observations(y, num_times=times.shape[0], num_space=x_s.shape[0])
        block_horizon = self._resolve_block_horizon(times, horizon)

        kfu_t_local = self.temporal_builder.compute_kfu_t(times, block_horizon)
        kuu_t_local = self.temporal_builder.compute_kuu_t(block_horizon)
        inv_kuu_t_local = cholesky_inverse(kuu_t_local, jitter=self.jitter)
        temporal_transfer, cross_kuu_t, residual_kuu_t = self._temporal_transfer_matrix(block_horizon)

        a_t_local = kfu_t_local @ inv_kuu_t_local
        a_t = a_t_local @ temporal_transfer
        a_s = kxz_s @ state.inv_kzz_s

        delta_lambda, delta_h = self._residual_corrected_summary_terms(
            a_t_reference=a_t,
            a_t_local=a_t_local,
            residual_kuu_t=residual_kuu_t,
            a_s=a_s,
            y_grid=y_grid,
        )

        state.lambda_precision = state.lambda_precision + delta_lambda
        state.h = state.h + delta_h
        state.m, state.chol_lambda = self._recover_posterior()
        state.num_blocks_processed += 1

        mean_matrix = state.m.reshape(state.kuu_t.shape[0], state.kzz_s.shape[0])
        block_mean = a_t @ mean_matrix @ a_s.transpose(-1, -2)
        block_rmse = torch.sqrt(torch.mean((block_mean - y_grid) ** 2))

        pred = self.predict(times, x_s, include_noise=True)
        block_nll = 0.5 * torch.mean(
            torch.log(2.0 * torch.pi * pred["obs_var"]) + (y_grid - pred["mean"]) ** 2 / pred["obs_var"]
        )

        return {
            "Lambda": state.lambda_precision,
            "h": state.h,
            "m": state.m,
            "chol_lambda": state.chol_lambda,
            "Kuu_t": kuu_t_local,
            "Kuu_t_cross": cross_kuu_t,
            "Kuu_t_residual": residual_kuu_t,
            "Kfu_t": kfu_t_local,
            "A_t": a_t,
            "A_t_local": a_t_local,
            "A_s": a_s,
            "temporal_transfer": temporal_transfer,
            "rmse": block_rmse,
            "pred_nll": block_nll,
            "runtime_s": time.perf_counter() - start_time,
            "block_index": state.num_blocks_processed - 1,
            "block_horizon": block_horizon,
        }

    def predict(
        self,
        t_star: torch.Tensor,
        s_star: torch.Tensor,
        include_noise: bool = True,
        horizon: Optional[TemporalBlockSpec] = None,
    ) -> dict[str, torch.Tensor]:
        """Predict from the current online posterior summary."""

        state = self._require_initialized()
        if state.m is None or state.chol_lambda is None:
            raise RuntimeError("Posterior state has not been recovered yet.")

        t_star = self._as_times(t_star)
        s_star = self._as_spatial(s_star)
        pred_horizon = state.reference_horizon
        if horizon is not None and self.enforce_shared_horizon:
            pred_horizon = state.reference_horizon

        kfu_t_star = self.temporal_builder.compute_kfu_t(t_star, pred_horizon)
        kxz_s_star = self.spatial_kernel.compute_kxz_s(s_star, self.z_s)

        a_t_star = kfu_t_star @ state.inv_kuu_t
        a_s_star = kxz_s_star @ state.inv_kzz_s

        mean_matrix = state.m.reshape(state.kuu_t.shape[0], state.kzz_s.shape[0])
        mean = a_t_star @ mean_matrix @ a_s_star.transpose(-1, -2)

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

        # TODO: chunk this dense Kronecker factor when prediction grids become large.
        a_star_dense = torch.kron(a_t_star, a_s_star)
        posterior_correction = rowwise_quadratic_form_from_precision_cholesky(
            a_star_dense,
            state.chol_lambda,
        ).reshape(mean.shape)
        latent_var = torch.clamp(prior_diag - projected_prior_diag + posterior_correction, min=1e-9)
        obs_var = latent_var + (self.sigma2 if include_noise else 0.0)

        return {
            "mean": mean,
            "latent_var": latent_var,
            "obs_var": obs_var,
        }

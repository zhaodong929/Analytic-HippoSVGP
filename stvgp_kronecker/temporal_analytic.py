"""Temporal analytic wrapper for the Kronecker HiPPO-SVGP prototype."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


def _miller_rescale(
    f_nm1: torch.Tensor,
    f_n: torch.Tensor,
    f_np1: torch.Tensor,
    large_thr: float,
    small_thr: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = torch.maximum(torch.maximum(f_nm1.abs(), f_n.abs()), f_np1.abs())
    scale = torch.where(
        scale > large_thr,
        scale,
        torch.where(scale < small_thr, torch.full_like(scale, small_thr), torch.ones_like(scale)),
    )
    return f_nm1 / scale, f_n / scale, f_np1 / scale, torch.log(scale)


def _miller_recurrence_python(x_safe: torch.Tensor, lmax: int, n_start: int) -> tuple[torch.Tensor, torch.Tensor]:
    out_shape = (lmax + 1,) + tuple(x_safe.shape)
    f = torch.zeros(out_shape, device=x_safe.device, dtype=x_safe.dtype)
    log_s = torch.zeros(out_shape, device=x_safe.device, dtype=x_safe.dtype)

    f_np1 = torch.zeros_like(x_safe)
    f_n = torch.ones_like(x_safe)
    cur_log_s = torch.zeros_like(x_safe)

    if x_safe.dtype == torch.float64:
        large_thr = 1e100
        small_thr = 1e-100
    else:
        large_thr = 1e15
        small_thr = 1e-15

    for n in range(n_start, 0, -1):
        f_nm1 = ((2 * n + 1) / x_safe) * f_n - f_np1
        f_nm1, f_n, f_np1, log_scale = _miller_rescale(
            f_nm1,
            f_n,
            f_np1,
            large_thr=large_thr,
            small_thr=small_thr,
        )
        cur_log_s = cur_log_s + log_scale
        if n - 1 <= lmax:
            f[n - 1] = f_nm1
            log_s[n - 1] = cur_log_s
        f_np1 = f_n
        f_n = f_nm1
    return f, log_s


def spherical_bessel_j(lmax: int, x: torch.Tensor) -> torch.Tensor:
    """Pure-torch spherical Bessel `j_0, ..., j_lmax` via Miller recurrence.

    This logic is adapted from the existing analytic solar prototype under
    `scripts/onedim/solar/test_ohsgp_analytic_solar.py`.
    """

    eps = 1e-8
    x_abs = x.abs()
    mask_nonzero = x_abs > eps
    x_safe = torch.where(mask_nonzero, x_abs, torch.full_like(x_abs, eps))

    max_x = float(x_abs.max().detach().cpu()) if x_abs.numel() > 0 else 0.0
    n_start = int(lmax + max_x + 50)

    f, log_s = _miller_recurrence_python(x_safe, lmax=lmax, n_start=n_start)
    true_j0 = torch.sin(x_safe) / x_safe
    scale0 = true_j0 / (f[0] + 1e-300)
    corr = torch.exp(torch.clamp(log_s - log_s[0].unsqueeze(0), min=-700.0, max=700.0))
    j = f * corr * scale0.unsqueeze(0)

    parity = torch.where(
        (torch.arange(lmax + 1, device=x.device) % 2) == 0,
        torch.ones(lmax + 1, device=x.device, dtype=x.dtype),
        -torch.ones(lmax + 1, device=x.device, dtype=x.dtype),
    ).view(-1, *([1] * x.ndim))
    j = torch.where((x >= 0).unsqueeze(0), j, j * parity)

    j_at_0 = torch.zeros_like(j)
    j_at_0[0] = 1.0
    return torch.where(mask_nonzero.unsqueeze(0), j, j_at_0)


@dataclass
class TemporalBlockSpec:
    """Specification for one temporal block or horizon."""

    start: float
    end: float
    num_discrete_steps: int
    prev_discrete_steps: int = 0
    phase_origin: Optional[float] = None

    @classmethod
    def from_times(
        cls,
        times: torch.Tensor,
        num_discrete_steps: Optional[int] = None,
        prev_discrete_steps: int = 0,
        padding: float = 0.0,
    ) -> "TemporalBlockSpec":
        times = torch.as_tensor(times, dtype=torch.float64).reshape(-1)
        if times.numel() == 0:
            raise ValueError("Cannot build a TemporalBlockSpec from an empty time tensor.")
        start = float(times.min().item()) - padding
        end = float(times.max().item()) + padding
        if math.isclose(start, end):
            end = start + 1.0
        num_steps = int(num_discrete_steps or max(times.numel(), 1))
        return cls(
            start=start,
            end=end,
            num_discrete_steps=num_steps,
            prev_discrete_steps=int(prev_discrete_steps),
            phase_origin=start,
        )


@dataclass
class TemporalAnalyticConfig:
    """Configuration for the analytic temporal builder."""

    inducing_size: int
    rff_sample_size: int = 256
    variance: float = 1.0
    lengthscale: float = 1.0
    globalstart_wt_mode: str = "w"
    phase_origin_mode: str = "global_start"
    num_discrete_steps: Optional[int] = None
    prev_discrete_steps: int = 0
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    jitter: float = 1e-6
    seed: int = 0


class AnalyticTemporalBuilder(nn.Module):
    """Stateful temporal builder using fixed base frequencies and analytic HiPPO features."""

    def __init__(self, config: TemporalAnalyticConfig) -> None:
        super().__init__()
        self.config = config
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(config.seed))
        base_freq = torch.randn(
            1,
            config.rff_sample_size,
            generator=generator,
            dtype=config.dtype,
        )
        self.register_buffer("base_frequencies", base_freq)
        self.log_variance = nn.Parameter(torch.log(torch.as_tensor(config.variance, dtype=config.dtype)))
        self.log_lengthscale = nn.Parameter(torch.log(torch.as_tensor(config.lengthscale, dtype=config.dtype)))

    @property
    def variance(self) -> torch.Tensor:
        return torch.exp(self.log_variance)

    @property
    def lengthscale(self) -> torch.Tensor:
        return torch.exp(self.log_lengthscale)

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(tensor, dtype=self.config.dtype, device=self.base_frequencies.device)

    def current_frequencies(self) -> torch.Tensor:
        return self.base_frequencies / self.lengthscale

    def resolve_horizon(self, times_or_horizon: torch.Tensor | TemporalBlockSpec) -> TemporalBlockSpec:
        if isinstance(times_or_horizon, TemporalBlockSpec):
            return times_or_horizon
        return TemporalBlockSpec.from_times(
            times_or_horizon,
            num_discrete_steps=self.config.num_discrete_steps,
            prev_discrete_steps=self.config.prev_discrete_steps,
        )

    def get_time_stamps(self, horizon: TemporalBlockSpec) -> torch.Tensor:
        start = torch.as_tensor(horizon.start, dtype=self.config.dtype, device=self.base_frequencies.device)
        step = (horizon.end - horizon.start) / max(horizon.num_discrete_steps, 1)
        offsets = torch.arange(
            horizon.num_discrete_steps,
            dtype=self.config.dtype,
            device=self.base_frequencies.device,
        )
        return (start + step * (1.0 + offsets)).unsqueeze(-1)

    def _phase_origin_term(self, w: torch.Tensor, w_eff: torch.Tensor, horizon: TemporalBlockSpec) -> torch.Tensor:
        if self.config.phase_origin_mode != "global_start":
            return torch.zeros_like(w)
        phase_origin = horizon.phase_origin if horizon.phase_origin is not None else horizon.start
        phase_origin = torch.as_tensor(phase_origin, dtype=self.config.dtype, device=w.device)
        if self.config.globalstart_wt_mode == "none":
            return torch.zeros_like(w)
        if self.config.globalstart_wt_mode == "w_eff":
            return w_eff * phase_origin
        return w * phase_origin

    def compute_temporal_basis(self, horizon: TemporalBlockSpec) -> tuple[torch.Tensor, torch.Tensor]:
        w = self.current_frequencies()
        step_size = (horizon.end - horizon.start) / max(horizon.num_discrete_steps, 1)
        w_eff = w * step_size
        t_index = int(horizon.prev_discrete_steps) + int(horizon.num_discrete_steps)
        kappa = 0.5 * w_eff * t_index
        phase_origin_term = self._phase_origin_term(w, w_eff, horizon)

        levels = torch.arange(
            self.config.inducing_size,
            dtype=self.config.dtype,
            device=w.device,
        )
        j = spherical_bessel_j(self.config.inducing_size - 1, kappa).squeeze(1)
        prefactor = torch.sqrt(2.0 * levels + 1.0)[:, None]
        phase = phase_origin_term + kappa + levels[:, None] * math.pi / 2.0
        z_sin = prefactor * j * torch.sin(phase)
        z_cos = prefactor * j * torch.cos(phase)
        scale = (1.0 / self.config.rff_sample_size) ** 0.5
        z = scale * torch.cat([z_sin, z_cos], dim=1).transpose(0, 1)
        return z, self.get_time_stamps(horizon)

    def compute_feature_matrix(self, query_times: torch.Tensor) -> torch.Tensor:
        query_times = self._to_device(query_times).reshape(-1, 1)
        w = self.current_frequencies()
        scale = (1.0 / self.config.rff_sample_size) ** 0.5
        return scale * torch.cat([torch.sin(w * query_times), torch.cos(w * query_times)], dim=1)

    def compute_kuu_t(self, times_or_horizon: torch.Tensor | TemporalBlockSpec) -> torch.Tensor:
        horizon = self.resolve_horizon(times_or_horizon)
        z, _ = self.compute_temporal_basis(horizon)
        return self.variance * (z.transpose(0, 1) @ z)

    def compute_kuu_t_cross(
        self,
        left_horizon: torch.Tensor | TemporalBlockSpec,
        right_horizon: torch.Tensor | TemporalBlockSpec,
    ) -> torch.Tensor:
        """Cross-covariance between two temporal inducing systems.

        When two blocks use different local horizons, both inducing systems are
        still linear functionals of the same underlying RFF-expanded temporal GP.
        Their cross-covariance is therefore given by the inner product of the two
        temporal basis matrices in that shared feature space.
        """

        left = self.resolve_horizon(left_horizon)
        right = self.resolve_horizon(right_horizon)
        z_left, _ = self.compute_temporal_basis(left)
        z_right, _ = self.compute_temporal_basis(right)
        return self.variance * (z_left.transpose(0, 1) @ z_right)

    def compute_kfu_t(
        self,
        query_times: torch.Tensor,
        horizon: torch.Tensor | TemporalBlockSpec,
    ) -> torch.Tensor:
        horizon = self.resolve_horizon(horizon)
        z, _ = self.compute_temporal_basis(horizon)
        z_f = self.compute_feature_matrix(query_times)
        return self.variance * (z_f @ z)

    def compute_ktt_diag(self, query_times: torch.Tensor) -> torch.Tensor:
        query_times = self._to_device(query_times).reshape(-1)
        return self.variance.expand(query_times.shape[0])


def compute_kuu_t(
    times_or_horizon: torch.Tensor | TemporalBlockSpec,
    config: TemporalAnalyticConfig,
) -> torch.Tensor:
    """Convenience wrapper for the requested Stage 1 interface."""
    builder = AnalyticTemporalBuilder(config)
    return builder.compute_kuu_t(times_or_horizon)


def compute_kfu_t(
    query_times: torch.Tensor,
    horizon: torch.Tensor | TemporalBlockSpec,
    config: TemporalAnalyticConfig,
) -> torch.Tensor:
    """Convenience wrapper for the requested Stage 1 interface."""
    builder = AnalyticTemporalBuilder(config)
    return builder.compute_kfu_t(query_times, horizon)


def compute_kuu_t_cross(
    left_horizon: torch.Tensor | TemporalBlockSpec,
    right_horizon: torch.Tensor | TemporalBlockSpec,
    config: TemporalAnalyticConfig,
) -> torch.Tensor:
    """Convenience wrapper for temporal inducing cross-covariances."""

    builder = AnalyticTemporalBuilder(config)
    return builder.compute_kuu_t_cross(left_horizon, right_horizon)

"""Spatial kernels for the Kronecker spatio-temporal prototype."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SpatialKernelConfig:
    """Configuration for the spatial kernel."""

    input_dim: int
    kernel_type: str = "rbf"
    variance: float = 1.0
    lengthscale: float | list[float] = 1.0
    matern_nu: float = 1.5


class BaseSpatialKernel(nn.Module):
    """Common spatial kernel interface."""

    def __init__(self, input_dim: int, variance: float, lengthscale: float | list[float]) -> None:
        super().__init__()
        lengthscale_tensor = torch.as_tensor(lengthscale, dtype=torch.float64)
        if lengthscale_tensor.ndim == 0:
            lengthscale_tensor = lengthscale_tensor.repeat(input_dim)
        self.input_dim = input_dim
        self.log_variance = nn.Parameter(torch.log(torch.as_tensor(float(variance), dtype=torch.float64)))
        self.log_lengthscale = nn.Parameter(torch.log(lengthscale_tensor.to(dtype=torch.float64)))

    @property
    def variance(self) -> torch.Tensor:
        return torch.exp(self.log_variance)

    @property
    def lengthscale(self) -> torch.Tensor:
        return torch.exp(self.log_lengthscale)

    def _scaled_inputs(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = x1 / self.lengthscale
        x2 = x2 / self.lengthscale
        return x1, x2

    def diag(self, x: torch.Tensor) -> torch.Tensor:
        return self.variance.expand(x.shape[0])

    def compute_kzz_s(self, z_s: torch.Tensor) -> torch.Tensor:
        return self.forward(z_s, z_s)

    def compute_kxz_s(self, x_s: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:
        return self.forward(x_s, z_s)


class RBFKernel(BaseSpatialKernel):
    """ARD RBF kernel."""

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1, x2 = self._scaled_inputs(x1, x2)
        sq_dist = (
            torch.sum(x1**2, dim=-1, keepdim=True)
            + torch.sum(x2**2, dim=-1).unsqueeze(0)
            - 2.0 * (x1 @ x2.transpose(-1, -2))
        )
        sq_dist = torch.clamp(sq_dist, min=0.0)
        return self.variance * torch.exp(-0.5 * sq_dist)


class MaternKernel(BaseSpatialKernel):
    """Matern kernel with `nu` in `{1.5, 2.5}`."""

    def __init__(
        self,
        input_dim: int,
        variance: float = 1.0,
        lengthscale: float | list[float] = 1.0,
        nu: float = 1.5,
    ) -> None:
        super().__init__(input_dim=input_dim, variance=variance, lengthscale=lengthscale)
        if nu not in {1.5, 2.5}:
            raise ValueError("Only Matern-1.5 and Matern-2.5 are implemented in Stage 1.")
        self.nu = nu

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1, x2 = self._scaled_inputs(x1, x2)
        sq_dist = (
            torch.sum(x1**2, dim=-1, keepdim=True)
            + torch.sum(x2**2, dim=-1).unsqueeze(0)
            - 2.0 * (x1 @ x2.transpose(-1, -2))
        )
        sq_dist = torch.clamp(sq_dist, min=0.0)
        dist = torch.sqrt(sq_dist + 1e-12)
        if self.nu == 1.5:
            scaled = math.sqrt(3.0) * dist
            return self.variance * (1.0 + scaled) * torch.exp(-scaled)
        scaled = math.sqrt(5.0) * dist
        return self.variance * (1.0 + scaled + (5.0 / 3.0) * sq_dist) * torch.exp(-scaled)


def build_spatial_kernel(config: SpatialKernelConfig) -> BaseSpatialKernel:
    """Factory for supported spatial kernels."""
    kernel_type = config.kernel_type.lower()
    if kernel_type == "rbf":
        return RBFKernel(
            input_dim=config.input_dim,
            variance=config.variance,
            lengthscale=config.lengthscale,
        )
    if kernel_type == "matern":
        return MaternKernel(
            input_dim=config.input_dim,
            variance=config.variance,
            lengthscale=config.lengthscale,
            nu=config.matern_nu,
        )
    raise ValueError(f"Unsupported spatial kernel type: {config.kernel_type}")

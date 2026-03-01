"""Frequency-space parameterised kernel.

Spectral density: S(w) = sigma^2 * sum_i gamma_i * delta(w - omega_i)
Time domain:      k(tau) = sigma^2 * sum_i gamma_i * cos(omega_i * tau)
"""

import math

import torch
import torch.nn as nn

from hipposvgp.kernels.base import Kernel
from hipposvgp.rff import SpectralSample


class FrequencySpace(Kernel):
    """Kernel with learnable frequencies and weights in spectral domain."""

    def __init__(self, num_components, device="cpu", dtype=torch.float64):
        super().__init__()
        self.num_components = num_components
        self._dtype = dtype

        self.omega = nn.Parameter(
            torch.randn(num_components, dtype=dtype, device=device)
        )
        self.log_gamma_raw = nn.Parameter(
            torch.full(
                (num_components,),
                math.log(1.0 / num_components),
                dtype=dtype,
                device=device,
            )
        )
        self.log_sf = nn.Parameter(torch.tensor(0.0, dtype=dtype, device=device))

    @property
    def gamma(self):
        """Normalized spectral weights (sum to 1)."""
        return torch.softmax(self.log_gamma_raw, dim=0)

    def sample_from_spectral(self, num_rff=None):
        """Return learned frequencies and weights as SpectralSample."""
        return SpectralSample(
            frequencies=self.omega.unsqueeze(0),
            scale=torch.sqrt(self.gamma),
        )

    def forward(self, X1, X2=None, diag=False):
        if X2 is None:
            X2 = X1

        omega = self.omega
        gamma = self.gamma
        sf = torch.exp(self.log_sf)

        if diag:
            if X1.dim() == 2:
                return sf * torch.ones(X1.shape[0], dtype=self._dtype, device=self.device)
            return sf * torch.ones(X1.shape[:-1], dtype=self._dtype, device=self.device)

        if X1.dim() == 3:
            diff = (X1.unsqueeze(2) - X2.unsqueeze(1)).sum(-1)
            cos_terms = torch.cos(omega.view(1, 1, 1, -1) * diff.unsqueeze(-1))
            return sf * (cos_terms * gamma.view(1, 1, 1, -1)).sum(-1)
        else:
            diff = (X1.unsqueeze(1) - X2.unsqueeze(0)).sum(-1)
            cos_terms = torch.cos(omega.view(1, 1, -1) * diff.unsqueeze(-1))
            return sf * (cos_terms * gamma.view(1, 1, -1)).sum(-1)

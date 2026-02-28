"""Abstract base class for kernels."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Kernel(nn.Module, ABC):
    """Base class for all kernels.

    Subclasses must implement ``forward`` and ``sample_from_spectral``.
    """

    def __init__(self):
        super().__init__()

    @property
    def device(self):
        """Infer device from parameters."""
        return next(self.parameters()).device

    @property
    def signal_variance(self):
        """Return exp(log_sf) if the kernel defines log_sf."""
        return torch.exp(self.log_sf)

    @abstractmethod
    def forward(self, X1, X2=None, diag=False):
        """Compute kernel matrix k(X1, X2).

        Args:
            X1: [N, D] or [B, N, D]
            X2: [M, D] or [B, M, D], defaults to X1
            diag: if True, return only diagonal elements

        Returns:
            Kernel matrix [N, M] or diagonal [N].
        """

    def get_weights(self):
        """RFF component weights. Override for weighted kernels (e.g., FreqSpace)."""
        return None

    @abstractmethod
    def sample_from_spectral(self, num_rff):
        """Sample from the spectral density of this kernel.

        Args:
            num_rff: number of random Fourier features

        Returns:
            Frequency samples tensor.
        """

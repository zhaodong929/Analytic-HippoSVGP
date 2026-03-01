"""Random Fourier Feature (RFF) utilities."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SpectralSample:
    """Spectral representation of a kernel: frequencies + per-component scale."""

    frequencies: torch.Tensor  # [1, R]
    scale: torch.Tensor  # [R]

    @property
    def num_rff(self):
        return self.frequencies.shape[-1]


def rff_features(sample, X):
    """Compute RFF features: phi(X) = scale * [sin(wX), cos(wX)].

    Args:
        sample: SpectralSample with frequencies [1, R] and scale [R] or scalar
        X: [N, D] data points

    Returns:
        Z_f: [N, 2R] RFF features
    """
    w = sample.frequencies
    scale = sample.scale

    sin_term = torch.sin(w * X)  # [N, R]
    cos_term = torch.cos(w * X)  # [N, R]

    if isinstance(scale, torch.Tensor) and scale.dim() > 0:
        return torch.cat([scale * sin_term, scale * cos_term], dim=1)
    else:
        return scale * torch.cat([sin_term, cos_term], dim=1)

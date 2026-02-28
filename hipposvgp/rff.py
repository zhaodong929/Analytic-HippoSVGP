"""Random Fourier Feature (RFF) utilities."""

import torch


def rff_features(w, X, scale=None):
    """Compute RFF features: phi(X) = scale * [sin(wX), cos(wX)].

    Args:
        w: [1, R] or [R] frequency tensor
        X: [N, D] data points
        scale: scalar or [R] tensor. If None, uses (1/R)^0.5.

    Returns:
        Z_f: [N, 2R] RFF features
    """
    num_rff = w.shape[-1]
    if scale is None:
        scale = (1.0 / num_rff) ** 0.5

    sin_term = torch.sin(w * X)  # [N, R]
    cos_term = torch.cos(w * X)  # [N, R]

    if isinstance(scale, torch.Tensor) and scale.dim() > 0:
        return torch.cat([scale * sin_term, scale * cos_term], dim=1)
    else:
        return scale * torch.cat([sin_term, cos_term], dim=1)

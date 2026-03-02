"""Random Fourier Feature (RFF) utilities."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from hipposvgp.types import SpectralSample


def rff_features(
    sample: SpectralSample,
    X: Float[Array, "N D"],
) -> Float[Array, "N TwoR"]:
    """Compute RFF features: phi(X) = scale * [sin(wX), cos(wX)].

    Args:
        sample: SpectralSample with frequencies [1, R] and scale [R]
        X: [N, D] data points

    Returns:
        Z_f: [N, 2R] RFF features
    """
    w = sample.frequencies  # [1, R]
    scale = sample.scale  # [R]

    sin_term = jnp.sin(w * X)  # [N, R]
    cos_term = jnp.cos(w * X)  # [N, R]

    return jnp.concatenate([scale * sin_term, scale * cos_term], axis=1)

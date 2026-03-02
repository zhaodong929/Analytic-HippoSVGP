"""Core data types for hipposvgp."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class SpectralSample(eqx.Module):
    """Spectral representation of a kernel: frequencies + per-component scale."""

    frequencies: Float[Array, "1 R"]
    scale: Float[Array, "R"]

    @property
    def num_rff(self) -> int:
        return self.frequencies.shape[-1]


class KernelMatrices(eqx.Module):
    """Container for all kernel matrices needed by the model."""

    Kuu: Float[Array, "M M"]
    Kfu: Float[Array, "N M"]
    Kff: Float[Array, "N N"]
    Kuu_old: Float[Array, "M Mp"] | None = None
    Kold_old: Float[Array, "Mp Mp"] | None = None
    Kold_old_newk: Float[Array, "Mp Mp"] | None = None
    Z: Float[Array, "TwoR M"] | None = None


class PosteriorState(eqx.Module):
    """Transfer variational posterior between tasks (SVGP)."""

    mv: Float[Array, "M"]
    Lv: Float[Array, "M M"]
    Kaa: Float[Array, "M M"]
    Z: Float[Array, "TwoR M"]


class SGPRPosteriorState(eqx.Module):
    """Transfer posterior state between tasks (SGPR)."""

    mu: Float[Array, "M 1"]
    Su: Float[Array, "M M"]
    Kaa: Float[Array, "M M"]
    Z: Float[Array, "TwoR M"]

"""Abstract base class for kernels."""

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from hipposvgp.types import SpectralSample


class Kernel(eqx.Module):
    """Base class for all kernels.

    Subclasses must implement ``__call__`` and ``sample_from_spectral``.
    """

    log_sf: Float[Array, "..."]

    @property
    def signal_variance(self) -> Float[Array, ""]:
        return jnp.exp(self.log_sf).squeeze()

    @abc.abstractmethod
    def __call__(
        self,
        X1: Float[Array, "N D"],
        X2: Float[Array, "M D"] | None = None,
        *,
        diag: bool = False,
    ) -> Float[Array, "..."]:
        """Compute kernel matrix k(X1, X2)."""
        ...

    @abc.abstractmethod
    def sample_from_spectral(
        self,
        num_rff: int,
        key: PRNGKeyArray,
    ) -> SpectralSample:
        """Sample from the spectral density of this kernel."""
        ...

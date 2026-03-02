"""Frequency-space parameterised kernel.

Spectral density: S(w) = sigma^2 * sum_i gamma_i * delta(w - omega_i)
Time domain:      k(tau) = sigma^2 * sum_i gamma_i * cos(omega_i * tau)
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from hipposvgp.kernels.base import Kernel
from hipposvgp.types import SpectralSample


class FrequencySpace(Kernel):
    """Kernel with learnable frequencies and weights in spectral domain."""

    log_sf: Float[Array, ""]
    omega: Float[Array, "C"]
    log_gamma_raw: Float[Array, "C"]

    def __init__(self, num_components: int):
        self.omega = jnp.zeros(num_components)
        self.log_gamma_raw = jnp.full(
            num_components, math.log(1.0 / num_components)
        )
        self.log_sf = jnp.array(0.0)

    @property
    def gamma(self) -> Float[Array, "C"]:
        """Normalized spectral weights (sum to 1)."""
        return jax.nn.softmax(self.log_gamma_raw)

    def sample_from_spectral(
        self, num_rff: int | None = None, key: PRNGKeyArray | None = None
    ) -> SpectralSample:
        """Return learned frequencies and weights as SpectralSample."""
        return SpectralSample(
            frequencies=self.omega[None, :],
            scale=jnp.sqrt(self.gamma),
        )

    def __call__(
        self,
        X1: Float[Array, "N D"],
        X2: Float[Array, "M D"] | None = None,
        *,
        diag: bool = False,
    ) -> Float[Array, "..."]:
        if X2 is None:
            X2 = X1
        omega = self.omega
        gamma = self.gamma
        sf = jnp.exp(self.log_sf)

        if diag:
            return sf * jnp.ones(X1.shape[0])

        # diff: [N, M] pairwise differences (sum over D)
        diff = (X1[:, None, :] - X2[None, :, :]).sum(axis=-1)
        # cos_terms: [N, M, C]
        cos_terms = jnp.cos(omega[None, None, :] * diff[:, :, None])
        return sf * (cos_terms * gamma[None, None, :]).sum(axis=-1)

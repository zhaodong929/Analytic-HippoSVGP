"""Squared Exponential (RBF) kernel."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from hipposvgp.kernels.base import Kernel
from hipposvgp.types import SpectralSample


class SquaredExponential(Kernel):
    """k(x1, x2) = sf * exp(-0.5 * sum_d ((x1_d - x2_d) / ls_d)^2)"""

    log_sf: Float[Array, "1"]
    log_ls: Float[Array, "D"]

    def __init__(self, input_dim: int):
        self.log_sf = jnp.zeros(1)
        self.log_ls = jnp.zeros(input_dim)

    def __call__(
        self,
        X1: Float[Array, "N D"],
        X2: Float[Array, "M D"] | None = None,
        *,
        diag: bool = False,
    ) -> Float[Array, "..."]:
        if X2 is None:
            X2 = X1
        X1_s = X1 * jnp.exp(-self.log_ls)
        X2_s = X2 * jnp.exp(-self.log_ls)

        if diag:
            sq_dist = jnp.sum((X1_s - X2_s) ** 2, axis=-1)
            return jnp.exp(self.log_sf) * jnp.exp(-0.5 * sq_dist)

        f1 = jnp.sum(X1_s**2, axis=-1)  # [N]
        f2 = jnp.sum(X2_s**2, axis=-1)  # [M]
        sq_dist = f1[:, None] + f2[None, :] - 2 * X1_s @ X2_s.T
        return jnp.exp(self.log_sf) * jnp.exp(-0.5 * sq_dist)

    def sample_from_spectral(
        self, num_rff: int, key: PRNGKeyArray
    ) -> SpectralSample:
        w = jax.random.normal(key, (num_rff,)) / jnp.exp(self.log_ls).squeeze()
        scale = jnp.full(num_rff, (1.0 / num_rff) ** 0.5)
        return SpectralSample(frequencies=w[None, :], scale=scale)

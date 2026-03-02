"""Spectral Mixture Kernel."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from hipposvgp.kernels.base import Kernel
from hipposvgp.types import SpectralSample


class SpectralMixture(Kernel):
    """k(tau) = sum_q w_q exp(-2 pi^2 tau^2 v_q) cos(2 pi mu_q tau)"""

    mu_smk: Float[Array, "Q 1 1"]
    log_var_smk: Float[Array, "Q 1 1"]
    w_smk_raw: Float[Array, "Q"]
    log_sf: Float[Array, "1"]

    def __init__(self, num_mixture: int, *, key: PRNGKeyArray):
        k1, k2, k3 = jax.random.split(key, 3)
        self.mu_smk = 0.1 * jax.random.normal(k1, (num_mixture, 1, 1))
        self.log_var_smk = -10.0 + jax.random.normal(k2, (num_mixture, 1, 1))
        self.w_smk_raw = jax.random.normal(k3, (num_mixture,))
        self.log_sf = jnp.zeros(1)

    def __call__(
        self,
        X1: Float[Array, "N D"],
        X2: Float[Array, "M D"] | None = None,
        *,
        diag: bool = False,
    ) -> Float[Array, "..."]:
        if X2 is None:
            X2 = X1
        two_pi_sq = 2 * math.pi**2
        two_pi = 2.0 * math.pi
        weights = jax.nn.softmax(self.w_smk_raw)  # [Q]

        diff = X1[:, None, :] - X2[None, :, :]  # [N, M, D]
        diff_sq = (diff**2).sum(axis=-1)  # [N, M]
        diff_sum = diff.sum(axis=-1)  # [N, M]

        mu = self.mu_smk.squeeze()  # [Q]
        var = jnp.exp(self.log_var_smk.squeeze())  # [Q]

        # term1: [Q, N, M]
        term1 = jnp.exp(
            -two_pi_sq * diff_sq[None, :, :] * var[:, None, None]
        )
        # term2: [Q, N, M]
        term2 = jnp.cos(two_pi * diff_sum[None, :, :] * mu[:, None, None])

        result = jnp.sum(weights[:, None, None] * term1 * term2, axis=0)

        if diag:
            return jnp.diag(result)
        return result

    def sample_from_spectral(
        self, num_rff: int, key: PRNGKeyArray
    ) -> SpectralSample:
        weights = jax.nn.softmax(self.w_smk_raw)  # [Q]
        Q = self.mu_smk.shape[0]
        noise = jax.random.normal(key, (Q, num_rff))
        mu = self.mu_smk.squeeze()  # [Q]
        std = jnp.sqrt(jnp.exp(self.log_var_smk.squeeze()))  # [Q]
        S = mu[:, None] + std[:, None] * noise  # [Q, num_rff]
        S_flat = S.reshape(-1)  # [Q * num_rff]
        scale = jnp.sqrt(jnp.repeat(weights, num_rff) / num_rff)
        return SpectralSample(frequencies=S_flat[None, :], scale=scale)

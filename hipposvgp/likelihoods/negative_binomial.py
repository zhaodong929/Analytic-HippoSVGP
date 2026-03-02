"""Negative Binomial likelihood."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from hipposvgp.likelihoods.base import Likelihood


class NegativeBinomialLikelihood(Likelihood):
    """Negative Binomial likelihood with learnable dispersion.

    Uses softplus link function: mu = softplus(f).
    Dispersion parameter nu in (0, 1) via sigmoid.
    """

    _nu: Float[Array, ""]

    def __init__(self, initial_nu: float = 0.5):
        self._nu = jnp.array(initial_nu)

    @property
    def nu(self) -> Float[Array, ""]:
        return jax.nn.sigmoid(self._nu)

    @staticmethod
    def link_fn(f: Float[Array, "..."]) -> Float[Array, "..."]:
        return jax.nn.softplus(f)

    def log_prob(
        self, logmu: Float[Array, "N 1"], y: Float[Array, "N 1"]
    ) -> Float[Array, "N 1"]:
        mu = self.link_fn(logmu)
        nu = self.nu
        alpha = mu * (1 - nu) / nu
        log_gamma_term = (
            jax.lax.lgamma(y + alpha)
            - jax.lax.lgamma(alpha)
            - jax.lax.lgamma(y + 1)
        )
        return log_gamma_term + y * jnp.log(nu) + alpha * jnp.log(1 - nu)

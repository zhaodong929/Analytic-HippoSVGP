"""Gaussian likelihood."""

from __future__ import annotations

import math

import jax.numpy as jnp
from jaxtyping import Array, Float

from hipposvgp.likelihoods.base import Likelihood


class GaussianLikelihood(Likelihood):
    """Gaussian likelihood with learnable noise variance."""

    log_variance: Float[Array, ""]

    def __init__(self, variance: float = 1.0):
        self.log_variance = jnp.log(jnp.array(variance))

    @property
    def variance(self) -> Float[Array, ""]:
        return jnp.exp(self.log_variance)

    def log_prob(
        self, f: Float[Array, "N 1"], y: Float[Array, "N 1"]
    ) -> Float[Array, "N 1"]:
        return -0.5 * (
            jnp.log(2 * math.pi * self.variance)
            + ((y - f) ** 2) / self.variance
        )

    def variational_expectations(
        self,
        f_mean: Float[Array, "N 1"],
        f_var: Float[Array, "N 1"],
        y: Float[Array, "N 1"],
    ) -> Float[Array, "N 1"]:
        """E_q[log p(y|f)] in closed form for Gaussian likelihood."""
        return -0.5 * (
            jnp.log(2 * math.pi * self.variance)
            + ((y - f_mean) ** 2 + f_var) / self.variance
        )

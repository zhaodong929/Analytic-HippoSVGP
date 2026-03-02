"""Bernoulli likelihood."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from hipposvgp.likelihoods.base import Likelihood


class BernoulliLikelihood(Likelihood):
    """Bernoulli likelihood with sigmoid link function."""

    JITTER: float = 1e-9

    def log_prob(
        self, f: Float[Array, "N 1"], y: Float[Array, "N 1"]
    ) -> Float[Array, "N 1"]:
        mu = jax.nn.sigmoid(f)
        return y * jnp.log(mu + self.JITTER) + (1 - y) * jnp.log(
            1 - mu + self.JITTER
        )

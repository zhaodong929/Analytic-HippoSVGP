"""Abstract base class for likelihoods."""

from __future__ import annotations

import abc

import equinox as eqx
from jaxtyping import Array, Float


class Likelihood(eqx.Module):
    """Base class for all likelihoods."""

    @abc.abstractmethod
    def log_prob(
        self, f: Float[Array, "N 1"], y: Float[Array, "N 1"]
    ) -> Float[Array, "N 1"]:
        """Compute log p(y | f)."""
        ...

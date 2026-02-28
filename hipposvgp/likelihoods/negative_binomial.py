"""Negative Binomial likelihood."""

import torch
import torch.nn as nn
from torch.distributions import NegativeBinomial as NegativeBinomial_

from hipposvgp.likelihoods.base import Likelihood


class NegativeBinomialLikelihood(Likelihood):
    """Negative Binomial likelihood with learnable dispersion.

    Uses softplus link function: mu = softplus(f).
    Dispersion parameter nu in (0, 1) via sigmoid.
    """

    def __init__(self, initial_nu=0.5):
        super().__init__()
        self._nu = nn.Parameter(torch.tensor(initial_nu, dtype=torch.float32))

    @property
    def nu(self):
        return torch.sigmoid(self._nu)

    @staticmethod
    def link_fn(f):
        return torch.nn.functional.softplus(f)

    def log_prob(self, logmu, y):
        mu = self.link_fn(logmu)
        nu = self.nu
        alpha = mu * (1 - nu) / nu

        log_gamma_term = (
            torch.special.gammaln(y + alpha)
            - torch.special.gammaln(alpha)
            - torch.special.gammaln(y + 1)
        )
        return log_gamma_term + y * torch.log(nu) + alpha * torch.log(1 - nu)

    def sample(self, logmu, num_samples=30):
        """Vectorized sampling from the negative binomial distribution."""
        mu = self.link_fn(logmu)  # [S, N]
        nu = self.nu
        alpha = mu * (1 - nu) / nu  # [S, N]
        dist = NegativeBinomial_(total_count=alpha, probs=nu)
        return dist.sample(sample_shape=(num_samples,))  # [num_samples, S, N]

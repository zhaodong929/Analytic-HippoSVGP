"""Bernoulli likelihood."""

import torch

from hipposvgp.likelihoods.base import Likelihood


class BernoulliLikelihood(Likelihood):
    """Bernoulli likelihood with sigmoid link function."""

    JITTER = 1e-9

    def log_prob(self, f, y):
        mu = torch.sigmoid(f)
        return y * torch.log(mu + self.JITTER) + (1 - y) * torch.log(1 - mu + self.JITTER)

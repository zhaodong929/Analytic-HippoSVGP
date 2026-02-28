"""Gaussian likelihood."""

import math

import torch
import torch.nn as nn

from hipposvgp.likelihoods.base import Likelihood


class GaussianLikelihood(Likelihood):
    """Gaussian likelihood with learnable noise variance."""

    def __init__(self, variance=1.0):
        super().__init__()
        self.log_variance = nn.Parameter(torch.log(torch.tensor(variance)))

    @property
    def variance(self):
        return torch.exp(self.log_variance)

    def log_prob(self, f, y):
        return -0.5 * (
            torch.log(2 * math.pi * self.variance)
            + ((y - f) ** 2) / self.variance
        )

    def variational_expectations(self, f_mean, f_var, y):
        """E_q[log p(y|f)] in closed form for Gaussian likelihood."""
        return -0.5 * (
            torch.log(2 * math.pi * self.variance)
            + ((y - f_mean) ** 2 + f_var) / self.variance
        )

    def conditional_moments(self, f):
        mean_y = f
        var_y = torch.ones_like(f) * self.variance
        return mean_y, var_y

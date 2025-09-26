import torch
import math
import torch.nn as nn
from torch.distributions import NegativeBinomial as NegativeBinomial_
import numpy as np

class Bernoulli(nn.Module):
    def __init__(self):
        super().__init__()

    def link_fn(self, mean):
        # Define your link function here
        return torch.sigmoid(mean)

    def log_prob(self, pred, y):
        jitter = 1e-9
        # Compute mu using the link function
        mu = self.link_fn(pred)
        # Compute the negative binomial log-likelihood
        likelihood = y* torch.log(mu + jitter) + (1 - y)* torch.log(1 - mu + jitter)
        return likelihood

    def forward(self, f, y):
        y = y.double()
        f = f.double()
        pred = self.link_fn(f)
        # Return the negative binomial likelihood
        loglik = self.log_prob(pred, y)
        return loglik.float()

class GaussianLikelihood(torch.nn.Module):
    """
    Gaussian likelihood for SVGP with diagonal noise.
    """

    def __init__(self, variance=1.0):
        """
        Initialize the Gaussian likelihood.

        Args:
            variance (float): Initial observation noise variance
        """
        super().__init__()
        self.log_variance = nn.Parameter(torch.log(torch.tensor(variance)))

    @property
    def variance(self):
        """Return the observation noise variance."""
        return torch.exp(self.log_variance)

    def forward(self, f_mean):
        """
        Compute predictive mean and variance.

        Args:
            f_mean: Mean of the latent GP at prediction locations

        Returns:
            tuple: (predictive_mean, predictive_variance)
        """
        return f_mean, self.variance

    def log_prob(self, F, Y):
        """
        Compute log likelihood: log p(Y|F).

        Args:
            F: Function values (mean, variance)
            Y: Observed values

        Returns:
            torch.Tensor: Log likelihood
        """
        # Compute log likelihood using Gaussian PDF
        return -0.5 * (
            torch.log(2 * math.pi * self.variance) + ((Y - F) ** 2) / self.variance
        )

    def variational_expectations(self, f_mean, f_var, Y):
        """
        Compute E_q[log p(Y|F)].

        Args:
            f_mean: Variational mean
            f_var: Variational variance
            Y: Observed values

        Returns:
            torch.Tensor: Expected log likelihood
        """
        return -0.5 * (
            torch.log(2 * math.pi * self.variance)
            + ((Y - f_mean) ** 2 + f_var) / self.variance
        )
    def conditional_moments(self, F):
        mean_y = F 
        # var_y = self.variance but as shape of F
        var_y = torch.ones_like(F) * self.variance
        return mean_y, var_y
    
class NegativeBinomial(nn.Module):
    def __init__(self, initial_nu=0.5):
        super().__init__()
        # Initialize nu as a trainable parameter
        self._nu = nn.Parameter(torch.tensor(initial_nu, dtype=torch.float32))

    @property
    def nu(self):
        # Ensure that nu is between 0 and 1
        # return torch.nn.functional.softplus(self._nu)
        return torch.sigmoid(self._nu)

    def link_fn(self, mean):
        # Define your link function here
        return torch.nn.functional.softplus(mean)
        # return mean.exp()

    def log_prob(self, logmu, y):#, logmu):
        # y = y.unsqueeze(-1)

        # Compute mu using the link function
        mu = self.link_fn(logmu)
        nu = self.nu
        alpha = mu * (1 - nu) / nu

        # Compute the log-gamma terms
        log_gamma_term = (
            torch.special.gammaln(y + alpha)
            - torch.special.gammaln(alpha)
            - torch.special.gammaln(y + 1)
        )

        # Compute the negative binomial log-likelihood
        likelihood = log_gamma_term + y * torch.log(nu) + alpha * torch.log(1 - nu)

        return likelihood

    def sample(self, logmu, numy_perf=30):
        mu = self.link_fn(logmu) # [sample_size, N]
        nu = self.nu # probs
        alpha = mu * (1 - nu) / nu # total count [sample_size, N]
        sample_res = []
        sample_size, N = mu.shape[0], mu.shape[1]
        for i in range(sample_size):
            sample_res.append([])
            for j in range(N):
                dist = NegativeBinomial_(total_count=alpha[i][j], probs=nu)
                sample_res[i].append(dist.sample(sample_shape=(numy_perf,)).cpu())
        return torch.tensor(np.array(sample_res))

    def forward(self, y, f):
        y = y.double()
        f = f.double()
        logmu = f.log()
        # Return the negative binomial likelihood
        loglik = self.negbin_likelihood(y, logmu)
        return loglik.float()
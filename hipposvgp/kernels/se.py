"""Squared Exponential (RBF) kernel."""

import torch
import torch.nn as nn

from hipposvgp.kernels.base import Kernel


class SquaredExponential(Kernel):
    """Squared exponential kernel with ARD lengthscales.

    k(x1, x2) = sf * exp(-0.5 * sum_d ((x1_d - x2_d) / ls_d)^2)
    """

    def __init__(self, input_dim, device="cpu"):
        super().__init__()
        self.input_dim = input_dim
        self.log_sf = nn.Parameter(
            torch.zeros(1, dtype=torch.float64, device=device)
        )
        self.log_ls = nn.Parameter(
            torch.zeros(self.input_dim, dtype=torch.float64, device=device)
        )

    def forward(self, X1, X2=None, diag=False):
        if X2 is None:
            X2 = X1

        X1_s = X1 * torch.exp(-self.log_ls)
        X2_s = X2 * torch.exp(-self.log_ls)

        if diag:
            sq_dist = torch.sum((X1_s - X2_s).pow(2), -1)
            return torch.exp(self.log_sf) * torch.exp(-0.5 * sq_dist)

        if X1_s.dim() == 3:
            f1 = torch.sum(X1_s.pow(2), -1, keepdim=True)
            f2 = torch.sum(X2_s.pow(2), -1, keepdim=True)
            cross = X1_s @ X2_s.transpose(-1, -2)
            return torch.exp(self.log_sf) * torch.exp(
                -0.5 * (f1 + f2.transpose(-1, -2) - 2 * cross)
            )
        else:
            f1 = torch.sum(X1_s.pow(2), -1)
            f2 = torch.sum(X2_s.pow(2), -1)
            return torch.exp(self.log_sf) * torch.exp(
                -0.5 * (f1.unsqueeze(1) + f2.unsqueeze(0) - 2 * X1_s @ X2_s.T)
            )

    def sample_from_spectral(self, num_rff):
        return (1.0 / torch.exp(self.log_ls)) * torch.randn(
            num_rff, device=self.device
        )

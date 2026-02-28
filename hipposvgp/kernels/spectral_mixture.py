"""Spectral Mixture Kernel."""

import math

import torch
import torch.nn as nn

from hipposvgp.kernels.base import Kernel


class SpectralMixture(Kernel):
    """Spectral mixture kernel.

    k(tau) = sum_q w_q exp(-2 pi^2 tau^2 v_q) cos(2 pi mu_q tau)
    """

    def __init__(self, num_mixture, device="cpu"):
        super().__init__()
        self.num_mixture = num_mixture
        self.mu_smk = nn.Parameter(
            0.1 * torch.randn(self.num_mixture, 1, 1, device=device)
        )
        self.log_var_smk = nn.Parameter(
            -10.0 + 1.0 * torch.randn(self.num_mixture, 1, 1, device=device)
        )
        self.w_smk_raw = nn.Parameter(torch.randn(self.num_mixture, device=device))
        # SMK does not have a single log_sf; provide a dummy for compatibility
        self.log_sf = nn.Parameter(
            torch.tensor(0.0, device=device), requires_grad=False
        )

    def forward(self, X1, X2=None, diag=False):
        if X2 is None:
            X2 = X1

        two_pi_sq = 2 * math.pi ** 2
        two_pi = 2.0 * math.pi

        if X1.dim() == 3:
            diff = X1.unsqueeze(2) - X2.unsqueeze(1)
            diff_sq = diff.pow(2).sum(-1)
            diff = diff.sum(-1)
            term1 = torch.exp(
                -two_pi_sq
                * diff_sq.unsqueeze(1) * torch.exp(self.log_var_smk).unsqueeze(0)
            )
            term2 = torch.cos(
                two_pi
                * diff.unsqueeze(1) * self.mu_smk.unsqueeze(0)
            )
            return torch.sum(
                torch.softmax(self.w_smk_raw, 0).view(1, -1, 1, 1) * term1 * term2,
                dim=1,
            )
        else:
            diff = X1.unsqueeze(1) - X2.unsqueeze(0)
            diff_sq = diff.pow(2).sum(-1)
            diff = diff.sum(-1)
            term1 = torch.exp(
                -two_pi_sq
                * diff_sq.unsqueeze(0) * torch.exp(self.log_var_smk)
            )
            term2 = torch.cos(
                two_pi
                * diff.unsqueeze(0) * self.mu_smk
            )
            return torch.sum(
                torch.softmax(self.w_smk_raw, 0).view(-1, 1, 1) * term1 * term2,
                dim=0,
            )

    def sample_from_spectral(self, num_rff):
        S = self.mu_smk.view(self.num_mixture, 1) + torch.sqrt(
            torch.exp(self.log_var_smk)
        ).view(self.num_mixture, 1) * torch.randn(
            self.num_mixture, num_rff, device=self.device
        )
        return S, torch.softmax(self.w_smk_raw, 0)

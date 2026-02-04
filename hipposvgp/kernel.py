import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import math


class SMK_kernel(nn.Module):
    def __init__(self, num_mixture, device="cpu"):
        super().__init__()
        self.num_mixture = num_mixture
        self.device = device
        self.mu_smk = nn.Parameter(
            0.1 * torch.tensor(npr.randn(self.num_mixture, 1, 1))
        )
        self.log_var_smk = nn.Parameter(
            -10.0 + 1.0 * torch.tensor(npr.randn(self.num_mixture, 1, 1))
        )
        self.w_smk_raw = nn.Parameter(torch.tensor(npr.randn(self.num_mixture)))

    def forward(self, X1, X2):
        if len(X1.shape) == 3:  # If batch dimension exists
            diff = X1.unsqueeze(2) - X2.unsqueeze(1)  # [batch, N, M, input_dim]
            diff_sq = diff.pow(2).sum(-1)  # [batch, N, M]
            diff = diff.sum(-1)  # [batch, N, M]
            term1 = torch.exp(
                -2
                * torch.tensor(np.pi, dtype=torch.float32).pow(2)
                * diff_sq.unsqueeze(1)
                * torch.exp(self.log_var_smk).unsqueeze(0)
            )  # [num_mixture, batch, N, M]
            term2 = torch.cos(
                torch.tensor(2.0 * np.pi, dtype=torch.float32)
                * diff.unsqueeze(1)
                * self.mu_smk.unsqueeze(0)
            )  # [num_mixture, batch, N, M]
            res = torch.sum(
                torch.softmax(self.w_smk_raw, 0).view(1, -1, 1, 1) * term1 * term2,
                dim=1,
            )  # [batch, N, M]
        else:  # No batch dimension
            diff = X1.unsqueeze(1) - X2.unsqueeze(0)  # [N, M, input_dim]
            diff_sq = diff.pow(2).sum(-1)  # [N, M]
            diff = diff.sum(-1)  # [N, M]
            term1 = torch.exp(
                -2
                * torch.tensor(np.pi, dtype=torch.float32).pow(2)
                * diff_sq.unsqueeze(0)
                * torch.exp(self.log_var_smk)
            )  # [num_mixture, N, M]
            term2 = torch.cos(
                torch.tensor(2.0 * np.pi, dtype=torch.float32)
                * diff.unsqueeze(0)
                * self.mu_smk
            )  # [num_mixture, N, M]
            res = torch.sum(
                torch.softmax(self.w_smk_raw, 0).view(-1, 1, 1) * term1 * term2, dim=0
            )  # [N, M]
        return res

    def sample_from_spectral(self, num_rff):
        S = self.mu_smk.view(self.num_mixture, 1) + torch.sqrt(
            torch.exp(self.log_var_smk)
        ).view(self.num_mixture, 1) * torch.randn(
            self.num_mixture, num_rff, device=self.device
        )  # [num_mixture, num_rff]
        return S, torch.softmax(self.w_smk_raw, 0)  # [num_mixture]


class SE_kernel(nn.Module):
    def __init__(self, input_dim, device="cpu"):
        super().__init__()
        self.input_dim = input_dim
        self.log_sf = nn.Parameter(
            0.0 + 0.0 * torch.tensor(npr.randn(1), dtype=torch.float64)
        )
        self.log_ls = nn.Parameter(
            0.0 + 0.0 * torch.tensor(npr.randn(self.input_dim), dtype=torch.float64)
        )
        self.device = device

    def forward_slow(self, X1, X2):
        X1 = X1 * torch.exp(-self.log_ls)
        X2 = X2 * torch.exp(-self.log_ls)

        if len(X1.shape) == 3:  # If batch dimension exists
            X1 = X1.unsqueeze(2)  # [batch, N, 1, input_dim]
            X2 = X2.unsqueeze(3)  # [batch, 1, M, input_dim]
            diff = X2 - X1  # Broadcasting subtraction
            return torch.exp(self.log_sf) * torch.exp(-0.5 * torch.sum(diff**2, dim=-1))
        else:  # No batch dimension
            X1 = torch.unsqueeze(X1, 1)  # [N, 1, input_dim]
            X2 = torch.unsqueeze(X2.T, 2)  # [input_dim, M, 1]
            return (
                torch.exp(self.log_sf)
                * torch.exp(-0.5 * torch.sum((X2 - X1.T) ** 2, 0))
            ).T

    def forward(self, X1: torch.Tensor, X2: torch.Tensor = None, diag: bool = False):
        """
        Compute the Squared Exponential (RBF) kernel between X1 and X2.
        
        Args:
            X1 (torch.Tensor): First input tensor [batch, N, D] or [N, D]
            X2 (torch.Tensor, optional): Second input tensor, defaults to X1
            diag (bool): If True, return only diagonal elements
            
        Returns:
            torch.Tensor: Kernel matrix or diagonal elements
        """
        if diag:
            if X2 is None:
                X2 = X1  # Ensuring X2 is not None
                # Scale inputs by length scale
                X1 = X1 * torch.exp(-self.log_ls)
                X2 = X2 * torch.exp(-self.log_ls)

                # Compute squared differences
                sq_dist = torch.sum((X1 - X2).pow(2), -1)  # [batch, N] or [N]
                return torch.exp(self.log_sf) * torch.exp(-0.5 * sq_dist)
            else: 
                raise ValueError("Cannot compute diagonal elements for two inputs")
        else:
            if X2 is None:
                X2 = X1  # Ensuring X2 is not None

            # Scale inputs by length scale
            X1 = X1 * torch.exp(-self.log_ls)
            X2 = X2 * torch.exp(-self.log_ls)

            if len(X1.shape) == 3:  # If batch dimension exists
                factor1 = torch.sum(X1.pow(2), -1, keepdim=True)  # [batch, N, 1]
                factor2 = torch.sum(X2.pow(2), -1, keepdim=True)  # [batch, M, 1]
                cross = X1 @ X2.transpose(-1, -2)  # [batch, N, M]
                return torch.exp(self.log_sf) * torch.exp(
                    -0.5 * (factor1 + factor2.transpose(-1, -2) - 2 * cross)
                )
            else:
                # No batch dimension
                factor1 = torch.sum(X1.pow(2), -1)  # [N]
                factor2 = torch.sum(X2.pow(2), -1)  # [M]
                return torch.exp(self.log_sf) * torch.exp(
                    -0.5 * (factor1.unsqueeze(1) + factor2.unsqueeze(0) - 2 * X1 @ X2.T)
                )


    def sample_from_spectral(self, num_rff):
        # For SE kernel, the spectral density is Gaussian(mean = 0, std = 1./ length-scale)
        # num_rff: the number of RFF samples
        return (1.0 / torch.exp(self.log_ls)) * torch.randn(
            num_rff, device=self.device
        )  # we ignore signal variance here, but will multiply it back later


class FreqSpaceKernel(nn.Module):
    """Kernel parameterized directly in frequency space.

    Spectral density: S(ω) = σ² Σ_i γ_i δ(ω - ω_i)
    Time domain: k(τ) = σ² Σ_i γ_i cos(ω_i τ)

    Uses learnable frequencies and weights directly, without sampling like RFF.
    """
    def __init__(self, num_components, device="cpu", dtype=torch.float64,
                 init_omega_scale=1.0, init_gamma_scale=0.0):
        super().__init__()
        self.num_components = num_components
        self.device = device
        self.dtype = dtype

        # Learnable frequencies (log scale for positivity)
        # Initial values: approximately in the range of log(|N(0, init_omega_scale)|)
        self.log_omega = nn.Parameter(
            torch.log(torch.abs(torch.randn(num_components, dtype=dtype, device=device) * init_omega_scale) + 1e-6)
        )

        # Learnable weights (log scale, normalized via softmax)
        self.log_gamma_raw = nn.Parameter(
            torch.randn(num_components, dtype=dtype, device=device) * init_gamma_scale
        )

        # Signal variance (for compatibility with other kernels)
        self.log_sf = nn.Parameter(
            torch.tensor(0.0, dtype=dtype, device=device)
        )

    @property
    def omega(self):
        """Frequencies (positive values)."""
        return torch.exp(self.log_omega)

    @property
    def gamma(self):
        """Normalized weights (sum to 1)."""
        return torch.softmax(self.log_gamma_raw, dim=0)

    def sample_from_spectral(self, num_rff=None):
        """For compatibility - returns learned omega without sampling.

        Args:
            num_rff: Ignored (exists for compatibility).

        Returns:
            omega: [num_components] Learned frequencies.
        """
        return self.omega

    def get_weights(self):
        """Returns weights (called from HSVGP).

        Returns:
            gamma: [num_components] Normalized weights.
        """
        return self.gamma

    def forward(self, X1: torch.Tensor, X2: torch.Tensor = None, diag: bool = False):
        """Compute kernel matrix.

        k(x1, x2) = σ² Σ_i γ_i cos(ω_i (x1 - x2))
        """
        if X2 is None:
            X2 = X1

        omega = self.omega      # [R]
        gamma = self.gamma      # [R]
        sf = torch.exp(self.log_sf)

        if diag:
            # Diagonal elements: k(x, x) = σ² Σ_i γ_i cos(0) = σ² Σ_i γ_i = σ²
            if X1.dim() == 2:
                return sf * torch.ones(X1.shape[0], dtype=self.dtype, device=self.device)
            else:
                return sf * torch.ones(X1.shape[:-1], dtype=self.dtype, device=self.device)

        # Compute differences
        if X1.dim() == 3:  # [batch, N, D]
            diff = X1.unsqueeze(2) - X2.unsqueeze(1)  # [batch, N, M, D]
            diff = diff.sum(-1)  # [batch, N, M] (assuming 1D input)
            # cos(ω_i * diff) for each ω_i
            cos_terms = torch.cos(omega.view(1, 1, 1, -1) * diff.unsqueeze(-1))  # [batch, N, M, R]
            # Weighted sum
            return sf * (cos_terms * gamma.view(1, 1, 1, -1)).sum(-1)  # [batch, N, M]
        else:  # [N, D]
            diff = X1.unsqueeze(1) - X2.unsqueeze(0)  # [N, M, D]
            diff = diff.sum(-1)  # [N, M]
            cos_terms = torch.cos(omega.view(1, 1, -1) * diff.unsqueeze(-1))  # [N, M, R]
            return sf * (cos_terms * gamma.view(1, 1, -1)).sum(-1)  # [N, M]


if __name__ == "__main__":
    torch.manual_seed(42)
    x1 = torch.randn(5, 1)
    x2 = torch.randn(20, 1)

    # RFF for SMK
    mykernel = SMK_kernel(2)
    res = mykernel(x1, x1)
    s, w = mykernel.sample_from_spectral(1000)  # [num_mixture, num_rff], [num_mix]

    Z_f = torch.cos(2 * math.pi * s.unsqueeze(0) * x1.unsqueeze(-1))
    Z_f = (torch.sqrt(w).view(1, -1, 1) * Z_f).permute(2, 0, 1)  # [num_rff, 5, num_mix]
    Kff = Z_f @ Z_f.permute(0, 2, 1)

    Z_f2 = torch.sin(2 * math.pi * s.unsqueeze(0) * x1.unsqueeze(-1))
    Z_f2 = (torch.sqrt(w).view(1, -1, 1) * Z_f2).permute(
        2, 0, 1
    )  # [num_rff, 5, num_mix]
    Kff2 = Z_f2 @ Z_f2.permute(0, 2, 1)
    res2 = torch.mean(Kff + Kff2, 0)
    print(res)
    print(res2)

    np.testing.assert_allclose(res.detach().numpy(), res2.detach().numpy(), atol=1e-3)
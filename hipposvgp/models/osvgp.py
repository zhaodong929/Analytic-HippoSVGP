"""Unified Online Sparse Variational GP model.

Consolidates ELBO, prediction, correction, and KL computations
from hsvgp.py, hsvgp_analytic.py, and svgp.py into a single class.
The Z matrix computation strategy is delegated to an InducingStrategy.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from hipposvgp.linalg import add_jitter, cholesky, whitened_kl


@dataclass
class PosteriorState:
    """Transfer variational posterior between tasks."""

    mv: torch.Tensor  # [M]
    Lv: torch.Tensor  # [M, M]
    Kaa: torch.Tensor  # [M, M]
    Z: torch.Tensor  # feature matrix


class OnlineSVGP(nn.Module):
    """Unified online sparse variational GP.

    Supports HiPPO-ODE, analytic HiPPO, and standard inducing strategies
    via the strategy pattern.

    Args:
        kernel: Kernel module
        likelihood: Likelihood module
        inducing_strategy: InducingStrategy module
        device: torch device
        prior_state: PosteriorState from previous task (None for first task)
        use_robust_cholesky: use robust Cholesky with progressive jitter
        jitter: base jitter for numerical stability
    """

    def __init__(
        self,
        kernel,
        likelihood,
        inducing_strategy,
        device,
        prior_state=None,
        use_robust_cholesky=False,
        jitter=1e-6,
        dtype=torch.float64,
        flag_update_kernel=True,
    ):
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.jitter = jitter
        self.use_robust_cholesky = use_robust_cholesky
        self.flag_update_kernel = flag_update_kernel

        self.kernel = kernel
        self.likelihood = likelihood
        self.inducing_strategy = inducing_strategy

        inducing_size = inducing_strategy.inducing_size

        # Variational parameters q(v) = N(mv, Lv @ Lv^T)
        self.mv = nn.Parameter(
            torch.zeros(inducing_size, dtype=dtype), requires_grad=True
        )
        self.Lv = nn.Parameter(
            torch.eye(inducing_size, dtype=dtype), requires_grad=True
        )

        # Prior state from previous task
        if prior_state is not None:
            self.mv_old = prior_state.mv.clone().detach().to(dtype=dtype, device=device)
            self.Lv_old = prior_state.Lv.clone().detach().to(dtype=dtype, device=device)
            self.Kaa_old = (
                prior_state.Kaa.clone().detach().to(dtype=dtype, device=device)
                if prior_state.Kaa is not None
                else None
            )
            self.Z_old = (
                prior_state.Z.clone().detach().to(dtype=dtype, device=device)
                if prior_state.Z is not None
                else None
            )
        else:
            self.mv_old = None
            self.Lv_old = None
            self.Kaa_old = None
            self.Z_old = None

        # Kernel matrix cache
        self._cached_km = None
        self._cached_w = None

    @property
    def _has_prior(self):
        """Whether this model has a prior from a previous task."""
        return self.mv_old is not None

    def _cholesky(self, K):
        """Cholesky decomposition with optional robustness."""
        return cholesky(K, use_robust=self.use_robust_cholesky, jitter=self.jitter)

    def _compute_qf(self, Kuu, Kfu, Kff, L_Kuu):
        """Compute mean and covariance of q(f).

        Args:
            Kuu: [M, M]
            Kfu: [N, M]
            Kff: [N, N]
            L_Kuu: [M, M] Cholesky factor of Kuu

        Returns:
            mean_f: [N, 1]
            Cov_f: [N, N]
        """
        # Unwhiten: mu_u = L_Kuu @ mv
        mu_u = L_Kuu @ self.mv.unsqueeze(-1)
        L_Kuu_inv_mu_u = torch.cholesky_solve(mu_u, L_Kuu)
        mean_f = Kfu @ L_Kuu_inv_mu_u

        # Cov(f) = Kff - Qff + Kfu Kuu^-1 (Lv Lv^T) Kuu^-1 Kfu^T
        L_Kuu_inv_Kfu_T = torch.cholesky_solve(Kfu.T, L_Kuu)
        Qff = Kfu @ L_Kuu_inv_Kfu_T

        L_u = L_Kuu @ self.Lv
        temp = torch.cholesky_solve(L_u, L_Kuu)
        V = Kfu @ temp
        Cov_f = Kff - Qff + V @ V.T

        return mean_f, Cov_f

    def _quold_new(self, Kuu, Kuu_old, Kold_old_newk, full_cov=True):
        """Predict old inducing points under new posterior.

        Args:
            Kuu: [M, M] current inducing kernel
            Kuu_old: [M, M'] cross kernel
            Kold_old_newk: [M', M'] old inducing re-evaluated with new kernel

        Returns:
            (mean, var): predictive mean and (co)variance at old inducing locations
        """
        Kff_new = Kold_old_newk
        Kfu_new = Kuu_old.T

        Luu = self._cholesky(Kuu)

        mu_u = Luu @ self.mv
        mu_u = mu_u.unsqueeze(1)
        L_u = Luu @ self.Lv

        alpha = torch.cholesky_solve(mu_u, Luu)
        mean = Kfu_new @ alpha

        Kuu_inv_Kfu = torch.cholesky_solve(Kfu_new.T, Luu)
        var = Kff_new - Kfu_new @ Kuu_inv_Kfu

        if full_cov:
            temp = torch.cholesky_solve(L_u, Luu)
            V = Kfu_new @ temp
            var += V @ V.T
        else:
            temp = torch.cholesky_solve(L_u, Luu)
            var_diag_extra = torch.sum((Kfu_new @ temp) ** 2, dim=1)
            var = (
                torch.diag(Kff_new)
                - torch.sum(Kfu_new * Kuu_inv_Kfu.T, dim=1)
                + var_diag_extra
            )
            var = var.unsqueeze(1)

        return mean, var

    def _correction_term(self, km):
        """Compute online correction: log ratio between old and new posterior.

        Args:
            km: KernelMatrices dataclass

        Returns:
            Scalar correction term.
        """
        if not self._has_prior:
            return torch.tensor(0.0, device=self.device)

        mv_old = self.mv_old.unsqueeze(-1)
        Lv_old = self.Lv_old

        Luu_old = self._cholesky(km.Kold_old)
        m_old = Luu_old @ mv_old
        S_old = Luu_old @ Lv_old @ Lv_old.T @ Luu_old.T

        mu, Sigma = self._quold_new(
            km.Kuu, km.Kuu_old, km.Kold_old_newk, full_cov=True
        )
        Smm = Sigma + mu @ mu.T

        LSa = self._cholesky(S_old)
        LKa = self._cholesky(km.Kold_old)

        obj = torch.sum(torch.log(torch.diagonal(LKa))) - torch.sum(
            torch.log(torch.diagonal(LSa))
        )
        Sainv_ma = torch.cholesky_solve(m_old, LSa).squeeze(-1)
        obj -= 0.5 * torch.sum(m_old.squeeze(-1) * Sainv_ma)
        obj += torch.sum(mu.squeeze(-1) * Sainv_ma)

        Sainv_Smm = torch.cholesky_solve(Smm, LSa)
        Kainv_Smm = torch.cholesky_solve(Smm, LKa)
        obj -= 0.5 * (torch.trace(Sainv_Smm) - torch.trace(Kainv_Smm))

        return obj

    def elbo(self, X, y, w=None, recompute_k=True, cache_k=False, beta=1.0):
        """Compute the evidence lower bound.

        Args:
            X: [N, D] training inputs
            y: [N, 1] training targets
            w: pre-sampled RFF frequencies (None to sample)
            recompute_k: whether to recompute kernel matrices
            cache_k: whether to cache kernel matrices
            beta: scaling for KL term (for tempering)

        Returns:
            (elbo, ve_mean, kl): ELBO, variational expectation, KL divergence
        """
        # 1. Compute or retrieve kernel matrices
        if recompute_k:
            km, w = self.inducing_strategy.compute_kernel_matrices(
                self.kernel,
                X,
                Z_old=self.Z_old,
                Kaa_old=self.Kaa_old,
                flag_update_kernel=self.flag_update_kernel,
            )
            # Always cache the latest km for get_posterior_state()
            self._cached_km = km
            self._cached_w = w
        else:
            km = self._cached_km
            w = self._cached_w

        # 2. Cholesky of Kuu
        L_Kuu = self._cholesky(km.Kuu)

        # 3. KL divergence
        kl = whitened_kl(self.mv, self.Lv)

        # 4. q(f) distribution
        mean_f, Cov_f = self._compute_qf(km.Kuu, km.Kfu, km.Kff, L_Kuu)
        Cov_f = add_jitter(Cov_f, jitter=self.jitter)

        # 5. Sample f ~ q(f)
        L_Cov_f = torch.linalg.cholesky(Cov_f)
        eps = torch.randn_like(mean_f)
        f_samples = mean_f + L_Cov_f @ eps

        # 6. Variational expectations
        variational_expectations = self.likelihood.log_prob(f_samples, y)
        ve_mean = torch.sum(variational_expectations)

        # 7. ELBO
        elbo = ve_mean - beta * kl

        # 8. Online correction
        if self.Z_old is not None and km.Kuu_old is not None:
            corr = self._correction_term(km)
            elbo = elbo + corr

        return elbo, ve_mean, kl

    def predict(self, Xnew, w=None, full_cov=False, include_noise=False):
        """Compute predictive mean and variance.

        Args:
            Xnew: [N*, D] test inputs
            w: pre-sampled RFF frequencies (None to sample)
            full_cov: if True, return full covariance matrix
            include_noise: if True, add observation noise to variance

        Returns:
            mean: [N*, 1] predictive mean
            var: [N*, 1] or [N*, N*] predictive variance
        """
        Kuu, Kfu_new, Kff_new = self.inducing_strategy.compute_prediction_matrices(
            self.kernel, Xnew, w=w
        )
        Luu = self._cholesky(Kuu)

        mu_u = Luu @ self.mv
        mu_u = mu_u.unsqueeze(1)
        L_u = Luu @ self.Lv

        alpha = torch.cholesky_solve(mu_u, Luu)
        mean = Kfu_new @ alpha

        Kuu_inv_Kfu = torch.cholesky_solve(Kfu_new.T, Luu)

        if full_cov:
            var = Kff_new - Kfu_new @ Kuu_inv_Kfu
            temp = torch.cholesky_solve(L_u, Luu)
            V = Kfu_new @ temp
            var += V @ V.T
            if include_noise and hasattr(self.likelihood, "variance"):
                var = var + self.likelihood.variance * torch.eye(
                    var.shape[0], device=self.device, dtype=self.dtype
                )
        else:
            temp = torch.cholesky_solve(L_u, Luu)
            var_diag_extra = torch.sum((Kfu_new @ temp) ** 2, dim=1)
            var = (
                torch.diag(Kff_new)
                - torch.sum(Kfu_new * Kuu_inv_Kfu.T, dim=1)
                + var_diag_extra
            )
            var = var.unsqueeze(1)
            if include_noise and hasattr(self.likelihood, "variance"):
                var = var + self.likelihood.variance

        return mean, var

    def get_posterior_state(self):
        """Extract current posterior state for transfer to next task."""
        # Get current Kuu from cached or recompute
        if self._cached_km is not None:
            Kaa = self._cached_km.Kuu.detach()
            Z = self._cached_km.Z.detach() if self._cached_km.Z is not None else None
        else:
            Kaa = None
            Z = None

        return PosteriorState(
            mv=self.mv.detach().clone(),
            Lv=self.Lv.detach().clone(),
            Kaa=Kaa,
            Z=Z,
        )

"""Online Sparse GP Regression (VFE collapsed bound).

Separate from OnlineSVGP because SGPR uses a collapsed ELBO
(no explicit variational parameters mv, Lv) with Woodbury-based formulas.
Gaussian likelihood only -- noise variance is a learned parameter.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from hipposvgp.linalg import add_jitter
from hipposvgp.rff import SpectralSample


@dataclass
class SGPRPosteriorState:
    """Transfer posterior state between tasks for SGPR."""

    mu: torch.Tensor  # [M, 1] optimal posterior mean at inducing points
    Su: torch.Tensor  # [M, M] optimal posterior covariance
    Kaa: torch.Tensor  # [M, M] inducing kernel matrix
    Z: torch.Tensor  # feature matrix


@dataclass
class _SGPRTerms:
    """Intermediate terms shared by ELBO, prediction, and qb_opt."""

    Kff: torch.Tensor
    Kbf: torch.Tensor
    Kba: Optional[torch.Tensor]
    Kaa: Optional[torch.Tensor]
    Kaa_cur: Optional[torch.Tensor]
    La: Optional[torch.Tensor]
    Kbb: torch.Tensor
    Lb: torch.Tensor
    D: torch.Tensor
    LD: torch.Tensor
    Lbinv_Kba: Optional[torch.Tensor]
    LDinv_Lbinv_c: torch.Tensor
    err: torch.Tensor
    Qff: torch.Tensor


class OnlineSGPR(nn.Module):
    """Online Sparse GP Regression with HiPPO inducing features.

    Uses a collapsed bound (VFE) instead of explicit variational inference.
    Gaussian likelihood only (noise variance is a learned parameter).

    For non-Gaussian likelihoods, use OnlineSVGP instead.

    Args:
        kernel: Kernel module
        inducing_strategy: InducingStrategy module
        device: torch device
        X: [N, D] training inputs
        y: [N, 1] training targets
        prior_state: SGPRPosteriorState from previous task (None for first task)
        spectral_sample: fixed SpectralSample (None to sample from kernel)
    """

    def __init__(
        self,
        kernel,
        inducing_strategy,
        device,
        X,
        y,
        prior_state=None,
        spectral_sample=None,
        dtype=torch.float64,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.kernel = kernel
        self.inducing_strategy = inducing_strategy
        self.spectral_sample = spectral_sample

        self.X = torch.as_tensor(X, dtype=dtype, device=device)
        self.y = torch.as_tensor(y, dtype=dtype, device=device)
        self.num_data = self.X.shape[0]

        self.log_noise = nn.Parameter(torch.tensor([-3.0], dtype=dtype))
        self.jitter = 1e-6

        # Prior state from previous task
        if prior_state is not None:
            self.mu_old = prior_state.mu.clone().detach().to(dtype=dtype, device=device)
            self.Su_old = prior_state.Su.clone().detach().to(dtype=dtype, device=device)
            self.Kaa_old = prior_state.Kaa.clone().detach().to(dtype=dtype, device=device)
            self.Z_old = (
                prior_state.Z.clone().detach().to(dtype=dtype, device=device)
                if prior_state.Z is not None
                else None
            )
        else:
            self.mu_old = None
            self.Su_old = None
            self.Kaa_old = None
            self.Z_old = None

        self._cached_km = None
        self._cached_sample = None

    @property
    def _has_prior(self):
        """Whether this model has a prior from a previous task."""
        return self.mu_old is not None

    def _add_jitter(self, K):
        return add_jitter(K, jitter=self.jitter)

    def _common_terms(self, sample=None):
        """Compute intermediate terms shared by ELBO, prediction, and qb_opt.

        Handles both first task (no prior) and online tasks (with prior).
        Results are cached for reuse by predict/qb_opt.
        """
        Mb = self.inducing_strategy.inducing_size
        sigma2 = torch.exp(self.log_noise)
        sigma = torch.sqrt(sigma2)

        # Compute kernel matrices via inducing strategy
        effective_sample = self.spectral_sample if sample is None else sample
        km, resolved_sample = self.inducing_strategy.compute_kernel_matrices(
            self.kernel,
            self.X,
            Z_old=self.Z_old if self._has_prior else None,
            Kaa_old=self.Kaa_old if self._has_prior else None,
            flag_update_kernel=True,
            sample=effective_sample,
        )
        self._cached_km = km
        self._cached_sample = resolved_sample

        Kff = km.Kff
        Kfb = km.Kfu
        Kbf = Kfb.T
        Kbb = self._add_jitter(km.Kuu)

        # Data term: c = Kbf y / sigma^2
        Sinv_y = self.y / sigma2
        c = Kbf @ Sinv_y

        Lb = torch.linalg.cholesky(Kbb)
        Lbinv_Kbf = torch.linalg.solve_triangular(Lb, Kbf, upper=False) / sigma
        d1 = Lbinv_Kbf @ Lbinv_Kbf.T

        # D starts as I + d1 (standard SGPR)
        D = torch.eye(Mb, device=self.device, dtype=self.dtype) + d1

        # Online correction terms (only with prior from previous task)
        Kba = None
        Kaa = None
        Kaa_cur = None
        La = None
        Lbinv_Kba = None

        if self._has_prior:
            Saa = self.Su_old
            ma = self.mu_old
            Kba = km.Kuu_old
            Kaa_cur = km.Kold_old_newk if km.Kold_old_newk is not None else km.Kold_old
            if Kaa_cur is not None:
                Kaa_cur = self._add_jitter(Kaa_cur)
            Kaa = self._add_jitter(self.Kaa_old)

            # Prior contribution to c
            Sainv_ma = torch.linalg.solve(Saa, ma)
            c = c + Kba @ Sainv_ma

            Lbinv_Kba = torch.linalg.solve_triangular(Lb, Kba, upper=False)

            # d2: prior precision term
            LSa = torch.linalg.cholesky(Saa)
            Kab_Lbinv = Lbinv_Kba.T
            LSainv_Kab_Lbinv = torch.linalg.solve_triangular(
                LSa, Kab_Lbinv, upper=False
            )
            d2 = LSainv_Kab_Lbinv.T @ LSainv_Kab_Lbinv

            # d3: prior kernel inverse term
            La = torch.linalg.cholesky(Kaa)
            Lainv_Kab_Lbinv = torch.linalg.solve_triangular(
                La, Kab_Lbinv, upper=False
            )
            d3 = Lainv_Kab_Lbinv.T @ Lainv_Kab_Lbinv

            D = D + d2 - d3

        D = self._add_jitter(D)
        LD = torch.linalg.cholesky(D)

        Lbinv_c = torch.linalg.solve_triangular(Lb, c, upper=False)
        LDinv_Lbinv_c = torch.linalg.solve_triangular(LD, Lbinv_c, upper=False)

        return _SGPRTerms(
            Kff=Kff, Kbf=Kbf, Kba=Kba, Kaa=Kaa, Kaa_cur=Kaa_cur,
            La=La, Kbb=Kbb, Lb=Lb, D=D, LD=LD,
            Lbinv_Kba=Lbinv_Kba, LDinv_Lbinv_c=LDinv_Lbinv_c,
            err=self.y, Qff=d1,
        )

    def elbo(self):
        """Compute collapsed ELBO (no explicit variational parameters).

        Returns:
            Scalar ELBO value.
        """
        sigma2 = torch.exp(self.log_noise)
        N = self.num_data

        t = self._common_terms()
        Kfdiag = torch.diag(t.Kff)

        # Standard SGPR terms
        bound = -0.5 * N * math.log(2 * math.pi)
        bound -= 0.5 * torch.sum(t.err.pow(2)) / sigma2
        bound += 0.5 * torch.sum(t.LDinv_Lbinv_c.pow(2))
        bound -= 0.5 * N * torch.sum(torch.log(sigma2))
        bound -= torch.sum(torch.log(torch.diag(t.LD)))
        # Trace correction: -1/(2sigma^2) tr(Kff - Qff)
        bound -= 0.5 * torch.sum(Kfdiag) / sigma2
        bound += 0.5 * torch.sum(torch.diag(t.Qff))

        # Online correction (only with prior)
        if self._has_prior:
            Saa = self.Su_old
            ma = self.mu_old

            LSa = torch.linalg.cholesky(Saa)
            Lainv_ma = torch.linalg.solve_triangular(LSa, ma, upper=False)
            bound -= 0.5 * torch.sum(Lainv_ma.pow(2))

            # Log-det ratio: log|Kaa|^{1/2} - log|Saa|^{1/2}
            bound += torch.sum(torch.log(torch.diag(t.La)))
            bound -= torch.sum(torch.log(torch.diag(LSa)))

            # Inducing point correction
            Kaadiff = t.Kaa_cur - t.Lbinv_Kba.T @ t.Lbinv_Kba
            Sainv_Kaadiff = torch.linalg.solve(Saa, Kaadiff)
            Kainv_Kaadiff = torch.linalg.solve(t.Kaa, Kaadiff)
            bound -= 0.5 * torch.sum(
                torch.diag(Sainv_Kaadiff) - torch.diag(Kainv_Kaadiff)
            )

        return bound

    def predict(self, Xnew, sample=None, full_cov=False, include_noise=True):
        """Compute predictive mean and variance at new points.

        Args:
            Xnew: [N*, D] test inputs
            sample: SpectralSample (None to use cached from last _common_terms call)
            full_cov: if True, return full covariance matrix
            include_noise: if True, add observation noise to variance

        Returns:
            mean: [N*, 1] predictive mean
            var: [N*] or [N*, N*] predictive variance
        """
        effective_sample = self.spectral_sample if sample is None else sample
        t = self._common_terms(sample=effective_sample)

        resolved_sample = self._cached_sample

        Kuu, Kfu_new, Kff_new = self.inducing_strategy.compute_prediction_matrices(
            self.kernel, Xnew, sample=resolved_sample
        )
        Kbs = Kfu_new.T

        Lbinv_Kbs = torch.linalg.solve_triangular(t.Lb, Kbs, upper=False)
        LDinv_Lbinv_Kbs = torch.linalg.solve_triangular(
            t.LD, Lbinv_Kbs, upper=False
        )
        mean = LDinv_Lbinv_Kbs.T @ t.LDinv_Lbinv_c

        if full_cov:
            Kss = self._add_jitter(Kff_new)
            var = (
                Kss
                - Lbinv_Kbs.T @ Lbinv_Kbs
                + LDinv_Lbinv_Kbs.T @ LDinv_Lbinv_Kbs
            )
            if include_noise:
                var += torch.exp(self.log_noise) * torch.eye(
                    var.shape[0], device=self.device, dtype=self.dtype
                )
        else:
            # Use exact kernel diagonal (not RFF approximation) for numerical stability
            var1 = self.kernel(Xnew, diag=True)
            var2 = -torch.sum(Lbinv_Kbs.pow(2), 0)
            var3 = torch.sum(LDinv_Lbinv_Kbs.pow(2), 0)
            var = var1 + var2 + var3
            if include_noise:
                var += torch.exp(self.log_noise) * torch.ones_like(var)

        return mean, var

    def qb_opt(self, full_cov=True):
        """Compute optimal posterior at inducing points.

        Returns:
            mean: [M, 1] posterior mean
            var: [M, M] or [M] posterior (co)variance
            Kbb: [M, M] inducing kernel matrix
        """
        t = self._common_terms()
        Kss = t.Kbb.clone()
        Kbs = t.Kbb.clone()

        Lbinv_Kbs = torch.linalg.solve_triangular(t.Lb, Kbs, upper=False)
        LDinv_Lbinv_Kbs = torch.linalg.solve_triangular(
            t.LD, Lbinv_Kbs, upper=False
        )
        mean = LDinv_Lbinv_Kbs.T @ t.LDinv_Lbinv_c

        if full_cov:
            var = (
                Kss
                - Lbinv_Kbs.T @ Lbinv_Kbs
                + LDinv_Lbinv_Kbs.T @ LDinv_Lbinv_Kbs
            )
        else:
            var1 = torch.diag(Kss)
            var2 = -torch.sum(Lbinv_Kbs.pow(2), 0)
            var3 = torch.sum(LDinv_Lbinv_Kbs.pow(2), 0)
            var = var1 + var2 + var3

        return mean, var, t.Kbb

    def get_posterior_state(self):
        """Extract current optimal posterior state for transfer to next task."""
        mu, Su, Kbb = self.qb_opt(full_cov=True)
        Z = (
            self._cached_km.Z.detach()
            if self._cached_km is not None and self._cached_km.Z is not None
            else None
        )
        return SGPRPosteriorState(
            mu=mu.detach().clone(),
            Su=Su.detach().clone(),
            Kaa=Kbb.detach().clone(),
            Z=Z,
        )

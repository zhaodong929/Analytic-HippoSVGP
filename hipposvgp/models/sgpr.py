"""Online Sparse GP Regression (VFE collapsed bound).

Uses a collapsed ELBO (no explicit variational parameters mv, Lv)
with Woodbury-based formulas. Gaussian likelihood only.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from hipposvgp.inducing.base import InducingStrategy
from hipposvgp.kernels.base import Kernel
from hipposvgp.linalg import add_jitter
from hipposvgp.types import KernelMatrices, SGPRPosteriorState, SpectralSample


class _SGPRTerms(eqx.Module):
    """Intermediate terms shared by ELBO, prediction, and qb_opt."""

    Kff: Float[Array, "N N"]
    Kbf: Float[Array, "M N"]
    Kba: Float[Array, "M Mp"] | None
    Kaa: Float[Array, "Mp Mp"] | None
    Kaa_cur: Float[Array, "Mp Mp"] | None
    La: Float[Array, "Mp Mp"] | None
    Kbb: Float[Array, "M M"]
    Lb: Float[Array, "M M"]
    D: Float[Array, "M M"]
    LD: Float[Array, "M M"]
    Lbinv_Kba: Float[Array, "M Mp"] | None
    LDinv_Lbinv_c: Float[Array, "M 1"]
    err: Float[Array, "N 1"]
    Qff: Float[Array, "M M"]


class OnlineSGPR(eqx.Module):
    """Online Sparse GP Regression with HiPPO inducing features.

    Uses a collapsed bound (VFE) instead of explicit variational inference.
    Gaussian likelihood only.
    """

    kernel: Kernel
    inducing_strategy: InducingStrategy
    log_noise: Float[Array, ""]
    X: Float[Array, "N D"]
    y: Float[Array, "N 1"]
    # Prior state from previous task
    mu_old: Float[Array, "M 1"] | None = None
    Su_old: Float[Array, "M M"] | None = None
    Kaa_old: Float[Array, "M M"] | None = None
    Z_old: Float[Array, "TwoR M"] | None = None
    # Config
    jitter: float = eqx.field(static=True, default=1e-6)

    def __init__(
        self,
        kernel: Kernel,
        inducing_strategy: InducingStrategy,
        X: Float[Array, "N D"],
        y: Float[Array, "N 1"],
        *,
        prior_state: SGPRPosteriorState | None = None,
        spectral_sample: SpectralSample | None = None,
        jitter: float = 1e-6,
    ):
        self.kernel = kernel
        self.inducing_strategy = inducing_strategy
        self.X = X
        self.y = y
        self.log_noise = jnp.array(-3.0)
        self.jitter = jitter

        if prior_state is not None:
            self.mu_old = prior_state.mu
            self.Su_old = prior_state.Su
            self.Kaa_old = prior_state.Kaa
            self.Z_old = prior_state.Z
        else:
            self.mu_old = None
            self.Su_old = None
            self.Kaa_old = None
            self.Z_old = None

    @property
    def _has_prior(self) -> bool:
        return self.mu_old is not None

    @property
    def num_data(self) -> int:
        return self.X.shape[0]

    def _add_jitter(self, K: Float[Array, "N N"]) -> Float[Array, "N N"]:
        return add_jitter(K, jitter=self.jitter)

    def _common_terms(
        self,
        *,
        sample: SpectralSample | None = None,
        key: PRNGKeyArray | None = None,
    ) -> tuple[_SGPRTerms, KernelMatrices, SpectralSample]:
        """Compute intermediate terms shared by ELBO, prediction, and qb_opt."""
        Mb = self.inducing_strategy.inducing_size
        sigma2 = jnp.exp(self.log_noise)
        sigma = jnp.sqrt(sigma2)

        km, resolved_sample = self.inducing_strategy.compute_kernel_matrices(
            self.kernel,
            self.X,
            Z_old=self.Z_old if self._has_prior else None,
            Kaa_old=self.Kaa_old if self._has_prior else None,
            flag_update_kernel=True,
            sample=sample,
            key=key,
        )

        Kff = km.Kff
        Kfb = km.Kfu
        Kbf = Kfb.T
        Kbb = self._add_jitter(km.Kuu)

        # Data term
        Sinv_y = self.y / sigma2
        c = Kbf @ Sinv_y

        Lb = jnp.linalg.cholesky(Kbb)
        Lbinv_Kbf = jax.scipy.linalg.solve_triangular(Lb, Kbf, lower=True) / sigma
        d1 = Lbinv_Kbf @ Lbinv_Kbf.T

        D = jnp.eye(Mb) + d1

        # Online correction terms
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
            Sainv_ma = jnp.linalg.solve(Saa, ma)
            c = c + Kba @ Sainv_ma

            Lbinv_Kba = jax.scipy.linalg.solve_triangular(Lb, Kba, lower=True)

            # d2: prior precision term
            LSa = jnp.linalg.cholesky(Saa)
            Kab_Lbinv = Lbinv_Kba.T
            LSainv_Kab_Lbinv = jax.scipy.linalg.solve_triangular(
                LSa, Kab_Lbinv, lower=True
            )
            d2 = LSainv_Kab_Lbinv.T @ LSainv_Kab_Lbinv

            # d3: prior kernel inverse term
            La = jnp.linalg.cholesky(Kaa)
            Lainv_Kab_Lbinv = jax.scipy.linalg.solve_triangular(
                La, Kab_Lbinv, lower=True
            )
            d3 = Lainv_Kab_Lbinv.T @ Lainv_Kab_Lbinv

            D = D + d2 - d3

        D = self._add_jitter(D)
        LD = jnp.linalg.cholesky(D)

        Lbinv_c = jax.scipy.linalg.solve_triangular(Lb, c, lower=True)
        LDinv_Lbinv_c = jax.scipy.linalg.solve_triangular(LD, Lbinv_c, lower=True)

        terms = _SGPRTerms(
            Kff=Kff, Kbf=Kbf, Kba=Kba, Kaa=Kaa, Kaa_cur=Kaa_cur,
            La=La, Kbb=Kbb, Lb=Lb, D=D, LD=LD,
            Lbinv_Kba=Lbinv_Kba, LDinv_Lbinv_c=LDinv_Lbinv_c,
            err=self.y, Qff=d1,
        )
        return terms, km, resolved_sample

    def elbo(
        self,
        *,
        sample: SpectralSample | None = None,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, ""], KernelMatrices, SpectralSample]:
        """Compute collapsed ELBO.

        Returns:
            (elbo, km, sample): ELBO value, kernel matrices, resolved sample
        """
        sigma2 = jnp.exp(self.log_noise)
        N = self.num_data

        t, km, resolved_sample = self._common_terms(sample=sample, key=key)
        Kfdiag = jnp.diag(t.Kff)

        # Standard SGPR terms
        bound = -0.5 * N * math.log(2 * math.pi)
        bound = bound - 0.5 * jnp.sum(t.err**2) / sigma2
        bound = bound + 0.5 * jnp.sum(t.LDinv_Lbinv_c**2)
        bound = bound - 0.5 * N * jnp.sum(jnp.log(sigma2))
        bound = bound - jnp.sum(jnp.log(jnp.diag(t.LD)))
        bound = bound - 0.5 * jnp.sum(Kfdiag) / sigma2
        bound = bound + 0.5 * jnp.sum(jnp.diag(t.Qff))

        # Online correction
        if self._has_prior:
            Saa = self.Su_old
            ma = self.mu_old

            LSa = jnp.linalg.cholesky(Saa)
            Lainv_ma = jax.scipy.linalg.solve_triangular(LSa, ma, lower=True)
            bound = bound - 0.5 * jnp.sum(Lainv_ma**2)

            # Log-det ratio
            bound = bound + jnp.sum(jnp.log(jnp.diag(t.La)))
            bound = bound - jnp.sum(jnp.log(jnp.diag(LSa)))

            # Inducing point correction
            Kaadiff = t.Kaa_cur - t.Lbinv_Kba.T @ t.Lbinv_Kba
            Sainv_Kaadiff = jnp.linalg.solve(Saa, Kaadiff)
            Kainv_Kaadiff = jnp.linalg.solve(t.Kaa, Kaadiff)
            bound = bound - 0.5 * jnp.sum(
                jnp.diag(Sainv_Kaadiff) - jnp.diag(Kainv_Kaadiff)
            )

        return bound, km, resolved_sample

    def predict(
        self,
        Xnew: Float[Array, "Ns D"],
        *,
        sample: SpectralSample | None = None,
        key: PRNGKeyArray | None = None,
        full_cov: bool = False,
        include_noise: bool = True,
    ) -> tuple[Float[Array, "Ns 1"], Float[Array, "..."]]:
        """Compute predictive mean and variance at new points."""
        t, km, resolved_sample = self._common_terms(sample=sample, key=key)

        Kuu, Kfu_new, Kff_new = self.inducing_strategy.compute_prediction_matrices(
            self.kernel, Xnew, sample=resolved_sample, key=key
        )
        Kbs = Kfu_new.T

        Lbinv_Kbs = jax.scipy.linalg.solve_triangular(t.Lb, Kbs, lower=True)
        LDinv_Lbinv_Kbs = jax.scipy.linalg.solve_triangular(
            t.LD, Lbinv_Kbs, lower=True
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
                var = var + jnp.exp(self.log_noise) * jnp.eye(var.shape[0])
        else:
            var1 = self.kernel(Xnew, diag=True)
            var2 = -jnp.sum(Lbinv_Kbs**2, axis=0)
            var3 = jnp.sum(LDinv_Lbinv_Kbs**2, axis=0)
            var = var1 + var2 + var3
            if include_noise:
                var = var + jnp.exp(self.log_noise) * jnp.ones_like(var)

        return mean, var

    def qb_opt(
        self,
        *,
        sample: SpectralSample | None = None,
        key: PRNGKeyArray | None = None,
        full_cov: bool = True,
    ) -> tuple[Float[Array, "M 1"], Float[Array, "M M"], Float[Array, "M M"]]:
        """Compute optimal posterior at inducing points."""
        t, km, _ = self._common_terms(sample=sample, key=key)
        Kss = t.Kbb
        Kbs = t.Kbb

        Lbinv_Kbs = jax.scipy.linalg.solve_triangular(t.Lb, Kbs, lower=True)
        LDinv_Lbinv_Kbs = jax.scipy.linalg.solve_triangular(
            t.LD, Lbinv_Kbs, lower=True
        )
        mean = LDinv_Lbinv_Kbs.T @ t.LDinv_Lbinv_c

        if full_cov:
            var = (
                Kss
                - Lbinv_Kbs.T @ Lbinv_Kbs
                + LDinv_Lbinv_Kbs.T @ LDinv_Lbinv_Kbs
            )
        else:
            var1 = jnp.diag(Kss)
            var2 = -jnp.sum(Lbinv_Kbs**2, axis=0)
            var3 = jnp.sum(LDinv_Lbinv_Kbs**2, axis=0)
            var = var1 + var2 + var3

        return mean, var, t.Kbb

    def get_posterior_state(
        self,
        *,
        sample: SpectralSample | None = None,
        key: PRNGKeyArray | None = None,
    ) -> SGPRPosteriorState:
        """Extract current optimal posterior state for transfer."""
        mu, Su, Kbb = self.qb_opt(sample=sample, key=key, full_cov=True)
        _, km, _ = self._common_terms(sample=sample, key=key)
        return SGPRPosteriorState(
            mu=mu,
            Su=Su,
            Kaa=Kbb,
            Z=km.Z,
        )

"""Online Sparse Variational GP model.

Consolidates ELBO, prediction, correction, and KL computations
into a single Equinox module. Z matrix computation strategy is
delegated to an InducingStrategy.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from hipposvgp.inducing.base import InducingStrategy
from hipposvgp.kernels.base import Kernel
from hipposvgp.linalg import add_jitter, cholesky, whitened_kl
from hipposvgp.likelihoods.base import Likelihood
from hipposvgp.types import KernelMatrices, PosteriorState, SpectralSample


class OnlineSVGP(eqx.Module):
    """Online sparse variational GP.

    Supports HiPPO-ODE, analytic HiPPO, and standard inducing strategies
    via the strategy pattern. All state is immutable (Equinox convention).
    """

    kernel: Kernel
    likelihood: Likelihood
    inducing_strategy: InducingStrategy
    mv: Float[Array, "M"]
    Lv: Float[Array, "M M"]
    # Prior state from previous task (None if first task)
    mv_old: Float[Array, "M"] | None = None
    Lv_old: Float[Array, "M M"] | None = None
    Kaa_old: Float[Array, "M M"] | None = None
    Z_old: Float[Array, "TwoR M"] | None = None
    # Config
    jitter: float = eqx.field(static=True, default=1e-6)
    use_robust_cholesky: bool = eqx.field(static=True, default=False)
    flag_update_kernel: bool = eqx.field(static=True, default=True)

    def __init__(
        self,
        kernel: Kernel,
        likelihood: Likelihood,
        inducing_strategy: InducingStrategy,
        *,
        prior_state: PosteriorState | None = None,
        jitter: float = 1e-6,
        use_robust_cholesky: bool = False,
        flag_update_kernel: bool = True,
    ):
        self.kernel = kernel
        self.likelihood = likelihood
        self.inducing_strategy = inducing_strategy
        self.jitter = jitter
        self.use_robust_cholesky = use_robust_cholesky
        self.flag_update_kernel = flag_update_kernel

        M = inducing_strategy.inducing_size
        self.mv = jnp.zeros(M)
        self.Lv = jnp.eye(M)

        if prior_state is not None:
            self.mv_old = prior_state.mv
            self.Lv_old = prior_state.Lv
            self.Kaa_old = prior_state.Kaa
            self.Z_old = prior_state.Z
        else:
            self.mv_old = None
            self.Lv_old = None
            self.Kaa_old = None
            self.Z_old = None

    @property
    def _has_prior(self) -> bool:
        return self.mv_old is not None

    def _cholesky(self, K: Float[Array, "N N"]) -> Float[Array, "N N"]:
        return cholesky(K, use_robust=self.use_robust_cholesky, jitter=self.jitter)

    def _compute_qf(
        self,
        Kuu: Float[Array, "M M"],
        Kfu: Float[Array, "N M"],
        Kff: Float[Array, "N N"],
        L_Kuu: Float[Array, "M M"],
    ) -> tuple[Float[Array, "N 1"], Float[Array, "N N"]]:
        """Compute mean and covariance of q(f)."""
        mu_u = L_Kuu @ self.mv[:, None]  # [M, 1]
        L_Kuu_inv_mu_u = jax.scipy.linalg.cho_solve((L_Kuu, True), mu_u)
        mean_f = Kfu @ L_Kuu_inv_mu_u  # [N, 1]

        L_Kuu_inv_Kfu_T = jax.scipy.linalg.cho_solve((L_Kuu, True), Kfu.T)
        Qff = Kfu @ L_Kuu_inv_Kfu_T

        L_u = L_Kuu @ self.Lv
        temp = jax.scipy.linalg.cho_solve((L_Kuu, True), L_u)
        V = Kfu @ temp
        Cov_f = Kff - Qff + V @ V.T

        return mean_f, Cov_f

    def _quold_new(
        self,
        Kuu: Float[Array, "M M"],
        Kuu_old: Float[Array, "M Mp"],
        Kold_old_newk: Float[Array, "Mp Mp"],
    ) -> tuple[Float[Array, "Mp 1"], Float[Array, "Mp Mp"]]:
        """Predict old inducing points under new posterior."""
        Kff_new = Kold_old_newk
        Kfu_new = Kuu_old.T

        Luu = self._cholesky(Kuu)
        mu_u = (Luu @ self.mv)[:, None]
        L_u = Luu @ self.Lv

        alpha = jax.scipy.linalg.cho_solve((Luu, True), mu_u)
        mean = Kfu_new @ alpha

        Kuu_inv_Kfu = jax.scipy.linalg.cho_solve((Luu, True), Kfu_new.T)
        var = Kff_new - Kfu_new @ Kuu_inv_Kfu

        temp = jax.scipy.linalg.cho_solve((Luu, True), L_u)
        V = Kfu_new @ temp
        var = var + V @ V.T

        return mean, var

    def _correction_term(self, km: KernelMatrices) -> Float[Array, ""]:
        """Compute online correction: log ratio between old and new posterior."""
        mv_old = self.mv_old[:, None]
        Lv_old = self.Lv_old

        Luu_old = self._cholesky(km.Kold_old)
        m_old = Luu_old @ mv_old
        S_old = Luu_old @ Lv_old @ Lv_old.T @ Luu_old.T

        mu, Sigma = self._quold_new(km.Kuu, km.Kuu_old, km.Kold_old_newk)
        Smm = Sigma + mu @ mu.T

        LSa = self._cholesky(S_old)
        LKa = self._cholesky(km.Kold_old)

        obj = jnp.sum(jnp.log(jnp.diag(LKa))) - jnp.sum(jnp.log(jnp.diag(LSa)))

        Sainv_ma = jax.scipy.linalg.cho_solve((LSa, True), m_old).squeeze(-1)
        obj = obj - 0.5 * jnp.sum(m_old.squeeze(-1) * Sainv_ma)
        obj = obj + jnp.sum(mu.squeeze(-1) * Sainv_ma)

        Sainv_Smm = jax.scipy.linalg.cho_solve((LSa, True), Smm)
        Kainv_Smm = jax.scipy.linalg.cho_solve((LKa, True), Smm)
        obj = obj - 0.5 * (jnp.trace(Sainv_Smm) - jnp.trace(Kainv_Smm))

        return obj

    def elbo(
        self,
        X: Float[Array, "N D"],
        y: Float[Array, "N 1"],
        *,
        sample: SpectralSample | None = None,
        key: PRNGKeyArray | None = None,
        beta: float = 1.0,
    ) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""], KernelMatrices]:
        """Compute the evidence lower bound.

        Args:
            X: [N, D] training inputs
            y: [N, 1] training targets
            sample: pre-sampled SpectralSample (None to sample)
            key: PRNG key for sampling (needed if sample is None or for f samples)
            beta: scaling for KL term

        Returns:
            (elbo, ve_sum, kl, km): ELBO, variational expectation, KL, kernel matrices
        """
        if key is not None:
            key_sample, key_eps = jax.random.split(key)
        else:
            key_sample = None
            key_eps = None

        # 1. Compute kernel matrices
        km, sample = self.inducing_strategy.compute_kernel_matrices(
            self.kernel,
            X,
            Z_old=self.Z_old,
            Kaa_old=self.Kaa_old,
            flag_update_kernel=self.flag_update_kernel,
            sample=sample,
            key=key_sample,
        )

        # 2. Cholesky of Kuu
        L_Kuu = self._cholesky(km.Kuu)

        # 3. KL divergence
        kl = whitened_kl(self.mv, self.Lv)

        # 4. q(f) distribution
        mean_f, Cov_f = self._compute_qf(km.Kuu, km.Kfu, km.Kff, L_Kuu)
        Cov_f = add_jitter(Cov_f, jitter=self.jitter)

        # 5. Sample f ~ q(f)
        L_Cov_f = jnp.linalg.cholesky(Cov_f)
        eps = jax.random.normal(key_eps, mean_f.shape) if key_eps is not None else jnp.zeros_like(mean_f)
        f_samples = mean_f + L_Cov_f @ eps

        # 6. Variational expectations
        ve = self.likelihood.log_prob(f_samples, y)
        ve_sum = jnp.sum(ve)

        # 7. ELBO
        elbo_val = ve_sum - beta * kl

        # 8. Online correction
        if self.Z_old is not None and km.Kuu_old is not None:
            corr = self._correction_term(km)
            elbo_val = elbo_val + corr

        return elbo_val, ve_sum, kl, km

    def predict(
        self,
        Xnew: Float[Array, "Ns D"],
        *,
        sample: SpectralSample | None = None,
        key: PRNGKeyArray | None = None,
        full_cov: bool = False,
        include_noise: bool = False,
    ) -> tuple[Float[Array, "Ns 1"], Float[Array, "..."]]:
        """Compute predictive mean and variance."""
        Kuu, Kfu_new, Kff_new = self.inducing_strategy.compute_prediction_matrices(
            self.kernel, Xnew, sample=sample, key=key
        )
        Luu = self._cholesky(Kuu)

        mu_u = (Luu @ self.mv)[:, None]
        L_u = Luu @ self.Lv

        alpha = jax.scipy.linalg.cho_solve((Luu, True), mu_u)
        mean = Kfu_new @ alpha

        Kuu_inv_Kfu = jax.scipy.linalg.cho_solve((Luu, True), Kfu_new.T)

        if full_cov:
            var = Kff_new - Kfu_new @ Kuu_inv_Kfu
            temp = jax.scipy.linalg.cho_solve((Luu, True), L_u)
            V = Kfu_new @ temp
            var = var + V @ V.T
            if include_noise and hasattr(self.likelihood, "variance"):
                var = var + self.likelihood.variance * jnp.eye(var.shape[0])
        else:
            temp = jax.scipy.linalg.cho_solve((Luu, True), L_u)
            var_diag_extra = jnp.sum((Kfu_new @ temp) ** 2, axis=1)
            var = (
                jnp.diag(Kff_new)
                - jnp.sum(Kfu_new * Kuu_inv_Kfu.T, axis=1)
                + var_diag_extra
            )
            var = var[:, None]
            if include_noise and hasattr(self.likelihood, "variance"):
                var = var + self.likelihood.variance

        return mean, var

    def get_posterior_state(self, km: KernelMatrices) -> PosteriorState:
        """Extract current posterior state for transfer to next task."""
        return PosteriorState(
            mv=self.mv,
            Lv=self.Lv,
            Kaa=km.Kuu,
            Z=km.Z,
        )

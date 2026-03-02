"""Exact (full-rank) GP regression.

Pure functional design: training data is passed explicitly to methods,
not stored as mutable state.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from hipposvgp.kernels.base import Kernel
from hipposvgp.linalg import add_jitter, cholesky
from hipposvgp.likelihoods.gaussian import GaussianLikelihood


class ExactGP(eqx.Module):
    """Exact GP regression with Gaussian observation noise.

    Pure functional: training data is passed to each method explicitly.
    """

    kernel: Kernel
    likelihood: GaussianLikelihood
    jitter: float = eqx.field(static=True, default=1e-6)
    use_robust_cholesky: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        kernel: Kernel,
        likelihood: GaussianLikelihood | None = None,
        *,
        jitter: float = 1e-6,
        use_robust_cholesky: bool = False,
    ):
        self.kernel = kernel
        self.likelihood = likelihood if likelihood is not None else GaussianLikelihood(1.0)
        self.jitter = jitter
        self.use_robust_cholesky = use_robust_cholesky

    def _cholesky(self, K: Float[Array, "N N"]) -> Float[Array, "N N"]:
        return cholesky(K, use_robust=self.use_robust_cholesky, jitter=self.jitter)

    def _train_factor(
        self,
        X: Float[Array, "N D"],
        y: Float[Array, "N P"],
    ) -> tuple[Float[Array, "N N"], Float[Array, "N P"]]:
        """Compute Cholesky factor and alpha = (K + sigma^2 I)^{-1} y."""
        Kxx = self.kernel(X)
        noise = self.likelihood.variance
        K_noise = Kxx + noise * jnp.eye(Kxx.shape[0])
        L = self._cholesky(K_noise)
        alpha = jax.scipy.linalg.cho_solve((L, True), y)
        return L, alpha

    def nll(
        self,
        X: Float[Array, "N D"],
        y: Float[Array, "N P"],
    ) -> Float[Array, ""]:
        """Negative log marginal likelihood."""
        L, alpha = self._train_factor(X, y)
        n = X.shape[0]
        p = y.shape[1]

        data_fit = 0.5 * jnp.sum(y * alpha)
        log_det = p * jnp.sum(jnp.log(jnp.diag(L)))
        normalizer = 0.5 * n * p * math.log(2.0 * math.pi)

        return data_fit + log_det + normalizer

    def predict(
        self,
        Xnew: Float[Array, "Ns D"],
        X_train: Float[Array, "N D"],
        y_train: Float[Array, "N P"],
        *,
        full_cov: bool = False,
        include_noise: bool = True,
    ) -> tuple[Float[Array, "Ns P"], Float[Array, "..."]]:
        """Predictive mean and uncertainty at test inputs."""
        L, alpha = self._train_factor(X_train, y_train)
        Kxs = self.kernel(X_train, Xnew)  # [N, Ns]
        mean = Kxs.T @ alpha  # [Ns, P]

        v = jax.scipy.linalg.solve_triangular(L, Kxs, lower=True)

        if full_cov:
            Kss = self.kernel(Xnew)
            cov = Kss - v.T @ v
            cov = add_jitter(cov, jitter=self.jitter)
            if include_noise:
                cov = cov + self.likelihood.variance * jnp.eye(cov.shape[0])
            return mean, cov

        var = self.kernel(Xnew, diag=True) - jnp.sum(v**2, axis=0)
        var = jnp.maximum(var, 0.0)[:, None]
        if include_noise:
            var = var + self.likelihood.variance
        return mean, var

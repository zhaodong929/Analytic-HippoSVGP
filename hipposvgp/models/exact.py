"""Exact (full-rank) GP regression.

This module provides a non-online GP baseline with exact marginal
likelihood training (Gaussian likelihood only).
"""

import math

import torch
import torch.nn as nn

from hipposvgp.linalg import add_jitter, cholesky
from hipposvgp.likelihoods import GaussianLikelihood


class ExactGP(nn.Module):
    """Exact GP regression with Gaussian observation noise.

    Args:
        kernel: kernel module (e.g. SquaredExponential)
        likelihood: GaussianLikelihood (optional; created if None)
        device: torch device
        use_robust_cholesky: use retry-based Cholesky with growing jitter
        jitter: base jitter level
        dtype: floating point dtype
    """

    def __init__(
        self,
        kernel,
        likelihood=None,
        device="cpu",
        use_robust_cholesky=False,
        jitter=1e-6,
        dtype=torch.float64,
    ):
        super().__init__()

        self.device = torch.device(device)
        self.dtype = dtype
        self.use_robust_cholesky = use_robust_cholesky
        self.jitter = jitter

        self.kernel = kernel.to(device=self.device, dtype=self.dtype)

        if likelihood is None:
            likelihood = GaussianLikelihood(variance=1.0)
        if not isinstance(likelihood, GaussianLikelihood):
            raise TypeError(
                "ExactGP supports GaussianLikelihood only. "
                "Use OnlineSVGP for non-Gaussian likelihoods."
            )
        self.likelihood = likelihood.to(device=self.device, dtype=self.dtype)

        self.X_train = None
        self.y_train = None

    def set_train_data(self, X, y):
        """Set or replace the training set."""
        X_t = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        y_t = torch.as_tensor(y, dtype=self.dtype, device=self.device)

        if X_t.dim() != 2:
            raise ValueError(f"X must be [N, D], got shape {tuple(X_t.shape)}")
        if y_t.dim() == 1:
            y_t = y_t.unsqueeze(-1)
        if y_t.dim() != 2:
            raise ValueError(f"y must be [N, P], got shape {tuple(y_t.shape)}")
        if X_t.shape[0] != y_t.shape[0]:
            raise ValueError(
                f"X and y must have same number of rows, got {X_t.shape[0]} and {y_t.shape[0]}"
            )

        self.X_train = X_t
        self.y_train = y_t

    def _resolve_train_data(self, X=None, y=None):
        if X is None and y is None:
            if self.X_train is None or self.y_train is None:
                raise RuntimeError(
                    "Training data is not set. "
                    "Call set_train_data(X, y) or pass X and y to the method."
                )
            return self.X_train, self.y_train

        if X is None or y is None:
            raise ValueError("Both X and y must be provided together.")

        X_t = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        y_t = torch.as_tensor(y, dtype=self.dtype, device=self.device)
        if y_t.dim() == 1:
            y_t = y_t.unsqueeze(-1)
        if X_t.dim() != 2 or y_t.dim() != 2:
            raise ValueError("X must be [N, D] and y must be [N, P].")
        if X_t.shape[0] != y_t.shape[0]:
            raise ValueError("X and y must have same number of rows.")
        return X_t, y_t

    def _cholesky(self, K):
        return cholesky(K, use_robust=self.use_robust_cholesky, jitter=self.jitter)

    def _train_factor(self, X, y):
        Kxx = self.kernel(X)
        noise = self.likelihood.variance
        K_noise = Kxx + noise * torch.eye(
            Kxx.shape[0], device=self.device, dtype=self.dtype
        )
        L = self._cholesky(K_noise)
        alpha = torch.cholesky_solve(y, L)
        return L, alpha

    def nll(self, X=None, y=None):
        """Negative log marginal likelihood."""
        X_t, y_t = self._resolve_train_data(X, y)
        L, alpha = self._train_factor(X_t, y_t)

        n = X_t.shape[0]
        p = y_t.shape[1]

        data_fit = 0.5 * torch.sum(y_t * alpha)
        log_det = p * torch.sum(torch.log(torch.diagonal(L)))
        normalizer = 0.5 * n * p * math.log(2.0 * math.pi)

        return data_fit + log_det + normalizer

    def log_marginal_likelihood(self, X=None, y=None):
        """Exact log marginal likelihood."""
        return -self.nll(X, y)

    def posterior(self, Xnew, X=None, y=None, full_cov=False):
        """Posterior over latent function values f* (without observation noise)."""
        return self.predict(
            Xnew, X=X, y=y, full_cov=full_cov, include_noise=False
        )

    def predict(self, Xnew, X=None, y=None, full_cov=False, include_noise=True):
        """Predictive mean and uncertainty at test inputs.

        Args:
            Xnew: [N*, D] test inputs
            X: optional training inputs [N, D]
            y: optional training targets [N, P]
            full_cov: if True, return full covariance [N*, N*]
            include_noise: include observation noise in predictive uncertainty

        Returns:
            mean: [N*, P]
            var: [N*, 1] if ``full_cov=False`` else [N*, N*]
        """
        X_t, y_t = self._resolve_train_data(X, y)
        Xnew_t = torch.as_tensor(Xnew, dtype=self.dtype, device=self.device)
        if Xnew_t.dim() == 1:
            Xnew_t = Xnew_t.unsqueeze(-1)
        if Xnew_t.dim() != 2:
            raise ValueError(f"Xnew must be [N*, D], got {tuple(Xnew_t.shape)}")
        if Xnew_t.shape[1] != X_t.shape[1]:
            raise ValueError(
                f"Input dimension mismatch: train D={X_t.shape[1]}, test D={Xnew_t.shape[1]}"
            )

        L, alpha = self._train_factor(X_t, y_t)
        Kxs = self.kernel(X_t, Xnew_t)  # [N, N*]
        mean = Kxs.T @ alpha  # [N*, P]

        v = torch.linalg.solve_triangular(L, Kxs, upper=False)

        if full_cov:
            Kss = self.kernel(Xnew_t)
            cov = Kss - v.T @ v
            cov = add_jitter(cov, jitter=self.jitter)
            if include_noise:
                cov = cov + self.likelihood.variance * torch.eye(
                    cov.shape[0], device=self.device, dtype=self.dtype
                )
            return mean, cov

        var = self.kernel(Xnew_t, diag=True) - torch.sum(v.pow(2), dim=0)
        var = torch.clamp(var, min=0.0).unsqueeze(-1)
        if include_noise:
            var = var + self.likelihood.variance
        return mean, var

    def fit(self, X, y, num_epochs=1000, lr=0.01, verbose=False, log_every=100):
        """Simple full-batch optimizer loop over exact NLL.

        Returns:
            List of NLL values per epoch.
        """
        self.set_train_data(X, y)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        history = []

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.nll()
            loss.backward()
            optimizer.step()

            history.append(loss.item())
            if verbose and ((epoch + 1) % log_every == 0 or epoch == 0):
                print(f"  epoch {epoch + 1:5d}  NLL={loss.item():.4f}")

        return history

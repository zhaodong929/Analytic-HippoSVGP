"""Common evaluation metrics for GP models."""

import math

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import norm


def rmse(y_true, y_pred):
    """Root mean squared error."""
    y_true = jnp.squeeze(y_true)
    y_pred = jnp.squeeze(y_pred)
    return float(jnp.sqrt(jnp.mean((y_pred - y_true) ** 2)))


def r_squared(y_true, y_pred):
    """Coefficient of determination R^2."""
    y_true = jnp.squeeze(y_true)
    y_pred = jnp.squeeze(y_pred)
    mse = jnp.mean((y_pred - y_true) ** 2)
    var = jnp.mean((y_true - jnp.mean(y_true)) ** 2)
    if float(var) > 0:
        return float(1 - mse / var)
    return float("nan")


def nlpd(mean, var, y_test, likelihood=None, num_mc_samples=1000, key=None):
    """Negative log predictive density.

    If likelihood is provided, uses MC sampling through the likelihood.
    Otherwise assumes Gaussian observations.

    Args:
        mean: [N, 1] predictive mean
        var: [N, 1] predictive variance
        y_test: [N, 1] test targets
        likelihood: optional likelihood module for non-Gaussian observation models
        num_mc_samples: MC samples for likelihood integration
        key: JAX PRNG key (required when likelihood is provided)

    Returns:
        Scalar NLPD value.
    """
    if likelihood is not None and hasattr(likelihood, "log_prob"):
        if key is None:
            key = jax.random.PRNGKey(0)
        # MC integration through likelihood
        std = jnp.sqrt(var)
        f_samples = mean + std * jax.random.normal(
            key, (num_mc_samples, var.shape[0], var.shape[1])
        )
        y_expanded = jnp.broadcast_to(y_test[None, :, :], (num_mc_samples,) + y_test.shape)

        log_probs = likelihood.log_prob(
            f_samples.reshape(-1, 1), y_expanded.reshape(-1, 1)
        ).reshape(num_mc_samples, -1)
        # log-mean-exp over samples
        return float(
            -jax.scipy.special.logsumexp(log_probs, axis=0).mean()
            + np.log(num_mc_samples)
        )
    else:
        # Gaussian NLPD
        return float(
            jnp.mean(
                0.5 * (jnp.log(2 * math.pi * var) + (y_test - mean) ** 2 / var)
            )
        )


def gaussian_ece(mean, var, y_test, levels=None):
    """Expected calibration error from Gaussian predictive distribution.

    Computes ECE by checking what fraction of test points fall within
    each confidence interval of the Gaussian predictive distribution.

    Args:
        mean: [N, 1] or [N] predictive mean
        var: [N, 1] or [N] predictive variance
        y_test: [N, 1] or [N] test targets
        levels: confidence levels to check (default: 0.1 to 1.0)

    Returns:
        Scalar ECE value.
    """
    if levels is None:
        levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    mean = jnp.squeeze(mean)
    var = jnp.squeeze(var)
    y_test = jnp.squeeze(y_test)
    std = jnp.sqrt(jnp.clip(var, a_min=1e-10))

    # Standardized residuals
    z = (y_test - mean) / std

    ece_list = []
    for conf in levels:
        q = norm.ppf(0.5 + conf / 2)
        in_interval = jnp.mean((jnp.abs(z) < q).astype(jnp.float64))
        ece_list.append(abs(float(in_interval) - conf))

    return float(np.mean(ece_list))


def expected_calibration_error(y_samples, y_test, levels=None):
    """Expected calibration error from posterior samples.

    Args:
        y_samples: [S, N] posterior predictive samples
        y_test: [N] or [N, 1] test targets
        levels: confidence levels to check (default: 0.1 to 1.0)

    Returns:
        Scalar ECE value.
    """
    if levels is None:
        levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    y_test = jnp.squeeze(y_test)
    ece_list = []
    for conf in levels:
        lower = jnp.quantile(y_samples, 0.5 - conf / 2, axis=0)
        upper = jnp.quantile(y_samples, 0.5 + conf / 2, axis=0)
        in_interval = jnp.mean(
            ((y_test >= lower) & (y_test <= upper)).astype(jnp.float64)
        )
        ece_list.append(abs(float(in_interval) - conf))

    return float(np.mean(ece_list))

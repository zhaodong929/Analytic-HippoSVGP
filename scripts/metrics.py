"""Common evaluation metrics for GP models."""

import math

import numpy as np
import torch


def rmse(y_true, y_pred):
    """Root mean squared error."""
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()


def r_squared(y_true, y_pred):
    """Coefficient of determination R^2."""
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()
    mse = torch.mean((y_pred - y_true) ** 2)
    var = torch.mean((y_true - y_true.mean()) ** 2)
    if var.item() > 0:
        return (1 - mse / var).item()
    return float("nan")


def nlpd(mean, var, y_test, likelihood=None, num_mc_samples=1000, device="cpu"):
    """Negative log predictive density.

    If likelihood is provided, uses MC sampling through the likelihood.
    Otherwise assumes Gaussian observations.

    Args:
        mean: [N, 1] predictive mean
        var: [N, 1] predictive variance
        y_test: [N, 1] test targets
        likelihood: optional likelihood module for non-Gaussian observation models
        num_mc_samples: MC samples for likelihood integration
        device: torch device

    Returns:
        Scalar NLPD value.
    """
    if likelihood is not None and hasattr(likelihood, "log_prob"):
        # MC integration through likelihood
        std = torch.sqrt(var)
        f_samples = mean + std * torch.randn(
            num_mc_samples, var.shape[0], var.shape[1], device=device
        )
        y_expanded = y_test.unsqueeze(0).expand(num_mc_samples, -1, -1)

        log_probs = likelihood.log_prob(
            f_samples.reshape(-1, 1), y_expanded.reshape(-1, 1)
        ).reshape(num_mc_samples, -1)
        # log-mean-exp over samples
        return -torch.logsumexp(log_probs, dim=0).mean().item() + np.log(
            num_mc_samples
        )
    else:
        # Gaussian NLPD
        return (
            0.5
            * (
                torch.log(2 * math.pi * var)
                + (y_test - mean) ** 2 / var
            )
            .mean()
            .item()
        )


def expected_calibration_error(
    y_samples, y_test, levels=None
):
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

    y_test = y_test.squeeze()
    ece_list = []
    for conf in levels:
        lower = torch.quantile(y_samples, 0.5 - conf / 2, dim=0)
        upper = torch.quantile(y_samples, 0.5 + conf / 2, dim=0)
        in_interval = ((y_test >= lower) & (y_test <= upper)).float().mean()
        ece_list.append(abs(in_interval.item() - conf))

    return float(np.mean(ece_list))

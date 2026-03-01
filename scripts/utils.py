"""Shared utilities for training and evaluation scripts."""

import math
import random

import numpy as np
import torch


def seed_everything(seed=0):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_time_window(X_train, step_size, prev_end=None):
    """Compute discrete time window from training data.

    Args:
        X_train: training input tensor
        step_size: discretization step size
        prev_end: end of previous task's time window (None for first task)

    Returns:
        (start, end, num_steps) tuple
    """
    if prev_end is not None:
        start = prev_end
    else:
        start = torch.min(X_train).item()
    end_data = torch.max(X_train).item()
    maxT = end_data - start
    num_steps = max(1, math.ceil(maxT / step_size))
    end = start + num_steps * step_size
    return start, end, num_steps


def train_loop(model, num_epochs, lr, *, compute_elbo, log_every=1000):
    """Common training loop for SVGP and SGPR models.

    Args:
        model: nn.Module to train
        num_epochs: number of training epochs
        lr: learning rate
        compute_elbo: callable () -> (elbo_scalar,) or () -> (elbo, ve, kl)
        log_every: print interval

    Returns:
        history: list of dicts with epoch-level metrics
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        try:
            result = compute_elbo()
            if isinstance(result, tuple):
                loss = result[0]
            else:
                loss = result
            neg_loss = -loss
        except torch.linalg.LinAlgError as e:
            print(f"LinAlgError at epoch {epoch}: {e}")
            break
        neg_loss.backward()
        optimizer.step()

        # Record history
        entry = {"epoch": epoch, "elbo": loss.item()}
        if isinstance(result, tuple) and len(result) == 3:
            entry["ve"] = result[1].item()
            entry["kl"] = result[2].item()
        history.append(entry)

        if epoch % log_every == 0 or epoch == num_epochs - 1:
            if isinstance(result, tuple) and len(result) == 3:
                elbo, ve, kl = result
                print(
                    f"  epoch {epoch:5d}  ELBO={elbo.item():.2f}  "
                    f"VE={ve.item():.2f}  KL={kl.item():.4f}"
                )
            else:
                print(f"  epoch {epoch:5d}  ELBO={loss.item():.2f}")

    return history

"""Shared utilities for training and evaluation scripts."""

import math
import random

import jax.numpy as jnp
import numpy as np


def seed_everything(seed=0):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def compute_time_window(X_train, step_size, prev_end=None):
    """Compute discrete time window from training data.

    Args:
        X_train: training input array
        step_size: discretization step size
        prev_end: end of previous task's time window (None for first task)

    Returns:
        (start, end, num_steps) tuple
    """
    if prev_end is not None:
        start = prev_end
    else:
        start = float(jnp.min(X_train))
    end_data = float(jnp.max(X_train))
    maxT = end_data - start
    num_steps = max(1, math.ceil(maxT / step_size))
    end = start + num_steps * step_size
    return start, end, num_steps

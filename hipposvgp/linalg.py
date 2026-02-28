"""Shared linear algebra utilities for all models."""

import torch


def add_jitter(K, jitter=1e-6):
    """K + jitter * mean(diag(K)) * I for numerical stability."""
    return K + jitter * torch.mean(torch.diag(K)).detach() * torch.eye(
        K.shape[0], device=K.device, dtype=K.dtype
    )


def robust_cholesky(matrix, base_jitter=1e-6, max_retries=10):
    """Cholesky decomposition with progressively increasing jitter."""
    for i in range(max_retries):
        try:
            factor = 10 ** i if i > 0 else 1.0
            current_jitter = base_jitter * factor
            mat_reg = matrix + current_jitter * torch.eye(
                matrix.shape[0], device=matrix.device, dtype=matrix.dtype
            )
            return torch.linalg.cholesky(mat_reg)
        except torch.linalg.LinAlgError:
            continue
    raise torch.linalg.LinAlgError(
        f"Robust Cholesky failed after {max_retries} retries."
    )


def whitened_kl(mv, Lv):
    """KL[N(mv, Lv @ Lv^T) || N(0, I)] in whitened space.

    Args:
        mv: [M] variational mean
        Lv: [M, M] lower-triangular factor of variational covariance

    Returns:
        Scalar KL divergence.
    """
    M = mv.shape[0]
    Sv = Lv @ Lv.T
    trace_term = torch.trace(Sv)
    quadratic_term = torch.sum(mv ** 2)
    log_det_Sv = 2.0 * torch.sum(torch.log(torch.abs(torch.diagonal(Lv))))
    return 0.5 * (trace_term + quadratic_term - M - log_det_Sv)


def cholesky(matrix, use_robust=False, jitter=1e-6):
    """Cholesky decomposition with optional robustness.

    Args:
        matrix: positive-definite matrix
        use_robust: if True, use progressive jitter retries
        jitter: base jitter level
    """
    if use_robust:
        return robust_cholesky(matrix, base_jitter=jitter)
    return torch.linalg.cholesky(add_jitter(matrix, jitter=jitter))

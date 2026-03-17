"""Utilities for Kronecker-aware linear algebra."""

from __future__ import annotations

import math
from typing import Tuple

import torch


def add_diagonal_jitter(matrix: torch.Tensor, jitter: float = 1e-6) -> torch.Tensor:
    """Add scale-aware diagonal jitter."""
    diag_mean = torch.mean(torch.diag(matrix)).detach()
    if not torch.isfinite(diag_mean) or diag_mean <= 0:
        diag_mean = torch.tensor(1.0, dtype=matrix.dtype, device=matrix.device)
    eye = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    return matrix + jitter * diag_mean * eye


def safe_cholesky(
    matrix: torch.Tensor,
    jitter: float = 1e-6,
    retries: int = 6,
) -> torch.Tensor:
    """Run Cholesky with escalating jitter."""
    for retry in range(retries):
        try:
            return torch.linalg.cholesky(add_diagonal_jitter(matrix, jitter * (10.0**retry)))
        except torch.linalg.LinAlgError:
            continue
    return torch.linalg.cholesky(add_diagonal_jitter(matrix, jitter * (10.0**retries)))


def cholesky_inverse(matrix: torch.Tensor, jitter: float = 1e-6) -> torch.Tensor:
    """Compute a numerically stable inverse from a Cholesky factorization."""
    chol = safe_cholesky(matrix, jitter=jitter)
    eye = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    return torch.cholesky_solve(eye, chol)


def logdet_from_cholesky(chol: torch.Tensor) -> torch.Tensor:
    """Return log-determinant from a Cholesky factor."""
    return 2.0 * torch.log(torch.diag(chol)).sum()


def kron_logdet(k_t: torch.Tensor, k_s: torch.Tensor, jitter: float = 1e-6) -> torch.Tensor:
    """Compute log|K_t ⊗ K_s| without forming the full Kronecker matrix."""
    chol_t = safe_cholesky(k_t, jitter=jitter)
    chol_s = safe_cholesky(k_s, jitter=jitter)
    m_t = k_t.shape[0]
    m_s = k_s.shape[0]
    return m_s * logdet_from_cholesky(chol_t) + m_t * logdet_from_cholesky(chol_s)


def kron_inverse(k_t: torch.Tensor, k_s: torch.Tensor, jitter: float = 1e-6) -> torch.Tensor:
    """Materialize (K_t ⊗ K_s)^(-1) from factor inverses."""
    return torch.kron(
        cholesky_inverse(k_t, jitter=jitter),
        cholesky_inverse(k_s, jitter=jitter),
    )


def flatten_grid(matrix: torch.Tensor) -> torch.Tensor:
    """Flatten a `[N_t, N_s]` grid using PyTorch's row-major convention."""
    return matrix.reshape(-1)


def unflatten_grid(vector: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """Undo `flatten_grid`."""
    return vector.reshape(shape)


def kron_factor_matmul(a_t: torch.Tensor, a_s: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Apply `(A_t ⊗ A_s)` to `vec_r(matrix)` without materializing the Kronecker product."""
    return a_t @ matrix @ a_s.transpose(-1, -2)


def kron_factor_vec(a_t: torch.Tensor, a_s: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Apply `(A_t ⊗ A_s)` to a row-major flattened vector."""
    matrix = vector.reshape(a_t.shape[1], a_s.shape[1])
    return flatten_grid(kron_factor_matmul(a_t, a_s, matrix))


def kron_factor_transpose_vec(
    a_t: torch.Tensor,
    a_s: torch.Tensor,
    vector: torch.Tensor,
    grid_shape: Tuple[int, int],
) -> torch.Tensor:
    """Apply `(A_t ⊗ A_s)^T` to a row-major flattened grid."""
    matrix = vector.reshape(grid_shape)
    return flatten_grid(a_t.transpose(-1, -2) @ matrix @ a_s)


def kron_solve(
    k_t: torch.Tensor,
    k_s: torch.Tensor,
    rhs: torch.Tensor,
    jitter: float = 1e-6,
) -> torch.Tensor:
    """Solve `(K_t ⊗ K_s) x = rhs` for row-major flattened `rhs`."""
    chol_t = safe_cholesky(k_t, jitter=jitter)
    chol_s = safe_cholesky(k_s, jitter=jitter)
    rhs_matrix = rhs.reshape(k_t.shape[0], k_s.shape[0])
    left_solved = torch.cholesky_solve(rhs_matrix, chol_t)
    full_solved = torch.cholesky_solve(left_solved.transpose(-1, -2), chol_s).transpose(-1, -2)
    return flatten_grid(full_solved)


def kron_gram(a_t: torch.Tensor, a_s: torch.Tensor) -> torch.Tensor:
    """Return `(A_t ⊗ A_s)^T (A_t ⊗ A_s)` via factor Gramians."""
    return torch.kron(a_t.transpose(-1, -2) @ a_t, a_s.transpose(-1, -2) @ a_s)


def rowwise_quadratic_form(x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Compute `diag(X M X^T)` for dense `X`."""
    return torch.sum((x @ matrix) * x, dim=-1)


def rowwise_quadratic_form_from_precision_cholesky(
    x: torch.Tensor,
    chol_precision: torch.Tensor,
) -> torch.Tensor:
    """Compute `diag(X Precision^{-1} X^T)` from a Cholesky factor of precision."""

    solved_t = torch.cholesky_solve(x.transpose(-1, -2), chol_precision)
    solved = solved_t.transpose(-1, -2)
    return torch.sum(solved * x, dim=-1)


def kron_rowwise_prior_diag(
    a_t: torch.Tensor,
    k_t: torch.Tensor,
    a_s: torch.Tensor,
    k_s: torch.Tensor,
) -> torch.Tensor:
    """Compute diag((A_t⊗A_s)(K_t⊗K_s)(A_t⊗A_s)^T) as an outer product."""
    diag_t = rowwise_quadratic_form(a_t, k_t)
    diag_s = rowwise_quadratic_form(a_s, k_s)
    return torch.outer(diag_t, diag_s)


def gaussian_nll(
    y: torch.Tensor,
    sigma2: torch.Tensor,
    logdet_kuu: torch.Tensor,
    logdet_precision: torch.Tensor,
    info_dot_mean: torch.Tensor,
) -> torch.Tensor:
    """Compute the linear-Gaussian negative log-likelihood used in Stage 1."""
    n = y.numel()
    noise_precision = torch.reciprocal(sigma2)
    quadratic = noise_precision * torch.dot(y, y) - info_dot_mean
    return 0.5 * (
        n * math.log(2.0 * math.pi)
        + n * torch.log(sigma2)
        + logdet_kuu
        + logdet_precision
        + quadratic
    )

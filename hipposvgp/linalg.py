"""Shared linear algebra utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def add_jitter(
    K: Float[Array, "N N"],
    jitter: float = 1e-6,
) -> Float[Array, "N N"]:
    """K + jitter * mean(diag(K)) * I for numerical stability."""
    diag_mean = jnp.mean(jnp.diag(K))
    return K + jitter * jax.lax.stop_gradient(diag_mean) * jnp.eye(K.shape[0])


def robust_cholesky(
    matrix: Float[Array, "N N"],
    base_jitter: float = 1e-6,
    max_retries: int = 10,
) -> Float[Array, "N N"]:
    """Cholesky decomposition with progressively increasing jitter.

    Uses a scan-based approach to find the minimum jitter that works.
    Falls back to adding base_jitter * 10^i for i in 0..max_retries-1.
    """
    n = matrix.shape[0]
    eye = jnp.eye(n)

    def try_cholesky(matrix_reg):
        L = jnp.linalg.cholesky(matrix_reg)
        return jnp.all(jnp.isfinite(L)), L

    def body_fn(carry, factor):
        done, best_L = carry
        current_jitter = base_jitter * factor
        mat_reg = matrix + current_jitter * eye
        success, L = try_cholesky(mat_reg)
        # Only update if not already done and this attempt succeeded
        should_update = (~done) & success
        new_L = jnp.where(should_update, L, best_L)
        new_done = done | success
        return (new_done, new_L), None

    factors = jnp.array([10.0**i if i > 0 else 1.0 for i in range(max_retries)])
    init_L = jnp.zeros_like(matrix)
    (_, result_L), _ = jax.lax.scan(body_fn, (False, init_L), factors)
    return result_L


def whitened_kl(
    mv: Float[Array, "M"],
    Lv: Float[Array, "M M"],
) -> Float[Array, ""]:
    """KL[N(mv, Lv @ Lv^T) || N(0, I)] in whitened space."""
    M = mv.shape[0]
    Sv = Lv @ Lv.T
    trace_term = jnp.trace(Sv)
    quadratic_term = jnp.sum(mv**2)
    log_det_Sv = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(Lv))))
    return 0.5 * (trace_term + quadratic_term - M - log_det_Sv)


def cholesky(
    matrix: Float[Array, "N N"],
    use_robust: bool = False,
    jitter: float = 1e-6,
) -> Float[Array, "N N"]:
    """Cholesky decomposition with optional robustness."""
    if use_robust:
        return robust_cholesky(matrix, base_jitter=jitter)
    return jnp.linalg.cholesky(add_jitter(matrix, jitter=jitter))

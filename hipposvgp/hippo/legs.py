"""
HiPPO-LegS (Legendre Scaled) module for JAX.

Adapted from HazyResearch/hippo-code (Apache License 2.0).
See THIRD_PARTY_NOTICES.md for details.

Uses jax.lax.scan for efficient sequential computation.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from scipy import linalg as la
from scipy import special as ss


def transition(measure: str, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute A, B transition matrices for HiPPO measures."""
    if measure == "legs":
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
    elif measure == "legt":
        Q = np.arange(N, dtype=np.float64)
        R = (2 * Q + 1) ** 0.5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.0) ** (i - j), 1) * R[None, :]
        B = R[:, None]
        A = -A
    elif measure == "lmu":
        Q = np.arange(N, dtype=np.float64)
        R = (2 * Q + 1)[:, None]
        j, i = np.meshgrid(Q, Q)
        A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
        B = (-1.0) ** Q[:, None] * R
    else:
        raise ValueError(f"Unknown measure: {measure}")
    return A, B


class HiPPO_LegS(eqx.Module):
    """HiPPO-LegS model (scale invariant).

    Pre-computes discretized transition matrices for all timesteps.
    Uses jax.lax.scan for the sequential recurrence.
    """

    N: int = eqx.field(static=True)
    A_stacked: Float[Array, "L N N"]
    B_stacked: Float[Array, "L N"]
    eval_matrix: Float[Array, "L N"]

    def __init__(
        self,
        N: int,
        max_length: int = 2048,
        measure: str = "legs",
        discretization: str = "bilinear",
    ):
        self.N = N
        A, B = transition(measure, N)
        B = B.squeeze(-1)
        A_stacked = np.empty((max_length, N, N), dtype=A.dtype)
        B_stacked = np.empty((max_length, N), dtype=B.dtype)

        eye = np.eye(N)
        for t in range(1, max_length + 1):
            At = A / t
            Bt = B / t
            if discretization == "forward":
                A_stacked[t - 1] = eye + At
                B_stacked[t - 1] = Bt
            elif discretization == "backward":
                A_stacked[t - 1] = la.solve_triangular(
                    eye - At, eye, lower=True
                )
                B_stacked[t - 1] = la.solve_triangular(
                    eye - At, Bt, lower=True
                )
            elif discretization == "bilinear":
                A_stacked[t - 1] = la.solve_triangular(
                    eye - At / 2, eye + At / 2, lower=True
                )
                B_stacked[t - 1] = la.solve_triangular(
                    eye - At / 2, Bt, lower=True
                )
            elif discretization == "zoh":
                A_stacked[t - 1] = la.expm(
                    A * (math.log(t + 1) - math.log(t))
                )
                B_stacked[t - 1] = la.solve_triangular(
                    A, A_stacked[t - 1] @ B - B, lower=True
                )
            else:
                raise ValueError(f"Unknown discretization: {discretization}")

        self.A_stacked = jnp.array(A_stacked)
        self.B_stacked = jnp.array(B_stacked)

        vals = np.linspace(0.0, 1.0, max_length)
        self.eval_matrix = jnp.array(
            (B[:, None] * ss.eval_legendre(np.arange(N)[:, None], 2 * vals - 1)).T
        )

    def __call__(
        self,
        inputs: Float[Array, "L ..."],
        *,
        prev_discrete_steps: int = 0,
        ini: Float[Array, "... N"] | None = None,
        fast: bool = False,
    ) -> Float[Array, "L ... N"]:
        """Run HiPPO sequential scan.

        Args:
            inputs: [L, ...] input signal
            prev_discrete_steps: number of previous steps (for time indexing)
            ini: initial state [... N] or None
            fast: unused (kept for API compatibility; scan is always used)

        Returns:
            outputs: [L, ..., N] HiPPO coefficients at each timestep
        """
        L = inputs.shape[0]
        current_end = prev_discrete_steps + L

        # Slice pre-computed matrices for the relevant timesteps
        A_slice = self.A_stacked[prev_discrete_steps:current_end]  # [L, N, N]
        B_slice = self.B_stacked[prev_discrete_steps:current_end]  # [L, N]

        # Compute u = inputs[..., None] * B -> [L, ..., N]
        # For multi-dim inputs (e.g. [L, 2R]), we need to swap the time
        # dimension with the second-to-last to broadcast B correctly.
        inputs_expanded = inputs[..., None]  # [L, ..., 1]
        if inputs_expanded.ndim > 2:
            # [L, ..., 1] -> [..., L, 1] (swap time to second-to-last)
            u = jnp.swapaxes(inputs_expanded, 0, -2)
            u = u * B_slice  # [..., L, N]  (B_slice is [L, N])
            u = jnp.swapaxes(u, 0, -2)  # [L, ..., N]
        else:
            u = inputs_expanded * B_slice  # [L, 1] * [L, N] = [L, N]

        if ini is None:
            ini = jnp.zeros_like(u[0])

        def scan_fn(carry, x):
            # carry: [..., N] state
            # x: (A_t, u_t) where A_t is [N, N], u_t is [..., N]
            A_t, u_t = x
            # c[t] = A_t @ c[t-1] + u_t
            new_carry = jnp.einsum("ij,...j->...i", A_t, carry) + u_t
            return new_carry, new_carry

        _, outputs = jax.lax.scan(scan_fn, ini, (A_slice, u))
        return outputs

    def reconstruct(
        self, c: Float[Array, "... N"]
    ) -> Float[Array, "... L"]:
        """Reconstruct signal from HiPPO coefficients."""
        return (self.eval_matrix @ c[..., None]).squeeze(-1)

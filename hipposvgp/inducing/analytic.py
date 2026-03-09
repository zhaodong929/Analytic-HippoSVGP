"""Analytic HiPPO inducing strategy using spherical Bessel functions.

Core contribution: closed-form Z matrix computation
instead of numerical ODE integration.

Three backends for spherical Bessel functions:
- "forward_recurrence": Pure JAX forward recurrence (unstable for high orders)
- "scipy": scipy.special.jv via pure_callback + custom_jvp (stable, not JIT-native)
- "spbessax": spbessax library with Cai's descending recurrence (stable, JIT-native)
"""

from __future__ import annotations

import functools
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.special
from jaxtyping import Array, Float

from hipposvgp.inducing.base import RFFInducingBase
from hipposvgp.rff import rff_features
from hipposvgp.types import SpectralSample

# ---------------------------------------------------------------------------
# Backend 1: Forward recurrence (original, unstable for n > x)
# ---------------------------------------------------------------------------


def _spherical_bessel_j_impl(
    lmax: int, x: Float[Array, "..."]
) -> Float[Array, "Lp1 ..."]:
    """Compute spherical Bessel j_0(x)...j_lmax(x) via forward recurrence.

    WARNING: Numerically unstable for large lmax when |x| < lmax.
    """
    x_abs = jnp.abs(x)
    x_safe = jnp.where(x_abs > 1e-30, x_abs, jnp.ones_like(x_abs))

    j0 = jnp.sin(x_safe) / x_safe
    j0 = jnp.where(x_abs > 1e-30, j0, jnp.ones_like(x_abs))

    if lmax == 0:
        result = j0[None, ...]
    else:
        j1 = jnp.sin(x_safe) / x_safe**2 - jnp.cos(x_safe) / x_safe
        j1 = jnp.where(x_abs > 1e-30, j1, jnp.zeros_like(x_abs))

        if lmax == 1:
            result = jnp.stack([j0, j1], axis=0)
        else:
            results = [j0, j1]
            j_prev = j0
            j_curr = j1
            for n in range(1, lmax):
                j_next = (2 * n + 1) / x_safe * j_curr - j_prev
                j_next = jnp.where(
                    x_abs > 1e-30, j_next, jnp.zeros_like(x_abs)
                )
                results.append(j_next)
                j_prev = j_curr
                j_curr = j_next
            result = jnp.stack(results, axis=0)

    # Parity: j_n(-x) = (-1)^n j_n(x)
    ns = jnp.arange(lmax + 1)
    shape = (lmax + 1,) + (1,) * x.ndim
    parity = jnp.where(ns.reshape(shape) % 2 == 0, 1.0, -1.0)
    sign = jnp.where(x >= 0, 1.0, parity)

    return result * sign


def _make_spherical_bessel_j_recurrence(lmax: int):
    """Forward recurrence backend with custom_jvp."""

    @jax.custom_jvp
    def bessel_fn(x):
        return _spherical_bessel_j_impl(lmax, x)

    @bessel_fn.defjvp
    def bessel_fn_jvp(primals, tangents):
        (x,) = primals
        (x_dot,) = tangents

        j_extended = _spherical_bessel_j_impl(lmax + 1, x)
        j = j_extended[: lmax + 1]

        x_safe = jnp.where(jnp.abs(x) > 1e-30, x, jnp.ones_like(x))
        ns = jnp.arange(lmax + 1)
        shape = (lmax + 1,) + (1,) * x.ndim
        ns_shaped = ns.reshape(shape)

        dj = ns_shaped / x_safe * j - j_extended[1 : lmax + 2]
        dj_at_zero = jnp.zeros_like(j)
        dj_at_zero = dj_at_zero.at[1].set(1.0 / 3.0)
        dj = jnp.where(jnp.abs(x) > 1e-30, dj, dj_at_zero)

        return j, dj * x_dot

    return bessel_fn


# ---------------------------------------------------------------------------
# Backend 2: scipy.special.jv via pure_callback + custom_jvp
# ---------------------------------------------------------------------------


def _scipy_spherical_bessel_callback(
    lmax: int, x: Float[Array, "..."]
) -> Float[Array, "Lp1 ..."]:
    """Compute j_0...j_lmax via scipy.special.jv using jax.pure_callback."""

    def _impl(x_arr):
        x_np = np.asarray(x_arr)
        orig_shape = x_np.shape
        x_flat = x_np.ravel()
        N = len(x_flat)

        ns = np.arange(lmax + 1)
        nu = ns + 0.5  # half-integer orders

        x_abs = np.abs(x_flat)
        mask = x_abs > 1e-30
        x_safe = np.where(mask, x_abs, 1.0)

        # scipy.special.jv broadcasts: [L+1, 1] x [1, N] -> [L+1, N]
        jv_vals = scipy.special.jv(nu[:, None], x_safe[None, :])
        prefactor = np.sqrt(np.pi / (2 * x_safe))[None, :]
        j_vals = prefactor * jv_vals

        # Parity for negative x: j_n(-x) = (-1)^n j_n(x)
        sign = np.where(
            x_flat[None, :] >= 0, 1.0, ((-1.0) ** ns)[:, None]
        )
        j_vals *= sign

        # x=0 boundary: j_0(0)=1, j_n(0)=0
        j_at_zero = np.zeros((lmax + 1, N), dtype=np.float64)
        j_at_zero[0] = 1.0
        j_vals = np.where(mask[None, :], j_vals, j_at_zero)

        return j_vals.reshape((lmax + 1,) + orig_shape)

    result_shape = jax.ShapeDtypeStruct(
        (lmax + 1,) + x.shape, x.dtype
    )
    return jax.pure_callback(_impl, result_shape, x)


def _make_spherical_bessel_j_scipy(lmax: int):
    """Scipy backend: stable for all orders, uses pure_callback."""

    @jax.custom_jvp
    def bessel_fn(x):
        return _scipy_spherical_bessel_callback(lmax, x)

    @bessel_fn.defjvp
    def bessel_fn_jvp(primals, tangents):
        (x,) = primals
        (x_dot,) = tangents

        # Compute j_0...j_{lmax+1} for the derivative formula
        j_extended = _scipy_spherical_bessel_callback(lmax + 1, x)
        j = j_extended[: lmax + 1]

        x_safe = jnp.where(jnp.abs(x) > 1e-30, x, jnp.ones_like(x))
        ns = jnp.arange(lmax + 1)
        shape = (lmax + 1,) + (1,) * x.ndim
        ns_shaped = ns.reshape(shape)

        # dj_n/dx = (n/x)*j_n - j_{n+1}
        dj = ns_shaped / x_safe * j - j_extended[1 : lmax + 2]
        dj_at_zero = jnp.zeros_like(j)
        dj_at_zero = dj_at_zero.at[1].set(1.0 / 3.0)
        dj = jnp.where(jnp.abs(x) > 1e-30, dj, dj_at_zero)

        return j, dj * x_dot

    return bessel_fn


# ---------------------------------------------------------------------------
# Backend 3: spbessax (Cai's descending recurrence, stable + JIT-native)
# ---------------------------------------------------------------------------


def _make_spherical_bessel_j_spbessax(lmax: int):
    """spbessax backend: stable for all orders, fully JIT-compatible."""
    from spbessax.functions import create_j_l

    j_l_fn = create_j_l(lmax, output_all=True, dtype=jnp.float64)

    def bessel_fn(x):
        x_abs = jnp.abs(x)
        x_safe = jnp.where(x_abs > 1e-30, x_abs, jnp.ones_like(x_abs))

        result = j_l_fn(x_safe)  # (*x_shape, lmax+1)
        # Move order axis to front: (lmax+1, *x_shape)
        result = jnp.moveaxis(result, -1, 0)

        # Replace NaN with 0 (spbessax can produce NaN for small x + high order)
        result = jnp.where(jnp.isnan(result), 0.0, result)

        # x=0 boundary: j_0(0)=1, j_n(0)=0
        at_zero = jnp.zeros_like(result)
        at_zero = at_zero.at[0].set(1.0)
        result = jnp.where(x_abs > 1e-30, result, at_zero)

        # Parity for negative x: j_n(-x) = (-1)^n j_n(x)
        ns = jnp.arange(lmax + 1)
        shape = (lmax + 1,) + (1,) * x.ndim
        parity = jnp.where(ns.reshape(shape) % 2 == 0, 1.0, -1.0)
        sign = jnp.where(x >= 0, 1.0, parity)

        return result * sign

    return bessel_fn


# ---------------------------------------------------------------------------
# Unified factory with backend selection
# ---------------------------------------------------------------------------

_BACKEND_FACTORIES = {
    "forward_recurrence": _make_spherical_bessel_j_recurrence,
    "scipy": _make_spherical_bessel_j_scipy,
    "spbessax": _make_spherical_bessel_j_spbessax,
}

BESSEL_BACKENDS = tuple(_BACKEND_FACTORIES.keys())


@functools.lru_cache(maxsize=32)
def spherical_bessel_j(lmax: int, backend: str = "spbessax"):
    """Get a spherical Bessel function for a given lmax and backend.

    Args:
        lmax: Maximum order (returns j_0 through j_lmax).
        backend: One of "forward_recurrence", "scipy", "spbessax".

    Returns:
        A function f(x) -> j[0..lmax](x) with shape (lmax+1, *x.shape).
    """
    if backend not in _BACKEND_FACTORIES:
        raise ValueError(
            f"Unknown Bessel backend: {backend!r}. "
            f"Choose from {BESSEL_BACKENDS}"
        )
    return _BACKEND_FACTORIES[backend](lmax)


# ---------------------------------------------------------------------------
# AnalyticHiPPOInducing
# ---------------------------------------------------------------------------


class AnalyticHiPPOInducing(RFFInducingBase):
    """Closed-form Z matrix computation via spherical Bessel functions."""

    phase_origin: float = eqx.field(static=True, default=0.0)
    bessel_backend: str = eqx.field(static=True, default="spbessax")

    def __init__(
        self,
        inducing_size: int,
        rff_sample_size: int,
        start: float,
        end: float,
        num_discrete_steps: int,
        prev_discrete_steps: int = 0,
        phase_origin: float = 0.0,
        bessel_backend: str = "spbessax",
    ):
        self.inducing_size = inducing_size
        self.rff_sample_size = rff_sample_size
        self.start = start
        self.end = end
        self.num_discrete_steps = num_discrete_steps
        self.prev_discrete_steps = prev_discrete_steps
        self.phase_origin = phase_origin
        self.bessel_backend = bessel_backend

    def _compute_Z_at_t(
        self,
        w_eff: Float[Array, "1 R"],
        t_index: Float[Array, ""],
        phase_start: Float[Array, "1 R"],
        scale: Float[Array, "R"],
    ) -> Float[Array, "TwoR M"]:
        """Compute Z matrix at a single time index using analytic formula."""
        M = self.inducing_size

        kappa = 0.5 * w_eff * t_index  # [1, R]
        L = jnp.arange(M, dtype=jnp.float64)

        # Spherical Bessel j_0...j_{M-1}(kappa)
        bessel_fn = spherical_bessel_j(M - 1, self.bessel_backend)
        j = bessel_fn(kappa)  # [M, 1, R]
        j = j.squeeze(1)  # [M, R]

        # Prefactor sqrt(2l+1)
        prefactor = jnp.sqrt(2 * L + 1)[:, None]  # [M, 1]

        # Phase shift
        phase = phase_start + kappa + L[:, None] * math.pi / 2  # [M, R]
        cos_term = jnp.cos(phase)
        sin_term = jnp.sin(phase)

        # Combine
        Z_cos = prefactor * j * cos_term  # [M, R]
        Z_sin = prefactor * j * sin_term  # [M, R]

        # Stack [Z_sin | Z_cos] and transpose: [2R, M]
        scale_full = jnp.concatenate([scale, scale])  # [2R]
        Z = scale_full[:, None] * jnp.concatenate([Z_sin, Z_cos], axis=1).T
        return Z

    def _compute_Z(
        self,
        sample: SpectralSample,
        Z_old: Float[Array, "TwoR M"] | None = None,
        flag_update_kernel: bool = True,
    ) -> tuple[Float[Array, "TwoR M"], Float[Array, "TwoR M"] | None]:
        """Compute Z matrix analytically."""
        w = sample.frequencies  # [1, R]
        scale = sample.scale  # [R]

        step_size = self.maxT / self.num_discrete_steps
        w_eff = w * step_size

        phase_start = w * self.phase_origin

        t_index = jnp.array(
            self.prev_discrete_steps + self.num_discrete_steps,
            dtype=jnp.float64,
        )
        Z = self._compute_Z_at_t(w_eff, t_index, phase_start, scale)

        Z_old_newk = None
        if self.prev_discrete_steps > 0:
            t_index_old = jnp.array(
                self.prev_discrete_steps, dtype=jnp.float64
            )
            Z_old_newk = self._compute_Z_at_t(
                w_eff, t_index_old, phase_start, scale
            )

        return Z, Z_old_newk

    def _compute_Z_f(
        self, sample: SpectralSample, X: Float[Array, "N D"]
    ) -> Float[Array, "N TwoR"]:
        return rff_features(sample, X)

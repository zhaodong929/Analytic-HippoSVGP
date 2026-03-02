"""HiPPO ODE-based inducing strategy.

Computes Z matrix via numerical HiPPO-LegS sequential scan on RFF features.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from hipposvgp.hippo.legs import HiPPO_LegS
from hipposvgp.inducing.base import RFFInducingBase
from hipposvgp.rff import rff_features
from hipposvgp.types import SpectralSample


class HiPPOInducing(RFFInducingBase):
    """Numerical ODE (HiPPO_LegS sequential scan) for Z matrix computation."""

    hippo: HiPPO_LegS
    flag_fast: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        inducing_size: int,
        hippo: HiPPO_LegS,
        rff_sample_size: int,
        start: float,
        end: float,
        num_discrete_steps: int,
        prev_discrete_steps: int = 0,
        flag_fast: bool = False,
    ):
        self.inducing_size = inducing_size
        self.hippo = hippo
        self.rff_sample_size = rff_sample_size
        self.start = start
        self.end = end
        self.num_discrete_steps = num_discrete_steps
        self.prev_discrete_steps = prev_discrete_steps
        self.flag_fast = flag_fast

    def _make_h(
        self,
        sample: SpectralSample,
        time_stamps: Float[Array, "T 1"],
    ) -> Float[Array, "T TwoR"]:
        """RFF features at given timestamps for ODE input."""
        w = sample.frequencies  # [1, R]
        scale = sample.scale  # [R]
        sin_term = jnp.sin(w * time_stamps)  # [T, R]
        cos_term = jnp.cos(w * time_stamps)  # [T, R]
        return jnp.concatenate([scale * sin_term, scale * cos_term], axis=1)

    def _get_full_time_stamps(self) -> Float[Array, "T 1"]:
        """Generate timestamps covering full history (prev + current steps)."""
        total_steps = self.prev_discrete_steps + self.num_discrete_steps
        step_size = self.maxT / self.num_discrete_steps
        global_start = self.start - self.prev_discrete_steps * step_size
        return (
            global_start
            + step_size
            * (1 + jnp.arange(total_steps, dtype=jnp.float64))[:, None]
        )

    def _compute_Z(
        self,
        sample: SpectralSample,
        Z_old: Float[Array, "TwoR M"] | None = None,
        flag_update_kernel: bool = True,
    ) -> tuple[Float[Array, "TwoR M"], Float[Array, "TwoR M"] | None]:
        """Compute Z via HiPPO ODE scan."""
        if not flag_update_kernel and Z_old is not None:
            # Continue from Z_old -- only need current task's timestamps
            time_stamps = self._get_time_stamps()
            h = self._make_h(sample, time_stamps)
            res = self.hippo(
                h,
                ini=Z_old,
                prev_discrete_steps=self.prev_discrete_steps,
            )
            Z = res[-1]
            Z_old_newk = None
        elif self.prev_discrete_steps > 0:
            # Full re-scan from scratch
            full_time_stamps = self._get_full_time_stamps()
            h = self._make_h(sample, full_time_stamps)
            res = self.hippo(h, ini=None, prev_discrete_steps=0)
            Z = res[-1]
            Z_old_newk = res[self.prev_discrete_steps - 1]
        else:
            # First task
            time_stamps = self._get_time_stamps()
            h = self._make_h(sample, time_stamps)
            res = self.hippo(h, ini=None, prev_discrete_steps=0)
            Z = res[-1]
            Z_old_newk = None
        return Z, Z_old_newk

    def _compute_Z_f(
        self, sample: SpectralSample, X: Float[Array, "N D"]
    ) -> Float[Array, "N TwoR"]:
        return rff_features(sample, X)

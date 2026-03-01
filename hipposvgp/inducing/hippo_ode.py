"""HiPPO ODE-based inducing strategy.

Computes Z matrix via numerical HiPPO-LegS sequential scan on RFF features.
"""

import torch

from hipposvgp.inducing.base import RFFInducingBase
from hipposvgp.rff import rff_features


class HiPPOInducing(RFFInducingBase):
    """Numerical ODE (HiPPO_LegS sequential scan) for Z matrix computation."""

    def __init__(
        self,
        inducing_size,
        device,
        hippo,
        rff_sample_size,
        start,
        end,
        num_discrete_steps,
        prev_discrete_steps=0,
        flag_fast=False,
        dtype=torch.float64,
    ):
        super().__init__(
            inducing_size, device, rff_sample_size,
            start, end, num_discrete_steps, prev_discrete_steps, dtype,
        )
        self.hippo = hippo
        self.flag_fast = flag_fast

    def _make_h(self, sample, time_stamps):
        """RFF features at given timestamps for ODE input."""
        w = sample.frequencies
        scale = sample.scale
        sin_term = torch.sin(w * time_stamps)
        cos_term = torch.cos(w * time_stamps)
        return torch.cat([scale * sin_term, scale * cos_term], 1)

    def _get_full_time_stamps(self):
        """Generate timestamps covering full history (prev + current steps)."""
        total_steps = self.prev_discrete_steps + self.num_discrete_steps
        step_size = self.maxT / self.num_discrete_steps
        global_start = self.start - self.prev_discrete_steps * step_size
        return global_start + step_size * (
            1
            + torch.arange(total_steps, dtype=self.dtype, device=self.device)
        ).unsqueeze(1)

    def _compute_Z(self, sample, Z_old=None, flag_update_kernel=True):
        """Compute Z via HiPPO ODE scan.

        Returns:
            (Z, Z_old_newk): current Z and old Z re-evaluated with new kernel
        """
        if not flag_update_kernel and Z_old is not None:
            # Continue from Z_old -- only need current task's timestamps
            time_stamps = self._get_time_stamps()
            h = self._make_h(sample, time_stamps)
            Z = self.hippo(
                h,
                ini=Z_old,
                prev_discrete_steps=self.prev_discrete_steps,
                fast=self.flag_fast,
            )[-1]
            Z_old_newk = None
        elif self.prev_discrete_steps > 0:
            # Full re-scan from scratch -- need ALL timestamps (prev + current)
            full_time_stamps = self._get_full_time_stamps()
            h = self._make_h(sample, full_time_stamps)
            res = self.hippo(
                h, ini=None, prev_discrete_steps=0, fast=self.flag_fast
            )
            Z = res[-1]
            Z_old_newk = res[self.prev_discrete_steps - 1]
        else:
            # First task -- only current timestamps
            time_stamps = self._get_time_stamps()
            h = self._make_h(sample, time_stamps)
            res = self.hippo(
                h, ini=None, prev_discrete_steps=0, fast=self.flag_fast
            )
            Z = res[-1]
            Z_old_newk = None
        return Z, Z_old_newk

    def _compute_Z_f(self, sample, X):
        """Compute RFF features for data points."""
        return rff_features(sample, X)

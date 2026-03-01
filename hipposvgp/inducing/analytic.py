"""Analytic HiPPO inducing strategy using spherical Bessel functions.

Core contribution of the paper: closed-form Z matrix computation
instead of numerical ODE integration.
"""

import math

import scipy.special
import torch

from hipposvgp.inducing.base import RFFInducingBase
from hipposvgp.rff import rff_features


class AnalyticHiPPOInducing(RFFInducingBase):
    """Closed-form Z matrix computation via spherical Bessel functions."""

    def __init__(
        self,
        inducing_size,
        device,
        rff_sample_size,
        start,
        end,
        num_discrete_steps,
        prev_discrete_steps=0,
        phase_origin=0.0,
        dtype=torch.float64,
    ):
        super().__init__(
            inducing_size, device, rff_sample_size,
            start, end, num_discrete_steps, prev_discrete_steps, dtype,
        )
        self.phase_origin = float(phase_origin)

    @staticmethod
    def spherical_bessel_j(lmax, x):
        """Compute spherical Bessel functions j_0...j_lmax.

        Uses scipy.special.jv with custom autograd backward pass.

        Args:
            lmax: maximum order
            x: input tensor

        Returns:
            j: [(lmax+1), *x.shape] spherical Bessel values
        """

        class BesselJFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, nu, inp):
                ctx.save_for_backward(nu, inp)
                nu_np = nu.detach().cpu().numpy()
                inp_np = inp.detach().cpu().numpy()
                res = scipy.special.jv(nu_np, inp_np)
                return torch.from_numpy(res).to(device=inp.device, dtype=inp.dtype)

            @staticmethod
            def backward(ctx, grad_output):
                nu, inp = ctx.saved_tensors
                nu_np = nu.detach().cpu().numpy()
                inp_np = inp.detach().cpu().numpy()
                # dJ_v/dx = 0.5 * (J_{v-1} - J_{v+1})
                j_minus = scipy.special.jv(nu_np - 1, inp_np)
                j_plus = scipy.special.jv(nu_np + 1, inp_np)
                grad_inp_np = 0.5 * (j_minus - j_plus)
                grad_inp = torch.from_numpy(grad_inp_np).to(
                    device=inp.device, dtype=inp.dtype
                )
                return None, grad_inp * grad_output

        # Handle x=0 to avoid division by zero
        mask_nonzero = x.abs() > 1e-8
        x_safe = torch.where(mask_nonzero, x, torch.ones_like(x))

        # Order nu = n + 0.5
        L = torch.arange(lmax + 1, device=x.device, dtype=x.dtype).view(-1, 1, 1)
        nu = L + 0.5

        # Prefactor sqrt(pi / (2|x|))
        x_abs = x_safe.abs()
        prefactor = torch.sqrt(math.pi / (2 * x_abs))

        # Compute J_nu(|x|) via scipy wrapper
        bessel_val = BesselJFunction.apply(nu, x_abs)

        # For negative x: j_l(-x) = (-1)^l j_l(x)
        parity = torch.where((L % 2) == 0, 1.0, -1.0)
        x_nonneg = (x_safe >= 0).unsqueeze(0)
        sign_factor = torch.where(
            x_nonneg, torch.ones_like(x_abs).unsqueeze(0), parity
        )
        j = prefactor * bessel_val * sign_factor

        # Boundary condition at x=0: j_0(0)=1, j_n(0)=0
        j_at_0 = torch.zeros(
            (lmax + 1,) + x.shape, device=x.device, dtype=x.dtype
        )
        j_at_0[0] = 1.0
        j = torch.where(mask_nonzero, j, j_at_0)

        return j

    def _compute_Z_at_t(self, w_eff, t_index, phase_start, scale):
        """Compute Z matrix at a single time index using analytic formula.

        Args:
            w_eff: [1, R] effective frequencies (w * step_size)
            t_index: scalar time index
            phase_start: [1, R] phase offset
            scale: [R] tensor or scalar RFF scale

        Returns:
            Z: [2R, M] feature matrix
        """
        kappa = 0.5 * w_eff * t_index  # [1, R]
        L = torch.arange(self.inducing_size, dtype=self.dtype, device=self.device)

        # Spherical Bessel
        j = self.spherical_bessel_j(self.inducing_size - 1, kappa)
        if len(j.shape) == 3:
            j = j.squeeze(1)  # [M, R]

        # Prefactor sqrt(2l+1)
        prefactor = torch.sqrt(2 * L + 1)[:, None]

        # Phase shift (global phase origin)
        phase = phase_start + kappa + L[:, None] * math.pi / 2
        cos_term = torch.cos(phase)
        sin_term = torch.sin(phase)

        # Combine
        Z_cos = prefactor * j * cos_term
        Z_sin = prefactor * j * sin_term

        # Stack and scale
        if isinstance(scale, torch.Tensor) and scale.dim() > 0:
            scale_full = torch.cat([scale, scale])  # [2R]
            Z = scale_full[:, None] * torch.cat([Z_sin, Z_cos], dim=1).T
        else:
            Z = scale * torch.cat([Z_sin, Z_cos], dim=1).T

        return Z

    def _compute_Z(self, sample, Z_old=None, flag_update_kernel=True):
        """Compute Z matrix analytically.

        Returns:
            (Z, Z_old_newk): current Z [2R, M] and old Z [2R, M] or None
        """
        w = sample.frequencies
        scale = sample.scale

        step_size = self.maxT / self.num_discrete_steps
        w_eff = w * step_size

        phase_origin = torch.as_tensor(
            self.phase_origin, dtype=self.dtype, device=self.device
        )
        phase_start = w * phase_origin

        t_index = torch.as_tensor(
            self.prev_discrete_steps + self.num_discrete_steps,
            dtype=self.dtype,
            device=self.device,
        )
        Z = self._compute_Z_at_t(w_eff, t_index, phase_start, scale)

        Z_old_newk = None
        if self.prev_discrete_steps > 0:
            t_index_old = torch.as_tensor(
                self.prev_discrete_steps,
                dtype=self.dtype,
                device=self.device,
            )
            Z_old_newk = self._compute_Z_at_t(
                w_eff, t_index_old, phase_start, scale
            )

        return Z, Z_old_newk

    def _compute_Z_f(self, sample, X):
        """Compute RFF features for data points."""
        return rff_features(sample, X)

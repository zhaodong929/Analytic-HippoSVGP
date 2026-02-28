"""Abstract base class and data structures for inducing strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class KernelMatrices:
    """Container for all kernel matrices needed by the model.

    Attributes:
        Kuu: [M, M] inducing-inducing kernel matrix
        Kfu: [N, M] data-inducing kernel matrix
        Kff: [N, N] data-data kernel matrix
        Kuu_old: [M, M'] cross kernel between new and old inducing (None if no prior)
        Kold_old: [M', M'] old inducing kernel from previous task (None if no prior)
        Kold_old_newk: [M', M'] old inducing re-evaluated with new kernel (None if no prior)
        Z: [2R, M] feature matrix (for state transfer)
    """

    Kuu: torch.Tensor
    Kfu: torch.Tensor
    Kff: torch.Tensor
    Kuu_old: Optional[torch.Tensor] = None
    Kold_old: Optional[torch.Tensor] = None
    Kold_old_newk: Optional[torch.Tensor] = None
    Z: Optional[torch.Tensor] = None


class InducingStrategy(nn.Module, ABC):
    """Base class for inducing point strategies.

    Subclasses define how to compute Z (inducing features) and
    the resulting kernel matrices Kuu, Kfu, Kff.
    """

    def __init__(self, inducing_size):
        super().__init__()
        self.inducing_size = inducing_size

    @abstractmethod
    def compute_kernel_matrices(
        self,
        kernel,
        X,
        *,
        Z_old=None,
        Kaa_old=None,
        flag_update_kernel=True,
        w=None,
    ):
        """Compute all kernel matrices for the ELBO.

        Args:
            kernel: Kernel module
            X: [N, D] training data
            Z_old: [2R, M'] previous Z feature matrix (or None)
            Kaa_old: [M', M'] previous Kuu matrix (or None)
            flag_update_kernel: if True, recompute Z from scratch
            w: optional fixed spectral sample (for RFF-based strategies)

        Returns:
            KernelMatrices dataclass
        """

    @abstractmethod
    def compute_prediction_matrices(self, kernel, Xnew, w=None):
        """Compute kernel matrices for prediction.

        Args:
            kernel: Kernel module
            Xnew: [N*, D] test data
            w: optional fixed spectral sample (for RFF-based strategies)

        Returns:
            (Kuu, Kfu_new, Kff_new) tuple
        """


class RFFInducingBase(InducingStrategy, ABC):
    """Base class for RFF-based inducing strategies (HiPPO-ODE and Analytic).

    Provides shared logic for spectral sample resolution, time window management,
    and kernel matrix assembly from Z features.
    Subclasses implement ``_compute_Z`` and ``_compute_Z_f``.
    """

    def __init__(
        self,
        inducing_size,
        device,
        rff_sample_size,
        start,
        end,
        num_discrete_steps,
        prev_discrete_steps=0,
        dtype=torch.float64,
    ):
        super().__init__(inducing_size)
        self.device = device
        self.dtype = dtype
        self.rff_sample_size = rff_sample_size
        self.start = start
        self.end = end
        self.maxT = end - start
        self.num_discrete_steps = num_discrete_steps
        self.prev_discrete_steps = int(prev_discrete_steps)

    def update_time_window(self, new_start, new_end, new_n_steps, new_prev_steps):
        """Update the time window for the next task."""
        self.start = new_start
        self.end = new_end
        self.maxT = new_end - new_start
        self.num_discrete_steps = new_n_steps
        self.prev_discrete_steps = int(new_prev_steps)

    def _get_time_stamps(self):
        return self.start + (self.maxT / self.num_discrete_steps) * (
            1
            + torch.arange(
                self.num_discrete_steps, dtype=self.dtype, device=self.device
            )
        ).unsqueeze(1)

    def _resolve_spectral_sample(self, kernel, w):
        """Resolve spectral sample: use provided w, or sample from kernel."""
        if w is None:
            w = kernel.sample_from_spectral(self.rff_sample_size)
            if isinstance(w, tuple):
                raise TypeError(
                    "SpectralMixture kernel returns (S, weights) from sample_from_spectral, "
                    "which is not supported by RFF inducing strategies. "
                    "Use SE or FreqSpace kernel."
                )
        else:
            w = torch.as_tensor(w, dtype=self.dtype, device=self.device)
        if w.dim() == 1:
            w = w.unsqueeze(0)
        return w

    @abstractmethod
    def _compute_Z(self, w, kernel=None, Z_old=None, flag_update_kernel=True):
        """Compute Z matrix from spectral sample.

        Returns:
            (Z, Z_old_newk): current Z [2R, M] and old Z re-evaluated or None
        """

    @abstractmethod
    def _compute_Z_f(self, w, X, kernel=None):
        """Compute RFF features for data points.

        Returns:
            Z_f: [N, 2R] data feature matrix
        """

    def _assemble_kernel_matrices(
        self, sf, Z, Z_f, Z_old_newk, Z_old, Kaa_old, flag_update_kernel
    ):
        """Assemble kernel matrices from Z features."""
        Kuu = sf * (Z.T @ Z)
        Kfu = sf * (Z_f @ Z)
        Kff = sf * (Z_f @ Z_f.T)

        Kuu_old = None
        Kold_old = None
        Kold_old_newk_mat = None

        if flag_update_kernel:
            if Kaa_old is not None and Z_old_newk is not None:
                Kuu_old = sf * (Z.T @ Z_old_newk)
                Kold_old = Kaa_old
                Kold_old_newk_mat = sf * (Z_old_newk.T @ Z_old_newk)
        else:
            if Z_old is not None:
                Kuu_old = sf * (Z.T @ Z_old)
                Kold_old = Kaa_old
                Kold_old_newk_mat = (
                    sf * (Z_old.T @ Z_old) if Kaa_old is not None else None
                )

        return KernelMatrices(
            Kuu=Kuu,
            Kfu=Kfu,
            Kff=Kff,
            Kuu_old=Kuu_old,
            Kold_old=Kold_old,
            Kold_old_newk=Kold_old_newk_mat,
            Z=Z,
        )

    def compute_kernel_matrices(
        self,
        kernel,
        X,
        *,
        Z_old=None,
        Kaa_old=None,
        flag_update_kernel=True,
        w=None,
    ):
        w = self._resolve_spectral_sample(kernel, w)
        Z, Z_old_newk = self._compute_Z(
            w, kernel=kernel, Z_old=Z_old, flag_update_kernel=flag_update_kernel
        )
        Z_f = self._compute_Z_f(w, X, kernel=kernel)
        sf = torch.exp(kernel.log_sf)
        km = self._assemble_kernel_matrices(
            sf, Z, Z_f, Z_old_newk, Z_old, Kaa_old, flag_update_kernel
        )
        return km, w

    def compute_prediction_matrices(self, kernel, Xnew, w=None):
        w = self._resolve_spectral_sample(kernel, w)
        Z, _ = self._compute_Z(w, kernel=kernel)
        Z_f = self._compute_Z_f(w, Xnew, kernel=kernel)
        sf = torch.exp(kernel.log_sf)
        Kuu = sf * (Z.T @ Z)
        Kfu_new = sf * (Z_f @ Z)
        Kff_new = sf * (Z_f @ Z_f.T)
        return Kuu, Kfu_new, Kff_new

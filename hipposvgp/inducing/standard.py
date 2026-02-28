"""Standard learnable inducing points (no HiPPO)."""

import torch
import torch.nn as nn

from hipposvgp.inducing.base import InducingStrategy, KernelMatrices


class StandardInducing(InducingStrategy):
    """Learnable inducing point locations optimised directly.

    This is a baseline strategy where Z are free parameters.
    """

    def __init__(self, inducing_size, input_dim, device, dtype=torch.float64):
        super().__init__(inducing_size)
        self.device = device
        self.dtype = dtype
        self.Z = nn.Parameter(
            torch.zeros(inducing_size, input_dim, dtype=dtype, device=device),
            requires_grad=True,
        )

    def compute_kernel_matrices(
        self, kernel, X, *, Z_old=None, Kaa_old=None, flag_update_kernel=True, w=None
    ):
        Kuu = kernel(self.Z, self.Z)
        Kfu = kernel(X, self.Z)
        Kff = kernel(X, X)

        Kuu_old = None
        Kold_old = None
        Kold_old_newk = None
        if Z_old is not None and Kaa_old is not None:
            Kuu_old = kernel(self.Z, Z_old)
            Kold_old = Kaa_old
            Kold_old_newk = kernel(Z_old, Z_old)

        return KernelMatrices(
            Kuu=Kuu,
            Kfu=Kfu,
            Kff=Kff,
            Kuu_old=Kuu_old,
            Kold_old=Kold_old,
            Kold_old_newk=Kold_old_newk,
            Z=self.Z,
        ), None  # no w for standard inducing

    def compute_prediction_matrices(self, kernel, Xnew, w=None):
        Kuu = kernel(self.Z, self.Z)
        Kfu_new = kernel(Xnew, self.Z)
        Kff_new = kernel(Xnew, Xnew)
        return Kuu, Kfu_new, Kff_new

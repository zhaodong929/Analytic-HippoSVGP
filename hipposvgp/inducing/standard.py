"""Standard learnable inducing points (no HiPPO)."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from hipposvgp.inducing.base import InducingStrategy
from hipposvgp.kernels.base import Kernel
from hipposvgp.types import KernelMatrices, SpectralSample


class StandardInducing(InducingStrategy):
    """Learnable inducing point locations optimised directly."""

    Z: Float[Array, "M D"]

    def __init__(self, inducing_size: int, input_dim: int):
        self.inducing_size = inducing_size
        self.Z = jnp.zeros((inducing_size, input_dim))

    def compute_kernel_matrices(
        self,
        kernel: Kernel,
        X: Float[Array, "N D"],
        *,
        Z_old: Float[Array, "Mo D"] | None = None,
        Kaa_old: Float[Array, "Mo Mo"] | None = None,
        flag_update_kernel: bool = True,
        sample: SpectralSample | None = None,
        key: PRNGKeyArray | None = None,
    ) -> tuple[KernelMatrices, None]:
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
        ), None

    def compute_prediction_matrices(
        self,
        kernel: Kernel,
        Xnew: Float[Array, "Ns D"],
        sample: SpectralSample | None = None,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "M M"], Float[Array, "Ns M"], Float[Array, "Ns Ns"]]:
        Kuu = kernel(self.Z, self.Z)
        Kfu_new = kernel(Xnew, self.Z)
        Kff_new = kernel(Xnew, Xnew)
        return Kuu, Kfu_new, Kff_new

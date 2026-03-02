"""Abstract base class and shared logic for inducing strategies."""

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from hipposvgp.kernels.base import Kernel
from hipposvgp.types import KernelMatrices, SpectralSample


class InducingStrategy(eqx.Module):
    """Base class for inducing point strategies."""

    inducing_size: int = eqx.field(static=True)

    @abc.abstractmethod
    def compute_kernel_matrices(
        self,
        kernel: Kernel,
        X: Float[Array, "N D"],
        *,
        Z_old: Float[Array, "TwoR M"] | None = None,
        Kaa_old: Float[Array, "M M"] | None = None,
        flag_update_kernel: bool = True,
        sample: SpectralSample | None = None,
        key: PRNGKeyArray | None = None,
    ) -> tuple[KernelMatrices, SpectralSample | None]:
        """Compute all kernel matrices for the ELBO."""
        ...

    @abc.abstractmethod
    def compute_prediction_matrices(
        self,
        kernel: Kernel,
        Xnew: Float[Array, "Ns D"],
        sample: SpectralSample | None = None,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "M M"], Float[Array, "Ns M"], Float[Array, "Ns Ns"]]:
        """Compute kernel matrices for prediction."""
        ...


class RFFInducingBase(InducingStrategy):
    """Base class for RFF-based inducing strategies (HiPPO-ODE and Analytic).

    Provides shared logic for spectral sample resolution, time window management,
    and kernel matrix assembly from Z features.
    """

    rff_sample_size: int = eqx.field(static=True)
    start: float = eqx.field(static=True)
    end: float = eqx.field(static=True)
    num_discrete_steps: int = eqx.field(static=True)
    prev_discrete_steps: int = eqx.field(static=True)

    def with_time_window(
        self,
        new_start: float,
        new_end: float,
        new_n_steps: int,
        new_prev_steps: int,
    ):
        """Return a new instance with updated time window."""
        return eqx.tree_at(
            lambda s: (s.start, s.end, s.num_discrete_steps, s.prev_discrete_steps),
            self,
            (new_start, new_end, new_n_steps, new_prev_steps),
        )

    @property
    def maxT(self) -> float:
        return self.end - self.start

    def _get_time_stamps(self) -> Float[Array, "L 1"]:
        """Evenly spaced timestamps in [start, end]."""
        step = self.maxT / self.num_discrete_steps
        return (
            self.start
            + step
            * (1 + jnp.arange(self.num_discrete_steps, dtype=jnp.float64))[:, None]
        )

    def _resolve_spectral_sample(
        self,
        kernel: Kernel,
        sample: SpectralSample | None,
        key: PRNGKeyArray | None = None,
    ) -> SpectralSample:
        """Resolve spectral sample: use provided sample, or sample from kernel."""
        if sample is not None:
            return sample
        return kernel.sample_from_spectral(self.rff_sample_size, key)

    @abc.abstractmethod
    def _compute_Z(
        self,
        sample: SpectralSample,
        Z_old: Float[Array, "TwoR M"] | None = None,
        flag_update_kernel: bool = True,
    ) -> tuple[Float[Array, "TwoR M"], Float[Array, "TwoR M"] | None]:
        """Compute Z matrix from spectral sample."""
        ...

    @abc.abstractmethod
    def _compute_Z_f(
        self, sample: SpectralSample, X: Float[Array, "N D"]
    ) -> Float[Array, "N TwoR"]:
        """Compute RFF features for data points."""
        ...

    def _assemble_kernel_matrices(
        self,
        sf: Float[Array, ""],
        Z: Float[Array, "TwoR M"],
        Z_f: Float[Array, "N TwoR"],
        Z_old_newk: Float[Array, "TwoR M"] | None,
        Z_old: Float[Array, "TwoR M"] | None,
        Kaa_old: Float[Array, "M M"] | None,
        flag_update_kernel: bool,
    ) -> KernelMatrices:
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
        kernel: Kernel,
        X: Float[Array, "N D"],
        *,
        Z_old: Float[Array, "TwoR M"] | None = None,
        Kaa_old: Float[Array, "M M"] | None = None,
        flag_update_kernel: bool = True,
        sample: SpectralSample | None = None,
        key: PRNGKeyArray | None = None,
    ) -> tuple[KernelMatrices, SpectralSample]:
        sample = self._resolve_spectral_sample(kernel, sample, key)
        Z, Z_old_newk = self._compute_Z(
            sample, Z_old=Z_old, flag_update_kernel=flag_update_kernel
        )
        Z_f = self._compute_Z_f(sample, X)
        sf = jnp.exp(kernel.log_sf).squeeze()
        km = self._assemble_kernel_matrices(
            sf, Z, Z_f, Z_old_newk, Z_old, Kaa_old, flag_update_kernel
        )
        return km, sample

    def compute_prediction_matrices(
        self,
        kernel: Kernel,
        Xnew: Float[Array, "Ns D"],
        sample: SpectralSample | None = None,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "M M"], Float[Array, "Ns M"], Float[Array, "Ns Ns"]]:
        sample = self._resolve_spectral_sample(kernel, sample, key)
        Z, _ = self._compute_Z(sample)
        Z_f = self._compute_Z_f(sample, Xnew)
        sf = jnp.exp(kernel.log_sf).squeeze()
        Kuu = sf * (Z.T @ Z)
        Kfu_new = sf * (Z_f @ Z)
        Kff_new = sf * (Z_f @ Z_f.T)
        return Kuu, Kfu_new, Kff_new

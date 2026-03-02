"""hipposvgp: JAX + Equinox reimplementation of HiPPO-SVGP."""

from hipposvgp.types import (
    SpectralSample,
    KernelMatrices,
    PosteriorState,
    SGPRPosteriorState,
)
from hipposvgp.rff import rff_features
from hipposvgp.linalg import add_jitter, robust_cholesky, whitened_kl, cholesky
from hipposvgp.kernels import (
    Kernel,
    SquaredExponential,
    FrequencySpace,
    SpectralMixture,
)
from hipposvgp.likelihoods import (
    Likelihood,
    GaussianLikelihood,
    BernoulliLikelihood,
    NegativeBinomialLikelihood,
)
from hipposvgp.hippo import HiPPO_LegS
from hipposvgp.inducing import (
    InducingStrategy,
    RFFInducingBase,
    HiPPOInducing,
    AnalyticHiPPOInducing,
    StandardInducing,
)
from hipposvgp.models import OnlineSVGP, OnlineSGPR, ExactGP
from hipposvgp.training import (
    make_svgp_train_step,
    make_sgpr_train_step,
    make_exact_gp_train_step,
    train_loop,
)

__all__ = [
    # Types
    "SpectralSample",
    "KernelMatrices",
    "PosteriorState",
    "SGPRPosteriorState",
    # RFF
    "rff_features",
    # Linalg
    "add_jitter",
    "robust_cholesky",
    "whitened_kl",
    "cholesky",
    # Kernels
    "Kernel",
    "SquaredExponential",
    "FrequencySpace",
    "SpectralMixture",
    # Likelihoods
    "Likelihood",
    "GaussianLikelihood",
    "BernoulliLikelihood",
    "NegativeBinomialLikelihood",
    # HiPPO
    "HiPPO_LegS",
    # Inducing
    "InducingStrategy",
    "RFFInducingBase",
    "HiPPOInducing",
    "AnalyticHiPPOInducing",
    "StandardInducing",
    # Models
    "OnlineSVGP",
    "OnlineSGPR",
    "ExactGP",
    # Training
    "make_svgp_train_step",
    "make_sgpr_train_step",
    "make_exact_gp_train_step",
    "train_loop",
]

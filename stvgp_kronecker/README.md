# analytic-kron-stgp

Public-facing name for this staged prototype: `analytic-kron-stgp`.

This folder contains a staged Kronecker-separable spatio-temporal HiPPO-SVGP
prototype in PyTorch.

Current status:

- `temporal_analytic.py`: wraps the existing analytic spherical-Bessel /
  HiPPO-RFF temporal construction behind a small interface.
- `spatial_kernel.py`: spatial RBF and Matern kernels with `Kzz_s` and `Kxz_s`
  helpers.
- `kron_ops.py`: Kronecker-aware utilities for solves, log-determinants, and
  reshape-safe matrix products.
- `st_model_batch.py`: Stage 1 batch Gaussian model using the closed-form
  posterior update from the reference document.
- `era5_dataset.py`: ERA5/xarray loading prototype and temporal block helpers.
- `train_batch.py`: small synthetic experiment and optional blockwise batch
  training loop.
- `st_model_online.py`: Stage 2 Gaussian online posterior-summary recursion.
- `train_online.py`: small synthetic online experiment and batch comparison.
- `INTERFACE_SUMMARY.md`: explicit summary of what must be provided by the
  existing temporal analytic code and existing SVGP code before Stage 2.
- `tests/`: minimal Stage 1 tests.

Suggested structure for this prototype:

```text
stvgp_kronecker/
├── README.md
├── STAGE2_ROADMAP.md
├── temporal_analytic.py
├── spatial_kernel.py
├── kron_ops.py
├── st_model_batch.py
├── st_model_online.py
├── era5_dataset.py
├── train_batch.py
├── train_online.py
├── INTERFACE_SUMMARY.md
└── tests/
    └── test_stage1.py
```

Important implementation note:

- Stage 1 and the current Stage 2 implementation keep the full inducing
  posterior precision as a dense
  `(M_t M_s) x (M_t M_s)` matrix for correctness and simplicity.
- The code avoids materializing large data-space Kronecker matrices where it is
  easy to do so.
- The current Stage 2 recursion assumes one shared inducing coordinate system
  across blocks, implemented through a fixed reference horizon.
- Ambiguous math-to-code points are marked with `TODO` comments instead of being
  silently guessed.

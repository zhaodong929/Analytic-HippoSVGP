# Existing Code Interface Summary

This note records the exact interfaces needed from the existing codebase before
moving beyond Stage 1.

## Temporal analytic code

Source being reused:
- `scripts/onedim/solar/test_ohsgp_analytic_solar.py`
- related reusable logic in `hipposvgp/hsvgp_analytic.py`

Stage 1 wrapper expectations:

1. `compute_kuu_t(times_or_horizon, config) -> Tensor[M_t, M_t]`
2. `compute_kfu_t(query_times, horizon, config) -> Tensor[N_t, M_t]`
3. `compute_ktt_diag(query_times) -> Tensor[N_t]`
4. deterministic reuse of the same spectral samples / temporal basis during
   training and prediction
5. support for block-local horizons via `TemporalBlockSpec`

How this is mapped now:
- Implemented by [temporal_analytic.py](/vol/bitbucket/zg1425/myproject/Analytic-HippoSVGP/stvgp_kronecker/temporal_analytic.py)
- current wrapper uses fixed base frequencies plus learnable
  `log_variance` / `log_lengthscale`

Open TODOs:
- confirm whether future online mode should reuse one global spectral draw or a
  controlled resampling policy
- confirm whether local horizon should be indexed by physical time or by
  discrete block position for all datasets

## Existing SVGP code

Source being reused:
- `stvgp_kronecker/Multi-dimensional_svgp.py`
- Gaussian closed-form patterns from `scripts/onedim/solar/test_ohsgp_analytic_solar.py`

Stage 1 expectations:

1. Gaussian noise parameterization
2. optimizer / training-loop pattern
3. prediction utilities returning mean and variance
4. metric logging hooks

How this is mapped now:
- Gaussian noise is represented by `log_noise_std` in
  [st_model_batch.py](/vol/bitbucket/zg1425/myproject/Analytic-HippoSVGP/stvgp_kronecker/st_model_batch.py)
- training and reporting are handled in
  [train_batch.py](/vol/bitbucket/zg1425/myproject/Analytic-HippoSVGP/stvgp_kronecker/train_batch.py)
- prediction is exposed by `BatchKroneckerSTHiPPOSVGP.predict`

Open TODOs:
- if Stage 2 later needs variational optimization instead of pure conjugate
  recursion, decide whether to wrap or refactor the existing SVGP optimizer
- if non-Gaussian likelihoods are added later, the current closed-form Stage 1
  path will need a separate inference layer

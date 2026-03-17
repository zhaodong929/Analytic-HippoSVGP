# Stage 2 Roadmap

Stage 2 now has a first working Gaussian recursion in
[st_model_online.py](/vol/bitbucket/zg1425/myproject/Analytic-HippoSVGP/stvgp_kronecker/st_model_online.py).
The remaining items below are the follow-up work needed beyond this first
online version.

Required interfaces from the existing temporal analytic code:

1. `compute_kuu_t(times_or_horizon, config) -> Tensor[M_t, M_t]`
2. `compute_kfu_t(query_times, horizon, config) -> Tensor[N_t, M_t]`
3. `compute_ktt_diag(query_times) -> Tensor[N_t]`
4. A stable way to reuse the same temporal inducing construction for both
   training and prediction blocks
5. Optional later extension: support block-dependent horizons without changing
   the spatial side

Required interfaces from the existing SVGP code:

1. Gaussian noise parameterization and optimization loop
2. Metric logging for RMSE, predictive NLL, runtime, and memory
3. Prediction utilities that can consume a cached posterior mean/covariance
4. Optional later extension: mini-batch loop that iterates over temporal blocks

Stage 2 target state:

- Maintain:
  - `Lambda`: posterior precision summary
  - `h`: posterior information vector
- For block `n`:
  - build `Kuu_t_n`, `Kfu_t_n`
  - form Kronecker-aware `A_n`
  - update:
    - `Lambda_n = Lambda_{n-1} + (1 / sigma^2) A_n^T A_n`
    - `h_n = h_{n-1} + (1 / sigma^2) A_n^T y_n`
  - recover:
    - `S_n = inv(Lambda_n)`
    - `m_n = S_n @ h_n`

Open TODOs before Stage 2:

- Replace the dense `Lambda` with a more scalable structured representation if
  `M_t M_s` becomes large.
- Implement principled basis transfer when temporal inducing bases change across
  blocks instead of relying on one shared reference horizon.
- Benchmark whether block-level `A_n^T A_n` should be accumulated explicitly or
  via factorized `A_t` and `A_s` summaries.

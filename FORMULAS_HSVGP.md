# hsvgp.py: Formula-to-Code Map (One-Chain Derivation)

This document maps the math chain (model definition → RFF → HiPPO → SVGP → ELBO → prediction)
to concrete code blocks in `hipposvgp/hsvgp.py`.

## 1) Model Definition (GP Regression)

Formula:

```
y_i = f(t_i) + ε_i,   ε_i ~ N(0, σ^2)
f(t) ~ GP(0, k(t, t'))
```

Code:

- Likelihood definition: `hipposvgp/likelihood.py` (GaussianLikelihood)
- Model class: `hipposvgp/hsvgp.py` (class HIPPOOSVGP)

## 2) SVGP Inducing Variables

Formula:

```
u = f(Z),   p(u) = N(0, Kuu)
p(f | u) = N(Kfu Kuu^{-1} u, Kff - Qff)
Qff = Kfu Kuu^{-1} Kfu^T
q(u) = N(mv, Sv) with Sv = Lv Lv^T
```

Code:

- `hipposvgp/hsvgp.py`: `ELBO`, `pred_f`, `quold_new`
  - `mv`, `Lv` as variational parameters in `__init__`
  - `Kuu`, `Kfu`, `Kff` via `KuuKfuKff_rff_se`

## 3) RFF Approximation (Bochner)

Formula:

```
k(t, t') = ∫ e^{iw(t-t')} p(w) dw
k(t, t') ≈ (1/M) Σ [cos(w_n t)cos(w_n t') + sin(w_n t)sin(w_n t')]

φ(t) = sqrt(1/M) [sin(w_1 t),...,sin(w_M t), cos(w_1 t),...,cos(w_M t)]
k(t, t') ≈ φ(t)^T φ(t')
```

Code:

- `hipposvgp/kernel.py`: `SE_kernel.sample_from_spectral` (samples w)
- `hipposvgp/hsvgp.py`:
  - `KfuKff_rff_se`: constructs `Z_f = (1/sqrt(M)) [sin(wX), cos(wX)]`
  - `KuuKfuKff_rff_se`: same structure for Kff/Kfu/Kuu

## 4) HiPPO Projection (Time-Orthogonal Expansion)

Formula:

```
c_ℓ(t) = ∫_0^t f(x) φ_ℓ^{(t)}(x) dx
```

Code:

- `hipposvgp/hippo.py`: `HiPPO_LegS.forward`
- `hipposvgp/hsvgp.py`: `apply_hippo`, `get_Z`
  - `get_Z` builds `h = (1/sqrt(M)) [sin(w * t), cos(w * t)]`
  - `apply_hippo(h)` returns `Z` (the HiPPO features)

## 5) Kernel Construction in HiPPO Feature Space

Formula:

```
Kuu = Z^T Z
Kfu = Φ(X) Z
Kff = Φ(X) Φ(X)^T
```

Code:

- `hipposvgp/hsvgp.py`:
  - `Kuu_se`, `KfuKff_rff_se`, `KuuKfuKff_rff_se`

## 6) Variational ELBO

Formula:

```
L = E_{q(f)}[log p(y|f)] - KL(q(u) || p(u)) + CorrectionTerm
```

Code:

- `hipposvgp/hsvgp.py`: `ELBO`
  - Likelihood term: `self.likelihood.log_prob(...)`
  - KL term: computed from `mv`, `Lv`
  - Correction term: `correction_term(...)` if `Z_old` exists

## 7) KL(q(u) || p(u))

Formula:

```
KL = 1/2 [ tr(Sv) + mv^T mv - M - log|Sv| ]
```

Code:

- `hipposvgp/hsvgp.py`: `ELBO`
  - `Sv = Lv @ Lv.T`
  - `kl = 0.5 * (trace(Sv) + sum(mv^2) - M - logdet(Sv))`

## 8) Predictive Mean/Variance

Formula:

```
μ_* = K_*u Kuu^{-1} mv
Σ_* = K_** - K_*u Kuu^{-1} K_u* + K_*u Kuu^{-1} Sv Kuu^{-1} K_u*
```

Code:

- `hipposvgp/hsvgp.py`: `pred_f`
  - `mean = Kfu_new @ cholesky_solve(mu_u, Luu)`
  - `var = Kff_new - Kfu_new @ Kuu^{-1} Kfu_new^T + extra`

## 9) Online Correction (HiPPO-SVGP)

Formula (conceptual):

```
q_new(u) = ∫ p(u | u_old) q_old(u_old) du_old
```

Code:

- `hipposvgp/hsvgp.py`: `correction_term`
  - Computes correction using `mv_old`, `Lv_old`, `Kuu_old`, `Ku_old_old`

## 10) Full Training Objective

Formula:

```
max_{θ, mv, Lv} L = E_{q(f)}[log p(y|f)] - KL(q(u)||p(u)) + CorrectionTerm
```

Code:

- `hipposvgp/hsvgp.py`: `ELBO`
- Training loop in scripts, e.g. `scripts/onedim/covid/test_ohsgp.py`


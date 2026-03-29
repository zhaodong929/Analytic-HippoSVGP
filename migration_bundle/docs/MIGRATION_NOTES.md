# Migration Bundle Notes

This bundle is a standalone copy of the one-dimensional `covid` and `solar`
experiment code arranged under:

- `data/`
- `docs/`
- `hipposvgp/`
- `results/`
- `scripts/`

Key intent:

- Preserve the `solar` analytic implementation logic as the canonical analytic backend.
- Reuse that backend from the copied `covid` analytic model.
- Leave the original repository code unchanged.

What changed inside this bundle:

- Added `hipposvgp/analytic_backend.py`.
- Rewired `hipposvgp/hsvgp_analytic.py` to use the stable pure-torch spherical
  Bessel recurrence and the `solar`-style analytic `Z` construction.
- Added `hipposvgp/sgpr_analytic.py` so the `solar` analytic classes are now a
  formal module instead of living inside the script.
- Replaced the copied `scripts/onedim/solar/test_ohsgp_analytic_solar.py` with
  a module-driven runner that imports `hipposvgp/sgpr_analytic.py`.

Files copied for migration:

- Shared core: `hipposvgp/{__init__,hippo,kernel,likelihood,utils,sgpr,hsvgp,hsvgp_analytic}.py`
- Shared analytic backend: `hipposvgp/{analytic_backend,sgpr_analytic}.py`
- Solar scripts: `scripts/onedim/solar/{test_ohsgp_analytic_solar,test_hsgp_online_fixedkernel,baseline,kernel}.py`
- Covid scripts: `scripts/onedim/covid/{test_ohsgp,test_ohsgp_analytic,data}.py`
- Data: full copy of original `data/` including `covid`, `solar`, `era5`, `uci`, and `baseband.npy`

Suggested usage:

1. `cd migration_bundle`
2. Run scripts from this directory so local `hipposvgp/` is imported first.
3. Port this folder as the starting point into the new project.

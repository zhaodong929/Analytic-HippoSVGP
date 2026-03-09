"""Kernel matrix diagnostic script.

Compares ODE vs Analytic inducing Z/Kuu/Kfu/Kff matrices to verify correctness.

Usage:
    python -m scripts.verify solver --seed 42                       # ODE vs Analytic
    python -m scripts.verify solver --seed 42 --bessel-backend scipy  # Use scipy backend
    python -m scripts.verify feature --seed 42                      # FreqSpace vs SE+RFF
    python -m scripts.verify all --seed 42                          # Both
"""

import argparse
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import equinox as eqx

from hipposvgp.kernels import SquaredExponential, FrequencySpace
from hipposvgp.inducing import HiPPOInducing, AnalyticHiPPOInducing
from hipposvgp.hippo import HiPPO_LegS

from scripts.utils import seed_everything


def _fmt(val):
    """Format a numeric value for display."""
    if val < 1e-6:
        return f"{val:.2e}"
    return f"{val:.6f}"


def _check(name, mat_a, mat_b, tol=1e-3):
    """Compare two matrices and report pass/fail."""
    max_err = float(jnp.abs(mat_a - mat_b).max())
    fro_a = float(jnp.linalg.norm(mat_a))
    fro_ratio = float(jnp.linalg.norm(mat_a - mat_b)) / max(fro_a, 1e-12)

    status = "PASS" if max_err < tol else "FAIL"
    print(f"  {name:20s}  max_abs_error={_fmt(max_err)}  fro_ratio={_fmt(fro_ratio)}  [{status}]")
    return status == "PASS"


def verify_solver(seed=42, inducing_size=15, rff_sample_size=200, num_steps=20,
                  prev_steps_list=None, bessel_backend="spbessax"):
    """Compare ODE vs Analytic Z, Kuu, Kfu, Kff with shared SpectralSample.

    Tests multiple prev_discrete_steps values to check accuracy across tasks.
    """
    if prev_steps_list is None:
        prev_steps_list = [0, 10, 20]

    print("=" * 60)
    print("SOLVER VERIFICATION: ODE vs Analytic")
    print("=" * 60)

    seed_everything(seed)
    key = jax.random.PRNGKey(seed)

    all_pass = True

    for prev_steps in prev_steps_list:
        print(f"\n--- prev_discrete_steps = {prev_steps} ---")

        start = 0.0
        end = 10.0

        # Shared kernel
        kernel = SquaredExponential(1)

        # Sample a shared SpectralSample
        key, subkey = jax.random.split(key)
        sample = kernel.sample_from_spectral(rff_sample_size, subkey)

        # Build ODE
        hippo = HiPPO_LegS(inducing_size)
        ode_inducing = HiPPOInducing(
            inducing_size, hippo, rff_sample_size,
            start, end, num_steps, prev_steps,
        )

        # Build Analytic
        ana_inducing = AnalyticHiPPOInducing(
            inducing_size, rff_sample_size,
            start, end, num_steps, prev_steps,
            phase_origin=0.0,
            bessel_backend=bessel_backend,
        )

        # Generate test data
        N = 30
        X = jnp.linspace(start, end, N)[:, None]

        # Compute kernel matrices
        km_ode, _ = ode_inducing.compute_kernel_matrices(
            kernel, X, sample=sample,
        )
        km_ana, _ = ana_inducing.compute_kernel_matrices(
            kernel, X, sample=sample,
        )

        # ODE uses numerical integration; looser tolerance is expected.
        z_tol = 0.1 if prev_steps == 0 else 0.5
        k_tol = 0.1 if prev_steps == 0 else 0.5

        all_pass &= _check("Z", km_ode.Z, km_ana.Z, tol=z_tol)
        all_pass &= _check("Kuu", km_ode.Kuu, km_ana.Kuu, tol=k_tol)
        all_pass &= _check("Kfu", km_ode.Kfu, km_ana.Kfu, tol=k_tol)
        all_pass &= _check("Kff", km_ode.Kff, km_ana.Kff, tol=1e-6)

    return all_pass


def verify_feature(seed=42, num_components=50, inducing_size=15, num_steps=20,
                   bessel_backend="spbessax"):
    """Compare FreqSpace (omega=w, gamma=uniform) vs SE+RFF.

    When FreqSpace omega matches SE spectral sample and gamma is uniform,
    the kernel matrices should agree.
    """
    print("\n" + "=" * 60)
    print("FEATURE VERIFICATION: FreqSpace vs SE+RFF")
    print("=" * 60)

    seed_everything(seed)
    key = jax.random.PRNGKey(seed)

    # SE kernel
    se_kernel = SquaredExponential(1)
    key, subkey = jax.random.split(key)
    se_sample = se_kernel.sample_from_spectral(num_components, subkey)

    # FreqSpace kernel initialized to match SE
    fs_kernel = FrequencySpace(num_components)
    fs_kernel = eqx.tree_at(lambda k: k.omega, fs_kernel, jnp.squeeze(se_sample.frequencies, axis=0))
    # Set gamma to uniform (1/R each): softmax(0,...,0) = 1/R
    fs_kernel = eqx.tree_at(lambda k: k.log_gamma_raw, fs_kernel, jnp.zeros(num_components))
    fs_kernel = eqx.tree_at(lambda k: k.log_sf, fs_kernel, jnp.squeeze(se_kernel.log_sf))

    # FreqSpace sample
    fs_sample = fs_kernel.sample_from_spectral()

    # Verify the samples match conceptually
    print(f"\n  SE scale[0:3]:  {se_sample.scale[:3]}")
    print(f"  FS scale[0:3]:  {fs_sample.scale[:3]}")

    # Build analytic inducing for both
    start, end = 0.0, 10.0

    ana_se = AnalyticHiPPOInducing(
        inducing_size, num_components,
        start, end, num_steps, 0,
        bessel_backend=bessel_backend,
    )
    ana_fs = AnalyticHiPPOInducing(
        inducing_size, num_components,
        start, end, num_steps, 0,
        bessel_backend=bessel_backend,
    )

    N = 30
    X = jnp.linspace(start, end, N)[:, None]

    km_se, _ = ana_se.compute_kernel_matrices(se_kernel, X, sample=se_sample)
    km_fs, _ = ana_fs.compute_kernel_matrices(fs_kernel, X, sample=fs_sample)

    all_pass = True
    all_pass &= _check("Z", km_se.Z, km_fs.Z, tol=1e-6)
    all_pass &= _check("Kuu (pre-sf)", km_se.Z.T @ km_se.Z, km_fs.Z.T @ km_fs.Z, tol=1e-6)

    # Kuu with signal variance
    sf_se = float(jnp.exp(se_kernel.log_sf).squeeze())
    sf_fs = float(jnp.exp(fs_kernel.log_sf).squeeze())
    print(f"\n  sf_se={sf_se:.6f}  sf_fs={sf_fs:.6f}")
    all_pass &= _check("Kuu", km_se.Kuu, km_fs.Kuu, tol=1e-6)
    all_pass &= _check("Kfu", km_se.Kfu, km_fs.Kfu, tol=1e-6)
    all_pass &= _check("Kff", km_se.Kff, km_fs.Kff, tol=1e-6)

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Kernel matrix diagnostics")
    parser.add_argument(
        "mode", choices=["solver", "feature", "all"],
        help="Verification mode",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bessel-backend", type=str, default="spbessax",
        choices=["forward_recurrence", "scipy", "spbessax"],
        help="Spherical Bessel function backend (default: spbessax)",
    )
    args = parser.parse_args()

    results = {}

    if args.mode in ("solver", "all"):
        results["solver"] = verify_solver(
            seed=args.seed, bessel_backend=args.bessel_backend,
        )

    if args.mode in ("feature", "all"):
        results["feature"] = verify_feature(
            seed=args.seed, bessel_backend=args.bessel_backend,
        )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        all_pass &= passed

    if not all_pass:
        print("\nSome checks FAILED.")
        sys.exit(1)
    else:
        print("\nAll checks PASSED.")


if __name__ == "__main__":
    main()

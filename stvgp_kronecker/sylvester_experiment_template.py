import argparse
import csv
import time

import numpy as np
from scipy.linalg import cholesky, eigh, solve


def make_spd_matrix(rng, size, jitter=0.1):
    """Construct a symmetric positive definite matrix."""
    a = rng.standard_normal((size, size))
    return a @ a.T + jitter * np.eye(size)


def sylvester_qform(Ks, Kt, G, B, Q):
    """
    Solve Ks^{-1} Z Kt^{-1} + G Z B = Q
    and return <Q, Z>_F = vec(Q)^T Lambda^{-1} vec(Q).
    """
    Ls = cholesky(Ks, lower=True)   # Ks = Ls Ls^T
    Lt = cholesky(Kt, lower=True)   # Kt = Lt Lt^T

    # Whitening
    Qt = Ls.T @ Q @ Lt
    Gt = Ls.T @ G @ Ls
    Bt = Lt.T @ B @ Lt

    # Eigendecomposition
    gam, Us = eigh(Gt)
    bet, Ut = eigh(Bt)

    # Entrywise solve in rotated coordinates
    Qhat = Us.T @ Qt @ Ut
    denom = 1.0 + np.outer(gam, bet)
    Zhat = Qhat / denom

    # Transform back
    Zt = Us @ Zhat @ Ut.T
    Z = Ls @ Zt @ Lt.T

    qform = float(np.sum(Q * Z))   # Frobenius inner product
    return qform, Z


def dense_qform(Ks, Kt, G, B, Q):
    """
    Reference solve using the explicit Kronecker-structured dense system.

    vec(Ks^{-1} Z Kt^{-1}) = (Kt^{-T} kron Ks^{-1}) vec(Z)
    vec(G Z B)             = (B^T kron G) vec(Z)
    """
    ks_inv = np.linalg.inv(Ks)
    kt_inv = np.linalg.inv(Kt)

    lam = np.kron(kt_inv.T, ks_inv) + np.kron(B.T, G)
    q = Q.reshape(-1, order="F")
    z = solve(lam, q, assume_a="sym")
    Z = z.reshape(Q.shape, order="F")
    qform = float(q @ z)
    return qform, Z


def relative_error(reference, estimate):
    ref_norm = np.linalg.norm(reference)
    if ref_norm == 0.0:
        return float(np.linalg.norm(estimate))
    return float(np.linalg.norm(estimate - reference) / ref_norm)


def benchmark_once(rng, Ms, Mt):
    Ks = make_spd_matrix(rng, Ms)
    Kt = make_spd_matrix(rng, Mt)
    G = make_spd_matrix(rng, Ms)
    B = make_spd_matrix(rng, Mt)
    Q = rng.standard_normal((Ms, Mt))

    t0 = time.perf_counter()
    dense_qf, dense_Z = dense_qform(Ks, Kt, G, B, Q)
    dense_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    syl_qf, syl_Z = sylvester_qform(Ks, Kt, G, B, Q)
    syl_time = time.perf_counter() - t1

    return {
        "dense_time_s": dense_time,
        "sylvester_time_s": syl_time,
        "speedup_dense_over_sylvester": dense_time / syl_time,
        "rel_err_qform": abs(syl_qf - dense_qf) / max(abs(dense_qf), 1e-15),
        "rel_err_solution": relative_error(dense_Z, syl_Z),
        "theory_dense_(MtMs)^3": (Ms * Mt) ** 3,
        "theory_sylvester": Ms ** 3 + Mt ** 3 + (Ms ** 2) * Mt + Ms * (Mt ** 2),
    }


def aggregate_results(rows):
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def run_benchmark(sizes, repeats, seed):
    rng = np.random.default_rng(seed)
    results = []

    for size in sizes:
        rows = [benchmark_once(rng, size, size) for _ in range(repeats)]
        agg = aggregate_results(rows)
        agg.update({
            "Ms": size,
            "Mt": size,
            "N=Ms*Mt": size * size,
        })
        results.append(agg)

    return results


def write_csv(path, rows):
    fieldnames = [
        "Ms",
        "Mt",
        "N=Ms*Mt",
        "dense_time_s",
        "sylvester_time_s",
        "speedup_dense_over_sylvester",
        "rel_err_qform",
        "rel_err_solution",
        "theory_dense_(MtMs)^3",
        "theory_sylvester",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark the Sylvester solver against an explicit dense solve."
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[4, 6, 8, 10, 12, 14, 16],
        help="Matrix sizes to benchmark. Each size uses Ms = Mt = size.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of random trials averaged for each matrix size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the random number generator.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sylvester_benchmark_results.csv",
        help="CSV path for benchmark results.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run_benchmark(args.sizes, args.repeats, args.seed)
    write_csv(args.output, results)

    for row in results:
        print(
            f"Ms=Mt={int(row['Ms']):2d} | "
            f"dense={row['dense_time_s']:.6e}s | "
            f"sylvester={row['sylvester_time_s']:.6e}s | "
            f"speedup={row['speedup_dense_over_sylvester']:.3f}x | "
            f"rel_err_qform={row['rel_err_qform']:.3e} | "
            f"rel_err_solution={row['rel_err_solution']:.3e}"
        )

    print(f"\nSaved benchmark table to: {args.output}")

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _read_comparison_rows(path: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "eval_task": int(row["eval_task"]),
                    "after_task": int(row["after_task"]),
                    "analytic_variant": row["analytic_variant"],
                    "ode_rmse": float(row["ode_rmse"]),
                    "analytic_rmse": float(row["analytic_rmse"]),
                    "ode_r2": float(row["ode_r2"]),
                    "analytic_r2": float(row["analytic_r2"]),
                    "ode_nlpd": float(row["ode_nlpd"]),
                    "analytic_nlpd": float(row["analytic_nlpd"]),
                    "ode_ece": float(row["ode_ece"]),
                    "analytic_ece": float(row["analytic_ece"]),
                }
            )
    return rows


def _x_keys(rows: list[dict[str, float | int | str]]) -> list[tuple[int, int]]:
    keys = {(int(r["eval_task"]), int(r["after_task"])) for r in rows}
    return sorted(keys, key=lambda t: (t[1], t[0]))


def _series_from_keys(
    rows: list[dict[str, float | int | str]],
    metric_prefix: str,
    keys: list[tuple[int, int]],
) -> tuple[list[float], list[float]]:
    ode_lookup: dict[tuple[int, int], float] = {}
    analytic_lookup: dict[tuple[int, int], float] = {}
    for r in rows:
        key = (int(r["eval_task"]), int(r["after_task"]))
        ode_lookup[key] = float(r[f"ode_{metric_prefix}"])
        analytic_lookup[key] = float(r[f"analytic_{metric_prefix}"])
    return [ode_lookup[k] for k in keys], [analytic_lookup[k] for k in keys]


def _plot_metric(
    ax,
    x: list[int],
    x_labels: list[str],
    ode_y: list[float],
    analytic_y: list[float],
    title: str,
    ylabel: str,
) -> None:
    ax.plot(
        x,
        ode_y,
        marker="o",
        linewidth=2.0,
        markersize=5.5,
        linestyle="-",
        color="#1f77b4",
        label="Numerical ODE",
    )
    ax.plot(
        x,
        analytic_y,
        marker="s",
        linewidth=2.0,
        markersize=5.5,
        linestyle="--",
        color="#d62728",
        label="Analytic",
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Task Pair", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_comparison = repo_root / "ohsgp_analytic" / "models0" / "ODEvsAnalytic_comparison_metrics.csv"
    default_out = Path(__file__).resolve().parent / "new ODEvsAnalytic comparison.png"

    parser = argparse.ArgumentParser("Plot ODE vs Analytic metrics comparison from merged CSV")
    parser.add_argument("--comparison_csv", type=Path, default=default_comparison)
    parser.add_argument("--out_png", type=Path, default=default_out)
    args = parser.parse_args()

    rows = _read_comparison_rows(args.comparison_csv)
    keys = _x_keys(rows)
    x_labels = [f"T{eval_task}->A{after_task}" for eval_task, after_task in keys]
    x = list(range(len(keys)))

    ode_rmse, analytic_rmse = _series_from_keys(rows, "rmse", keys)
    ode_r2, analytic_r2 = _series_from_keys(rows, "r2", keys)
    ode_nlpd, analytic_nlpd = _series_from_keys(rows, "nlpd", keys)
    ode_ece, analytic_ece = _series_from_keys(rows, "ece", keys)

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    _plot_metric(axes[0, 0], x, x_labels, ode_rmse, analytic_rmse, "RMSE (Lower is Better)", "RMSE")
    _plot_metric(axes[0, 1], x, x_labels, ode_r2, analytic_r2, "R2 Score (Higher is Better)", "R2")
    _plot_metric(axes[1, 0], x, x_labels, ode_nlpd, analytic_nlpd, "NLPD (Lower is Better)", "NLPD")
    _plot_metric(axes[1, 1], x, x_labels, ode_ece, analytic_ece, "ECE (Lower is Better)", "ECE")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("ODE vs Analytical Comparison", fontsize=22, y=1.04, fontweight="bold")
    fig.tight_layout()

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.out_png}")


if __name__ == "__main__":
    main()

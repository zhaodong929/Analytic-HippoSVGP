#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _read_metrics(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "eval_task": int(row["eval_task"]),
                    "after_task": int(row["after_task"]),
                    "rmse": float(row["rmse"]),
                    "r2": float(row["r2"]),
                    "nlpd": float(row["nlpd"]),
                    "ece": float(row["ece"]),
                }
            )
    return rows


def _x_keys(rows: list[dict[str, float]]) -> list[tuple[int, int]]:
    # Match the ordering style in the reference figure:
    # T1->A1, T1->A2, T2->A2, T1->A3, T2->A3, T3->A3, ...
    keys = {(int(r["eval_task"]), int(r["after_task"])) for r in rows}
    return sorted(keys, key=lambda t: (t[1], t[0]))


def _series_from_keys(
    rows: list[dict[str, float]],
    metric: str,
    keys: list[tuple[int, int]],
) -> list[float]:
    lookup: dict[tuple[int, int], float] = {}
    for r in rows:
        k = (int(r["eval_task"]), int(r["after_task"]))
        lookup[k] = float(r[metric])
    return [lookup[k] for k in keys]


def _plot_metric(
    ax,
    x: list[int],
    x_labels: list[str],
    ys: dict[str, list[float]],
    title: str,
    ylabel: str,
):
    label_new = "Analytic (new, with phase-origin shift)"
    label_old = "Analytic (old, no phase-origin shift)"
    palette = {
        "Numerical ODE": "#1f77b4",
        label_old: "#d62728",
        label_new: "#2ca02c",
    }
    markers = {
        "Numerical ODE": "o",
        label_old: "s",
        label_new: "^",
    }
    linestyles = {
        "Numerical ODE": "-",
        label_old: "--",
        label_new: "--",
    }
    for name, y in ys.items():
        ax.plot(
            x,
            y,
            marker=markers[name],
            linewidth=2.0,
            markersize=5.5,
            linestyle=linestyles[name],
            color=palette[name],
            label=name,
        )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Task Pair", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45)


def main():
    repo_root = Path(__file__).resolve().parents[2]
    default_ode = repo_root / "ohsgp" / "models0" / "metrics.csv"
    default_global = repo_root / "ohsgp_analytic" / "models0" / "global_start" / "metrics.csv"
    default_zero = repo_root / "ohsgp_analytic" / "models0" / "zero" / "metrics.csv"
    default_out = Path(__file__).resolve().parent / "new ODEvsAnalytic 3way.png"

    parser = argparse.ArgumentParser("Plot ODE vs Analytic(2 branches) metrics comparison")
    parser.add_argument("--ode_csv", type=Path, default=default_ode)
    parser.add_argument("--analytic_global_csv", type=Path, default=default_global)
    parser.add_argument("--analytic_zero_csv", type=Path, default=default_zero)
    parser.add_argument("--out_png", type=Path, default=default_out)
    args = parser.parse_args()

    ode_rows = _read_metrics(args.ode_csv)
    ana_global_rows = _read_metrics(args.analytic_global_csv)
    ana_zero_rows = _read_metrics(args.analytic_zero_csv)
    keys = _x_keys(ode_rows)
    x_labels = [f"T{et}->A{at}" for et, at in keys]
    x = list(range(len(keys)))

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    _plot_metric(
        axes[0, 0],
        x,
        x_labels,
        {
            "Numerical ODE": _series_from_keys(ode_rows, "rmse", keys),
            "Analytic (old, no phase-origin shift)": _series_from_keys(ana_zero_rows, "rmse", keys),
            "Analytic (new, with phase-origin shift)": _series_from_keys(ana_global_rows, "rmse", keys),
        },
        title="RMSE (Lower is Better)",
        ylabel="RMSE",
    )
    _plot_metric(
        axes[0, 1],
        x,
        x_labels,
        {
            "Numerical ODE": _series_from_keys(ode_rows, "r2", keys),
            "Analytic (old, no phase-origin shift)": _series_from_keys(ana_zero_rows, "r2", keys),
            "Analytic (new, with phase-origin shift)": _series_from_keys(ana_global_rows, "r2", keys),
        },
        title="R2 Score (Higher is Better)",
        ylabel="R2",
    )
    _plot_metric(
        axes[1, 0],
        x,
        x_labels,
        {
            "Numerical ODE": _series_from_keys(ode_rows, "nlpd", keys),
            "Analytic (old, no phase-origin shift)": _series_from_keys(ana_zero_rows, "nlpd", keys),
            "Analytic (new, with phase-origin shift)": _series_from_keys(ana_global_rows, "nlpd", keys),
        },
        title="NLPD (Lower is Better)",
        ylabel="NLPD",
    )
    _plot_metric(
        axes[1, 1],
        x,
        x_labels,
        {
            "Numerical ODE": _series_from_keys(ode_rows, "ece", keys),
            "Analytic (old, no phase-origin shift)": _series_from_keys(ana_zero_rows, "ece", keys),
            "Analytic (new, with phase-origin shift)": _series_from_keys(ana_global_rows, "ece", keys),
        },
        title="ECE (Lower is Better)",
        ylabel="ECE",
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("ODE vs Analytical Comparison", fontsize=22, y=1.04, fontweight="bold")
    fig.tight_layout()

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.out_png}")


if __name__ == "__main__":
    main()

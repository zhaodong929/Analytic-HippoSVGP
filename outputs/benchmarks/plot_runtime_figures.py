from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"
AGG_DIR = ROOT / "aggregated"
FIG_DIR = ROOT / "figures"
CONFIG_PATH = ROOT / "accuracy_flops_config.json"


COLORS = {
    "ode": "#1f77b4",
    "analytic": "#c44e52",
}

MARKERS = {
    "ode": "o",
    "analytic": "s",
}

LINESTYLES = {
    "ode": "-",
    "analytic": "--",
}

LABELS = {
    "ode": "ODE baseline",
    "analytic": "Analytic HiPPO-SVGP",
}


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "legend.fontsize": 7.3,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "axes.linewidth": 0.8,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_grouped_runtime_rows(path: Path):
    grouped = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["mode"], row["method"], int(row["sweep_value"]))
            grouped[key].append(float(row["time_ms"]))
    return grouped


def summarize_runtime(grouped, sweep_values):
    summary = {}
    for mode in ("kernel_only", "full_update"):
        for method in ("ode", "analytic"):
            medians = []
            stds = []
            for sweep_value in sweep_values:
                values = np.asarray(grouped[(mode, method, sweep_value)], dtype=float)
                medians.append(float(np.median(values)))
                stds.append(float(np.std(values)))
            summary[(mode, method)] = {
                "median": np.asarray(medians, dtype=float),
                "std": np.asarray(stds, dtype=float),
            }
    return summary


def load_accuracy_summary(path: Path):
    rows = defaultdict(dict)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "rmse_mean" in row:
                rmse_mean = float(row["rmse_mean"])
                rmse_std = float(row.get("rmse_std", 0.0) or 0.0)
                nll_mean = float(row.get("nll_mean", row.get("nll", 0.0)) or 0.0)
            else:
                # Backward compatibility for older summary CSVs that stored a
                # single RMSE/NLL value instead of mean/std aggregates.
                rmse_mean = float(row["rmse"])
                rmse_std = 0.0
                nll_mean = float(row.get("nll", 0.0) or 0.0)
            rows[row["method"]][int(row["sweep_value"])] = {
                "rmse_mean": rmse_mean,
                "rmse_std": rmse_std,
                "nll_mean": nll_mean,
                "delta_t": float(row["delta_t"]),
            }
    return rows


def load_config():
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {}


def style_axes(ax, ylabel: str):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.6)
    ax.tick_params(axis="both", labelsize=8.0)
    ax.set_ylabel(ylabel, fontsize=9.0)


def add_series(ax, x, y, method, std=None):
    color = COLORS[method]
    if std is not None:
        lower = np.maximum(y - std, 0.0)
        upper = y + std
        ax.fill_between(x, lower, upper, color=color, alpha=0.14, linewidth=0)
    ax.plot(
        x,
        y,
        color=color,
        linestyle=LINESTYLES[method],
        linewidth=2.0,
        marker=MARKERS[method],
        markersize=4.8,
        markerfacecolor="white",
        markeredgewidth=1.3,
        label=LABELS[method],
    )


def add_embedded_legend(ax):
    leg = ax.legend(
        loc="upper left",
        frameon=True,
        fontsize=7.2,
        handlelength=1.8,
        borderpad=0.25,
        labelspacing=0.25,
    )
    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("#cccccc")
    frame.set_alpha(0.88)


def get_delta_t_tick_labels(t_values, delta_t_values):
    labels = []
    for t_value, delta_t in zip(t_values, delta_t_values):
        if delta_t >= 0.1:
            delta_str = f"{delta_t:.2f}"
        else:
            delta_str = f"{delta_t:.3f}"
        labels.append(f"{delta_str}\n({t_value})")
    return labels


def plot_runtime_kernel_vs_delta_t(t_values, delta_t_values, runtime_summary):
    fig, ax = plt.subplots(figsize=(2.85, 2.25))
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.28, top=0.96)

    x = np.arange(len(t_values), dtype=float)
    style_axes(ax, ylabel="Runtime (ms)")

    for method in ("ode", "analytic"):
        stats = runtime_summary[("kernel_only", method)]
        add_series(ax, x, stats["median"], method, std=stats["std"])

    ax.set_xticks(x, get_delta_t_tick_labels(t_values, delta_t_values))
    ax.set_xlabel(r"Integration step size $\Delta t$ ($T$ in parentheses)", fontsize=8.8)
    add_embedded_legend(ax)

    fig.savefig(FIG_DIR / "runtime_vs_delta_t_kernel.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "runtime_vs_delta_t_kernel.png", bbox_inches="tight")
    plt.close(fig)


def plot_rmse_vs_delta_t(t_values, delta_t_values, accuracy_summary):
    fig, ax = plt.subplots(figsize=(2.85, 2.25))
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.28, top=0.96)

    x = np.arange(len(t_values), dtype=float)
    style_axes(ax, ylabel="Test RMSE")

    for method in ("ode", "analytic"):
        y = np.asarray([accuracy_summary[method][value]["rmse_mean"] for value in t_values], dtype=float)
        std = np.asarray([accuracy_summary[method][value]["rmse_std"] for value in t_values], dtype=float)
        add_series(ax, x, y, method, std=std)

    ax.set_xticks(x, get_delta_t_tick_labels(t_values, delta_t_values))
    ax.set_xlabel(r"Integration step size $\Delta t$ ($T$ in parentheses)", fontsize=8.8)
    add_embedded_legend(ax)

    fig.savefig(FIG_DIR / "rmse_vs_delta_t.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "rmse_vs_delta_t.png", bbox_inches="tight")
    plt.close(fig)


def plot_runtime_kernel_vs_m(m_values, runtime_summary):
    fig, ax = plt.subplots(figsize=(2.85, 2.25))
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.22, top=0.96)

    x = np.arange(len(m_values), dtype=float)
    style_axes(ax, ylabel="Runtime (ms)")

    for method in ("ode", "analytic"):
        stats = runtime_summary[("kernel_only", method)]
        add_series(ax, x, stats["median"], method, std=stats["std"])

    ax.set_xticks(x, [str(v) for v in m_values])
    ax.set_xlabel(r"Basis size $M$", fontsize=8.8)
    add_embedded_legend(ax)

    fig.savefig(FIG_DIR / "runtime_vs_M_kernel.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "runtime_vs_M_kernel.png", bbox_inches="tight")
    plt.close(fig)


def plot_rmse_vs_m(m_values, accuracy_summary):
    fig, ax = plt.subplots(figsize=(2.85, 2.25))
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.22, top=0.96)

    x = np.arange(len(m_values), dtype=float)
    style_axes(ax, ylabel="Test RMSE")

    for method in ("ode", "analytic"):
        y = np.asarray([accuracy_summary[method][value]["rmse_mean"] for value in m_values], dtype=float)
        std = np.asarray([accuracy_summary[method][value]["rmse_std"] for value in m_values], dtype=float)
        add_series(ax, x, y, method, std=std)

    ax.set_xticks(x, [str(v) for v in m_values])
    ax.set_xlabel(r"Basis size $M$", fontsize=8.8)
    add_embedded_legend(ax)

    fig.savefig(FIG_DIR / "rmse_vs_M.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "rmse_vs_M.png", bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()

    t_values = config.get("t_values", [100, 300, 1000, 3000, 7000])
    t_runtime = summarize_runtime(load_grouped_runtime_rows(RAW_DIR / "runtime_vs_T_raw.csv"), t_values)
    t_accuracy = load_accuracy_summary(AGG_DIR / "accuracy_vs_T_summary.csv")
    delta_t_values = [t_accuracy["ode"][value]["delta_t"] for value in t_values]

    plot_runtime_kernel_vs_delta_t(t_values, delta_t_values, t_runtime)
    plot_rmse_vs_delta_t(t_values, delta_t_values, t_accuracy)

    m_values = config.get("m_values", [8, 16, 32, 64, 128])
    m_runtime = summarize_runtime(load_grouped_runtime_rows(RAW_DIR / "runtime_vs_M_raw.csv"), m_values)
    m_accuracy = load_accuracy_summary(AGG_DIR / "accuracy_vs_M_summary.csv")

    plot_runtime_kernel_vs_m(m_values, m_runtime)
    plot_rmse_vs_m(m_values, m_accuracy)


if __name__ == "__main__":
    main()

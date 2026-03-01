"""Visualization script for compare-mode results.

Reads saved results and generates posterior, metrics, and training plots.
Does NOT import model code -- works entirely from saved files.

Usage:
    python -m scripts.plot results/compare_solar_sgpr/ --type all
    python -m scripts.plot results/compare_solar_sgpr/ --type posterior
    python -m scripts.plot results/compare_solar_sgpr/ --type metrics
    python -m scripts.plot results/compare_solar_sgpr/ --type training
"""

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _load_comparison_csv(results_dir):
    """Load comparison.csv into list of dicts."""
    path = os.path.join(results_dir, "comparison.csv")
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({
                "strategy": row["strategy"],
                "task_id": int(row["task_id"]),
                "rmse": float(row["rmse"]),
                "r2": float(row["r2"]),
                "nlpd": float(row["nlpd"]),
                "ece": float(row["ece"]),
                "wall_time": float(row["wall_time"]),
                "final_elbo": float(row["final_elbo"]),
            })
    return rows


def _discover_strategies(results_dir):
    """Discover strategy directories."""
    strategies = []
    for name in sorted(os.listdir(results_dir)):
        d = os.path.join(results_dir, name)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "metrics.csv")):
            strategies.append(name)
    return strategies


def _load_predictions(results_dir, strategy):
    """Load prediction tensors for a strategy."""
    pred_dir = os.path.join(results_dir, strategy, "predictions")
    preds = {}
    for fname in sorted(os.listdir(pred_dir)):
        if fname.endswith(".pt"):
            task_id = int(fname.replace("task", "").replace(".pt", ""))
            preds[task_id] = torch.load(
                os.path.join(pred_dir, fname), map_location="cpu", weights_only=True,
            )
    return preds


def _load_training_log(results_dir, strategy):
    """Load training log JSON."""
    path = os.path.join(results_dir, strategy, "training_log.json")
    with open(path) as f:
        return json.load(f)


def plot_posterior(results_dir, strategies=None):
    """Plot posterior predictions: rows=tasks, columns=strategies.

    Each subplot shows the GP mean + 2sigma band on a dense grid covering
    all seen data, with training data (all seen + current task highlighted)
    and test data overlaid.
    """
    if strategies is None:
        strategies = _discover_strategies(results_dir)

    all_preds = {s: _load_predictions(results_dir, s) for s in strategies}
    comp = _load_comparison_csv(results_dir)

    task_ids = sorted(all_preds[strategies[0]].keys())
    num_tasks = len(task_ids)
    num_strats = len(strategies)

    fig, axes = plt.subplots(
        num_tasks, num_strats,
        figsize=(8 * num_strats, 3 * num_tasks),
        sharex=True, sharey=True,
        squeeze=False,
    )

    for col, strat in enumerate(strategies):
        preds = all_preds[strat]
        strat_metrics = [r for r in comp if r["strategy"] == strat]
        metrics_by_task = {r["task_id"]: r for r in strat_metrics}

        for row, tid in enumerate(task_ids):
            ax = axes[row, col]
            p = preds[tid]

            # Dense grid predictions
            X_grid = p["X_grid"].squeeze().numpy()
            mean = p["mean_grid"].squeeze().numpy()
            var = p["var_grid"].squeeze().numpy()
            std = np.sqrt(np.maximum(var, 1e-10))

            # Confidence band + mean
            ax.fill_between(
                X_grid, mean - 2 * std, mean + 2 * std,
                alpha=0.2, color="C0",
            )
            ax.plot(X_grid, mean, color="C0", linewidth=1.5)

            # All seen training data
            X_train_all = p["X_train_all"].squeeze().numpy()
            y_train_all = p["y_train_all"].squeeze().numpy()
            ax.scatter(
                X_train_all, y_train_all,
                s=12, c="C1", alpha=0.4, zorder=2, label="Train (seen)",
            )

            # Current task training data (highlighted)
            X_cur = p["X_train_current"].squeeze().numpy()
            y_cur = p["y_train_current"].squeeze().numpy()
            ax.scatter(
                X_cur, y_cur,
                s=25, c="C3", edgecolors="black", linewidths=0.5,
                zorder=3, label="Train (current)",
            )

            # All seen test data
            X_test_all = p["X_test_all"].squeeze().numpy()
            y_test_all = p["y_test_all"].squeeze().numpy()
            ax.scatter(
                X_test_all, y_test_all,
                s=15, marker="x", c="C2", alpha=0.4, zorder=4, label="Test (seen)",
            )

            # Current task test data (highlighted)
            X_test_cur = p["X_test_current"].squeeze().numpy()
            y_test_cur = p["y_test_current"].squeeze().numpy()
            ax.scatter(
                X_test_cur, y_test_cur,
                s=30, marker="x", c="C2", linewidths=1.5,
                zorder=5, label="Test (current)",
            )

            # Metrics annotation
            m = metrics_by_task.get(tid, {})
            rmse_val = m.get("rmse", float("nan"))
            nlpd_val = m.get("nlpd", float("nan"))
            ax.text(
                0.98, 0.95,
                f"RMSE={rmse_val:.3f}\nNLPD={nlpd_val:.3f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
            )

            if row == 0:
                ax.set_title(strat, fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"Task {tid}", fontsize=10)
            if row == 0 and col == num_strats - 1:
                ax.legend(fontsize=7, loc="upper left", ncol=2)

    fig.tight_layout()
    out = os.path.join(results_dir, "posterior.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_metrics(results_dir, strategies=None):
    """Plot RMSE, NLPD, R^2, ECE across tasks for each strategy."""
    if strategies is None:
        strategies = _discover_strategies(results_dir)

    comp = _load_comparison_csv(results_dir)
    metric_names = ["rmse", "nlpd", "r2", "ece"]
    metric_labels = ["RMSE", "NLPD", "R$^2$", "ECE"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    for ax, mname, mlabel in zip(axes, metric_names, metric_labels):
        for strat in strategies:
            rows = [r for r in comp if r["strategy"] == strat]
            rows.sort(key=lambda r: r["task_id"])
            tasks = [r["task_id"] for r in rows]
            vals = [r[mname] for r in rows]
            ax.plot(tasks, vals, marker="o", markersize=4, label=strat)
        ax.set_xlabel("Task")
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(results_dir, "metrics.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_training(results_dir, strategies=None):
    """Plot epoch-level ELBO learning curves."""
    if strategies is None:
        strategies = _discover_strategies(results_dir)

    # Determine number of tasks from first strategy
    log0 = _load_training_log(results_dir, strategies[0])
    num_tasks = len(log0)

    fig, axes = plt.subplots(
        1, num_tasks, figsize=(4 * num_tasks, 3.5), squeeze=False
    )

    for strat in strategies:
        log = _load_training_log(results_dir, strat)
        for i, task_log in enumerate(log):
            ax = axes[0, i]
            history = task_log["history"]
            if not history:
                continue
            epochs = [h["epoch"] for h in history]
            elbos = [h["elbo"] for h in history]
            ax.plot(epochs, elbos, label=strat, alpha=0.8)
            ax.set_title(f"Task {task_log['task_id']}", fontsize=10)
            ax.set_xlabel("Epoch")
            if i == 0:
                ax.set_ylabel("ELBO")

    axes[0, -1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    out = os.path.join(results_dir, "training.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot compare-mode results")
    parser.add_argument("results_dir", type=str, help="Path to results directory")
    parser.add_argument(
        "--type", choices=["posterior", "metrics", "training", "all"],
        default="all", help="Plot type",
    )
    args = parser.parse_args()

    strategies = _discover_strategies(args.results_dir)
    if not strategies:
        print(f"No strategy directories found in {args.results_dir}")
        return

    print(f"Strategies: {strategies}")

    if args.type in ("posterior", "all"):
        plot_posterior(args.results_dir, strategies)

    if args.type in ("metrics", "all"):
        plot_metrics(args.results_dir, strategies)

    if args.type in ("training", "all"):
        plot_training(args.results_dir, strategies)


if __name__ == "__main__":
    main()

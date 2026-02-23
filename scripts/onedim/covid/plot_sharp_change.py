import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_task(task_id: int, data_root: Path):
    fp = data_root / f"task_{task_id}" / "sequences" / "state_MG.npz"
    if not fp.exists():
        raise FileNotFoundError(f"Missing file: {fp}")
    data = np.load(fp)
    max_seq = int(data["max_seq_length"])
    days = data["days"][:max_seq].astype(float)
    deaths = data["deaths"][:max_seq].astype(float)
    return days, deaths


def _rolling_mean_std(x: np.ndarray, window: int):
    if len(x) < window:
        return None, None
    kernel = np.ones(window, dtype=float) / window
    mean = np.convolve(x, kernel, mode="valid")
    mean2 = np.convolve(x ** 2, kernel, mode="valid")
    var = np.maximum(mean2 - mean ** 2, 0.0)
    std = np.sqrt(var)
    return mean, std


def plot_timeseries(tasks, out_dir: Path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    # Full range per task
    for task_id, (days, deaths) in tasks.items():
        axes[0].plot(days, deaths, label=f"task{task_id}")
        axes[0].axvline(days[-1], color="gray", linestyle="--", alpha=0.3)
        axes[0].text(
            days[len(days) // 2],
            np.max(deaths),
            f"task{task_id}",
            fontsize=8,
            ha="center",
            va="bottom",
            alpha=0.7,
        )
    axes[0].set_title("Per-task time series with task boundaries")
    axes[0].set_xlabel("day")
    axes[0].set_ylabel("deaths")
    axes[0].legend(ncol=5, fontsize=8, frameon=False)

    # Task5 tail zoom
    days5, deaths5 = tasks[5]
    k = min(15, len(deaths5))
    axes[1].plot(days5[-k:], deaths5[-k:], color="tab:red")
    axes[1].scatter(days5[-k:], deaths5[-k:], color="tab:red", s=20)
    axes[1].set_title(f"Task5 tail zoom (last {k} points)")
    axes[1].set_xlabel("day")
    axes[1].set_ylabel("deaths")

    fig.tight_layout()
    fig.savefig(out_dir / "01_timeseries_task_boundaries.png", dpi=150)
    plt.close(fig)


def plot_differences(tasks, out_dir: Path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for task_id, (days, deaths) in tasks.items():
        diff = np.diff(deaths)
        days_diff = days[1:]
        axes[0].plot(days_diff, diff, label=f"task{task_id}")
        axes[1].plot(days_diff, np.abs(diff), label=f"task{task_id}")

    axes[0].set_title("First differences (deaths[t] - deaths[t-1])")
    axes[0].set_ylabel("delta deaths")
    axes[0].legend(ncol=5, fontsize=8, frameon=False)

    axes[1].set_title("Absolute first differences")
    axes[1].set_xlabel("day")
    axes[1].set_ylabel("abs delta deaths")

    fig.tight_layout()
    fig.savefig(out_dir / "02_first_difference.png", dpi=150)
    plt.close(fig)


def plot_distributions(tasks, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [tasks[i][1] for i in range(1, 6)]
    parts = ax.violinplot(data, showmeans=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)
    ax.set_title("Per-task deaths distribution (violin)")
    ax.set_xlabel("task")
    ax.set_ylabel("deaths")
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels([f"task{i}" for i in range(1, 6)])

    fig.tight_layout()
    fig.savefig(out_dir / "03_deaths_distribution_violin.png", dpi=150)
    plt.close(fig)


def plot_absdiff_ecdf(tasks, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for task_id, (days, deaths) in tasks.items():
        absdiff = np.abs(np.diff(deaths))
        x = np.sort(absdiff)
        y = np.arange(1, len(x) + 1) / len(x) if len(x) else np.array([])
        ax.plot(x, y, label=f"task{task_id}")

    ax.set_title("ECDF of absolute first differences")
    ax.set_xlabel("abs delta deaths")
    ax.set_ylabel("ECDF")
    ax.legend(ncol=5, fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(out_dir / "04_absdiff_ecdf.png", dpi=150)
    plt.close(fig)


def plot_rolling_stats(tasks, out_dir: Path, window: int = 7):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    for task_id, (days, deaths) in tasks.items():
        mean, std = _rolling_mean_std(deaths, window)
        if mean is None:
            continue
        days_roll = days[window - 1 :]
        axes[0].plot(days_roll, mean, label=f"task{task_id}")
        axes[1].plot(days_roll, std, label=f"task{task_id}")

    axes[0].set_title(f"Rolling mean (window={window})")
    axes[0].set_xlabel("day")
    axes[0].set_ylabel("mean deaths")
    axes[0].legend(ncol=5, fontsize=8, frameon=False)

    axes[1].set_title(f"Rolling std (window={window})")
    axes[1].set_xlabel("day")
    axes[1].set_ylabel("std deaths")

    fig.tight_layout()
    fig.savefig(out_dir / "05_rolling_mean_std.png", dpi=150)
    plt.close(fig)


def plot_scatter(tasks, out_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 5))
    for task_id, (days, deaths) in tasks.items():
        ax.scatter(days, deaths, s=18, alpha=0.8, label=f"task{task_id}")
    ax.set_title("Days vs deaths (color by task)")
    ax.set_xlabel("day")
    ax.set_ylabel("deaths")
    ax.legend(ncol=5, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "06_scatter_by_task.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize sharp changes in state_MG dataset.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="ohsgp_analytic/Analyze the dataset",
        help="Output directory for PNG files.",
    )
    parser.add_argument("--window", type=int, default=7, help="Rolling window size.")
    args = parser.parse_args()

    root = _repo_root()
    data_root = root / "data" / "covid" / "processed_timeseries_5tasks"
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = {}
    for task_id in range(1, 6):
        tasks[task_id] = _load_task(task_id, data_root)

    plot_timeseries(tasks, out_dir)
    plot_differences(tasks, out_dir)
    plot_distributions(tasks, out_dir)
    plot_absdiff_ecdf(tasks, out_dir)
    plot_rolling_stats(tasks, out_dir, window=args.window)
    plot_scatter(tasks, out_dir)

    print(f"Saved PNGs to: {out_dir}")


if __name__ == "__main__":
    main()

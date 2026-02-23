import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(csv_path):
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "model": r["model"],
                    "freq": int(r["freq"]),
                    "eval_task": int(r["eval_task"]),
                    "rmse": float(r["rmse"]),
                    "r2": float(r["r2"]),
                    "nlpd": float(r["nlpd"]),
                }
            )
    return rows


def main():
    out_dir = Path(__file__).resolve().parent
    csv_path = out_dir / "per_task.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    rows = load_rows(csv_path)
    freqs = sorted({r["freq"] for r in rows})
    models = sorted({r["model"] for r in rows})

    # Optional nicer labels
    label_map = {
        "rff_baseline": "RFF baseline",
        "freqspace_rsf": "Freq-space (RSF)",
    }

    metrics = [
        ("rmse", "RMSE (Lower is Better)"),
        ("r2", "R2 Score (Higher is Better)"),
        ("nlpd", "NLPD (Lower is Better)"),
    ]

    for freq in freqs:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
        for metric_idx, (metric_key, metric_title) in enumerate(metrics):
            ax = axes[metric_idx]
            for model in models:
                series = [
                    r
                    for r in rows
                    if r["freq"] == freq and r["model"] == model
                ]
                series = sorted(series, key=lambda x: x["eval_task"])
                x = [f"T{r['eval_task']}" for r in series]
                y = [r[metric_key] for r in series]
                ax.plot(x, y, marker="o", label=label_map.get(model, model))

            ax.set_title(metric_title)
            ax.tick_params(axis="x", rotation=45)

        axes[0].set_ylabel("value")
        fig.suptitle(f"Per-task comparison (freq={freq})")
        axes[-1].legend(loc="best")
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(out_dir / f"compare_freq_{freq}.png", dpi=150)
        plt.close(fig)

    print(f"Saved {len(freqs)} figures to {out_dir}")


if __name__ == "__main__":
    main()

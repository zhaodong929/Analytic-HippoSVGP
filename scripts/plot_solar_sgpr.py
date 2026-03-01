"""Plot Solar SGPR results with task-1 ExactGP hyperparameter pre-fitting.

Workflow:
1) Fit exact GP hyperparameters on task 1.
2) Freeze those hyperparameters.
3) Sample spectral features once and reuse across all tasks.
4) Run ODE / Analytic HiPPO-SGPR task by task.
"""

from dataclasses import dataclass

import numpy as np
import torch

from hipposvgp.kernels import SquaredExponential
from hipposvgp.inducing import HiPPOInducing, AnalyticHiPPOInducing
from hipposvgp.hippo import HiPPO_LegS
from hipposvgp.models import ExactGP
from hipposvgp.models.sgpr import OnlineSGPR
from hipposvgp.rff import SpectralSample

from scripts.datasets.solar import SolarOnlineDataset
from scripts.utils import seed_everything, compute_time_window


@dataclass(frozen=True)
class FixedHyperparameters:
    """Hyperparameters shared by ExactGP and SGPR."""

    log_sf: torch.Tensor
    log_ls: torch.Tensor
    log_noise: torch.Tensor


def resolve_global_phase_origin(dataset):
    """Global phase origin for analytic HiPPO: start of task 1."""
    task1 = next(dataset.tasks())
    return torch.min(task1.X_train).item()


def fit_task1_hyperparameters(dataset, device, cfg):
    """Fit kernel/noise hyperparameters on task 1 via exact GP."""
    task1 = next(dataset.tasks())
    input_dim = task1.X_train.shape[1]

    kernel = SquaredExponential(input_dim, device=device)
    model = ExactGP(kernel=kernel, device=device)

    exact_cfg = cfg["exact_gp"]
    with torch.no_grad():
        model.kernel.log_ls.fill_(1.4108)
        model.kernel.log_sf.fill_(-0.5334)
        model.likelihood.log_variance.fill_(-4.5)
    print("Fitting ExactGP hyperparameters on task 1...")
    model.fit(
        task1.X_train,
        task1.y_train,
        num_epochs=exact_cfg["num_epochs"],
        lr=exact_cfg["lr"],
        verbose=exact_cfg.get("verbose", True),
        log_every=exact_cfg.get("log_every", 100),
    )

    fixed = FixedHyperparameters(
        log_sf=model.kernel.log_sf.detach().clone(),
        log_ls=model.kernel.log_ls.detach().clone(),
        log_noise=model.likelihood.log_variance.detach().clone(),
    )
    print(
        "Fixed hyperparameters:"
        f" log_sf={fixed.log_sf.item():.4f},"
        f" log_ls={fixed.log_ls.detach().cpu().numpy()},"
        f" log_noise={fixed.log_noise.item():.4f}"
    )
    return fixed


def sample_fixed_spectral_sample(input_dim, device, fixed_hparams, rff_sample_size):
    """Draw one SpectralSample to be shared by all tasks."""
    kernel = build_frozen_kernel(input_dim, device, fixed_hparams)
    return kernel.sample_from_spectral(rff_sample_size)


def build_frozen_kernel(input_dim, device, fixed_hparams):
    """Create SE kernel with fixed hyperparameters."""
    kernel = SquaredExponential(input_dim, device=device)
    with torch.no_grad():
        kernel.log_sf.copy_(fixed_hparams.log_sf.to(device=device, dtype=torch.float64))
        kernel.log_ls.copy_(fixed_hparams.log_ls.to(device=device, dtype=torch.float64))
    for param in kernel.parameters():
        param.requires_grad_(False)
    return kernel


def apply_frozen_noise(model, fixed_hparams):
    """Set SGPR noise from fitted hyperparameters and freeze it."""
    with torch.no_grad():
        model.log_noise.copy_(fixed_hparams.log_noise.to(device=model.device, dtype=model.dtype))
    model.log_noise.requires_grad_(False)


def train_sgpr(model, num_epochs=500, lr=0.01):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        # With fixed hyperparameters SGPR reduces to closed-form posterior updates.
        model.eval()
        with torch.no_grad():
            _ = model.elbo()
        return model

    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        try:
            bound = model.elbo()
            (-bound).backward()
        except torch.linalg.LinAlgError:
            break
        optimizer.step()
    return model


def predict_on_grid(model, X_grid):
    model.eval()
    with torch.no_grad():
        mean, var = model.predict(X_grid, full_cov=False, include_noise=True)
        if var.dim() == 1:
            var = var.unsqueeze(1)
    return mean.squeeze().cpu().numpy(), var.squeeze().cpu().numpy()


def run_experiment(
    strategy_name,
    dataset,
    device,
    cfg,
    fixed_hparams,
    *,
    phase_origin,
    spectral_sample,
):
    """Run full online experiment, returning per-task predictions."""
    seed_everything(0)

    inducing_size = cfg["inducing_size"]
    rff_sample_size = cfg["rff_sample_size"]
    step_size = cfg["step_size"]
    num_epochs = cfg["num_epochs"]

    prior_state = None
    prev_end = None
    prev_steps = 0
    results = []

    # Collect all data for global grid
    all_X_train = []
    all_y_train = []

    for task_data in dataset.tasks():
        task_id = task_data.task_id
        X_train = task_data.X_train
        y_train = task_data.y_train

        start, end, num_steps = compute_time_window(X_train, step_size, prev_end)

        kernel = build_frozen_kernel(
            input_dim=X_train.shape[1],
            device=device,
            fixed_hparams=fixed_hparams,
        )

        if strategy_name == "ode":
            hippo = HiPPO_LegS(inducing_size, device)
            inducing = HiPPOInducing(
                inducing_size, device, hippo, rff_sample_size,
                start, end, num_steps, prev_steps,
            )
        else:
            inducing = AnalyticHiPPOInducing(
                inducing_size, device, rff_sample_size,
                start, end, num_steps, prev_steps,
                phase_origin=phase_origin,
            )

        model = OnlineSGPR(
            kernel, inducing, device, X_train, y_train,
            prior_state=prior_state,
            spectral_sample=spectral_sample,
        ).to(device)
        apply_frozen_noise(model, fixed_hparams)

        model = train_sgpr(model, num_epochs=num_epochs)

        # Accumulate training data
        all_X_train.append(X_train.cpu())
        all_y_train.append(y_train.cpu())

        # Predict on dense grid covering all seen data so far
        X_all = torch.cat(all_X_train)
        x_min = X_all.min().item() - 5
        x_max = X_all.max().item() + 5
        X_grid = torch.linspace(x_min, x_max, 300, dtype=torch.float64).unsqueeze(1).to(device)

        mean, var = predict_on_grid(model, X_grid)

        elbo_val = model.elbo().item()

        results.append({
            "task_id": task_id,
            "X_grid": X_grid.cpu().squeeze().numpy(),
            "mean": mean,
            "var": var,
            "X_train_current": X_train.cpu().squeeze().numpy(),
            "y_train_current": y_train.cpu().squeeze().numpy(),
            "X_train_all": torch.cat(all_X_train).squeeze().numpy(),
            "y_train_all": torch.cat(all_y_train).squeeze().numpy(),
            "X_test": task_data.X_test.cpu().squeeze().numpy(),
            "y_test": task_data.y_test.cpu().squeeze().numpy(),
            "elbo": elbo_val,
        })

        prior_state = model.get_posterior_state()
        prev_end = end
        prev_steps += num_steps

        print(f"  [{strategy_name}] Task {task_id}: ELBO={elbo_val:.2f}")

    return results


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device("cpu")

    dataset_for_hparams = SolarOnlineDataset(
        "data/solar_data_train.npy", "data/solar_data_test.npy",
        num_chunks=10, device=device,
    )
    dataset_ode = SolarOnlineDataset(
        "data/solar_data_train.npy", "data/solar_data_test.npy",
        num_chunks=10, device=device,
    )
    dataset_ana = SolarOnlineDataset(
        "data/solar_data_train.npy", "data/solar_data_test.npy",
        num_chunks=10, device=device,
    )

    cfg = {
        "inducing_size": 100,
        "rff_sample_size": 5000,
        "step_size": 2.45,
        "num_epochs": 500,
        "exact_gp": {
            "num_epochs": 500,
            "lr": 0.05,
            "verbose": True,
            "log_every": 100,
        },
    }
    phase_origin = resolve_global_phase_origin(dataset_for_hparams)
    fixed_hparams = fit_task1_hyperparameters(dataset_for_hparams, device, cfg)
    fixed_sample = sample_fixed_spectral_sample(
        input_dim=1,
        device=device,
        fixed_hparams=fixed_hparams,
        rff_sample_size=cfg["rff_sample_size"],
    )

    print("Running ODE...")
    results_ode = run_experiment(
        "ode",
        dataset_ode,
        device,
        cfg,
        fixed_hparams,
        phase_origin=phase_origin,
        spectral_sample=fixed_sample,
    )
    print("Running Analytic...")
    results_ana = run_experiment(
        "analytic",
        dataset_ana,
        device,
        cfg,
        fixed_hparams,
        phase_origin=phase_origin,
        spectral_sample=fixed_sample,
    )

    num_tasks = len(results_ode)

    # Load full data for background
    raw_train = np.load("data/solar_data_train.npy")

    # Plot: num_tasks rows x 2 columns
    fig, axes = plt.subplots(
        num_tasks, 2, figsize=(16, 3.2 * num_tasks),
        sharex=True, sharey=True,
    )

    for i, (res_ode, res_ana) in enumerate(zip(results_ode, results_ana)):
        for j, (res, label) in enumerate([(res_ode, "SGPR-ODE"), (res_ana, "SGPR-Analytic")]):
            ax = axes[i, j]

            x_g = res["X_grid"]
            mu = res["mean"]
            std = np.sqrt(np.maximum(res["var"], 1e-10))

            # Confidence band
            ax.fill_between(x_g, mu - 2 * std, mu + 2 * std,
                            alpha=0.2, color="C0", label=r"$\pm 2\sigma$")
            # Mean
            ax.plot(x_g, mu, color="C0", linewidth=1.5, label="Mean")

            # All data (background)
            ax.scatter(raw_train[:, 0], raw_train[:, 1],
                       s=4, c="gray", alpha=0.2, zorder=1)

            # Previous training data
            ax.scatter(res["X_train_all"], res["y_train_all"],
                       s=12, c="C1", alpha=0.5, zorder=2, label="Train (seen)")

            # Current task training data
            ax.scatter(res["X_train_current"], res["y_train_current"],
                       s=25, c="C3", edgecolors="black", linewidths=0.5,
                       zorder=3, label="Train (current)")

            # Test data
            ax.scatter(res["X_test"], res["y_test"],
                       s=20, marker="x", c="C2", zorder=4, label="Test")

            ax.set_ylabel(f"Task {res['task_id']}", fontsize=10)
            if i == 0:
                ax.set_title(f"{label}", fontsize=12, fontweight="bold")
            ax.text(0.98, 0.95, f"ELBO={res['elbo']:.1f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, bbox=dict(boxstyle="round,pad=0.3",
                                          facecolor="wheat", alpha=0.7))

            if i == 0 and j == 1:
                ax.legend(fontsize=7, loc="upper left", ncol=2)

            ax.set_ylim(-3, 4)

    axes[-1, 0].set_xlabel("X")
    axes[-1, 1].set_xlabel("X")

    fig.suptitle("Solar SGPR: ODE vs Analytic (task-by-task online learning)",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig("solar_sgpr_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved to solar_sgpr_comparison.png")


if __name__ == "__main__":
    main()

"""Unified training script for online sparse GP models.

Supports SVGP (variational) and SGPR (collapsed bound) model types,
in single-strategy mode or compare mode (multiple strategies from one YAML).

Usage:
    python -m scripts.train scripts/configs/covid_analytic.yaml
    python -m scripts.train scripts/configs/compare_solar_sgpr.yaml
"""

import argparse
import csv
import json
import os
import time
from copy import deepcopy

import torch
import yaml

from hipposvgp.kernels import SquaredExponential, SpectralMixture, FrequencySpace
from hipposvgp.likelihoods import (
    GaussianLikelihood,
    BernoulliLikelihood,
    NegativeBinomialLikelihood,
)
from hipposvgp.inducing import HiPPOInducing, AnalyticHiPPOInducing, StandardInducing
from hipposvgp.hippo import HiPPO_LegS
from hipposvgp.models import OnlineSVGP, ExactGP
from hipposvgp.models.sgpr import OnlineSGPR
from hipposvgp.rff import SpectralSample

from scripts.datasets.covid import COVIDOnlineDataset
from scripts.datasets.solar import SolarOnlineDataset
from scripts.metrics import rmse, r_squared, nlpd, gaussian_ece
from scripts.utils import seed_everything, compute_time_window, train_loop


# ---------------------------------------------------------------------------
# Config → object builders
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_kernel(cfg, device):
    ktype = cfg["type"]
    if ktype == "se":
        return SquaredExponential(cfg.get("input_dim", 1), device=device)
    elif ktype == "spectral_mixture":
        return SpectralMixture(cfg["num_mixture"], device=device)
    elif ktype == "freq_space":
        return FrequencySpace(cfg["num_components"], device=device)
    else:
        raise ValueError(f"Unknown kernel type: {ktype}")


def build_likelihood(cfg, device):
    ltype = cfg["type"]
    if ltype == "gaussian":
        return GaussianLikelihood(cfg.get("variance", 1.0)).to(device)
    elif ltype == "bernoulli":
        return BernoulliLikelihood().to(device)
    elif ltype == "negative_binomial":
        return NegativeBinomialLikelihood(cfg.get("initial_nu", 0.5)).to(device)
    else:
        raise ValueError(f"Unknown likelihood type: {ltype}")


def build_dataset(cfg, device):
    dname = cfg["name"]
    if dname == "covid":
        return COVIDOnlineDataset(
            cfg["path"], num_tasks=cfg.get("num_tasks", 5), device=device,
        )
    elif dname == "solar":
        return SolarOnlineDataset(
            cfg["train_path"], cfg["test_path"],
            num_chunks=cfg.get("num_chunks", 10), device=device,
        )
    else:
        raise ValueError(f"Unknown dataset: {dname}")


def build_inducing_strategy(
    cfg, device, hippo, start, end, num_steps, prev_steps, *, phase_origin=0.0,
):
    strategy = cfg["inducing_strategy"]
    M = cfg["inducing_size"]
    R = cfg["rff_sample_size"]

    if strategy == "hippo_ode":
        return HiPPOInducing(
            M, device, hippo, R, start, end, num_steps, prev_steps,
        )
    elif strategy == "analytic_hippo":
        return AnalyticHiPPOInducing(
            M, device, R, start, end, num_steps, prev_steps,
            phase_origin=phase_origin,
        )
    elif strategy == "standard":
        return StandardInducing(M, cfg.get("input_dim", 1), device)
    else:
        raise ValueError(f"Unknown inducing strategy: {strategy}")


def _draw_spectral_sample(kernel, R):
    """Draw and detach a spectral sample from a kernel."""
    s = kernel.sample_from_spectral(R)
    return SpectralSample(
        frequencies=s.frequencies.detach().clone(),
        scale=s.scale.detach().clone(),
    )


# ---------------------------------------------------------------------------
# Build model + train
# ---------------------------------------------------------------------------

def _build_and_train(
    model_type, kernel, task_data, inducing, device,
    prior_state, spectral_sample, model_cfg, train_cfg, task_id,
    *, fitted_log_noise=None,
):
    """Build model, train, return (model, history)."""
    num_epochs = train_cfg.get("num_epochs", 500)
    lr = train_cfg.get("lr", 0.01)

    if model_type == "sgpr":
        model = OnlineSGPR(
            kernel=kernel, inducing_strategy=inducing, device=device,
            X=task_data.X_train, y=task_data.y_train,
            prior_state=prior_state, spectral_sample=spectral_sample,
        ).to(device)
        # Transfer and freeze noise from ExactGP
        if fitted_log_noise is not None:
            model.log_noise.data.copy_(fitted_log_noise)
            model.log_noise.requires_grad_(False)
        history = train_loop(model, num_epochs, lr, compute_elbo=model.elbo)
    else:
        likelihood = build_likelihood(
            model_cfg.get("likelihood", {"type": "gaussian"}), device,
        )
        flag_update_kernel = (
            task_id == 1
            or not train_cfg.get("fix_kernel_after_task1", False)
        )
        model = OnlineSVGP(
            kernel=kernel, likelihood=likelihood, inducing_strategy=inducing,
            device=device, prior_state=prior_state,
            use_robust_cholesky=model_cfg.get("use_robust_cholesky", False),
            flag_update_kernel=flag_update_kernel,
        ).to(device)

        if task_id > 1 and train_cfg.get("fix_kernel_after_task1", False):
            if hasattr(model.likelihood, "_nu"):
                model.likelihood._nu.requires_grad = False
            if prior_state is not None:
                model.mv = torch.nn.Parameter(prior_state.mv.clone().to(device))
                model.Lv = torch.nn.Parameter(prior_state.Lv.clone().to(device))

        history = train_loop(
            model, num_epochs, lr,
            compute_elbo=lambda: model.elbo(
                task_data.X_train, task_data.y_train, sample=spectral_sample,
            ),
        )

    return model, history


def evaluate_task(model, task_data, device, *, model_type):
    """Evaluate model on task test data."""
    model.eval()
    with torch.no_grad():
        if model_type == "sgpr":
            mean, var = model.predict(
                task_data.X_test, full_cov=False, include_noise=True,
            )
            if var.dim() == 1:
                var = var.unsqueeze(1)
            pred_mean = mean
        else:
            mean, var = model.predict(task_data.X_test)
            pred_mean = (
                model.likelihood.link_fn(mean)
                if hasattr(model.likelihood, "link_fn") else mean
            )

        rmse_val = rmse(task_data.y_test, pred_mean)
        r2_val = r_squared(task_data.y_test, pred_mean)
        ece_val = gaussian_ece(mean, var, task_data.y_test)

        if model_type == "sgpr":
            nlpd_val = nlpd(mean, var, task_data.y_test)
        else:
            nlpd_val = nlpd(
                mean, var, task_data.y_test,
                likelihood=model.likelihood, device=device,
            )

    return {"rmse": rmse_val, "r2": r2_val, "nlpd": nlpd_val, "ece": ece_val}


# ---------------------------------------------------------------------------
# Strategy runner
# ---------------------------------------------------------------------------

def run_strategy(
    dataset_cfg, device, model_cfg, train_cfg, kernel_cfg,
    *, fitted_kernel=None, fitted_log_noise=None,
    spectral_sample=None, phase_origin=0.0, seed=0,
):
    """Run one inducing strategy across all tasks.

    Args:
        fitted_kernel: Pre-fitted kernel (from ExactGP). Deepcopied per task.
        spectral_sample: Fixed spectral sample shared across tasks (or None).
        phase_origin: Phase origin for analytic inducing.
        seed: Random seed.

    Returns:
        (task_metrics, training_logs) — lists of dicts.
    """
    seed_everything(seed)

    model_type = model_cfg.get("model_type", "svgp")
    fix_kernel = train_cfg.get("fix_kernel_after_task1", False)
    step_size = model_cfg["step_size"]

    hippo = None
    if model_cfg["inducing_strategy"] == "hippo_ode":
        hippo = HiPPO_LegS(model_cfg["inducing_size"], device)

    dataset = build_dataset(dataset_cfg, device)
    base_kernel = fitted_kernel  # None until task 1 completes (when no ExactGP)

    prior_state = None
    prev_end = None
    prev_steps = 0
    all_test_data = []
    all_X_train = []
    all_y_train = []
    all_X_test = []
    all_y_test = []
    task_metrics = []
    training_logs = []

    for task_data in dataset.tasks():
        task_id = task_data.task_id
        print(f"\n  --- Task {task_id}  "
              f"(train={task_data.X_train.shape[0]}, "
              f"test={task_data.X_test.shape[0]}) ---")

        # Kernel lifecycle:
        #   - fitted_kernel (ExactGP): frozen for all tasks
        #   - no fitted_kernel: learn on task 1, freeze for task 2+
        if base_kernel is not None:
            kernel = deepcopy(base_kernel)
            if fitted_kernel is not None or (task_id > 1 and fix_kernel):
                for p in kernel.parameters():
                    p.requires_grad_(False)
        else:
            kernel = build_kernel(kernel_cfg, device)

        # Time window + inducing
        start, end, num_steps = compute_time_window(
            task_data.X_train, step_size, prev_end,
        )
        inducing = build_inducing_strategy(
            model_cfg, device, hippo, start, end, num_steps, prev_steps,
            phase_origin=phase_origin,
        )

        # Train
        t0 = time.time()
        model, history = _build_and_train(
            model_type, kernel, task_data, inducing, device,
            prior_state, spectral_sample, model_cfg, train_cfg, task_id,
            fitted_log_noise=fitted_log_noise,
        )
        wall_time = time.time() - t0

        # Save learned kernel after task 1 (when no ExactGP pre-fitting)
        if task_id == 1 and fitted_kernel is None:
            base_kernel = deepcopy(model.kernel)

        # Evaluate on all seen test sets
        all_test_data.append(task_data)
        for td in all_test_data:
            metrics = evaluate_task(model, td, device, model_type=model_type)
            print(
                f"    [eval task{td.task_id}] "
                f"RMSE={metrics['rmse']:.4f}  R2={metrics['r2']:.4f}  "
                f"NLPD={metrics['nlpd']:.4f}  ECE={metrics['ece']:.4f}"
            )

        # Accumulate training and test data
        all_X_train.append(task_data.X_train.cpu())
        all_y_train.append(task_data.y_train.cpu())
        all_X_test.append(task_data.X_test.cpu())
        all_y_test.append(task_data.y_test.cpu())

        # Predict on dense grid covering all seen data
        X_all = torch.cat(all_X_train)
        x_lo = X_all.min().item() - 5
        x_hi = X_all.max().item() + 5
        X_grid = torch.linspace(x_lo, x_hi, 300, dtype=torch.float64).unsqueeze(1).to(device)

        model.eval()
        with torch.no_grad():
            if model_type == "sgpr":
                grid_mean, grid_var = model.predict(
                    X_grid, full_cov=False, include_noise=True,
                )
                if grid_var.dim() == 1:
                    grid_var = grid_var.unsqueeze(1)
            else:
                grid_mean, grid_var = model.predict(X_grid)

        final_elbo = history[-1]["elbo"] if history else float("nan")
        task_metrics.append({
            **metrics, "task_id": task_id,
            "wall_time": wall_time, "final_elbo": final_elbo,
            "_pred": {
                "X_grid": X_grid.cpu(),
                "mean_grid": grid_mean.cpu(),
                "var_grid": grid_var.cpu(),
                "X_train_all": torch.cat(all_X_train),
                "y_train_all": torch.cat(all_y_train),
                "X_train_current": task_data.X_train.cpu(),
                "y_train_current": task_data.y_train.cpu(),
                "X_test_all": torch.cat(all_X_test),
                "y_test_all": torch.cat(all_y_test),
                "X_test_current": task_data.X_test.cpu(),
                "y_test_current": task_data.y_test.cpu(),
            },
        })
        training_logs.append({"task_id": task_id, "history": history})

        prior_state = model.get_posterior_state()
        prev_end = end
        prev_steps += num_steps

    return task_metrics, training_logs


# ---------------------------------------------------------------------------
# Single-strategy mode
# ---------------------------------------------------------------------------

def run_single(config):
    seed = config["experiment"].get("seed", 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = config["model"]
    train_cfg = config["training"]

    print(f"Device: {device}")
    print(f"Experiment: {config['experiment']['name']}")

    full_model_cfg = {**model_cfg}
    if "likelihood" in config:
        full_model_cfg["likelihood"] = config["likelihood"]

    # Phase origin
    dataset = build_dataset(config["dataset"], device)
    task1 = next(dataset.tasks())
    phase_origin = (
        torch.min(task1.X_train).item()
        if model_cfg.get("inducing_strategy") == "analytic_hippo"
        else model_cfg.get("phase_origin", 0.0)
    )

    run_strategy(
        config["dataset"], device, full_model_cfg, train_cfg, config["kernel"],
        phase_origin=phase_origin, seed=seed,
    )
    print("\nExperiment complete.")


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------

def _fit_exact_gp_hparams(dataset, device, base_cfg):
    """Fit ExactGP kernel hyperparameters on task 1.

    Uses data-driven heuristic initialization (median pairwise distance
    for lengthscale, data std for signal variance) to avoid both the
    overly-smooth and overfitting local optima common with small N.
    """
    exact_cfg = base_cfg.get("exact_gp", {})
    if not exact_cfg:
        return None

    task1 = next(dataset.tasks())
    num_epochs = exact_cfg.get("num_epochs", 500)
    lr = exact_cfg.get("lr", 0.05)

    kernel = build_kernel(base_cfg["kernel"], device)
    model = ExactGP(kernel=kernel, device=device)

    # Initialization: use config values if provided, else data-driven heuristic
    with torch.no_grad():
        if "init_log_ls" in exact_cfg:
            kernel.log_ls.fill_(exact_cfg["init_log_ls"])
        else:
            dists = torch.cdist(task1.X_train, task1.X_train)
            median_dist = dists[dists > 0].median()
            kernel.log_ls.fill_(torch.log(median_dist).item())

        if "init_log_sf" in exact_cfg:
            kernel.log_sf.fill_(exact_cfg["init_log_sf"])
        else:
            kernel.log_sf.fill_(torch.log(task1.y_train.std()).item())

        if "init_log_noise" in exact_cfg:
            model.likelihood.log_variance.fill_(exact_cfg["init_log_noise"])
        else:
            model.likelihood.log_variance.fill_(
                torch.log(task1.y_train.var() * 0.1).item()
            )

    print(f"Fitting ExactGP on task 1 ({num_epochs} epochs)...")
    print(f"  init: log_ls={kernel.log_ls.data.cpu().numpy()}, "
          f"log_sf={kernel.log_sf.item():.4f}, "
          f"log_noise={model.likelihood.log_variance.item():.4f}")

    model.fit(
        task1.X_train, task1.y_train,
        num_epochs=num_epochs, lr=lr,
        verbose=True, log_every=exact_cfg.get("log_every", 100),
    )

    fitted_log_noise = model.likelihood.log_variance.detach().clone()
    # Clip noise from below (following HarrisonZhu's HIPPOSVGP repo)
    min_log_noise = exact_cfg.get("min_log_noise", None)
    if min_log_noise is not None and fitted_log_noise.item() < min_log_noise:
        print(f"  clipping log_noise: {fitted_log_noise.item():.4f} -> {min_log_noise}")
        fitted_log_noise.fill_(min_log_noise)
    final_nll = model.nll().item()
    print(f"  final: NLL={final_nll:.4f}, "
          f"log_ls={kernel.log_ls.data.cpu().numpy()}, "
          f"log_sf={kernel.log_sf.item():.4f}, "
          f"log_noise={model.likelihood.log_variance.item():.4f}")
    return model.kernel, fitted_log_noise


def _save_strategy_outputs(strat_dir, task_metrics, training_logs):
    """Save metrics.csv, training_log.json, and prediction .pt files."""
    pred_dir = os.path.join(strat_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    fieldnames = ["task_id", "rmse", "r2", "nlpd", "ece", "wall_time", "final_elbo"]
    with open(os.path.join(strat_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in task_metrics:
            writer.writerow({k: row[k] for k in fieldnames})

    with open(os.path.join(strat_dir, "training_log.json"), "w") as f:
        json.dump(training_logs, f, indent=2)

    for m in task_metrics:
        pred = m.pop("_pred", None)
        if pred is not None:
            torch.save(pred, os.path.join(pred_dir, f"task{m['task_id']:02d}.pt"))


def _write_comparison(output_dir, all_results):
    fieldnames = [
        "strategy", "task_id", "rmse", "r2", "nlpd", "ece", "wall_time", "final_elbo",
    ]
    rows = [
        {"strategy": name, **{k: m[k] for k in fieldnames if k != "strategy"}}
        for name, metrics_list in all_results.items()
        for m in metrics_list
    ]
    with open(os.path.join(output_dir, "comparison.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_compare(config):
    seed = config["experiment"].get("seed", 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = config["experiment"].get("output_dir", "results/compare")
    os.makedirs(output_dir, exist_ok=True)

    base_cfg = config["base"]
    train_cfg = base_cfg.get("training", {})
    R = base_cfg["rff_sample_size"]

    print(f"Device: {device}")
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Output: {output_dir}")

    dataset = build_dataset(config["dataset"], device)
    fit_result = _fit_exact_gp_hparams(dataset, device, base_cfg)
    if fit_result is not None:
        fitted_kernel, fitted_log_noise = fit_result
    else:
        fitted_kernel, fitted_log_noise = None, None

    # Phase origin
    task1 = next(dataset.tasks())
    phase_origin = torch.min(task1.X_train).item()

    # Shared spectral sample: draw once before strategy loop
    spectral_sample = None
    if config["experiment"].get("share_spectral_sample", False):
        seed_everything(seed)
        source = fitted_kernel or build_kernel(base_cfg["kernel"], device)
        spectral_sample = _draw_spectral_sample(source, R)
        torch.save(spectral_sample, os.path.join(output_dir, "spectral_sample.pt"))

    all_results = {}

    for strat_name, strat_cfg in config["strategies"].items():
        print(f"\n{'#' * 60}")
        print(f"Strategy: {strat_name}")
        print(f"{'#' * 60}")

        merged_cfg = {**base_cfg, **strat_cfg}
        if "likelihood" in base_cfg:
            merged_cfg["likelihood"] = base_cfg["likelihood"]

        task_metrics, training_logs = run_strategy(
            config["dataset"], device, merged_cfg, train_cfg, base_cfg["kernel"],
            fitted_kernel=fitted_kernel, fitted_log_noise=fitted_log_noise,
            spectral_sample=spectral_sample,
            phase_origin=phase_origin, seed=seed,
        )

        _save_strategy_outputs(
            os.path.join(output_dir, strat_name), task_metrics, training_logs,
        )
        all_results[strat_name] = task_metrics

    _write_comparison(output_dir, all_results)
    print(f"\nResults saved to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path):
    config = load_config(config_path)
    mode = config.get("experiment", {}).get("mode", "single")
    if mode == "compare":
        run_compare(config)
    else:
        run_single(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Online Sparse GP model")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)

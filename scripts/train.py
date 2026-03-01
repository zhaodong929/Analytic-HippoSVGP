"""Unified training script for online sparse GP models.

Supports both SVGP (variational) and SGPR (collapsed bound) model types.
Supports single-strategy mode (default) and compare mode (multiple strategies).

Usage:
    python -m scripts.train scripts/configs/covid_analytic.yaml
    python -m scripts.train scripts/configs/compare_solar_sgpr.yaml
"""

import argparse
import csv
import json
import os
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

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
    cfg, device, hippo, start, end, num_steps, prev_steps, *, phase_origin=0.0
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


# ---------------------------------------------------------------------------
# Model build + train (single entry point for both SGPR and SVGP)
# ---------------------------------------------------------------------------

def _build_and_train(
    model_type, kernel, task_data, inducing, device,
    prior_state, spectral_sample, model_cfg, train_cfg, task_id,
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
# Kernel lifecycle: init → learn on task 1 → freeze for task 2+
# ---------------------------------------------------------------------------

class KernelManager:
    """Manage kernel construction, learning, and freezing across tasks."""

    def __init__(self, kernel_cfg, device, train_cfg, fitted_kernel=None):
        self._kernel_cfg = kernel_cfg
        self._device = device
        self._fix_after_task1 = train_cfg.get("fix_kernel_after_task1", False)
        self._fitted_kernel = fitted_kernel
        self._learned_kernel = None

    def get_kernel(self, task_id):
        """Return an appropriately initialized kernel for this task."""
        if self._fitted_kernel is not None:
            kernel = deepcopy(self._fitted_kernel)
        elif task_id == 1:
            kernel = build_kernel(self._kernel_cfg, self._device)
        else:
            kernel = deepcopy(self._learned_kernel)

        if task_id > 1 and self._fix_after_task1:
            for p in kernel.parameters():
                p.requires_grad_(False)

        return kernel

    def after_task(self, task_id, model):
        """Save the learned kernel from task 1 when no ExactGP pre-fitting."""
        if task_id == 1 and self._fitted_kernel is None:
            self._learned_kernel = deepcopy(model.kernel)


# ---------------------------------------------------------------------------
# Shared spectral sample lifecycle
# ---------------------------------------------------------------------------

class SpectralSampleManager:
    """Manage spectral sample: immediate (ExactGP fitted) or deferred (task 1)."""

    def __init__(self, rff_sample_size, fitted_kernel=None, enabled=True):
        self._rff_sample_size = rff_sample_size
        self._enabled = enabled
        self.sample = None

        if enabled and fitted_kernel is not None:
            self.sample = self._draw(fitted_kernel)

    def maybe_create_after_task1(self, kernel_manager):
        """Create deferred sample from task 1 learned kernel."""
        if self._enabled and self.sample is None and kernel_manager._learned_kernel is not None:
            self.sample = self._draw(kernel_manager._learned_kernel)

    def _draw(self, kernel):
        s = kernel.sample_from_spectral(self._rff_sample_size)
        return SpectralSample(
            frequencies=s.frequencies.detach().clone(),
            scale=s.scale.detach().clone(),
        )

    def save(self, path):
        if self.sample is not None:
            torch.save(self.sample, path)


# ---------------------------------------------------------------------------
# Strategy runner (shared by single and compare modes)
# ---------------------------------------------------------------------------

@dataclass
class StrategyResult:
    task_metrics: list = field(default_factory=list)
    training_logs: list = field(default_factory=list)


def run_strategy(
    dataset_cfg, device, model_cfg, train_cfg, kernel_manager,
    spectral_manager, phase_origin, seed,
):
    """Run one inducing strategy across all tasks. Returns StrategyResult."""
    seed_everything(seed)

    model_type = model_cfg.get("model_type", "svgp")
    inducing_size = model_cfg["inducing_size"]
    step_size = model_cfg["step_size"]

    hippo = None
    if model_cfg["inducing_strategy"] == "hippo_ode":
        hippo = HiPPO_LegS(inducing_size, device)

    dataset = build_dataset(dataset_cfg, device)
    result = StrategyResult()

    prior_state = None
    prev_end = None
    prev_steps = 0
    all_test_data = []

    for task_data in dataset.tasks():
        task_id = task_data.task_id
        print(f"\n  --- Task {task_id}  "
              f"(train={task_data.X_train.shape[0]}, test={task_data.X_test.shape[0]}) ---")

        start, end, num_steps = compute_time_window(
            task_data.X_train, step_size, prev_end,
        )
        kernel = kernel_manager.get_kernel(task_id)
        inducing = build_inducing_strategy(
            model_cfg, device, hippo, start, end, num_steps, prev_steps,
            phase_origin=phase_origin,
        )

        t0 = time.time()
        model, history = _build_and_train(
            model_type, kernel, task_data, inducing, device,
            prior_state, spectral_manager.sample, model_cfg, train_cfg, task_id,
        )
        wall_time = time.time() - t0

        kernel_manager.after_task(task_id, model)
        if task_id == 1:
            spectral_manager.maybe_create_after_task1(kernel_manager)

        # Evaluate
        all_test_data.append(task_data)
        for td in all_test_data:
            metrics = evaluate_task(model, td, device, model_type=model_type)
            print(
                f"    [eval task{td.task_id}] "
                f"RMSE={metrics['rmse']:.4f}  R2={metrics['r2']:.4f}  "
                f"NLPD={metrics['nlpd']:.4f}  ECE={metrics['ece']:.4f}"
            )

        final_elbo = history[-1]["elbo"] if history else float("nan")
        result.task_metrics.append({
            **metrics, "task_id": task_id,
            "wall_time": wall_time, "final_elbo": final_elbo,
        })
        result.training_logs.append({"task_id": task_id, "history": history})

        # Attach prediction tensors for saving
        model.eval()
        with torch.no_grad():
            if model_type == "sgpr":
                mean, var = model.predict(
                    task_data.X_test, full_cov=False, include_noise=True,
                )
            else:
                mean, var = model.predict(task_data.X_test)
        result.task_metrics[-1]["_pred"] = {
            "X_test": task_data.X_test.cpu(),
            "y_test": task_data.y_test.cpu(),
            "mean": mean.cpu(),
            "var": (var if var.dim() > 0 else var.unsqueeze(0)).cpu(),
        }

        prior_state = model.get_posterior_state()
        prev_end = end
        prev_steps += num_steps

    return result


# ---------------------------------------------------------------------------
# Single-strategy mode
# ---------------------------------------------------------------------------

def run_single(config):
    seed = config["experiment"].get("seed", 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = config["model"]
    train_cfg = config["training"]
    model_type = model_cfg.get("model_type", "svgp")

    if model_type == "svgp" and config.get("likelihood", {}).get("type") == "gaussian":
        warnings.warn(
            "Using SVGP with Gaussian likelihood. Consider model_type: 'sgpr'.",
            stacklevel=2,
        )

    print(f"Device: {device}")
    print(f"Experiment: {config['experiment']['name']}  ({model_type})")

    # Merge likelihood into model_cfg for _build_and_train
    full_model_cfg = {**model_cfg}
    if "likelihood" in config:
        full_model_cfg["likelihood"] = config["likelihood"]

    # Phase origin
    dataset_tmp = build_dataset(config["dataset"], device)
    task1 = next(dataset_tmp.tasks())
    phase_origin = (
        torch.min(task1.X_train).item()
        if model_cfg.get("inducing_strategy") == "analytic_hippo"
        else model_cfg.get("phase_origin", 0.0)
    )

    km = KernelManager(config["kernel"], device, train_cfg)
    sm = SpectralSampleManager(
        model_cfg.get("rff_sample_size", 1000), enabled=False,
    )

    run_strategy(
        config["dataset"], device, full_model_cfg, train_cfg,
        km, sm, phase_origin, seed,
    )
    print("\nExperiment complete.")


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------

def _fit_exact_gp_hparams(dataset, device, base_cfg):
    exact_cfg = base_cfg.get("exact_gp", {})
    if not exact_cfg:
        return None
    task1 = next(dataset.tasks())
    kernel = build_kernel(base_cfg["kernel"], device)
    model = ExactGP(kernel=kernel, device=device)
    print("Fitting ExactGP hyperparameters on task 1...")
    model.fit(
        task1.X_train, task1.y_train,
        num_epochs=exact_cfg.get("num_epochs", 500),
        lr=exact_cfg.get("lr", 0.05),
        verbose=True, log_every=exact_cfg.get("log_every", 100),
    )
    return model.kernel


def _save_strategy_outputs(strat_dir, result):
    """Save metrics.csv, training_log.json, and prediction .pt files."""
    pred_dir = os.path.join(strat_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    fieldnames = ["task_id", "rmse", "r2", "nlpd", "ece", "wall_time", "final_elbo"]
    with open(os.path.join(strat_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.task_metrics:
            writer.writerow({k: row[k] for k in fieldnames})

    with open(os.path.join(strat_dir, "training_log.json"), "w") as f:
        json.dump(result.training_logs, f, indent=2)

    for m in result.task_metrics:
        pred = m.pop("_pred", None)
        if pred is not None:
            torch.save(pred, os.path.join(pred_dir, f"task{m['task_id']:02d}.pt"))


def _write_comparison(output_dir, all_results):
    fieldnames = ["strategy", "task_id", "rmse", "r2", "nlpd", "ece", "wall_time", "final_elbo"]
    rows = [
        {"strategy": name, **{k: m[k] for k in fieldnames if k != "strategy"}}
        for name, res in all_results.items()
        for m in res.task_metrics
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
    rff_sample_size = base_cfg["rff_sample_size"]

    print(f"Device: {device}")
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Output: {output_dir}")

    dataset = build_dataset(config["dataset"], device)
    fitted_kernel = _fit_exact_gp_hparams(dataset, device, base_cfg)

    # Phase origin
    task1 = next(dataset.tasks())
    phase_origin = torch.min(task1.X_train).item()

    all_results = {}

    for strat_name, strat_cfg in config["strategies"].items():
        print(f"\n{'#'*60}")
        print(f"Strategy: {strat_name}")
        print(f"{'#'*60}")

        merged_cfg = {**base_cfg, **strat_cfg}
        if "likelihood" in base_cfg:
            merged_cfg["likelihood"] = base_cfg["likelihood"]

        km = KernelManager(base_cfg["kernel"], device, train_cfg, fitted_kernel)
        sm = SpectralSampleManager(
            rff_sample_size, fitted_kernel,
            enabled=config["experiment"].get("share_spectral_sample", False),
        )

        result = run_strategy(
            config["dataset"], device, merged_cfg, train_cfg,
            km, sm, phase_origin, seed,
        )

        sm.save(os.path.join(output_dir, "spectral_sample.pt"))
        _save_strategy_outputs(os.path.join(output_dir, strat_name), result)
        all_results[strat_name] = result

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

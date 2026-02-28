"""Unified training script for online sparse GP models.

Supports both SVGP (variational) and SGPR (collapsed bound) model types.

Usage:
    python -m scripts.train scripts/configs/covid_analytic.yaml
    python -m scripts.train scripts/configs/solar_sgpr_ode.yaml
"""

import argparse
import warnings

import torch
import yaml
from copy import deepcopy

from hipposvgp.kernels import SquaredExponential, SpectralMixture, FrequencySpace
from hipposvgp.likelihoods import (
    GaussianLikelihood,
    BernoulliLikelihood,
    NegativeBinomialLikelihood,
)
from hipposvgp.inducing import HiPPOInducing, AnalyticHiPPOInducing, StandardInducing
from hipposvgp.hippo import HiPPO_LegS
from hipposvgp.models import OnlineSVGP, PosteriorState
from hipposvgp.models.sgpr import OnlineSGPR, SGPRPosteriorState

from scripts.datasets.covid import COVIDOnlineDataset
from scripts.datasets.solar import SolarOnlineDataset
from scripts.metrics import rmse, r_squared, nlpd
from scripts.utils import seed_everything, compute_time_window, train_loop


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
            cfg["path"],
            num_tasks=cfg.get("num_tasks", 5),
            device=device,
        )
    elif dname == "solar":
        return SolarOnlineDataset(
            cfg["train_path"],
            cfg["test_path"],
            num_chunks=cfg.get("num_chunks", 10),
            device=device,
        )
    else:
        raise ValueError(f"Unknown dataset: {dname}")


def build_inducing_strategy(
    cfg, device, hippo, start, end, num_steps, prev_steps, *, phase_origin=0.0
):
    """Build inducing strategy from config."""
    strategy = cfg["inducing_strategy"]
    inducing_size = cfg["inducing_size"]

    if strategy == "hippo_ode":
        return HiPPOInducing(
            inducing_size=inducing_size,
            device=device,
            hippo=hippo,
            rff_sample_size=cfg["rff_sample_size"],
            start=start,
            end=end,
            num_discrete_steps=num_steps,
            prev_discrete_steps=prev_steps,
        )
    elif strategy == "analytic_hippo":
        return AnalyticHiPPOInducing(
            inducing_size=inducing_size,
            device=device,
            rff_sample_size=cfg["rff_sample_size"],
            start=start,
            end=end,
            num_discrete_steps=num_steps,
            prev_discrete_steps=prev_steps,
            phase_origin=phase_origin,
        )
    elif strategy == "standard":
        return StandardInducing(
            inducing_size=inducing_size,
            input_dim=cfg.get("input_dim", 1),
            device=device,
        )
    else:
        raise ValueError(f"Unknown inducing strategy: {strategy}")


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_task(model, task_data, device, *, model_type):
    """Evaluate model on task test data.

    Args:
        model: trained model (OnlineSVGP or OnlineSGPR)
        task_data: TaskData with test inputs/outputs
        device: torch device
        model_type: "svgp" or "sgpr"
    """
    model.eval()
    with torch.no_grad():
        if model_type == "sgpr":
            mean, var = model.predict(
                task_data.X_test, full_cov=False, include_noise=True
            )
            if var.dim() == 1:
                var = var.unsqueeze(1)
            rmse_val = rmse(task_data.y_test, mean)
            r2_val = r_squared(task_data.y_test, mean)
            nlpd_val = nlpd(mean, var, task_data.y_test)
        else:
            mean, var = model.predict(task_data.X_test)
            if hasattr(model.likelihood, "link_fn"):
                pred_mean = model.likelihood.link_fn(mean)
            else:
                pred_mean = mean
            rmse_val = rmse(task_data.y_test, pred_mean)
            r2_val = r_squared(task_data.y_test, pred_mean)
            nlpd_val = nlpd(
                mean, var, task_data.y_test,
                likelihood=model.likelihood,
                device=device,
            )

    return {"rmse": rmse_val, "r2": r2_val, "nlpd": nlpd_val}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path):
    config = load_config(config_path)
    seed = config["experiment"].get("seed", 0)
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Experiment: {config['experiment']['name']}")

    # Build dataset
    dataset = build_dataset(config["dataset"], device)

    # Build HiPPO (shared across tasks for ODE strategy)
    model_cfg = config["model"]
    inducing_size = model_cfg["inducing_size"]
    step_size = model_cfg["step_size"]
    model_type = model_cfg.get("model_type", "svgp")

    # Warn if SVGP with Gaussian likelihood — SGPR gives tighter bounds
    if (
        model_type == "svgp"
        and config.get("likelihood", {}).get("type") == "gaussian"
    ):
        warnings.warn(
            "Using SVGP with Gaussian likelihood. Consider setting "
            "model_type: 'sgpr' for a tighter bound — SGPR analytically "
            "marginalizes the variational parameters, yielding a better "
            "ELBO with fewer parameters to optimize.",
            stacklevel=2,
        )

    print(f"Model type: {model_type}")

    hippo = None
    if model_cfg["inducing_strategy"] == "hippo_ode":
        hippo = HiPPO_LegS(inducing_size, device)

    prior_state = None
    prev_end = None
    prev_steps = 0
    all_test_data = []
    phase_origin = model_cfg.get("phase_origin", 0.0)

    train_cfg = config["training"]
    num_epochs = train_cfg["num_epochs"]
    lr = train_cfg["lr"]

    for task_data in dataset.tasks():
        task_id = task_data.task_id
        print(f"\n{'='*60}")
        print(f"Task {task_id}")
        print(f"{'='*60}")
        print(f"  Train: {task_data.X_train.shape[0]} points")
        print(f"  Test:  {task_data.X_test.shape[0]} points")

        # Compute time window
        start, end, num_steps = compute_time_window(
            task_data.X_train, step_size, prev_end
        )

        # AnalyticHiPPO always uses global start phase origin
        if task_id == 1 and model_cfg["inducing_strategy"] == "analytic_hippo":
            phase_origin = torch.min(task_data.X_train).item()

        # Build components
        if task_id == 1:
            kernel = build_kernel(config["kernel"], device)
            if model_type == "svgp":
                likelihood = build_likelihood(config["likelihood"], device)
        else:
            kernel = deepcopy(kernel)
            if model_type == "svgp":
                likelihood = deepcopy(likelihood)

        inducing = build_inducing_strategy(
            model_cfg, device, hippo, start, end, num_steps, prev_steps,
            phase_origin=phase_origin,
        )

        flag_update_kernel = (
            task_id == 1
            or not train_cfg.get("fix_kernel_after_task1", False)
        )

        # Build and train model
        if model_type == "sgpr":
            model = OnlineSGPR(
                kernel=kernel,
                inducing_strategy=inducing,
                device=device,
                X=task_data.X_train,
                y=task_data.y_train,
                prior_state=prior_state,
            ).to(device)

            # Fix kernel params after task 1 if configured
            if task_id > 1 and train_cfg.get("fix_kernel_after_task1", False):
                for p in model.kernel.parameters():
                    p.requires_grad = False

            train_loop(model, num_epochs, lr, compute_elbo=model.elbo)

        else:
            # SVGP
            model = OnlineSVGP(
                kernel=kernel,
                likelihood=likelihood,
                inducing_strategy=inducing,
                device=device,
                prior_state=prior_state,
                use_robust_cholesky=model_cfg.get("use_robust_cholesky", False),
                flag_update_kernel=flag_update_kernel,
            ).to(device)

            # Fix kernel params after task 1 if configured
            if task_id > 1 and train_cfg.get("fix_kernel_after_task1", False):
                for p in model.kernel.parameters():
                    p.requires_grad = False
                if hasattr(model.likelihood, "_nu"):
                    model.likelihood._nu.requires_grad = False
                # Initialize from previous posterior
                if prior_state is not None:
                    model.mv = torch.nn.Parameter(
                        prior_state.mv.clone().to(device)
                    )
                    model.Lv = torch.nn.Parameter(
                        prior_state.Lv.clone().to(device)
                    )

            train_loop(
                model, num_epochs, lr,
                compute_elbo=lambda: model.elbo(
                    task_data.X_train, task_data.y_train
                ),
            )

        # Evaluate on all accumulated test sets
        all_test_data.append(task_data)
        for prev_td in all_test_data:
            metrics = evaluate_task(model, prev_td, device, model_type=model_type)
            print(
                f"  [eval task{prev_td.task_id} after task{task_id}] "
                f"RMSE={metrics['rmse']:.4f}  R2={metrics['r2']:.4f}  "
                f"NLPD={metrics['nlpd']:.4f}"
            )

        # Transfer posterior
        prior_state = model.get_posterior_state()

        prev_end = end
        prev_steps += num_steps

    print("\nExperiment complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Online Sparse GP model")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)

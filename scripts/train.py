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

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import optax
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
from hipposvgp.training import (
    make_svgp_train_step,
    make_sgpr_train_step,
    make_exact_gp_train_step,
)

from scripts.datasets.covid import COVIDOnlineDataset
from scripts.datasets.solar import SolarOnlineDataset
from scripts.metrics import rmse, r_squared, nlpd, gaussian_ece
from scripts.utils import seed_everything, compute_time_window


# ---------------------------------------------------------------------------
# Config -> object builders
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_kernel(cfg, *, key=None):
    ktype = cfg["type"]
    if ktype == "se":
        return SquaredExponential(cfg.get("input_dim", 1))
    elif ktype == "spectral_mixture":
        if key is None:
            key = jax.random.PRNGKey(0)
        return SpectralMixture(cfg["num_mixture"], key=key)
    elif ktype == "freq_space":
        return FrequencySpace(cfg["num_components"])
    else:
        raise ValueError(f"Unknown kernel type: {ktype}")


def build_likelihood(cfg):
    ltype = cfg["type"]
    if ltype == "gaussian":
        return GaussianLikelihood(cfg.get("variance", 1.0))
    elif ltype == "bernoulli":
        return BernoulliLikelihood()
    elif ltype == "negative_binomial":
        return NegativeBinomialLikelihood(cfg.get("initial_nu", 0.5))
    else:
        raise ValueError(f"Unknown likelihood type: {ltype}")


def build_dataset(cfg):
    dname = cfg["name"]
    if dname == "covid":
        return COVIDOnlineDataset(
            cfg["path"], num_tasks=cfg.get("num_tasks", 5),
        )
    elif dname == "solar":
        return SolarOnlineDataset(
            cfg["train_path"], cfg["test_path"],
            num_chunks=cfg.get("num_chunks", 10),
        )
    else:
        raise ValueError(f"Unknown dataset: {dname}")


def build_inducing_strategy(
    cfg, hippo, start, end, num_steps, prev_steps, *, phase_origin=0.0,
):
    strategy = cfg["inducing_strategy"]
    M = cfg["inducing_size"]
    R = cfg["rff_sample_size"]

    if strategy == "hippo_ode":
        return HiPPOInducing(
            M, hippo, R, start, end, num_steps, prev_steps,
        )
    elif strategy == "analytic_hippo":
        return AnalyticHiPPOInducing(
            M, R, start, end, num_steps, prev_steps,
            phase_origin=phase_origin,
            bessel_backend=cfg.get("bessel_backend", "spbessax"),
        )
    elif strategy == "standard":
        return StandardInducing(M, cfg.get("input_dim", 1))
    else:
        raise ValueError(f"Unknown inducing strategy: {strategy}")


def _draw_spectral_sample(kernel, R, key):
    """Draw a spectral sample from a kernel."""
    return kernel.sample_from_spectral(R, key)


# ---------------------------------------------------------------------------
# Build model + train
# ---------------------------------------------------------------------------

def _build_and_train(
    model_type, kernel, task_data, inducing,
    prior_state, spectral_sample, model_cfg, train_cfg, task_id,
    *, fitted_log_noise=None, freeze_kernel=False, key,
):
    """Build model, train, return (model, history, km_or_none).

    For SVGP, returns km from the last training step (needed for get_posterior_state).
    For SGPR, returns None.
    """
    num_epochs = train_cfg.get("num_epochs", 500)
    lr = train_cfg.get("lr", 0.01)
    log_every = train_cfg.get("log_every", 1000)

    if model_type == "sgpr":
        model = OnlineSGPR(
            kernel=kernel, inducing_strategy=inducing,
            X=task_data.X_train, y=task_data.y_train,
            prior_state=prior_state, spectral_sample=spectral_sample,
        )

        if fitted_log_noise is not None:
            model = eqx.tree_at(lambda m: m.log_noise, model, fitted_log_noise)

        # Build filter_spec: only kernel and log_noise are potentially trainable.
        # X, y, old state, and inducing internals are always frozen.
        filter_spec = jax.tree.map(lambda _: False, model)
        if not freeze_kernel:
            filter_spec = eqx.tree_at(
                lambda m: m.kernel, filter_spec,
                jax.tree.map(lambda _: True, model.kernel),
            )
        if fitted_log_noise is None:
            filter_spec = eqx.tree_at(lambda m: m.log_noise, filter_spec, True)

        any_trainable = True in jax.tree.leaves(filter_spec)

        if not any_trainable:
            # No trainable params — single closed-form evaluation
            key, subkey = jax.random.split(key)
            elbo, km, resolved_sample = model.elbo(
                sample=spectral_sample, key=subkey,
            )
            elbo_val = float(elbo)
            print(f"  (no trainable params)  ELBO={elbo_val:.2f}")
            return model, [{"epoch": 0, "elbo": elbo_val}], None

        optimizer = optax.adam(lr)
        step_fn = make_sgpr_train_step(optimizer, filter_spec)
        trainable, _ = eqx.partition(model, filter_spec)
        opt_state = optimizer.init(trainable)

        history = []
        for epoch in range(num_epochs):
            key, subkey = jax.random.split(key)
            model, opt_state, loss, (km, resolved_sample) = step_fn(
                model, opt_state, spectral_sample, subkey,
            )
            elbo_val = float(-loss)
            history.append({"epoch": epoch, "elbo": elbo_val})
            if epoch % log_every == 0 or epoch == num_epochs - 1:
                print(f"  epoch {epoch:5d}  ELBO={elbo_val:.2f}")

        return model, history, None

    else:  # svgp
        likelihood = build_likelihood(
            model_cfg.get("likelihood", {"type": "gaussian"}),
        )
        flag_update_kernel = (
            task_id == 1
            or not train_cfg.get("fix_kernel_after_task1", False)
        )
        model = OnlineSVGP(
            kernel=kernel, likelihood=likelihood, inducing_strategy=inducing,
            prior_state=prior_state,
            use_robust_cholesky=model_cfg.get("use_robust_cholesky", False),
            flag_update_kernel=flag_update_kernel,
        )

        # Build filter_spec: old state and inducing internals are always frozen.
        filter_spec = jax.tree.map(lambda _: False, model)
        if freeze_kernel:
            # Only train mv and Lv
            filter_spec = eqx.tree_at(
                lambda m: (m.mv, m.Lv), filter_spec, (True, True),
            )
        else:
            # Train kernel, likelihood, mv, Lv
            filter_spec = eqx.tree_at(
                lambda m: (m.kernel, m.likelihood, m.mv, m.Lv), filter_spec,
                (
                    jax.tree.map(lambda _: True, model.kernel),
                    jax.tree.map(lambda _: True, model.likelihood),
                    True,
                    True,
                ),
            )

        optimizer = optax.adam(lr)
        step_fn = make_svgp_train_step(optimizer, filter_spec)
        trainable, _ = eqx.partition(model, filter_spec)
        opt_state = optimizer.init(trainable)

        history = []
        km = None
        for epoch in range(num_epochs):
            key, subkey = jax.random.split(key)
            model, opt_state, loss, (ve, kl, km) = step_fn(
                model, opt_state, task_data.X_train, task_data.y_train,
                spectral_sample, subkey,
            )
            elbo_val = float(-loss)
            entry = {"epoch": epoch, "elbo": elbo_val}
            entry["ve"] = float(ve)
            entry["kl"] = float(kl)
            history.append(entry)
            if epoch % log_every == 0 or epoch == num_epochs - 1:
                print(
                    f"  epoch {epoch:5d}  ELBO={elbo_val:.2f}  "
                    f"VE={float(ve):.2f}  KL={float(kl):.4f}"
                )

        return model, history, km


def evaluate_task(model, task_data, spectral_sample, key, *, model_type):
    """Evaluate model on task test data."""
    key, pred_key, nlpd_key = jax.random.split(key, 3)

    if model_type == "sgpr":
        mean, var = model.predict(
            task_data.X_test, sample=spectral_sample, key=pred_key,
            full_cov=False, include_noise=True,
        )
        if var.ndim == 1:
            var = var[:, None]
        pred_mean = mean
    else:
        mean, var = model.predict(
            task_data.X_test, sample=spectral_sample, key=pred_key,
        )
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
            likelihood=model.likelihood, key=nlpd_key,
        )

    return {"rmse": rmse_val, "r2": r2_val, "nlpd": nlpd_val, "ece": ece_val}


# ---------------------------------------------------------------------------
# Strategy runner
# ---------------------------------------------------------------------------

def run_strategy(
    dataset_cfg, model_cfg, train_cfg, kernel_cfg,
    *, fitted_kernel=None, fitted_log_noise=None,
    spectral_sample=None, phase_origin=0.0, seed=0,
):
    """Run one inducing strategy across all tasks.

    Args:
        fitted_kernel: Pre-fitted kernel (from ExactGP). Used for all tasks.
        spectral_sample: Fixed spectral sample shared across tasks (or None).
        phase_origin: Phase origin for analytic inducing.
        seed: Random seed.

    Returns:
        (task_metrics, training_logs) -- lists of dicts.
    """
    seed_everything(seed)
    key = jax.random.PRNGKey(seed)

    model_type = model_cfg.get("model_type", "svgp")
    fix_kernel = train_cfg.get("fix_kernel_after_task1", False)
    step_size = model_cfg["step_size"]

    hippo = None
    if model_cfg["inducing_strategy"] == "hippo_ode":
        hippo = HiPPO_LegS(model_cfg["inducing_size"])

    dataset = build_dataset(dataset_cfg)
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
        print(
            f"\n  --- Task {task_id}  "
            f"(train={task_data.X_train.shape[0]}, "
            f"test={task_data.X_test.shape[0]}) ---"
        )

        # Kernel lifecycle:
        #   - fitted_kernel (ExactGP): frozen for all tasks
        #   - no fitted_kernel: learn on task 1, freeze for task 2+
        freeze_kernel = (fitted_kernel is not None) or (task_id > 1 and fix_kernel)
        if base_kernel is not None:
            kernel = base_kernel
        else:
            key, kernel_key = jax.random.split(key)
            kernel = build_kernel(kernel_cfg, key=kernel_key)

        # Time window + inducing
        start, end, num_steps = compute_time_window(
            task_data.X_train, step_size, prev_end,
        )
        inducing = build_inducing_strategy(
            model_cfg, hippo, start, end, num_steps, prev_steps,
            phase_origin=phase_origin,
        )

        # Train
        key, train_key = jax.random.split(key)
        t0 = time.time()
        model, history, km = _build_and_train(
            model_type, kernel, task_data, inducing,
            prior_state, spectral_sample, model_cfg, train_cfg, task_id,
            fitted_log_noise=fitted_log_noise,
            freeze_kernel=freeze_kernel,
            key=train_key,
        )
        wall_time = time.time() - t0

        # Save learned kernel after task 1 (when no ExactGP pre-fitting)
        if task_id == 1 and fitted_kernel is None:
            base_kernel = model.kernel

        # Evaluate on all seen test sets
        all_test_data.append(task_data)
        for td in all_test_data:
            key, eval_key = jax.random.split(key)
            metrics = evaluate_task(
                model, td, spectral_sample, eval_key, model_type=model_type,
            )
            print(
                f"    [eval task{td.task_id}] "
                f"RMSE={metrics['rmse']:.4f}  R2={metrics['r2']:.4f}  "
                f"NLPD={metrics['nlpd']:.4f}  ECE={metrics['ece']:.4f}"
            )

        # Accumulate training and test data
        all_X_train.append(np.asarray(task_data.X_train))
        all_y_train.append(np.asarray(task_data.y_train))
        all_X_test.append(np.asarray(task_data.X_test))
        all_y_test.append(np.asarray(task_data.y_test))

        # Predict on dense grid covering all seen data
        X_all_np = np.concatenate(all_X_train)
        x_lo = float(X_all_np.min()) - 5
        x_hi = float(X_all_np.max()) + 5
        X_grid = jnp.linspace(x_lo, x_hi, 300)[:, None]

        key, grid_key = jax.random.split(key)
        if model_type == "sgpr":
            grid_mean, grid_var = model.predict(
                X_grid, sample=spectral_sample, key=grid_key,
                full_cov=False, include_noise=True,
            )
            if grid_var.ndim == 1:
                grid_var = grid_var[:, None]
        else:
            grid_mean, grid_var = model.predict(
                X_grid, sample=spectral_sample, key=grid_key,
            )

        final_elbo = history[-1]["elbo"] if history else float("nan")
        task_metrics.append({
            **metrics, "task_id": task_id,
            "wall_time": wall_time, "final_elbo": final_elbo,
            "_pred": {
                "X_grid": np.asarray(X_grid),
                "mean_grid": np.asarray(grid_mean),
                "var_grid": np.asarray(grid_var),
                "X_train_all": np.concatenate(all_X_train),
                "y_train_all": np.concatenate(all_y_train),
                "X_train_current": np.asarray(task_data.X_train),
                "y_train_current": np.asarray(task_data.y_train),
                "X_test_all": np.concatenate(all_X_test),
                "y_test_all": np.concatenate(all_y_test),
                "X_test_current": np.asarray(task_data.X_test),
                "y_test_current": np.asarray(task_data.y_test),
            },
        })
        training_logs.append({"task_id": task_id, "history": history})

        # Get posterior state for next task
        key, posterior_key = jax.random.split(key)
        if model_type == "svgp":
            prior_state = model.get_posterior_state(km)
        else:
            prior_state = model.get_posterior_state(
                sample=spectral_sample, key=posterior_key,
            )
        prev_end = end
        prev_steps += num_steps

    return task_metrics, training_logs


# ---------------------------------------------------------------------------
# Single-strategy mode
# ---------------------------------------------------------------------------

def run_single(config):
    seed = config["experiment"].get("seed", 0)
    model_cfg = config["model"]
    train_cfg = config["training"]

    print(f"Experiment: {config['experiment']['name']}")

    full_model_cfg = {**model_cfg}
    if "likelihood" in config:
        full_model_cfg["likelihood"] = config["likelihood"]

    # Phase origin
    dataset = build_dataset(config["dataset"])
    task1 = next(dataset.tasks())
    phase_origin = (
        float(jnp.min(task1.X_train))
        if model_cfg.get("inducing_strategy") == "analytic_hippo"
        else model_cfg.get("phase_origin", 0.0)
    )

    run_strategy(
        config["dataset"], full_model_cfg, train_cfg, config["kernel"],
        phase_origin=phase_origin, seed=seed,
    )
    print("\nExperiment complete.")


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------

def _fit_exact_gp_hparams(dataset, base_cfg):
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

    kernel = build_kernel(base_cfg["kernel"])
    model = ExactGP(kernel=kernel)

    # Initialization: use config values if provided, else data-driven heuristic
    if "init_log_ls" in exact_cfg:
        model = eqx.tree_at(
            lambda m: m.kernel.log_ls, model,
            jnp.full_like(model.kernel.log_ls, exact_cfg["init_log_ls"]),
        )
    else:
        dists = jnp.linalg.norm(
            task1.X_train[:, None, :] - task1.X_train[None, :, :], axis=-1,
        )
        median_dist = jnp.median(dists[dists > 0])
        model = eqx.tree_at(
            lambda m: m.kernel.log_ls, model,
            jnp.full_like(model.kernel.log_ls, jnp.log(median_dist)),
        )

    if "init_log_sf" in exact_cfg:
        model = eqx.tree_at(
            lambda m: m.kernel.log_sf, model,
            jnp.full_like(model.kernel.log_sf, exact_cfg["init_log_sf"]),
        )
    else:
        model = eqx.tree_at(
            lambda m: m.kernel.log_sf, model,
            jnp.full_like(model.kernel.log_sf, jnp.log(jnp.std(task1.y_train))),
        )

    if "init_log_noise" in exact_cfg:
        model = eqx.tree_at(
            lambda m: m.likelihood.log_variance, model,
            jnp.array(exact_cfg["init_log_noise"]),
        )
    else:
        model = eqx.tree_at(
            lambda m: m.likelihood.log_variance, model,
            jnp.log(jnp.var(task1.y_train) * 0.1),
        )

    print(f"Fitting ExactGP on task 1 ({num_epochs} epochs)...")
    print(
        f"  init: log_ls={np.asarray(model.kernel.log_ls)}, "
        f"log_sf={float(jnp.squeeze(model.kernel.log_sf)):.4f}, "
        f"log_noise={float(model.likelihood.log_variance):.4f}"
    )

    optimizer = optax.adam(lr)
    step_fn = make_exact_gp_train_step(optimizer)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    log_every = exact_cfg.get("log_every", 100)
    for epoch in range(num_epochs):
        model, opt_state, loss = step_fn(
            model, opt_state, task1.X_train, task1.y_train,
        )
        if epoch % log_every == 0 or epoch == num_epochs - 1:
            print(f"  epoch {epoch:5d}  NLL={float(loss):.4f}")

    fitted_log_noise = model.likelihood.log_variance
    # Clip noise from below (following HarrisonZhu's HIPPOSVGP repo)
    min_log_noise = exact_cfg.get("min_log_noise", None)
    if min_log_noise is not None and float(fitted_log_noise) < min_log_noise:
        print(
            f"  clipping log_noise: {float(fitted_log_noise):.4f} -> {min_log_noise}"
        )
        fitted_log_noise = jnp.array(min_log_noise)

    final_nll = float(model.nll(task1.X_train, task1.y_train))
    print(
        f"  final: NLL={final_nll:.4f}, "
        f"log_ls={np.asarray(model.kernel.log_ls)}, "
        f"log_sf={float(jnp.squeeze(model.kernel.log_sf)):.4f}, "
        f"log_noise={float(model.likelihood.log_variance):.4f}"
    )
    return model.kernel, fitted_log_noise


def _save_strategy_outputs(strat_dir, task_metrics, training_logs):
    """Save metrics.csv, training_log.json, and prediction .npz files."""
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
            np.savez(
                os.path.join(pred_dir, f"task{m['task_id']:02d}.npz"),
                **pred,
            )


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
    key = jax.random.PRNGKey(seed)
    output_dir = config["experiment"].get("output_dir", "results/compare")
    os.makedirs(output_dir, exist_ok=True)

    base_cfg = config["base"]
    train_cfg = base_cfg.get("training", {})
    R = base_cfg["rff_sample_size"]

    print(f"Experiment: {config['experiment']['name']}")
    print(f"Output: {output_dir}")

    dataset = build_dataset(config["dataset"])
    fit_result = _fit_exact_gp_hparams(dataset, base_cfg)
    if fit_result is not None:
        fitted_kernel, fitted_log_noise = fit_result
    else:
        fitted_kernel, fitted_log_noise = None, None

    # Phase origin
    task1 = next(dataset.tasks())
    phase_origin = float(jnp.min(task1.X_train))

    # Shared spectral sample: draw once before strategy loop
    spectral_sample = None
    if config["experiment"].get("share_spectral_sample", False):
        seed_everything(seed)
        if fitted_kernel is not None:
            source = fitted_kernel
        else:
            key, kernel_key = jax.random.split(key)
            source = build_kernel(base_cfg["kernel"], key=kernel_key)
        key, draw_key = jax.random.split(key)
        spectral_sample = _draw_spectral_sample(source, R, draw_key)
        np.savez(
            os.path.join(output_dir, "spectral_sample.npz"),
            frequencies=np.asarray(spectral_sample.frequencies),
            scale=np.asarray(spectral_sample.scale),
        )

    all_results = {}

    for strat_name, strat_cfg in config["strategies"].items():
        print(f"\n{'#' * 60}")
        print(f"Strategy: {strat_name}")
        print(f"{'#' * 60}")

        merged_cfg = {**base_cfg, **strat_cfg}
        if "likelihood" in base_cfg:
            merged_cfg["likelihood"] = base_cfg["likelihood"]

        task_metrics, training_logs = run_strategy(
            config["dataset"], merged_cfg, train_cfg, base_cfg["kernel"],
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

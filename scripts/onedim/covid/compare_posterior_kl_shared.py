import argparse
import csv
import importlib
import math
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch
from scripts.onedim.covid.data import COVIDDataset


VAR_FLOOR = 1e-8


def _prepare_module(module_name, seed, inducing_size, rff_sample_size, device, extra_args=None):
    mod = importlib.import_module(module_name)
    mod = importlib.reload(mod)
    args_dict = {"seed_num": seed}
    if extra_args is not None:
        args_dict.update(extra_args)
    mod.args = SimpleNamespace(**args_dict)
    mod.seed_everything(seed)
    mod.rff_sample_size = rff_sample_size
    mod.hippo = mod.HiPPO_LegS(inducing_size, device)
    return mod


def _reset_module_outputs(base_dir, seed):
    model_dir = Path(base_dir) / f"models{seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "metrics.csv").write_text("eval_task,after_task,rmse,r2,nlpd,ece\n")
    (model_dir / "training.log").write_text("")
    return model_dir


def _reset_output_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "metrics.csv").write_text("eval_task,after_task,rmse,r2,nlpd,ece\n")
    (path / "training.log").write_text("")
    return path


def _save_task1_checkpoint(path, model, w, prev_discrete_steps, prev_end):
    checkpoint = {
        "kernel_state_dict": {k: v.detach().cpu() for k, v in model.kernel.state_dict().items()},
        "likelihood_state_dict": {k: v.detach().cpu() for k, v in model.likelihood.state_dict().items()},
        "mv": model.mv.detach().cpu(),
        "Lv": model.Lv.detach().cpu(),
        "w": w.detach().cpu(),
        "prev_discrete_steps": int(prev_discrete_steps),
        "prev_end": float(prev_end),
        "inducing_size": int(model.inducing_size),
        "num_inducing_old": int(model.num_inducing_old) if model.num_inducing_old is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return checkpoint


def _load_task1_checkpoint(path):
    return torch.load(path, map_location="cpu")


def _apply_task1_checkpoint(prev_model, checkpoint, device):
    kernel_sd = {k: v.to(device=device) for k, v in checkpoint["kernel_state_dict"].items()}
    lik_sd = {k: v.to(device=device) for k, v in checkpoint["likelihood_state_dict"].items()}
    prev_model.kernel.load_state_dict(kernel_sd)
    prev_model.likelihood.load_state_dict(lik_sd)
    prev_model.mv = torch.nn.Parameter(checkpoint["mv"].to(device=device).clone(), requires_grad=True)
    prev_model.Lv = torch.nn.Parameter(checkpoint["Lv"].to(device=device).clone(), requires_grad=True)
    return prev_model


def _clone_task_cache(days_test_list, deaths_test_list, days_train, deaths_train):
    return (
        [x.clone().detach() for x in days_test_list],
        [x.clone().detach() for x in deaths_test_list],
        days_train.clone().detach(),
        deaths_train.clone().detach(),
    )


def _assert_shared_state_unchanged(model, checkpoint, solver_name, task_id, atol=1e-12, rtol=1e-9):
    for name, ref_val in checkpoint["kernel_state_dict"].items():
        cur_val = model.kernel.state_dict()[name].detach().cpu()
        ref_val = ref_val.detach().cpu()
        if not torch.allclose(cur_val, ref_val, atol=atol, rtol=rtol):
            diff = torch.max(torch.abs(cur_val - ref_val)).item()
            raise RuntimeError(
                f"[{solver_name}] kernel parameter drift detected at task {task_id}: "
                f"{name}, max_abs_diff={diff}"
            )
    for name, ref_val in checkpoint["likelihood_state_dict"].items():
        cur_val = model.likelihood.state_dict()[name].detach().cpu()
        ref_val = ref_val.detach().cpu()
        if not torch.allclose(cur_val, ref_val, atol=atol, rtol=rtol):
            diff = torch.max(torch.abs(cur_val - ref_val)).item()
            raise RuntimeError(
                f"[{solver_name}] likelihood parameter drift detected at task {task_id}: "
                f"{name}, max_abs_diff={diff}"
            )


def _set_fixed_kernel_hparams(model, lengthscale=None, variance=None):
    with torch.no_grad():
        if lengthscale is not None:
            ls = torch.as_tensor(lengthscale, dtype=model.kernel.log_ls.dtype, device=model.kernel.log_ls.device)
            if ls.ndim == 0:
                ls = ls.repeat(model.kernel.log_ls.numel())
            if ls.numel() != model.kernel.log_ls.numel():
                raise ValueError(
                    f"lengthscale has {ls.numel()} values but kernel expects {model.kernel.log_ls.numel()}"
                )
            model.kernel.log_ls.copy_(torch.log(ls))
        if variance is not None:
            var = float(variance)
            if var <= 0.0:
                raise ValueError(f"variance must be > 0, got {var}")
            model.kernel.log_sf.copy_(torch.tensor(math.log(var), dtype=model.kernel.log_sf.dtype, device=model.kernel.log_sf.device))
    model.kernel.log_ls.requires_grad = False
    model.kernel.log_sf.requires_grad = False
    if hasattr(model.likelihood, "_nu"):
        model.likelihood._nu.requires_grad = False


def _make_fixed_se_kernel_factory(se_kernel_cls, fixed_lengthscale, fixed_variance):
    if fixed_variance <= 0.0:
        raise ValueError(f"fixed_variance must be > 0, got {fixed_variance}")

    def _factory(input_dim, device="cpu"):
        kernel = se_kernel_cls(input_dim, device=device)
        with torch.no_grad():
            ls = torch.as_tensor(
                fixed_lengthscale, device=kernel.log_ls.device, dtype=kernel.log_ls.dtype
            )
            if ls.ndim == 0:
                ls = ls.repeat(kernel.log_ls.numel())
            if ls.numel() != kernel.log_ls.numel():
                raise ValueError(
                    f"fixed_lengthscale has {ls.numel()} values but kernel expects {kernel.log_ls.numel()}"
                )
            if torch.any(ls <= 0):
                raise ValueError("fixed_lengthscale must be > 0.")
            kernel.log_ls.copy_(torch.log(ls))
            kernel.log_sf.copy_(
                torch.tensor(
                    math.log(float(fixed_variance)),
                    device=kernel.log_sf.device,
                    dtype=kernel.log_sf.dtype,
                )
            )
        kernel.log_ls.requires_grad = False
        kernel.log_sf.requires_grad = False
        return kernel

    return _factory


def _assert_kernel_fixed(model, lengthscale, variance, atol=1e-10, rtol=1e-8):
    cur_ls = torch.exp(model.kernel.log_ls.detach().reshape(-1).cpu())
    ref_ls = torch.as_tensor(lengthscale, dtype=cur_ls.dtype).reshape(-1)
    if ref_ls.numel() == 1:
        ref_ls = ref_ls.repeat(cur_ls.numel())
    cur_var = float(torch.exp(model.kernel.log_sf.detach()).cpu().item())
    ref_var = float(variance)
    if not torch.allclose(cur_ls, ref_ls, atol=atol, rtol=rtol):
        raise RuntimeError(
            f"Kernel lengthscale drift detected. current={cur_ls.tolist()}, expected={ref_ls.tolist()}"
        )
    if not math.isclose(cur_var, ref_var, abs_tol=atol, rel_tol=rtol):
        raise RuntimeError(
            f"Kernel variance drift detected. current={cur_var}, expected={ref_var}"
        )


def _load_task_full_series(task_id, device, reference_filename=None):
    dataset = COVIDDataset(f"data/covid/processed_timeseries_5tasks/task_{task_id}")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Keep Full-GP data source consistent with train_ohsgp pipeline:
    # single state trajectory instead of mixing all states into one 1D GP.
    if reference_filename is None:
        batch = next(iter(dataloader))
    else:
        batch = None
        for b in dataloader:
            filename = str(b[9][0])
            if os.path.basename(filename) == reference_filename:
                batch = b
                break
        if batch is None:
            raise RuntimeError(
                f"reference sequence '{reference_filename}' not found in task_{task_id}."
            )

    deaths, days, _, _, _, num_days, _, _, _, filename, _, _, _ = batch
    deaths = deaths.to(device=device, dtype=torch.float64)
    days = days.to(device=device, dtype=torch.float64)
    num_days = num_days.to(device=device, dtype=torch.float64)
    n = int(num_days[0].item())
    days = days.permute(1, 0)[:n]
    deaths = deaths.permute(1, 0)[:n]
    selected_filename = os.path.basename(str(filename[0]))
    return days, deaths, selected_filename


def _fit_full_gp_hyperparams_task1(
    X_train,
    y_train,
    num_steps=1500,
    lr=0.03,
    noise_variance=None,
    noise_init=10.0,
    restart_id=0,
):
    X_train = X_train.to(dtype=torch.float64)
    y_train = y_train.to(dtype=torch.float64)
    device = X_train.device

    # Full GP with a constant mean term: center targets during fit,
    # then restore the mean in posterior prediction.
    y_mean = torch.mean(y_train, dim=0, keepdim=True)
    y_centered = y_train - y_mean

    # Normalize coordinates for robust hyperparameter optimization.
    x_scale = torch.std(X_train).clamp_min(1e-6)
    y_scale = torch.std(y_centered).clamp_min(1.0)
    X_fit = X_train / x_scale
    y_fit = y_centered / y_scale
    n = X_fit.shape[0]

    ls_scale_candidates = [0.25, 0.5, 1.0, 2.0, 4.0]
    var_scale_candidates = [0.5, 1.0, 2.0]
    ls_mul = ls_scale_candidates[restart_id % len(ls_scale_candidates)]
    var_mul = var_scale_candidates[(restart_id // len(ls_scale_candidates)) % len(var_scale_candidates)]

    # In normalized coordinates, base scales are O(1).
    ls0 = torch.tensor(ls_mul, device=device, dtype=torch.float64)
    var0 = torch.tensor(var_mul, device=device, dtype=torch.float64)

    if noise_variance is None:
        noise0_norm = float(noise_init) / float(y_scale.pow(2).item())
    else:
        noise0_norm = float(noise_variance) / float(y_scale.pow(2).item())
    noise0 = torch.tensor(max(noise0_norm, 1e-8), device=device, dtype=torch.float64)

    log_ls = torch.nn.Parameter(torch.log(ls0).reshape(1))
    log_var = torch.nn.Parameter(torch.log(var0).reshape(1))
    if noise_variance is None:
        log_noise = torch.nn.Parameter(torch.log(noise0).reshape(1))
        params = [log_ls, log_var, log_noise]
    else:
        log_noise = torch.log(noise0).reshape(1)
        params = [log_ls, log_var]

    optimizer = torch.optim.Adam(params, lr=lr)
    for _ in range(num_steps):
        optimizer.zero_grad()

        ls = torch.exp(log_ls).clamp_min(1e-6)
        var = torch.exp(log_var).clamp_min(1e-8)
        noise_var = torch.exp(log_noise).clamp_min(1e-8) if noise_variance is None else torch.exp(log_noise)

        K = _se_kernel(X_fit, X_fit, ls, var)
        C = K + noise_var * torch.eye(n, device=device, dtype=torch.float64)
        L = _cholesky_with_retry(C)
        alpha = torch.cholesky_solve(y_fit, L)

        nll = 0.5 * torch.sum(y_fit * alpha)
        nll = nll + torch.sum(torch.log(torch.diagonal(L)))
        nll = nll + 0.5 * n * math.log(2.0 * math.pi)
        nll.backward()
        optimizer.step()

        with torch.no_grad():
            log_ls.clamp_(math.log(1e-6), math.log(1e6))
            log_var.clamp_(math.log(1e-8), math.log(1e8))
            if noise_variance is None:
                log_noise.clamp_(math.log(1e-8), math.log(1e4))

    ls_norm = torch.exp(log_ls.detach()).item()
    var_norm = torch.exp(log_var.detach()).item()
    noise_norm = torch.exp(log_noise.detach()).item()

    # Map optimized normalized hyperparameters back to original units.
    x_scale_f = float(x_scale.item())
    y_scale_sq = float(y_scale.pow(2).item())
    return {
        "lengthscale": float(ls_norm * x_scale_f),
        "variance": float(var_norm * y_scale_sq),
        "noise_variance": float(noise_norm * y_scale_sq),
        "mean_constant": float(y_mean.squeeze().item()),
    }


def _select_full_gp_hyperparams_task1(
    days,
    deaths,
    num_steps=1500,
    lr=0.03,
    noise_grid=None,
    forced_noise=None,
    noise_init=10.0,
    num_restarts=5,
):
    x_train, y_train, x_test, y_test = _task1_split_like_ohsgp(days, deaths)
    x_fit = days
    y_fit = deaths
    y_train_t = _full_gp_target_transform(y_train)
    y_fit_t = _full_gp_target_transform(y_fit)

    if num_restarts < 1:
        raise ValueError(f"num_restarts must be >= 1, got {num_restarts}")

    if forced_noise is not None:
        best = None
        best_rmse = float("inf")
        best_r2 = float("-inf")
        for rid in range(num_restarts):
            params = _fit_full_gp_hyperparams_task1(
                x_fit,
                y_fit_t,
                num_steps=num_steps,
                lr=lr,
                noise_variance=forced_noise,
                noise_init=noise_init,
                restart_id=rid,
            )
            mu_test, _ = full_gp_posterior(x_train, y_train_t, x_test, params)
            pred_test = _full_gp_target_inverse(mu_test).unsqueeze(-1)
            rmse, r2 = _rmse_r2(y_test, pred_test)
            if rmse < best_rmse:
                best = params
                best_rmse = rmse
                best_r2 = r2
        return best, best_rmse, best_r2

    if noise_grid is None:
        # Preferred mode: jointly learn lengthscale / variance / noise via marginal likelihood.
        best = None
        best_rmse = float("inf")
        best_r2 = float("-inf")
        for rid in range(num_restarts):
            params = _fit_full_gp_hyperparams_task1(
                x_fit,
                y_fit_t,
                num_steps=num_steps,
                lr=lr,
                noise_variance=None,
                noise_init=noise_init,
                restart_id=rid,
            )
            mu_test, _ = full_gp_posterior(x_train, y_train_t, x_test, params)
            pred_test = _full_gp_target_inverse(mu_test).unsqueeze(-1)
            rmse, r2 = _rmse_r2(y_test, pred_test)
            if rmse < best_rmse:
                best = params
                best_rmse = rmse
                best_r2 = r2
        return best, best_rmse, best_r2

    best = None
    best_rmse = float("inf")
    best_r2 = float("-inf")
    for noise_var in noise_grid:
        for rid in range(num_restarts):
            params = _fit_full_gp_hyperparams_task1(
                x_fit,
                y_fit_t,
                num_steps=num_steps,
                lr=lr,
                noise_variance=noise_var,
                noise_init=noise_init,
                restart_id=rid,
            )
            mu_test, _ = full_gp_posterior(x_train, y_train_t, x_test, params)
            pred_test = _full_gp_target_inverse(mu_test).unsqueeze(-1)
            rmse, r2 = _rmse_r2(y_test, pred_test)
            if rmse < best_rmse:
                best = params
                best_rmse = rmse
                best_r2 = r2
    return best, best_rmse, best_r2


def _predict_mean(model, f_mean):
    if hasattr(model.likelihood, "link_fn"):
        return model.likelihood.link_fn(f_mean)
    return f_mean


def _rmse_r2(y_true, y_pred):
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()
    mse = torch.mean((y_pred - y_true) ** 2)
    rmse = torch.sqrt(mse)
    var = torch.mean((y_true - y_true.mean()) ** 2)
    r2 = 1 - mse / var if var.item() > 0 else torch.tensor(float("nan"), device=y_true.device)
    return rmse.item(), r2.item()


def _full_gp_target_transform(y):
    return y


def _full_gp_target_inverse(f_mean):
    return f_mean


def _task_eval_metrics(model, x_test, y_test, w, num_mc_nlpd=1000):
    with torch.no_grad():
        mean, var = model.pred_f(x_test, w)
        var = torch.clamp(var, min=VAR_FLOOR)
        pred_mean = _predict_mean(model, mean)
        rmse, r2 = _rmse_r2(y_test, pred_mean)
        eps = torch.randn(num_mc_nlpd, var.shape[0], var.shape[1], device=var.device, dtype=var.dtype)
        f_samples = mean + torch.sqrt(var) * eps
        y_rep = y_test.unsqueeze(0).tile(num_mc_nlpd, 1, 1).reshape(-1, 1)
        lp = model.likelihood.log_prob(f_samples.reshape(-1, 1), y_rep)
        nlpd = -torch.log(lp.view(num_mc_nlpd, -1).exp().mean(dim=0)).mean().item()
    return rmse, r2, nlpd


def _task1_split_like_ohsgp(days, deaths):
    n = int(days.shape[0])
    torch.manual_seed(42)
    rand_ind = torch.randperm(n)
    test_size = int(n * 0.2)
    days_test = days[rand_ind[:test_size]]
    days_train = days[rand_ind[test_size:]]
    deaths_test = deaths[rand_ind[:test_size]]
    deaths_train = deaths[rand_ind[test_size:]]
    return days_train, deaths_train, days_test, deaths_test


def _ensure_col_vector(x, device):
    if isinstance(x, torch.Tensor):
        out = x.to(device=device, dtype=torch.float64)
    else:
        out = torch.as_tensor(x, device=device, dtype=torch.float64)
    if out.ndim == 1:
        out = out.unsqueeze(-1)
    if out.ndim != 2 or out.shape[1] != 1:
        raise ValueError(f"Expected shape [N,1], got {tuple(out.shape)}")
    return out


def _se_kernel(X1, X2, lengthscale, variance):
    X1s = X1 / lengthscale
    X2s = X2 / lengthscale
    sq = (X1s.pow(2).sum(-1, keepdim=True) + X2s.pow(2).sum(-1).unsqueeze(0) - 2.0 * (X1s @ X2s.T))
    return variance * torch.exp(-0.5 * sq)


def _cholesky_with_retry(mat, base_jitter=1e-8, max_tries=8):
    eye = torch.eye(mat.shape[0], device=mat.device, dtype=mat.dtype)
    scale = torch.mean(torch.diag(mat)).detach().clamp_min(1.0)
    for i in range(max_tries):
        jitter = (10.0**i) * base_jitter * scale
        try:
            return torch.linalg.cholesky(mat + jitter * eye)
        except torch.linalg.LinAlgError:
            continue
    raise torch.linalg.LinAlgError("Cholesky failed in full_gp_posterior after jitter retries.")


def full_gp_posterior(X_train, y_train, X_test, kernel_params):
    """
    Exact GP latent posterior with fixed RBF kernel + Gaussian observation noise.
    Returns mean [T] and latent variance [T] (noise not added to test variance).
    """
    device = X_train.device if isinstance(X_train, torch.Tensor) else torch.device("cpu")
    X_train = _ensure_col_vector(X_train, device)
    y_train = _ensure_col_vector(y_train, device)
    X_test = _ensure_col_vector(X_test, device)

    lengthscale = torch.as_tensor(kernel_params["lengthscale"], device=device, dtype=torch.float64)
    if lengthscale.ndim == 0:
        lengthscale = lengthscale.repeat(X_train.shape[1])
    if lengthscale.numel() != X_train.shape[1]:
        raise ValueError(f"lengthscale must have {X_train.shape[1]} value(s), got {lengthscale.numel()}")
    variance = torch.as_tensor(kernel_params["variance"], device=device, dtype=torch.float64)
    noise_var = torch.as_tensor(kernel_params["noise_variance"], device=device, dtype=torch.float64)
    if torch.any(lengthscale <= 0):
        raise ValueError("All lengthscale values must be > 0.")
    if variance <= 0:
        raise ValueError("variance must be > 0.")
    if noise_var <= 0:
        raise ValueError("noise_variance must be > 0.")

    mean_constant = torch.as_tensor(
        kernel_params.get("mean_constant", 0.0), device=device, dtype=torch.float64
    ).reshape(1, 1)
    y_train_centered = y_train - mean_constant

    K_tt = _se_kernel(X_train, X_train, lengthscale, variance)
    C = K_tt + noise_var * torch.eye(X_train.shape[0], device=device, dtype=torch.float64)
    L = _cholesky_with_retry(C)

    K_tn = _se_kernel(X_train, X_test, lengthscale, variance)
    alpha = torch.cholesky_solve(y_train_centered, L)
    mu = (K_tn.T @ alpha).squeeze(-1) + mean_constant.squeeze()

    v = torch.cholesky_solve(K_tn, L)
    K_nn_diag = torch.diag(_se_kernel(X_test, X_test, lengthscale, variance))
    var = K_nn_diag - torch.sum(K_tn * v, dim=0)
    var = torch.clamp(var, min=VAR_FLOOR)

    assert mu.ndim == 1 and var.ndim == 1, "full_gp_posterior must return [T] tensors."
    assert mu.shape == var.shape, "full_gp_posterior mean/var shape mismatch."
    return mu, var


def hipp_svpg_ode_posterior(model, X_test, w):
    with torch.no_grad():
        mu, var = model.pred_f(X_test, w)
    mu = mu.squeeze(-1)
    var = torch.clamp(var.squeeze(-1), min=VAR_FLOOR)
    assert mu.ndim == 1 and var.ndim == 1, "hipp_svpg_ode_posterior must return [T] tensors."
    assert mu.shape == var.shape, "hipp_svpg_ode_posterior mean/var shape mismatch."
    return mu, var


def hipp_svpg_analytical_posterior(model, X_test, w):
    with torch.no_grad():
        mu, var = model.pred_f(X_test, w)
    mu = mu.squeeze(-1)
    var = torch.clamp(var.squeeze(-1), min=VAR_FLOOR)
    assert mu.ndim == 1 and var.ndim == 1, "hipp_svpg_analytical_posterior must return [T] tensors."
    assert mu.shape == var.shape, "hipp_svpg_analytical_posterior mean/var shape mismatch."
    return mu, var


def kl_gaussian_1d(mu_q, var_q, mu_p, var_p):
    mu_q = mu_q.reshape(-1)
    mu_p = mu_p.reshape(-1)
    var_q = torch.clamp(var_q.reshape(-1), min=VAR_FLOOR)
    var_p = torch.clamp(var_p.reshape(-1), min=VAR_FLOOR)

    assert mu_q.shape == mu_p.shape == var_q.shape == var_p.shape, "All KL inputs must share shape [T]."
    kl = 0.5 * (torch.log(var_p / var_q) + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0)
    return kl


def _build_sorted_test_grid(days_train, days_test_list, device):
    xs = [days_train] + list(days_test_list)
    x_all = torch.cat(xs, dim=0).reshape(-1)
    x_all = torch.unique(x_all)
    x_all = torch.sort(x_all).values
    return x_all.to(device=device, dtype=torch.float64).unsqueeze(-1)


def _write_kl_csv(path, x_test, kl_ode, kl_ana, mu_p, var_p, mu_ode, var_ode, mu_ana, var_ana):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestep",
                "kl_ode",
                "kl_analytical",
                "mu_full_gp",
                "var_full_gp",
                "mu_ode",
                "var_ode",
                "mu_analytical",
                "var_analytical",
            ]
        )
        for i in range(x_test.shape[0]):
            writer.writerow(
                [
                    float(x_test[i].item()),
                    float(kl_ode[i].item()),
                    float(kl_ana[i].item()),
                    float(mu_p[i].item()),
                    float(var_p[i].item()),
                    float(mu_ode[i].item()),
                    float(var_ode[i].item()),
                    float(mu_ana[i].item()),
                    float(var_ana[i].item()),
                ]
            )


def _plot_kl(path, x_test, kl_ode, kl_ana):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("matplotlib is required to generate KL plots.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(x_test, kl_ode, label="KL(ODE || Full GP)", linewidth=2.0)
    plt.plot(x_test, kl_ana, label="KL(Analytical || Full GP)", linewidth=2.0)
    plt.xlabel("Timestep")
    plt.ylabel("KL divergence")
    plt.title("Per-timestep KL to Full GP posterior")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _detect_kl_spike_intervals(kl_ode, kl_ana, quantile=0.95):
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"quantile must be in (0,1), got {quantile}")
    kl_joint = torch.maximum(kl_ode.reshape(-1), kl_ana.reshape(-1))
    threshold = torch.quantile(kl_joint, quantile).item()
    mask = (kl_joint >= threshold).tolist()
    intervals = []
    start = None
    for i, is_spike in enumerate(mask):
        if is_spike and start is None:
            start = i
        elif (not is_spike) and start is not None:
            intervals.append((start, i - 1))
            start = None
    if start is not None:
        intervals.append((start, len(mask) - 1))
    return intervals, threshold


def _plot_variance_comparison(path, x_test, var_p, var_ode, var_ana):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("matplotlib is required to generate variance plots.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(x_test, var_p, label="var_full_gp", linewidth=2.0)
    plt.plot(x_test, var_ode, label="var_ode", linewidth=2.0)
    plt.plot(x_test, var_ana, label="var_analytical", linewidth=2.0)
    plt.xlabel("Timestep")
    plt.ylabel("Posterior variance")
    plt.title("Variance comparison on common test grid")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_log_variance_ratio(path, x_test, var_p, var_ode, var_ana):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("matplotlib is required to generate variance-ratio plots.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    var_p = torch.clamp(var_p, min=VAR_FLOOR)
    var_ode = torch.clamp(var_ode, min=VAR_FLOOR)
    var_ana = torch.clamp(var_ana, min=VAR_FLOOR)
    log_ratio_ode = torch.log(var_ode / var_p)
    log_ratio_ana = torch.log(var_ana / var_p)
    plt.figure(figsize=(10, 5))
    plt.plot(x_test, log_ratio_ode, label="log(var_ode / var_full_gp)", linewidth=2.0)
    plt.plot(x_test, log_ratio_ana, label="log(var_analytical / var_full_gp)", linewidth=2.0)
    plt.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    plt.xlabel("Timestep")
    plt.ylabel("Log variance ratio")
    plt.title("Variance mismatch against Full GP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_mean_abs_error_with_kl_spikes(path, x_test, mu_p, mu_ode, mu_ana, spike_intervals, spike_threshold):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("matplotlib is required to generate mean-error plots.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    err_ode = torch.abs(mu_ode - mu_p)
    err_ana = torch.abs(mu_ana - mu_p)
    plt.figure(figsize=(10, 5))
    plt.plot(x_test, err_ode, label="abs(mu_ode - mu_full_gp)", linewidth=2.0)
    plt.plot(x_test, err_ana, label="abs(mu_analytical - mu_full_gp)", linewidth=2.0)
    for i, (s, e) in enumerate(spike_intervals):
        label = "KL spike interval" if i == 0 else None
        plt.axvspan(float(x_test[s]), float(x_test[e]), color="gray", alpha=0.18, label=label)
    plt.xlabel("Timestep")
    plt.ylabel("Absolute mean error")
    plt.title(f"Mean mismatch with KL spikes (threshold={spike_threshold:.4g})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare posterior KL against full GP with shared fixed hyperparameters: "
            "ODE-HiPPO-SVGP vs Analytical-HiPPO-SVGP."
        )
    )
    parser.add_argument("--seed_num", type=int, default=0)
    parser.add_argument("--inducing_size", type=int, default=15)
    parser.add_argument("--num_epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--step_size", type=float, default=2.0)
    parser.add_argument("--rff_sample_size", type=int, default=1000)
    parser.add_argument("--task_start", type=int, default=2)
    parser.add_argument("--task_end", type=int, default=5)
    parser.add_argument(
        "--noise_variance",
        type=float,
        default=None,
        help="If set, fixes Full-GP observation noise variance. If unset, noise is learned.",
    )
    parser.add_argument(
        "--full_gp_noise_init",
        type=float,
        default=10.0,
        help="Initial noise variance when Full GP learns noise (raw count space).",
    )
    parser.add_argument(
        "--lengthscale",
        type=float,
        default=None,
        help="If set, overrides Task1 Full GP learned lengthscale and fixes it for all models.",
    )
    parser.add_argument(
        "--variance",
        type=float,
        default=None,
        help="If set, overrides Task1 Full GP learned kernel variance and fixes it for all models.",
    )
    parser.add_argument("--shared_dir", type=str, default="ohsgp_shared")
    parser.add_argument("--checkpoint_name", type=str, default="task1_shared_state.pt")
    parser.add_argument("--ode_out_root", type=str, default="ohsgp_shared/ode")
    parser.add_argument("--analytic_out_root", type=str, default="ohsgp_shared/analytic")
    parser.add_argument(
        "--analytic_phase_origin_mode",
        type=str,
        default="global_start",
        choices=["zero", "global_start"],
        help="Phase-origin mode forwarded to scripts.onedim.covid.test_ohsgp_analytic.",
    )
    parser.add_argument("--full_gp_fit_steps", type=int, default=1500)
    parser.add_argument("--full_gp_fit_lr", type=float, default=0.03)
    parser.add_argument(
        "--full_gp_restarts",
        type=int,
        default=8,
        help="Number of deterministic restarts for Task1 Full GP hyperparameter fitting.",
    )
    parser.add_argument(
        "--share_rff_w",
        action="store_true",
        help=(
            "After Task1, force ODE/Analytical solvers to use the exact same RFF sample w "
            "for Task2+ training and final posterior/KL evaluation."
        ),
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Enable deterministic PyTorch algorithms. "
            "On CUDA, set CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8) before running."
        ),
    )
    parser.add_argument(
        "--allow_overwrite_baselines",
        action="store_true",
        help=(
            "Allow writing into baseline output roots (ohsgp / ohsgp_analytic). "
            "Default behavior protects baseline images from being overwritten."
        ),
    )
    args = parser.parse_args()

    if args.task_start < 2:
        raise ValueError("--task_start must be >= 2, because Task1 is handled separately before continual tasks.")
    if args.task_end < args.task_start:
        raise ValueError("--task_end must be >= --task_start.")
    if args.noise_variance is not None and args.noise_variance <= 0.0:
        raise ValueError("--noise_variance must be > 0.")
    if args.full_gp_noise_init <= 0.0:
        raise ValueError("--full_gp_noise_init must be > 0.")
    if args.full_gp_restarts < 1:
        raise ValueError("--full_gp_restarts must be >= 1.")
    if not args.allow_overwrite_baselines:
        if Path(args.ode_out_root).resolve() == Path("ohsgp").resolve():
            raise ValueError(
                "--ode_out_root points to baseline root 'ohsgp'. "
                "Use a different output root or pass --allow_overwrite_baselines."
            )
        if Path(args.analytic_out_root).resolve() == Path("ohsgp_analytic").resolve():
            raise ValueError(
                "--analytic_out_root points to baseline root 'ohsgp_analytic'. "
                "Use a different output root or pass --allow_overwrite_baselines."
            )

    # Optional strict determinism. Disabled by default to avoid CUDA/CuBLAS runtime errors.
    if args.deterministic:
        cublas_cfg = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        cuda_on = torch.cuda.is_available()
        cublas_ok = cublas_cfg in {":4096:8", ":16:8"}
        if cuda_on and not cublas_ok:
            print(
                "[warning] --deterministic ignored because CUBLAS_WORKSPACE_CONFIG is not set "
                "to ':4096:8' or ':16:8'. Running in non-deterministic mode."
            )
        else:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ode_mod = _prepare_module(
        "scripts.onedim.covid.test_ohsgp",
        args.seed_num,
        args.inducing_size,
        args.rff_sample_size,
        device,
        extra_args={"out_root": args.ode_out_root},
    )
    analytic_mod = _prepare_module(
        "scripts.onedim.covid.test_ohsgp_analytic",
        args.seed_num,
        args.inducing_size,
        args.rff_sample_size,
        device,
        extra_args={
            "phase_origin_mode": args.analytic_phase_origin_mode,
            "out_root": args.analytic_out_root,
        },
    )

    ode_output_dir = _reset_module_outputs(args.ode_out_root, args.seed_num)
    # scripts.onedim.covid.test_ohsgp writes through this module-global path.
    ode_mod.OUTPUT_DIR = str(ode_output_dir)
    analytic_output_dir = _reset_output_dir(
        Path(args.analytic_out_root) / f"models{args.seed_num}" / args.analytic_phase_origin_mode
    )
    # scripts.onedim.covid.test_ohsgp_analytic writes through this module-global path.
    analytic_mod.OUTPUT_DIR = str(analytic_output_dir)
    print(
        "[analytic-config] "
        f"phase_origin_mode={args.analytic_phase_origin_mode}, "
        f"output_dir={analytic_output_dir}"
    )
    print(f"[ode-config] output_dir={ode_output_dir}")
    print("[full-gp-config] target_space=raw, noise_init={}".format(args.full_gp_noise_init))
    print("[full-gp-config] restarts={}".format(args.full_gp_restarts))
    print("[rff-config] share_rff_w={}".format(args.share_rff_w))

    # Full GP fit on Task1 full data (learn noise by default; can be fixed via --noise_variance).
    task1_days_full, task1_deaths_full, ref_filename = _load_task_full_series(1, device)
    full_gp_task1, t1_sel_rmse, t1_sel_r2 = _select_full_gp_hyperparams_task1(
        days=task1_days_full,
        deaths=task1_deaths_full,
        num_steps=args.full_gp_fit_steps,
        lr=args.full_gp_fit_lr,
        forced_noise=args.noise_variance,
        noise_init=args.full_gp_noise_init,
        num_restarts=args.full_gp_restarts,
    )
    print(
        "[task1 full-gp-fit] learned hyperparams: "
        f"lengthscale={full_gp_task1['lengthscale']:.8f}, "
        f"variance={full_gp_task1['variance']:.8f}, "
        f"noise_variance={full_gp_task1['noise_variance']:.8f}, "
        f"mean_constant={full_gp_task1['mean_constant']:.8f}"
    )
    print(
        "[task1 full-gp-fit] split metric used for selection: "
        f"rmse={t1_sel_rmse}, r2={t1_sel_r2}"
    )
    print(f"[full-gp-data] reference sequence: {ref_filename}")

    # Direct Full GP Task1 metrics on the same split rule as test_ohsgp.py.
    t1_x_train, t1_y_train, t1_x_test, t1_y_test = _task1_split_like_ohsgp(
        task1_days_full, task1_deaths_full
    )
    t1_mu_lat, _ = full_gp_posterior(
        t1_x_train,
        _full_gp_target_transform(t1_y_train),
        t1_x_test,
        full_gp_task1,
    )
    t1_pred = _full_gp_target_inverse(t1_mu_lat).unsqueeze(-1)
    t1_rmse, t1_r2 = _rmse_r2(t1_y_test, t1_pred)
    print(
        "[task1 full-gp-direct] "
        f"test rmse: {t1_rmse}, r2: {t1_r2}"
    )

    # Fix shared kernel hyperparameters from Task1 Full GP fit (unless manually overridden).
    fixed_lengthscale = args.lengthscale if args.lengthscale is not None else full_gp_task1["lengthscale"]
    fixed_variance = args.variance if args.variance is not None else full_gp_task1["variance"]
    fixed_noise_variance = full_gp_task1["noise_variance"]

    # Force both solvers to use the same fixed SE kernel from Task1 onward.
    ode_se_cls = ode_mod.SE_kernel
    ana_se_cls = analytic_mod.SE_kernel
    ode_mod.SE_kernel = _make_fixed_se_kernel_factory(ode_se_cls, fixed_lengthscale, fixed_variance)
    analytic_mod.SE_kernel = _make_fixed_se_kernel_factory(ana_se_cls, fixed_lengthscale, fixed_variance)

    # Train Task1 independently for ODE / Analytic under the fixed kernel.
    ode_mod.seed_everything(args.seed_num)
    ode_prev_model, ode_days_test_list, ode_deaths_test_list, ode_days_train, ode_deaths_train, ode_w, ode_prev_discrete_steps, ode_prev_end = (
        ode_mod.train_task1(
            device=device,
            inducing_size=args.inducing_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            step_size=args.step_size,
        )
    )
    _assert_kernel_fixed(ode_prev_model, fixed_lengthscale, fixed_variance)

    analytic_mod.seed_everything(args.seed_num)
    analytic_prev_model, ana_days_test_list, ana_deaths_test_list, ana_days_train, ana_deaths_train, ana_w, analytic_prev_discrete_steps, analytic_prev_end = (
        analytic_mod.train_task1(
            device=device,
            inducing_size=args.inducing_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            step_size=args.step_size,
        )
    )
    _assert_kernel_fixed(analytic_prev_model, fixed_lengthscale, fixed_variance)

    # Explicit Task1 metrics under Full-GP-fixed kernel for direct comparison.
    ode_t1_rmse, ode_t1_r2, ode_t1_nlpd = _task_eval_metrics(
        ode_prev_model,
        ode_days_test_list[0].to(device=device, dtype=torch.float64),
        ode_deaths_test_list[0].to(device=device, dtype=torch.float64),
        ode_w,
    )
    ana_t1_rmse, ana_t1_r2, ana_t1_nlpd = _task_eval_metrics(
        analytic_prev_model,
        ana_days_test_list[0].to(device=device, dtype=torch.float64),
        ana_deaths_test_list[0].to(device=device, dtype=torch.float64),
        ana_w,
    )
    print(
        "[task1 fixed-kernel-from-full-gp][ode] "
        f"test rmse: {ode_t1_rmse}, r2: {ode_t1_r2}, nlpd: {ode_t1_nlpd}"
    )
    print(
        "[task1 fixed-kernel-from-full-gp][analytic] "
        f"test rmse: {ana_t1_rmse}, r2: {ana_t1_r2}, nlpd: {ana_t1_nlpd}"
    )

    if args.share_rff_w:
        shared_w = ode_w.detach().clone()
        ode_w = shared_w
        ana_w = shared_w
        print("[rff-config] using shared Task1 RFF sample w for Task2+ and final posterior.")

    shared_dir = Path(args.shared_dir) / f"models{args.seed_num}"
    checkpoint_path = shared_dir / args.checkpoint_name
    _save_task1_checkpoint(checkpoint_path, ode_prev_model, ode_w, ode_prev_discrete_steps, ode_prev_end)

    for task_id in range(args.task_start, args.task_end + 1):
        ode_w_step = ode_w
        if args.share_rff_w:
            ode_w_step = shared_w
        ode_prev_model, ode_days_test_list, ode_deaths_test_list, ode_days_train, ode_deaths_train, ode_w, ode_prev_discrete_steps, ode_prev_end = (
            ode_mod.train_nexttask(
                ode_prev_model,
                ode_days_test_list,
                ode_deaths_test_list,
                ode_days_train,
                ode_deaths_train,
                device,
                task_id,
                args.inducing_size,
                args.num_epochs,
                args.lr,
                args.step_size,
                ode_w_step,
                ode_prev_discrete_steps,
                ode_prev_end,
            )
        )
        _assert_kernel_fixed(ode_prev_model, fixed_lengthscale, fixed_variance)
        if args.share_rff_w:
            shared_w = ode_w.detach().clone()
            ode_w = shared_w

        ana_w_step = ana_w
        if args.share_rff_w:
            ana_w_step = shared_w
        analytic_prev_model, ana_days_test_list, ana_deaths_test_list, ana_days_train, ana_deaths_train, ana_w, analytic_prev_discrete_steps, analytic_prev_end = (
            analytic_mod.train_nexttask(
                analytic_prev_model,
                ana_days_test_list,
                ana_deaths_test_list,
                ana_days_train,
                ana_deaths_train,
                device,
                task_id,
                args.inducing_size,
                args.num_epochs,
                args.lr,
                args.step_size,
                ana_w_step,
                analytic_prev_discrete_steps,
                analytic_prev_end,
            )
        )
        _assert_kernel_fixed(analytic_prev_model, fixed_lengthscale, fixed_variance)
        if args.share_rff_w:
            ana_w = shared_w

    # Build common test grid (unchanged: all task train+test time union from the current script).
    X_test = _build_sorted_test_grid(ode_days_train, ode_days_test_list, device)

    # Full GP posterior uses all observations from task1~task5 (not split train subset).
    all_days = []
    all_deaths = []
    for task_id in range(1, args.task_end + 1):
        d, y, _ = _load_task_full_series(task_id, device, reference_filename=ref_filename)
        all_days.append(d)
        all_deaths.append(_full_gp_target_transform(y))
    X_train = _ensure_col_vector(torch.cat(all_days, dim=0), device)
    y_train = _ensure_col_vector(torch.cat(all_deaths, dim=0), device)

    kernel_params = {
        "lengthscale": fixed_lengthscale,
        "variance": fixed_variance,
        "noise_variance": fixed_noise_variance,
    }

    mu_p, var_p = full_gp_posterior(X_train, y_train, X_test, kernel_params)
    mu_ode, var_ode = hipp_svpg_ode_posterior(ode_prev_model, X_test, ode_w)
    mu_ana, var_ana = hipp_svpg_analytical_posterior(analytic_prev_model, X_test, ana_w)

    # Sanity checks.
    assert mu_p.shape == var_p.shape == mu_ode.shape == var_ode.shape == mu_ana.shape == var_ana.shape

    kl_ode = kl_gaussian_1d(mu_q=mu_ode, var_q=var_ode, mu_p=mu_p, var_p=var_p)
    kl_analytical = kl_gaussian_1d(mu_q=mu_ana, var_q=var_ana, mu_p=mu_p, var_p=var_p)
    spike_intervals, spike_threshold = _detect_kl_spike_intervals(kl_ode, kl_analytical, quantile=0.95)

    out_dir = Path(args.shared_dir) / f"models{args.seed_num}"
    plot_path = out_dir / f"kl_vs_timestep_task{args.task_end}.png"
    csv_path = out_dir / f"posterior_kl_task{args.task_end}.csv"
    var_plot_path = out_dir / f"variance_compare_task{args.task_end}.png"
    var_ratio_plot_path = out_dir / f"log_var_ratio_task{args.task_end}.png"
    mean_err_plot_path = out_dir / f"mean_abs_error_task{args.task_end}.png"

    x_test_cpu = X_test.squeeze(-1).detach().cpu()
    mu_p_cpu = mu_p.detach().cpu()
    mu_ode_cpu = mu_ode.detach().cpu()
    mu_ana_cpu = mu_ana.detach().cpu()
    var_p_cpu = var_p.detach().cpu()
    var_ode_cpu = var_ode.detach().cpu()
    var_ana_cpu = var_ana.detach().cpu()
    kl_ode_cpu = kl_ode.detach().cpu()
    kl_ana_cpu = kl_analytical.detach().cpu()

    _plot_kl(
        plot_path,
        x_test_cpu.numpy(),
        kl_ode_cpu.numpy(),
        kl_ana_cpu.numpy(),
    )
    _plot_variance_comparison(
        var_plot_path,
        x_test_cpu.numpy(),
        var_p_cpu.numpy(),
        var_ode_cpu.numpy(),
        var_ana_cpu.numpy(),
    )
    _plot_log_variance_ratio(
        var_ratio_plot_path,
        x_test_cpu.numpy(),
        var_p_cpu,
        var_ode_cpu,
        var_ana_cpu,
    )
    _plot_mean_abs_error_with_kl_spikes(
        mean_err_plot_path,
        x_test_cpu.numpy(),
        mu_p_cpu,
        mu_ode_cpu,
        mu_ana_cpu,
        spike_intervals,
        spike_threshold,
    )
    _write_kl_csv(
        csv_path,
        x_test_cpu,
        kl_ode_cpu,
        kl_ana_cpu,
        mu_p_cpu,
        var_p_cpu,
        mu_ode_cpu,
        var_ode_cpu,
        mu_ana_cpu,
        var_ana_cpu,
    )

    print("Shared fixed hyperparameters")
    print("  full_gp_target_space: raw")
    print(f"  lengthscale: {fixed_lengthscale:.8f}")
    print(f"  variance: {fixed_variance:.8f}")
    print(f"  noise_variance (full GP): {fixed_noise_variance:.8f}")
    if args.noise_variance is None:
        print(f"  (learned from Task1 Full GP; init={args.full_gp_noise_init})")
    else:
        print("  (fixed by --noise_variance)")
    print()
    print("KL summary")
    print(f"  ODE mean KL: {kl_ode.mean().item():.8f}")
    print(f"  ODE max  KL: {kl_ode.max().item():.8f}")
    print(f"  Analytical mean KL: {kl_analytical.mean().item():.8f}")
    print(f"  Analytical max  KL: {kl_analytical.max().item():.8f}")
    print()
    print(f"Saved KL plot: {plot_path}")
    print(f"Saved variance comparison plot: {var_plot_path}")
    print(f"Saved log-variance-ratio plot: {var_ratio_plot_path}")
    print(f"Saved mean-abs-error plot: {mean_err_plot_path}")
    if len(spike_intervals) == 0:
        print("KL spike intervals (95% quantile): none")
    else:
        spike_ranges = ", ".join(
            [
                f"[idx:{s}-{e}, t:{float(x_test_cpu[s].item()):.4f}-{float(x_test_cpu[e].item()):.4f}]"
                for s, e in spike_intervals
            ]
        )
        print(f"KL spike intervals (95% quantile, thr={spike_threshold:.6f}): {spike_ranges}")
    print(f"Saved KL table: {csv_path}")


if __name__ == "__main__":
    main()

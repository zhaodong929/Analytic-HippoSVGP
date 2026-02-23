import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm

from hipposvgp.hsvgp_analytic import HIPPOOSVGP
from hipposvgp.hippo import HiPPO_LegS
from hipposvgp.kernel import FreqSpaceKernel
from hipposvgp.likelihood import GaussianLikelihood
from scripts.gpvae.data import ERA5Dataset


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


def _append_metrics_row(path, header, row):
    write_header = not os.path.exists(path)
    with open(path, "a+") as f:
        if write_header:
            f.write(header + "\n")
        f.write(row + "\n")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _task_dirs(base_root):
    items = []
    for name in os.listdir(base_root):
        if name.startswith("task_"):
            try:
                idx = int(name.split("_")[1])
            except (IndexError, ValueError):
                continue
            items.append((idx, os.path.join(base_root, name)))
    return [p for _, p in sorted(items, key=lambda x: x[0])]


def _collect_task_data(task_folder, dim_idx, device, max_sequences=None, max_points=None, seed=0):
    dataset = ERA5Dataset(task_folder)
    if max_sequences is not None:
        dataset_len = min(len(dataset), max_sequences)
    else:
        dataset_len = len(dataset)
    x_train_list, y_train_list = [], []
    x_test_list, y_test_list = [], []
    for i in range(dataset_len):
        data_train, data_val, data_test, time_train, time_val, time_test, _ = dataset[i]
        y_train = torch.tensor(data_train[:, dim_idx], dtype=torch.float64, device=device)
        y_test = torch.tensor(data_test[:, dim_idx], dtype=torch.float64, device=device)
        x_train = torch.tensor(time_train, dtype=torch.float64, device=device)
        x_test = torch.tensor(time_test, dtype=torch.float64, device=device)
        x_train_list.append(x_train)
        y_train_list.append(y_train)
        x_test_list.append(x_test)
        y_test_list.append(y_test)

    x_train = torch.cat(x_train_list, dim=0).unsqueeze(1)
    y_train = torch.cat(y_train_list, dim=0).unsqueeze(1)
    x_test = torch.cat(x_test_list, dim=0).unsqueeze(1)
    y_test = torch.cat(y_test_list, dim=0).unsqueeze(1)

    if max_points is not None and x_train.shape[0] > max_points:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        idx = torch.randperm(x_train.shape[0], generator=gen, device=device)[:max_points]
        x_train = x_train[idx]
        y_train = y_train[idx]
    if max_points is not None and x_test.shape[0] > max_points:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed + 1)
        idx = torch.randperm(x_test.shape[0], generator=gen, device=device)[:max_points]
        x_test = x_test[idx]
        y_test = y_test[idx]
    return x_train, y_train, x_test, y_test


def train_task1(task_folder, device, dim_idx, inducing_size, num_epochs, lr, step_size, max_sequences, max_points, seed):
    x_train, y_train, x_test, y_test = _collect_task_data(
        task_folder, dim_idx, device, max_sequences=max_sequences, max_points=max_points, seed=seed
    )

    prev_discrete_steps = 0
    maxT = torch.max(x_train).item() - torch.min(x_train).item()
    start = torch.min(x_train).item()
    num_discrete_steps = math.ceil(maxT / step_size)
    end = start + num_discrete_steps * step_size

    kernel = FreqSpaceKernel(num_components=rff_sample_size, device=device)
    likelihood = GaussianLikelihood(variance=0.01)

    model = HIPPOOSVGP(
        kernel=kernel,
        likelihood=likelihood,
        start=start,
        end=end,
        rff_sample_size=rff_sample_size,
        hippo=hippo,
        num_discrete_steps=num_discrete_steps,
        inducing_size=inducing_size,
        device=device,
        flag_update_kernel=False,
        use_analytic_Z=True,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(num_epochs), desc="task1", leave=False):
        model.train()
        optimizer.zero_grad()
        try:
            loss, ve_mean, kl = model.ELBO(x_train, y_train, None)
            loss = -loss
        except torch.linalg.LinAlgError as e:
            print(f"Caught LinAlgError: {e}")
            print(f"Breaking training loop at epoch {epoch}")
            break
        loss.backward()
        optimizer.step()

    w = model.kernel.sample_from_spectral(rff_sample_size).unsqueeze(0).detach()
    mean, var = model.pred_f(x_test, w)
    pred_mean = _predict_mean(model, mean)
    rmse, r2 = _rmse_r2(y_test, pred_mean)

    f_samples = mean + torch.sqrt(var) * torch.randn(1000, var.shape[0], var.shape[1], device=device)
    nlpd = -torch.log(
        model.likelihood.log_prob(
            f_samples.view(-1, 1),
            y_test.unsqueeze(0).tile(1000, 1, 1).view(-1, 1),
        )
        .view(1000, -1)
        .exp()
        .mean(dim=0)
    ).mean()

    print(f"[task1] test rmse: {rmse}, r2: {r2}, nlpd: {nlpd}")
    return model, [x_test], [y_test], x_train, y_train, w, num_discrete_steps, end


def train_nexttask(task_folder, prev_model, x_test_list, y_test_list, prev_x_train, prev_y_train,
                   device, dim_idx, task_id, inducing_size, num_epochs, lr, step_size, w,
                   prev_discrete_steps, prev_end, max_sequences, max_points, seed):
    mv_old = prev_model.mv.clone().detach().to(device=device)
    Lv_old = prev_model.Lv.clone().detach().to(device=device)
    Z_old, _, Kuu_old, _, _ = prev_model.Kuu_se(w)
    Kuu_old *= torch.exp(prev_model.kernel.log_sf)

    likelihood = prev_model.likelihood

    x_train, y_train, x_test, y_test = _collect_task_data(
        task_folder, dim_idx, device, max_sequences=max_sequences, max_points=max_points, seed=seed
    )

    start_next = prev_end
    end_next = torch.max(x_train).item()
    num_discrete_steps = math.ceil((end_next - start_next) / step_size)
    end_next = start_next + num_discrete_steps * step_size

    model = HIPPOOSVGP(
        kernel=FreqSpaceKernel(num_components=rff_sample_size, device=device),
        likelihood=likelihood,
        start=start_next,
        end=end_next,
        rff_sample_size=rff_sample_size,
        hippo=hippo,
        num_discrete_steps=num_discrete_steps,
        prev_discrete_steps=prev_discrete_steps,
        inducing_size=inducing_size,
        device=device,
        num_inducing_old=inducing_size,
        mv_old=mv_old,
        Lv_old=Lv_old,
        Kaa_old=Kuu_old,
        Z_old=Z_old,
        flag_update_kernel=False,
        use_analytic_Z=True,
    ).to(device)

    prev_discrete_steps += num_discrete_steps

    model.kernel.log_sf.requires_grad = False
    if hasattr(model.likelihood, "log_variance"):
        model.likelihood.log_variance.requires_grad = False
    model.mv = nn.Parameter(torch.tensor(prev_model.mv.clone().detach().cpu().numpy(), device=device))
    model.Lv = nn.Parameter(torch.tensor(prev_model.Lv.clone().detach().cpu().numpy(), device=device))

    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(num_epochs), desc=f"task{task_id}", leave=False):
        model.train()
        optimizer.zero_grad()
        try:
            if epoch == 0:
                loss, ve_mean, kl = model.ELBO(x_train, y_train, w, recompute_k=True, cache_k=True)
            else:
                loss, ve_mean, kl = model.ELBO(x_train, y_train, w, recompute_k=False, cache_k=False)
            loss = -loss
        except torch.linalg.LinAlgError as e:
            print(f"Caught LinAlgError: {e}")
            print(f"Breaking training loop at epoch {epoch}")
            break
        loss.backward()
        optimizer.step()

    x_test_list.append(x_test)
    y_test_list.append(y_test)

    count = 0
    for x_test in x_test_list:
        mean, var = model.pred_f(x_test, w)
        pred_mean = _predict_mean(model, mean)
        rmse, r2 = _rmse_r2(y_test_list[count], pred_mean)

        f_samples = mean + torch.sqrt(var) * torch.randn(1000, var.shape[0], var.shape[1], device=device)
        nlpd = -torch.log(
            model.likelihood.log_prob(
                f_samples.view(-1, 1),
                y_test_list[count].unsqueeze(0).tile(1000, 1, 1).view(-1, 1),
            )
            .view(1000, -1)
            .exp()
            .mean(dim=0)
        ).mean()
        print(f"[task{task_id} eval{count+1}] test rmse: {rmse}, r2: {r2}, nlpd: {nlpd}")
        count += 1

    x_train = torch.concatenate([prev_x_train, x_train])
    y_train = torch.concatenate([prev_y_train, y_train])
    return model, x_test_list, y_test_list, x_train, y_train, w, prev_discrete_steps, end_next


if __name__ == "__main__":
    parser = argparse.ArgumentParser("HiPPO-SVGP FreqSpace on ERA5 (per-dimension)")
    parser.add_argument("--seed_num", type=int, default=0)
    parser.add_argument("--dim_idx", type=int, default=0)
    parser.add_argument("--inducing_size", type=int, default=15)
    parser.add_argument("--num_epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--step_size", type=float, default=4.65)
    parser.add_argument("--num_components", type=int, default=1000)
    parser.add_argument("--max_points", type=int, default=None)
    parser.add_argument("--max_sequences", type=int, default=None)
    args = parser.parse_args()

    seed_everything(args.seed_num)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inducing_size = args.inducing_size
    num_epochs = args.num_epochs
    lr = args.lr
    step_size = args.step_size
    rff_sample_size = args.num_components

    hippo = HiPPO_LegS(inducing_size, device)

    base_root = "data/era5/processed_timeseries_4"
    task_folders = _task_dirs(base_root)
    if not task_folders:
        raise RuntimeError(f"No task_* folders found under {base_root}")

    out_dir = (
        "./ohsgp_freqspace/era5_dataset/models"
        + str(args.seed_num)
        + "/dim"
        + str(args.dim_idx)
        + "/"
    )
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, f"metrics_dim{args.dim_idx}_seed{args.seed_num}.csv")

    prev_model, x_test_list, y_test_list, x_train, y_train, w, prev_discrete_steps, prev_end = train_task1(
        task_folders[0], device, args.dim_idx, inducing_size, num_epochs, lr, step_size,
        args.max_sequences, args.max_points, args.seed_num
    )

    # log task1 metrics and plot
    mean, var = prev_model.pred_f(x_test_list[0], w)
    pred_mean = _predict_mean(prev_model, mean)
    rmse, r2 = _rmse_r2(y_test_list[0], pred_mean)
    f_samples = mean + torch.sqrt(var) * torch.randn(1000, var.shape[0], var.shape[1], device=device)
    nlpd = -torch.log(
        prev_model.likelihood.log_prob(
            f_samples.view(-1, 1),
            y_test_list[0].unsqueeze(0).tile(1000, 1, 1).view(-1, 1),
        )
        .view(1000, -1)
        .exp()
        .mean(dim=0)
    ).mean()
    _append_metrics_row(
        metrics_path,
        "eval_task,after_task,rmse,r2,nlpd",
        f"1,1,{rmse},{r2},{nlpd}",
    )

    # visualization for task1
    x_plot = torch.linspace(torch.min(x_train), torch.max(x_train), 1000, device=device).unsqueeze(1)
    mean_plot, var_plot = prev_model.pred_f(x_plot, w)
    mean_plot = mean_plot.detach().cpu().numpy().flatten()
    std_plot = np.sqrt(var_plot.detach().cpu().numpy().flatten())
    plt.figure(figsize=(12, 5))
    plt.plot(x_plot.detach().cpu().numpy().flatten(), mean_plot, label="Predicted Mean", color="red")
    plt.fill_between(
        x_plot.detach().cpu().numpy().flatten(),
        mean_plot - 2 * std_plot,
        mean_plot + 2 * std_plot,
        color="blue",
        alpha=0.3,
        label="95% CI",
    )
    plt.scatter(x_train.detach().cpu().numpy(), y_train.detach().cpu().numpy(), s=5, c="black", label="train")
    plt.scatter(x_test_list[0].detach().cpu().numpy(), y_test_list[0].detach().cpu().numpy(), s=5, c="blue", label="test")
    plt.xlabel("time")
    plt.ylabel("value")
    plt.title(f"ERA5 FreqSpace (dim {args.dim_idx}) task1")
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"dim{args.dim_idx}_task1.png"), format="png")
    plt.close()

    for task_idx, folder in enumerate(task_folders[1:], start=2):
        prev_model, x_test_list, y_test_list, x_train, y_train, w, prev_discrete_steps, prev_end = train_nexttask(
            folder,
            prev_model,
            x_test_list,
            y_test_list,
            x_train,
            y_train,
            device,
            args.dim_idx,
            task_idx,
            inducing_size,
            num_epochs,
            lr,
            step_size,
            w,
            prev_discrete_steps,
            prev_end,
            args.max_sequences,
            args.max_points,
            args.seed_num,
        )

        # log metrics for each eval task
        count = 0
        for x_test in x_test_list:
            mean, var = prev_model.pred_f(x_test, w)
            pred_mean = _predict_mean(prev_model, mean)
            rmse, r2 = _rmse_r2(y_test_list[count], pred_mean)
            f_samples = mean + torch.sqrt(var) * torch.randn(1000, var.shape[0], var.shape[1], device=device)
            nlpd = -torch.log(
                prev_model.likelihood.log_prob(
                    f_samples.view(-1, 1),
                    y_test_list[count].unsqueeze(0).tile(1000, 1, 1).view(-1, 1),
                )
                .view(1000, -1)
                .exp()
                .mean(dim=0)
            ).mean()
            _append_metrics_row(
                metrics_path,
                "eval_task,after_task,rmse,r2,nlpd",
                f"{count+1},{task_idx},{rmse},{r2},{nlpd}",
            )
            count += 1

        # visualization for current task
        x_plot = torch.linspace(torch.min(x_train), torch.max(x_train), 1000, device=device).unsqueeze(1)
        mean_plot, var_plot = prev_model.pred_f(x_plot, w)
        mean_plot = mean_plot.detach().cpu().numpy().flatten()
        std_plot = np.sqrt(var_plot.detach().cpu().numpy().flatten())
        plt.figure(figsize=(12, 5))
        plt.plot(x_plot.detach().cpu().numpy().flatten(), mean_plot, label="Predicted Mean", color="red")
        plt.fill_between(
            x_plot.detach().cpu().numpy().flatten(),
            mean_plot - 2 * std_plot,
            mean_plot + 2 * std_plot,
            color="blue",
            alpha=0.3,
            label="95% CI",
        )
        plt.scatter(x_train.detach().cpu().numpy(), y_train.detach().cpu().numpy(), s=5, c="black", label="train")
        plt.scatter(x_test_list[-1].detach().cpu().numpy(), y_test_list[-1].detach().cpu().numpy(), s=5, c="blue", label="test")
        plt.xlabel("time")
        plt.ylabel("value")
        plt.title(f"ERA5 FreqSpace (dim {args.dim_idx}) task{task_idx}")
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"dim{args.dim_idx}_task{task_idx}.png"), format="png")
        plt.close()

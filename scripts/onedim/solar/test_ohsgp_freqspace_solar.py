import numpy as np
import torch
import sys
sys.path.append("../../")
import random
import matplotlib.pyplot as plt
from torch.optim import Adam
import torch.nn as nn
import os
import argparse
import math
from tqdm import tqdm

from hipposvgp.hsvgp_analytic import HIPPOOSVGP
from hipposvgp.hippo import HiPPO_LegS
from hipposvgp.likelihood import GaussianLikelihood
from hipposvgp.kernel import FreqSpaceKernel


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


def setup():
    parser = argparse.ArgumentParser("HiPPO-SVGP FreqSpace on Solar")
    parser.add_argument("--seed_num", type=int, default=0)
    parser.add_argument("--inducing_size", type=int, default=100)
    parser.add_argument("--num_components", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--step_size", type=float, default=2.45)
    args = parser.parse_args()
    return args


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_solar_data(seed):
    data_train = np.load("data/solar_data_train.npy")
    data_test = np.load("data/solar_data_test.npy")
    data = np.vstack([data_train, data_test])
    ind = np.argsort(data[:, 0])
    data = data[ind]

    np.random.seed(seed)
    ind_test = np.random.choice(np.arange(data.shape[0]), size=91, replace=False)
    ind_train = [i for i in range(data.shape[0]) if i not in set(ind_test)]
    data_test = data[ind_test]
    data_train = data[ind_train]

    ind = np.argsort(data_train[:, 0])
    data_train = data_train[ind]
    return data_train, data_test


def split_tasks(x, y, num_tasks=10):
    x_list = []
    y_list = []
    split_size = x.shape[0] // num_tasks
    for i in range(num_tasks - 1):
        x_list.append(x[i * split_size : (i + 1) * split_size])
        y_list.append(y[i * split_size : (i + 1) * split_size])
    x_list.append(x[(num_tasks - 1) * split_size :])
    y_list.append(y[(num_tasks - 1) * split_size :])
    return x_list, y_list


def get_task_test(x_t, y_t, x_max):
    mask = x_t.flatten() < x_max
    return x_t[mask], y_t[mask]


def train_task1(device, inducing_size, num_epochs, lr, step_size, rff_sample_size, hippo, x_list, y_list, x_t, y_t):
    prev_discrete_steps = 0
    maxT = torch.max(x_list[0]).item() - torch.min(x_list[0]).item()
    start = torch.min(x_list[0]).item()
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
            loss, ve_mean, kl = model.ELBO(x_list[0], y_list[0], None)
            loss = -loss
        except torch.linalg.LinAlgError as e:
            print(f"Caught LinAlgError: {e}")
            print(f"Breaking training loop at epoch {epoch}")
            break
        loss.backward()
        optimizer.step()

    w = model.kernel.sample_from_spectral(rff_sample_size).unsqueeze(0).detach()
    x_test_task, y_test_task = get_task_test(x_t, y_t, torch.max(x_list[0]).item())
    return model, [x_test_task], [y_test_task], x_list[0], y_list[0], w, num_discrete_steps, end


def train_nexttask(prev_model, x_test_list, y_test_list, prev_x_train, prev_y_train,
                   device, task_id, inducing_size, num_epochs, lr, step_size, rff_sample_size,
                   hippo, w, prev_discrete_steps, prev_end, x_list, y_list, x_t, y_t):
    mv_old = prev_model.mv.clone().detach().to(device=device)
    Lv_old = prev_model.Lv.clone().detach().to(device=device)
    Z_old, _, Kuu_old, _, _ = prev_model.Kuu_se(w)
    Kuu_old *= torch.exp(prev_model.kernel.log_sf)

    start_next = prev_end
    end_next = torch.max(x_list[task_id - 1]).item()
    num_discrete_steps = math.ceil((end_next - start_next) / step_size)
    end_next = start_next + num_discrete_steps * step_size

    model = HIPPOOSVGP(
        kernel=FreqSpaceKernel(num_components=rff_sample_size, device=device),
        likelihood=prev_model.likelihood,
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
                loss, ve_mean, kl = model.ELBO(x_list[task_id - 1], y_list[task_id - 1], w, recompute_k=True, cache_k=True)
            else:
                loss, ve_mean, kl = model.ELBO(x_list[task_id - 1], y_list[task_id - 1], w, recompute_k=False, cache_k=False)
            loss = -loss
        except torch.linalg.LinAlgError as e:
            print(f"Caught LinAlgError: {e}")
            print(f"Breaking training loop at epoch {epoch}")
            break
        loss.backward()
        optimizer.step()

    x_test_task, y_test_task = get_task_test(x_t, y_t, torch.max(x_list[task_id - 1]).item())
    x_test_list.append(x_test_task)
    y_test_list.append(y_test_task)

    x_train = torch.concatenate([prev_x_train, x_list[task_id - 1]])
    y_train = torch.concatenate([prev_y_train, y_list[task_id - 1]])
    return model, x_test_list, y_test_list, x_train, y_train, w, prev_discrete_steps, end_next


def eval_and_log(model, w, x_test_list, y_test_list, metrics_path, after_task):
    for idx, x_test in enumerate(x_test_list):
        mean, var = model.pred_f(x_test, w)
        pred_mean = _predict_mean(model, mean)
        rmse, r2 = _rmse_r2(y_test_list[idx], pred_mean)
        f_samples = mean + torch.sqrt(var) * torch.randn(1000, var.shape[0], var.shape[1], device=mean.device)
        nlpd = -torch.log(
            model.likelihood.log_prob(
                f_samples.view(-1, 1),
                y_test_list[idx].unsqueeze(0).tile(1000, 1, 1).view(-1, 1),
            )
            .view(1000, -1)
            .exp()
            .mean(dim=0)
        ).mean()
        _append_metrics_row(
            metrics_path,
            "eval_task,after_task,rmse,r2,nlpd",
            f"{idx+1},{after_task},{rmse},{r2},{nlpd}",
        )


def plot_task(model, w, x_plot, x_train, y_train, x_test, y_test, out_path, title):
    mean_plot, var_plot = model.pred_f(x_plot, w)
    mean_plot = mean_plot.detach().cpu().numpy().flatten()
    std_plot = np.sqrt(var_plot.detach().cpu().numpy().flatten())
    x_plot_np = x_plot.detach().cpu().numpy().flatten()
    plt.figure(figsize=(12, 5))
    plt.plot(x_plot_np, mean_plot, label="Predicted Mean", color="red")
    plt.fill_between(
        x_plot_np,
        mean_plot - 2 * std_plot,
        mean_plot + 2 * std_plot,
        color="blue",
        alpha=0.3,
        label="95% CI",
    )
    plt.scatter(x_train.detach().cpu().numpy().flatten(), y_train.detach().cpu().numpy().flatten(), s=5, c="black", label="train")
    plt.scatter(x_test.detach().cpu().numpy().flatten(), y_test.detach().cpu().numpy().flatten(), s=5, c="blue", label="test")
    plt.xlabel("time")
    plt.ylabel("value")
    plt.title(title)
    plt.legend()
    plt.savefig(out_path, format="png")
    plt.close()


if __name__ == "__main__":
    args = setup()
    seed_everything(args.seed_num)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_train, data_test = load_solar_data(args.seed_num)
    x = torch.tensor(np.expand_dims(data_train[:, 0], 1), dtype=torch.float64, device=device)
    y = torch.tensor(np.expand_dims(data_train[:, 1], 1), dtype=torch.float64, device=device)
    x_t = torch.tensor(np.expand_dims(data_test[:, 0], 1), dtype=torch.float64, device=device)
    y_t = torch.tensor(np.expand_dims(data_test[:, 1], 1), dtype=torch.float64, device=device)

    x_list, y_list = split_tasks(x, y, num_tasks=10)
    hippo = HiPPO_LegS(args.inducing_size, device)

    out_dir = "./ohsgp_freqspace/solar_dataset/models" + str(args.seed_num) + "/"
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.csv")

    prev_model, x_test_list, y_test_list, x_train, y_train, w, prev_discrete_steps, prev_end = train_task1(
        device,
        args.inducing_size,
        args.num_epochs,
        args.lr,
        args.step_size,
        args.num_components,
        hippo,
        x_list,
        y_list,
        x_t,
        y_t,
    )

    eval_and_log(prev_model, w, x_test_list, y_test_list, metrics_path, after_task=1)
    x_plot = torch.linspace(torch.min(x_train), torch.max(x_train), 1000, device=device).unsqueeze(1)
    plot_task(
        prev_model,
        w,
        x_plot,
        x_train,
        y_train,
        x_test_list[-1],
        y_test_list[-1],
        os.path.join(out_dir, "solar_task1.png"),
        "Solar FreqSpace task1",
    )

    for task_id in range(2, 11):
        prev_model, x_test_list, y_test_list, x_train, y_train, w, prev_discrete_steps, prev_end = train_nexttask(
            prev_model,
            x_test_list,
            y_test_list,
            x_train,
            y_train,
            device,
            task_id,
            args.inducing_size,
            args.num_epochs,
            args.lr,
            args.step_size,
            args.num_components,
            hippo,
            w,
            prev_discrete_steps,
            prev_end,
            x_list,
            y_list,
            x_t,
            y_t,
        )
        eval_and_log(prev_model, w, x_test_list, y_test_list, metrics_path, after_task=task_id)
        x_plot = torch.linspace(torch.min(x_train), torch.max(x_train), 1000, device=device).unsqueeze(1)
        plot_task(
            prev_model,
            w,
            x_plot,
            x_train,
            y_train,
            x_test_list[-1],
            y_test_list[-1],
            os.path.join(out_dir, f"solar_task{task_id}.png"),
            f"Solar FreqSpace task{task_id}",
        )

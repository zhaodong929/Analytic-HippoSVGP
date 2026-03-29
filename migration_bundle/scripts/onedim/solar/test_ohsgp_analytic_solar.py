import argparse
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from hipposvgp.hippo import HiPPO_LegS
from hipposvgp.sgpr_analytic import (
    ANALYTIC_FIXED_CFG,
    HIPPO_OSGPR_VFE_ANALYTIC,
    HIPPO_SVGP_ANALYTIC,
    apply_analytic_fixed_config,
    sample_w,
)
from scripts.onedim.solar.baseline import FullGP


def setup():
    parser = argparse.ArgumentParser("OHSGPR Analytic (module version)")
    parser.add_argument("--seed_num", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device. 'auto' uses cuda if available, otherwise cpu.",
    )
    parser.add_argument("--data", type=str, default="solar")
    parser.add_argument("--kernel_type", type=str, default="se")
    parser.add_argument("--inducing_size", type=int, default=100)
    parser.add_argument("--rff_sample_size", type=int, default=1000)
    parser.add_argument("--rff_sample_size_qb", type=int, default=5000)
    parser.add_argument("--rff_sample_size_pred", type=int, default=5000)
    parser.add_argument("--step_size", type=float, default=2.45)
    parser.add_argument("--model_full_dir", type=str, default="./models")
    parser.add_argument("--fullgp_ckpt", type=str, default="steps_nll2000")
    parser.add_argument(
        "--hyper_source",
        type=str,
        default="load_fullgp",
        choices=["load_fullgp", "manual"],
    )
    parser.add_argument("--log_ls", type=float, default=1.4108)
    parser.add_argument("--log_sf", type=float, default=-0.5334)
    parser.add_argument("--log_noise", type=float, default=-4.5)
    parser.add_argument(
        "--split_replace",
        type=int,
        default=1,
        choices=[0, 1],
        help="1 keeps ODE script behavior (sampling with replacement); 0 is without replacement.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./ohsgp_analytic/solar_dataset_compare/analytic version",
    )
    parser.add_argument(
        "--phase_origin_mode",
        type=str,
        default="global_start",
        choices=["global_start", "zero"],
        help="Phase origin used in analytic spherical-Bessel construction.",
    )
    parser.add_argument(
        "--spectral_sample_device",
        type=str,
        default="cpu",
        choices=["cpu", "match"],
        help=(
            "Device used to sample spectral frequencies w. "
            "'cpu' improves cross-device reproducibility; 'match' keeps legacy behavior."
        ),
    )
    return parser.parse_args()


def resolve_fixed_hypers(args, x, y, device):
    if args.hyper_source == "manual":
        return args.log_ls, args.log_sf, args.log_noise, "manual"

    ckpt_path = os.path.join(args.model_full_dir, args.fullgp_ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"FullGP checkpoint not found: {ckpt_path}")

    full_gp = FullGP(X=x, y=y, device=device, kernel_type=args.kernel_type).to(device)
    full_gp.X = full_gp.X.to(device=device, dtype=torch.float64)
    full_gp.y = full_gp.y.to(device=device, dtype=torch.float64)
    full_gp.load_state_dict(torch.load(ckpt_path, map_location=device))
    return (
        full_gp.Kernel.log_ls.item(),
        full_gp.Kernel.log_sf.item(),
        full_gp.log_noise.item(),
        f"fullgp_loaded:{ckpt_path}",
    )


def load_dataset(args, device):
    if args.data == "solar":
        data_train = np.load("data/solar_data_train.npy")
        data_test = np.load("data/solar_data_test.npy")
        data = np.vstack([data_train, data_test])
        data = data[np.argsort(data[:, 0])]
        ind_test = np.random.choice(
            np.arange(data.shape[0]), size=91, replace=bool(args.split_replace)
        )
        ind_train = [i for i in range(data.shape[0]) if i not in ind_test]
        data_test = data[ind_test]
        data_train = data[ind_train]
        data_train = data_train[np.argsort(data_train[:, 0])]
        x_test = torch.tensor(np.arange(-200.0, 200.0, 2.0).reshape((-1, 1))).to(device)
    else:
        data = np.load("./baseband.npy")
        y_total = data[:18000]
        x_total = np.expand_dims((1 + np.arange(18000)) / 1800, 1)
        data = np.concatenate([x_total, y_total], 1)
        data_train = data[::2]
        data_test = data[1::2]
        x_test = torch.tensor(np.arange(-0.02, 10.02, 0.002).reshape((-1, 1))).to(device)

    x = torch.tensor(np.expand_dims(data_train[:, 0], 1)).to(device)
    y = torch.tensor(np.expand_dims(data_train[:, 1], 1)).to(device)
    x_t = np.expand_dims(data_test[:, 0], 1)
    y_t = np.expand_dims(data_test[:, 1], 1)
    return x, y, x_t, y_t, x_test


def split_stream(x, y):
    x_list = []
    y_list = []
    split_size = x.shape[0] // 10
    for i in range(9):
        x_list.append(x[i * split_size : (i + 1) * split_size])
        y_list.append(y[i * split_size : (i + 1) * split_size])
    x_list.append(x[9 * split_size :])
    y_list.append(y[9 * split_size :])
    return x_list, y_list


def plot_prediction(args, x_test, x, y, x_t, y_t, y_mean, y_cov, x_list, task_idx):
    plt.figure(figsize=(9, 3))
    plt.plot(
        x_test.detach().cpu().numpy().flatten(),
        y_mean.flatten(),
        label=f"SGP{task_idx}",
        color="orange",
    )
    plt.fill_between(
        x_test.detach().cpu().numpy().flatten(),
        y_mean.flatten() - 1.96 * np.sqrt(np.abs(y_cov).flatten()),
        y_mean.flatten() + 1.96 * np.sqrt(np.abs(y_cov).flatten()),
        alpha=0.3,
        color="orange",
    )
    if args.data == "solar":
        plt.scatter(
            x.detach().cpu().numpy().flatten(),
            y.detach().cpu().numpy().flatten(),
            color="blue",
            label="train",
            marker=".",
        )
        plt.scatter(x_t.flatten(), y_t.flatten(), color="black", label="test", marker=".")
    else:
        plt.scatter(
            x.detach().cpu().numpy().flatten(),
            y.detach().cpu().numpy().flatten(),
            color="blue",
            label="train",
            marker=".",
            s=1,
        )
        plt.scatter(x_t.flatten(), y_t.flatten(), color="black", label="test", marker=".", s=1)

    plt.axvline(x=torch.max(x_list[0]).item(), color="grey", linestyle="--", label="splits")
    for j in range(1, 10):
        plt.axvline(x=torch.max(x_list[j]).item(), color="grey", linestyle="--")
    plt.legend(bbox_to_anchor=(1.01, 0.75), loc="upper left")
    suffix = f"hsgp{task_idx}_analytic_{args.kernel_type}_{args.inducing_size}z.pdf"
    plt.savefig(os.path.join(args.out_dir, suffix), format="pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    args = setup()
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("`--device cuda` requested but CUDA is not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] using {device}")

    np.random.seed(args.seed_num)
    torch.manual_seed(args.seed_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed_num)

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    with open(metrics_path, "w") as f:
        f.write("task,rmse,nll\n")

    with torch.no_grad():
        x, y, x_t, y_t, x_test = load_dataset(args, device)
        x_list, y_list = split_stream(x, y)

        log_ls, log_sf, log_noise, hyper_source = resolve_fixed_hypers(args, x, y, device)
        print(
            "[Hyper] source={}, log_ls={:.4f}, log_sf={:.4f}, log_noise={:.4f}".format(
                hyper_source, log_ls, log_sf, log_noise
            )
        )
        phase_origin = torch.min(x_list[0]).item() if args.phase_origin_mode == "global_start" else 0.0
        print(
            "[Analytic] phase_origin_mode={}, phase_origin={:.6f}, wt_mode={}".format(
                args.phase_origin_mode,
                phase_origin,
                ANALYTIC_FIXED_CFG["globalstart_wt_mode"],
            )
        )

        hippo = HiPPO_LegS(args.inducing_size, device)
        prev_discrete_steps = 0.0
        start_t = time.time()

        maxT = torch.max(x_list[0]).item() - torch.min(x_list[0]).item()
        start = torch.min(x_list[0]).item()
        num_discrete_steps = math.ceil(maxT / args.step_size)
        end = start + num_discrete_steps * args.step_size
        maxT = end - start

        model = HIPPO_SVGP_ANALYTIC(
            X=x_list[0],
            y=y_list[0],
            maxT=maxT,
            inducing_size=args.inducing_size,
            rff_sample_size=args.rff_sample_size,
            num_discrete_steps=num_discrete_steps,
            start=start,
            device=device,
            hippo=hippo,
            kernel_type=args.kernel_type,
        ).to(device)
        apply_analytic_fixed_config(model, phase_origin=phase_origin)
        prev_discrete_steps += num_discrete_steps

        model.log_noise = nn.Parameter(
            torch.tensor(np.array([log_noise]), dtype=torch.float64, device=device),
            requires_grad=False,
        )
        model.Kernel.log_ls = nn.Parameter(
            torch.tensor(np.array([log_ls]), dtype=torch.float64, device=device),
            requires_grad=False,
        )
        model.Kernel.log_sf = nn.Parameter(
            torch.tensor(np.array([log_sf]), dtype=torch.float64, device=device),
            requires_grad=False,
        )

        w = sample_w(model, args.rff_sample_size_qb, sample_device_mode=args.spectral_sample_device)

        y_mean, y_cov = model.pred(x_test, w=w)
        y_mean = y_mean.detach().cpu().numpy()
        y_cov = y_cov.detach().cpu().numpy()
        plot_prediction(args, x_test, x, y, x_t, y_t, y_mean, y_cov, x_list, 1)

        mu_old, Su_old, Kaa_old = model.qb_opt(w)
        Kaa_old = Kaa_old.detach()
        mu_old = mu_old.detach()
        Su_old = Su_old.detach()
        Z_old, _ = model.get_Z(w)
        Z_old = Z_old.detach()

        nll, rmse = model.nll_rmse(
            torch.tensor(x_t[x_t.flatten() < torch.max(x_list[0]).item()]).to(device),
            torch.tensor(y_t[x_t.flatten() < torch.max(x_list[0]).item()]).to(device),
            w=w,
        )
        print("HGP1, rmse = {:.4f}, nll = {:.4f}".format(rmse, nll))
        with open(metrics_path, "a") as f:
            f.write("1,{:.8f},{:.8f}\n".format(rmse.item(), nll.item()))

        for i in range(1, 10):
            start = end
            end = torch.max(x_list[i]).item()
            num_discrete_steps = math.ceil((end - start) / args.step_size)
            end = start + num_discrete_steps * args.step_size

            cur_model = HIPPO_OSGPR_VFE_ANALYTIC(
                X=x_list[i],
                y=y_list[i],
                prev_discrete_steps=prev_discrete_steps,
                mu_old=mu_old,
                Su_old=Su_old,
                Kaa_old=Kaa_old,
                Z_old=Z_old,
                end=end,
                start=start,
                rff_sample_size=args.rff_sample_size,
                hippo=hippo,
                num_discrete_steps=num_discrete_steps,
                inducing_size=args.inducing_size,
                device=device,
                kernel_type=args.kernel_type,
            ).to(device)
            apply_analytic_fixed_config(cur_model, phase_origin=phase_origin)
            prev_discrete_steps += num_discrete_steps

            cur_model.log_noise = nn.Parameter(
                torch.tensor(np.array([log_noise]), dtype=torch.float64, device=device),
                requires_grad=False,
            )
            cur_model.Kernel.log_ls = nn.Parameter(
                torch.tensor(np.array([log_ls]), dtype=torch.float64, device=device),
                requires_grad=False,
            )
            cur_model.Kernel.log_sf = nn.Parameter(
                torch.tensor(np.array([log_sf]), dtype=torch.float64, device=device),
                requires_grad=False,
            )

            y_mean, y_cov = cur_model.pred(x_test, w=w)
            y_mean = y_mean.detach().cpu().numpy()
            y_cov = y_cov.detach().cpu().numpy()
            plot_prediction(args, x_test, x, y, x_t, y_t, y_mean, y_cov, x_list, i + 1)

            mu_old, Su_old, Kaa_old = cur_model.qb_opt(w)
            Kaa_old = Kaa_old.detach()
            mu_old = mu_old.detach()
            Su_old = Su_old.detach()
            Z_old, _ = cur_model.get_Z(w)
            Z_old = Z_old.detach()

            nll, rmse = cur_model.nll_rmse(
                torch.tensor(x_t[x_t.flatten() < torch.max(x_list[i]).item()]).to(device),
                torch.tensor(y_t[x_t.flatten() < torch.max(x_list[i]).item()]).to(device),
                w=w,
            )
            print("HGP{}, rmse = {:.4f}, nll = {:.4f}".format(i + 1, rmse, nll))
            with open(metrics_path, "a") as f:
                f.write("{},{:.8f},{:.8f}\n".format(i + 1, rmse.item(), nll.item()))

        end_t = time.time()
        print("M={}, time={:.3f}s".format(args.inducing_size, end_t - start_t))

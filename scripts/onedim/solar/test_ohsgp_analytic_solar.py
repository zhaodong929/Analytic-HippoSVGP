import argparse
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy import special as ss

from hipposvgp.hippo import HiPPO_LegS
from hipposvgp.sgpr import HIPPO_OSGPR_VFE, HIPPO_SVGP
from scripts.onedim.solar.baseline import FullGP


def setup():
    parser = argparse.ArgumentParser("OHSGPR Analytic (fixed-kernel style)")
    parser.add_argument("--seed_num", type=int, default=0)
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
        "--analytic_z_type",
        type=str,
        default="bessel",
        choices=["bessel"],
        help="Use spherical-Bessel analytic feature construction.",
    )
    parser.add_argument(
        "--phase_origin_mode",
        type=str,
        default="global_start",
        choices=["global_start", "zero"],
        help="Phase origin used in analytic spherical-Bessel construction.",
    )
    return parser.parse_args()


def _spherical_bessel_j(lmax, x):
    # x: [1, R]
    x_np = x.detach().cpu().numpy()
    x_abs = np.abs(x_np)
    x_safe = np.where(x_abs > 1e-8, x_abs, 1.0)

    L = np.arange(lmax + 1, dtype=np.float64).reshape(-1, 1, 1)
    nu = L + 0.5
    pref = np.sqrt(np.pi / (2.0 * x_safe))
    jv = ss.jv(nu, x_safe)
    j = pref * jv

    parity = np.where((L % 2) == 0, 1.0, -1.0)
    sign_factor = np.where(x_np >= 0.0, 1.0, parity)
    j = j * sign_factor

    j0 = np.zeros_like(j)
    j0[0] = 1.0
    j = np.where(x_abs > 1e-8, j, j0)

    return torch.tensor(j, dtype=x.dtype, device=x.device)


def _analytic_get_Z_bessel(model, w):
    rff_sample_size = w.shape[1]
    step_size = model.maxT / model.num_discrete_steps
    w_eff = w * step_size
    t_index = int(getattr(model, "prev_discrete_steps", 0)) + int(model.num_discrete_steps)
    kappa = 0.5 * w_eff * t_index
    phase_origin = torch.as_tensor(
        float(getattr(model, "phase_origin", 0.0)),
        dtype=model.dtype,
        device=model.device,
    )

    L = torch.arange(model.inducing_size, dtype=model.dtype, device=model.device)
    j = _spherical_bessel_j(model.inducing_size - 1, kappa).squeeze(1)
    pref = torch.sqrt(2.0 * L + 1.0)[:, None]
    phase = w * phase_origin + kappa + L[:, None] * math.pi / 2.0
    z_sin = pref * j * torch.sin(phase)
    z_cos = pref * j * torch.cos(phase)
    scale = (1.0 / rff_sample_size) ** 0.5
    Z = scale * torch.cat([z_sin, z_cos], dim=1).T

    time_stamps = model.start + (model.maxT / model.num_discrete_steps) * (
        1
        + torch.arange(model.num_discrete_steps, dtype=model.dtype, device=model.device)
    ).unsqueeze(1)
    return Z, time_stamps


def _analytic_get_Z(model, w):
    return _analytic_get_Z_bessel(model, w)


def _analytic_feature(w, X):
    rff_sample_size = w.shape[1]
    return (1.0 / rff_sample_size) ** 0.5 * torch.cat([torch.sin(w * X), torch.cos(w * X)], 1)


class HIPPO_SVGP_ANALYTIC(HIPPO_SVGP):
    def get_Z(self, w):
        return _analytic_get_Z(self, w)

    def Kuu_se(self, w, b=None, reducevar=True):
        if not reducevar:
            return super().Kuu_se(w, b=b, reducevar=reducevar)
        Z, _ = self.get_Z(w)
        return Z.T @ Z

    def KfuKff_rff_se(self, w, X, b=None, reducevar=True):
        if not reducevar:
            return super().KfuKff_rff_se(w, X, b=b, reducevar=reducevar)
        Z, _ = self.get_Z(w)
        Z_f = _analytic_feature(w, X)
        Kfu = Z_f @ Z
        Kff = Z_f @ Z_f.T
        return Kff, Kfu


class HIPPO_OSGPR_VFE_ANALYTIC(HIPPO_OSGPR_VFE):
    def get_Z(self, w):
        return _analytic_get_Z(self, w)

    def Kuu_se(self, w):
        Z, _ = self.get_Z(w)
        Kuu = Z.T @ Z
        Kuuold = Z.T @ self.Z_old
        Kuolduold = self.Z_old.T @ self.Z_old
        return Kuu, Kuuold, Kuolduold

    def KfuKff_rff_se(self, w, X):
        Z, _ = self.get_Z(w)
        Z_f = _analytic_feature(w, X)
        Kfu = Z_f @ Z
        Kff = Z_f @ Z_f.T
        return Kff, Kfu

    def KuuKfuKff_rff_se(self, w, X):
        Z, _ = self.get_Z(w)
        Z_f = _analytic_feature(w, X)
        Kfu = Z_f @ Z
        Kff = Z_f @ Z_f.T
        Kuu = Z.T @ Z
        Kuuold = Z.T @ self.Z_old
        Kuolduold = self.Z_old.T @ self.Z_old
        return Kff, Kfu, Kuu, Kuuold, Kuolduold


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


if __name__ == "__main__":
    args = setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed_num)
    torch.manual_seed(args.seed_num)

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    with open(metrics_path, "w") as f:
        f.write("task,rmse,nll\n")

    with torch.no_grad():
        if args.data == "solar":
            data_train = np.load("data/solar_data_train.npy")
            data_test = np.load("data/solar_data_test.npy")
            data = np.vstack([data_train, data_test])
            data = data[np.argsort(data[:, 0])]
            ind_test = np.random.choice(
                np.arange(data.shape[0]), size=91, replace=bool(args.split_replace)
            )
            ind_train = []
            for i in range(data.shape[0]):
                if i in ind_test:
                    continue
                ind_train.append(i)
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

        x_list = []
        y_list = []
        split_size = x.shape[0] // 10
        for i in range(9):
            x_list.append(x[i * split_size : (i + 1) * split_size])
            y_list.append(y[i * split_size : (i + 1) * split_size])
        x_list.append(x[9 * split_size :])
        y_list.append(y[9 * split_size :])

        log_ls, log_sf, log_noise, hyper_source = resolve_fixed_hypers(
            args,
            x,
            y,
            device,
        )
        print(
            "[Hyper] source={}, log_ls={:.4f}, log_sf={:.4f}, log_noise={:.4f}".format(
                hyper_source, log_ls, log_sf, log_noise
            )
        )
        print(
            "[Analytic] z_type={}, split_replace={}, seed={}".format(
                args.analytic_z_type, args.split_replace, args.seed_num
            )
        )
        if args.phase_origin_mode == "global_start":
            phase_origin = torch.min(x_list[0]).item()
        else:
            phase_origin = 0.0
        print(
            "[Analytic] phase_origin_mode={}, phase_origin={:.6f}".format(
                args.phase_origin_mode, phase_origin
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

        model1 = HIPPO_SVGP_ANALYTIC(
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
        model1.analytic_z_type = args.analytic_z_type
        model1.phase_origin = phase_origin
        prev_discrete_steps += num_discrete_steps

        model1.log_noise = nn.Parameter(
            torch.tensor(np.array([log_noise]), dtype=torch.float64, device=device),
            requires_grad=False,
        )
        model1.Kernel.log_ls = nn.Parameter(
            torch.tensor(np.array([log_ls]), dtype=torch.float64, device=device),
            requires_grad=False,
        )
        model1.Kernel.log_sf = nn.Parameter(
            torch.tensor(np.array([log_sf]), dtype=torch.float64, device=device),
            requires_grad=False,
        )

        w = model1.Kernel.sample_from_spectral(args.rff_sample_size_qb).unsqueeze(0)

        y_mean, y_cov = model1.pred(x_test, w=w)
        y_mean = y_mean.detach().cpu().numpy()
        y_cov = y_cov.detach().cpu().numpy()

        plt.figure(figsize=(9, 3))
        plt.plot(x_test.detach().cpu().numpy().flatten(), y_mean.flatten(), label="SGP1", color="orange")
        plt.fill_between(
            x_test.detach().cpu().numpy().flatten(),
            (y_mean.flatten() - 1.96 * np.sqrt(np.abs(y_cov).flatten())),
            (y_mean.flatten() + 1.96 * np.sqrt(np.abs(y_cov).flatten())),
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
        for i in range(1, 10):
            plt.axvline(x=torch.max(x_list[i]).item(), color="grey", linestyle="--")
        plt.legend(bbox_to_anchor=(1.01, 0.75), loc="upper left")
        plt.savefig(
            os.path.join(
                args.out_dir,
                "hsgp1_analytic_{}_{}z.pdf".format(args.kernel_type, args.inducing_size),
            ),
            format="pdf",
            bbox_inches="tight",
        )
        plt.close()

        mu_old, Su_old, Kaa_old = model1.qb_opt(w)
        Kaa_old = Kaa_old.detach()
        mu_old = mu_old.detach()
        Su_old = Su_old.detach()
        Z_old, _ = model1.get_Z(w)
        Z_old = Z_old.detach()

        nll, rmse = model1.nll_rmse(
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
            cur_model.analytic_z_type = args.analytic_z_type
            cur_model.phase_origin = phase_origin
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

            plt.figure(figsize=(9, 3))
            plt.plot(
                x_test.detach().cpu().numpy().flatten(),
                y_mean.flatten(),
                label="SGP{}".format(i + 1),
                color="orange",
            )
            plt.fill_between(
                x_test.detach().cpu().numpy().flatten(),
                (y_mean.flatten() - 1.96 * np.sqrt(np.abs(y_cov).flatten())),
                (y_mean.flatten() + 1.96 * np.sqrt(np.abs(y_cov).flatten())),
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
            plt.savefig(
                os.path.join(
                    args.out_dir,
                    "hsgp{}_analytic_{}_{}z.pdf".format(i + 1, args.kernel_type, args.inducing_size),
                ),
                format="pdf",
                bbox_inches="tight",
            )
            plt.close()

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

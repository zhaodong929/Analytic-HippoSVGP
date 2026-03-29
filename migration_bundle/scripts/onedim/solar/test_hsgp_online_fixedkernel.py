import numpy as np
import torch
import matplotlib.pyplot as plt
from scripts.onedim.solar.baseline import FullGP
from hipposvgp.sgpr import HIPPO_SVGP, HIPPO_OSGPR_VFE
from hipposvgp.hippo import HiPPO_LegS
import torch.nn as nn
import time
import math
import argparse
import os


def setup():
    parser = argparse.ArgumentParser("Argument Parser")
    parser.add_argument("--seed_num", type=int, default=0)
    parser.add_argument("--data", type=str, default="solar")
    parser.add_argument("--kernel_type", type=str, default="se")
    parser.add_argument("--inducing_size", type=int, default=100)
    parser.add_argument("--rff_sample_size", type=int, default=1000)
    parser.add_argument("--rff_sample_size_qb", type=int, default=5000)
    parser.add_argument("--rff_sample_size_pred", type=int, default=5000)
    parser.add_argument("--num_iters", type=int, default=5000)
    parser.add_argument("--step_size", type=float, default=2.45)  # 1/16 for audio data
    parser.add_argument("--model_full_dir", type=str, default="./models")
    parser.add_argument("--fullgp_ckpt", type=str, default="steps_nll2000")
    parser.add_argument("--use_hardcoded_hypers", action="store_true")
    parser.add_argument(
        "--split_replace",
        type=int,
        default=1,
        choices=[0, 1],
        help="1 keeps original behavior (sampling with replacement); 0 samples without replacement.",
    )
    parser.add_argument("--out_dir", type=str, default="./ohsgp/solar_dataset/models0")
    args = parser.parse_args()
    return args


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
            ind = np.argsort(data[:, 0])
            data = data[ind]
            ind_test = np.random.choice(
                np.arange(data.shape[0]), size=91, replace=bool(args.split_replace)
            )
            ind_train = []
            for i in range(data.shape[0]):
                if i in ind_test:
                    continue
                else:
                    ind_train.append(i)
            data_test = data[ind_test]
            data_train = data[ind_train]

            ind = np.argsort(data_train[:, 0])
            data_train = data_train[ind]
            x_test = torch.tensor(np.arange(-200.0, 200.0, 2.0).reshape((-1, 1))).to(device)
        else:
            data = np.load("./baseband.npy")  # [352672, 1]
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
        split_size_t = x_t.shape[0] // 10
        for i in range(9):
            x_list.append(x[i * split_size : (i + 1) * split_size])
            y_list.append(y[i * split_size : (i + 1) * split_size])
        x_list.append(x[9 * split_size :])
        y_list.append(y[9 * split_size :])

        # By default load hyperparameters from FullGP checkpoint trained beforehand.
        # Use --use_hardcoded_hypers to keep the legacy constants.
        if args.use_hardcoded_hypers:
            if args.data == "solar":
                log_ls = 1.4108
                log_sf = -0.5334
                log_noise = -4.5
            else:
                log_ls = -3.8722
                log_sf = 0.3478
                log_noise = -4.0
            hyper_source = "hardcoded"
        else:
            ckpt_path = os.path.join(args.model_full_dir, args.fullgp_ckpt)
            fullGP = FullGP(X=x, y=y, device=device, kernel_type=args.kernel_type).to(device)
            fullGP.X = fullGP.X.to(device=device)
            fullGP.y = fullGP.y.to(device=device)
            fullGP.load_state_dict(torch.load(ckpt_path, map_location=device))
            log_ls = fullGP.Kernel.log_ls.item()
            log_sf = fullGP.Kernel.log_sf.item()
            log_noise = fullGP.log_noise.item()
            hyper_source = f"fullgp:{ckpt_path}"
        print(
            "[Hyper] source={}, log_ls={:.4f}, log_sf={:.4f}, log_noise={:.4f}".format(
                hyper_source, log_ls, log_sf, log_noise
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

        model1 = HIPPO_SVGP(
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

        y_test_trained_mean_GP, y_test_trained_cov_GP = model1.pred(x_test, w=w)
        y_test_trained_mean_GP = y_test_trained_mean_GP.detach().cpu().numpy()
        y_test_trained_cov_GP = y_test_trained_cov_GP.detach().cpu().numpy()

        fig = plt.figure(figsize=(9, 3))

        plt.plot(
            x_test.detach().cpu().numpy().flatten(),
            y_test_trained_mean_GP.flatten(),
            label="SGP1",
            color="orange",
        )
        plt.fill_between(
            x_test.detach().cpu().numpy().flatten(),
            (y_test_trained_mean_GP.flatten() - 1.96 * np.sqrt(np.abs(y_test_trained_cov_GP).flatten())),
            (y_test_trained_mean_GP.flatten() + 1.96 * np.sqrt(np.abs(y_test_trained_cov_GP).flatten())),
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
            os.path.join(args.out_dir, "hsgp1_{}_{}z.pdf".format(args.kernel_type, args.inducing_size)),
            format="pdf",
            bbox_inches="tight",
        )

        mu_old, Su_old, Kaa_old = model1.qb_opt(w)
        Kaa_old = Kaa_old.detach()
        mu_old = mu_old.detach()
        Su_old = Su_old.detach()
        Z_old, time_stamps = model1.get_Z(w)
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

            cur_model = HIPPO_OSGPR_VFE(
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

            y_test_trained_mean_GP, y_test_trained_cov_GP = cur_model.pred(x_test, w=w)
            y_test_trained_mean_GP = y_test_trained_mean_GP.detach().cpu().numpy()
            y_test_trained_cov_GP = y_test_trained_cov_GP.detach().cpu().numpy()

            fig = plt.figure(figsize=(9, 3))
            plt.plot(
                x_test.detach().cpu().numpy().flatten(),
                y_test_trained_mean_GP.flatten(),
                label="SGP" + str(i + 1),
                color="orange",
            )
            plt.fill_between(
                x_test.detach().cpu().numpy().flatten(),
                (y_test_trained_mean_GP.flatten() - 1.96 * np.sqrt(np.abs(y_test_trained_cov_GP).flatten())),
                (y_test_trained_mean_GP.flatten() + 1.96 * np.sqrt(np.abs(y_test_trained_cov_GP).flatten())),
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
                os.path.join(args.out_dir, "hsgp{}_{}_{}z.pdf".format(i + 1, args.kernel_type, args.inducing_size)),
                format="pdf",
                bbox_inches="tight",
            )

            mu_old, Su_old, Kaa_old = cur_model.qb_opt(w)
            Kaa_old = Kaa_old.detach()
            mu_old = mu_old.detach()
            Su_old = Su_old.detach()
            Z_old, time_stamps = cur_model.get_Z(w)
            Z_old = Z_old.detach()

            nll, rmse = cur_model.nll_rmse(
                torch.tensor(x_t[x_t.flatten() < torch.max(x_list[i]).item()]).to(device),
                torch.tensor(y_t[x_t.flatten() < torch.max(x_list[i]).item()]).to(device),
                w=w,
            )
            print("HGP" + str(i + 1) + ", rmse = {:.4f}, nll = {:.4f}".format(rmse, nll))
            with open(metrics_path, "a") as f:
                f.write("{},{:.8f},{:.8f}\n".format(i + 1, rmse.item(), nll.item()))

        end_t = time.time()
        print("M={}, time={:.3f}s".format(args.inducing_size, end_t - start_t))
        print("[Train] total_time_sec={:.3f}".format(end_t - start_t))

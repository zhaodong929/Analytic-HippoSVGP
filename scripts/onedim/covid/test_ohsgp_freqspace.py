import numpy as np
import torch
import sys 
sys.path.append("../../")
from hipposvgp.hsvgp_analytic import HIPPOOSVGP
from hipposvgp.hsvgp import HIPPOOSVGP as HIPPOOSVGP_NUM
from hipposvgp.hippo import HiPPO_LegS
from hipposvgp.likelihood import NegativeBinomial
from hipposvgp.kernel import FreqSpaceKernel
from scripts.onedim.covid.data import COVIDDataset
import random
import matplotlib.pyplot as plt
from torch.optim import Adam
from copy import deepcopy
import torch.nn as nn
import os
import argparse
import math

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
    parser=argparse.ArgumentParser('Argument Parser')
    parser.add_argument('--seed_num',type=int,default=0)
    parser.add_argument('--num_components',type=int,default=1000)
    args=parser.parse_args()
    return args

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_task1(device, inducing_size, num_epochs, lr, step_size):
    # load data
    dataset = COVIDDataset("data/covid/processed_timeseries_5tasks/task_1")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for batch in dataloader:
        deaths, days, Rt, Rt_lower, Rt_upper, num_days, f, EpidemicStart, population, filename, start_date, state_index, week_index = batch
        deaths, days, f, pop, EpidemicStart, N, stateindex, week_index = deaths.to(device), days.to(device), f.to(device), population.to(device), EpidemicStart.to(device), num_days.to(device), state_index.to(device), week_index.to(device)
        deaths = deaths.double()
        days = days.double()
        f = f.double()
        pop = pop.double()
        EpidemicStart = EpidemicStart.double()
        N = N.double()
        week_index = week_index.double()
        break

    days = days.permute(1,0)[:int(N[0].item())]
    deaths = deaths.permute(1,0)[:int(N[0].item())]
    seed_everything(42)
    rand_ind = torch.randperm(int(N[0].item()))
    seed_everything(args.seed_num)
    test_size = int(N[0].item()* 0.2)
    days_test = days[rand_ind[:test_size]]
    days_train = days[rand_ind[test_size:]]
    deaths_test = deaths[rand_ind[:test_size]]
    deaths_train = deaths[rand_ind[test_size:]]


    prev_discrete_steps = 0
    maxT = torch.max(days_train).item()-torch.min(days_train).item()
    start = torch.min(days_train).item() 
    num_discrete_steps = math.ceil(maxT / step_size)
    end = start + num_discrete_steps * step_size
    maxT = end - start

    kernel = FreqSpaceKernel(num_components=rff_sample_size, device=device)
    likelihood = NegativeBinomial()
    likelihood = likelihood.to(device)

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

    model_num = HIPPOOSVGP_NUM(
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
    ).to(device)

    with torch.no_grad():
        w = model.kernel.sample_from_spectral(rff_sample_size).unsqueeze(0)

        # Z alignment
        Z_num, Zold_num, _ = model_num.get_Z(w)
        Z_ana, Zold_ana, _ = model.get_Z_analytic(w)

        def stats(name, Z):
            diag = torch.diag(Z.T @ Z)
            print(
                f"{name} norm={Z.norm().item():.6f}, diag mean={diag.mean().item():.6f}, "
                f"min={diag.min().item():.6f}, max={diag.max().item():.6f}"
            )

        print(
            f"rel_diff_Z = {(Z_num - Z_ana).norm().item() / (Z_num.norm().item() + 1e-12):.6f}"
        )
        stats("Z_num", Z_num)
        stats("Z_ana", Z_ana)
        diag_num = torch.diag(Z_num.T @ Z_num)
        diag_ana = torch.diag(Z_ana.T @ Z_ana)
        rel_diag = (diag_ana - diag_num) / (diag_num + 1e-12)
        print("diag_rel_diff (first 10):", rel_diag[:10].detach().cpu().numpy())
        print(
            f"diag_rel_diff mean={rel_diag.mean().item():.6f}, "
            f"min={rel_diag.min().item():.6f}, max={rel_diag.max().item():.6f}"
        )
        corr_num = (Z_num.T @ Z_num) / (diag_num.sqrt().unsqueeze(1) * diag_num.sqrt().unsqueeze(0))
        corr_ana = (Z_ana.T @ Z_ana) / (diag_ana.sqrt().unsqueeze(1) * diag_ana.sqrt().unsqueeze(0))
        corr_diff = corr_ana - corr_num
        corr_mask = ~torch.eye(corr_diff.shape[0], dtype=torch.bool, device=corr_diff.device)
        corr_vals = corr_diff[corr_mask]
        print(
            f"corr_diff mean={corr_vals.mean().item():.6f}, "
            f"abs mean={corr_vals.abs().mean().item():.6f}, "
            f"min={corr_vals.min().item():.6f}, max={corr_vals.max().item():.6f}"
        )

        Kuu_num = Z_num.T @ Z_num
        Kuu_ana = Z_ana.T @ Z_ana
        print(
            f"rel_diff_Kuu = {(Kuu_num - Kuu_ana).norm().item() / (Kuu_num.norm().item() + 1e-12):.6f}"
        )
        print(
            f"Kuu_num trace={Kuu_num.trace().item():.6f}, Kuu_ana trace={Kuu_ana.trace().item():.6f}"
        )
        eig_num = torch.linalg.eigvalsh(Kuu_num)
        eig_ana = torch.linalg.eigvalsh(Kuu_ana)
        print(
            f"Kuu_num eig min={eig_num.min().item():.6f}, max={eig_num.max().item():.6f}"
        )
        print(
            f"Kuu_ana eig min={eig_ana.min().item():.6f}, max={eig_ana.max().item():.6f}"
        )

        if Zold_num is not None and Zold_ana is not None:
            print(
                f"rel_diff_Zold = {(Zold_num - Zold_ana).norm().item() / (Zold_num.norm().item() + 1e-12):.6f}"
            )
            stats("Zold_num", Zold_num)
            stats("Zold_ana", Zold_ana)
            diag_old_num = torch.diag(Zold_num.T @ Zold_num)
            diag_old_ana = torch.diag(Zold_ana.T @ Zold_ana)
            corr_old_num = (Zold_num.T @ Zold_num) / (
                diag_old_num.sqrt().unsqueeze(1) * diag_old_num.sqrt().unsqueeze(0)
            )
            corr_old_ana = (Zold_ana.T @ Zold_ana) / (
                diag_old_ana.sqrt().unsqueeze(1) * diag_old_ana.sqrt().unsqueeze(0)
            )
            corr_old_diff = corr_old_ana - corr_old_num
            corr_old_mask = ~torch.eye(
                corr_old_diff.shape[0], dtype=torch.bool, device=corr_old_diff.device
            )
            corr_old_vals = corr_old_diff[corr_old_mask]
            print(
                f"corr_old_diff mean={corr_old_vals.mean().item():.6f}, "
                f"abs mean={corr_old_vals.abs().mean().item():.6f}, "
                f"min={corr_old_vals.min().item():.6f}, max={corr_old_vals.max().item():.6f}"
            )

    optimizer = Adam(model.parameters(), lr=lr)

    # Initialize the plot
    from tqdm import tqdm
    pbar = tqdm(range(num_epochs), desc="Task1 Training")

    log = []
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        try:
            loss, ve_mean, kl = model.ELBO(days_train, deaths_train, None)
            loss = -loss  # Negate the loss for minimization
        except torch.linalg.LinAlgError as e:
            print(f"Caught LinAlgError: {e}")
            print(f"Breaking training loop at epoch {epoch}")
            break
        loss.backward()
        optimizer.step()

        # Update progress bar
        pbar.set_postfix({
            'ELBO': f'{-loss.item():.2f}',
            'VE': f'{ve_mean.item():.2f}',
            'KL': f'{kl.item():.2f}'
        })
    
    #model.load_state_dict(torch.load('./ohsgp_'+str(inducing_size)+'z_dayu/models'+str(args.seed_num)+'/model1'))
    log.append('log_sf: {}, nu: {}\n'.format(model.kernel.log_sf.item(), model.likelihood._nu.item()))
    
    # compute test metric        
    w = model.kernel.sample_from_spectral(rff_sample_size).unsqueeze(0)
    w = w.detach()
    mean, var = model.pred_f(days_test, w)
    pred_mean = _predict_mean(model, mean)
    rmse, r2 = _rmse_r2(deaths_test, pred_mean)
    f_samples = mean + torch.sqrt(var) * torch.randn(1000, var.shape[0], var.shape[1]).to(device)  # [1000, Ntest, 1]

    nlpd = -torch.log(model.likelihood.log_prob(f_samples.view(-1,1), deaths_test.unsqueeze(0).tile(1000, 1, 1).view(-1,1)).view(1000,-1).exp().mean(dim=(0))).mean()
    log.append('test nlpd after task1: {}\n'.format(nlpd))
    log.append('test rmse after task1: {}\n'.format(rmse))
    log.append('test r2 after task1: {}\n'.format(r2))
    print(f"[task1] test rmse: {rmse}, r2: {r2}, nlpd: {nlpd}")
    
    f_samples = mean + torch.sqrt(var) * torch.randn(30, var.shape[0], var.shape[1]).to(device)  # [10, Ntest, 1]
    y_samples = model.likelihood.sample(f_samples.squeeze(-1), numy_perf=30).to(device)
    y_samples = y_samples.permute((1,0,2))
    y_samples = y_samples.reshape((y_samples.shape[0], -1))

    conf_level = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    ece_list = []
    for conf in conf_level:
        deaths_pred_lower = torch.quantile(y_samples, 0.5-conf/2, dim=1)
        deaths_pred_upper = torch.quantile(y_samples, 0.5+conf/2, dim=1)
        ece_list.append(torch.abs(torch.mean((((((deaths_pred_lower - deaths_test.squeeze(1)) < 0)+0.) + (((deaths_pred_upper - deaths_test.squeeze(1)) > 0)+0.)) > 1)+0.) - conf).item())
    ece = np.mean(np.array(ece_list))
    log.append('test ece after task1: {}\n'.format(ece))
    metrics_path = './ohsgp_freqspace/models'+str(args.seed_num)+'/metrics.csv'
    _append_metrics_row(
        metrics_path,
        "eval_task,after_task,rmse,r2,nlpd,ece",
        f"1,1,{rmse},{r2},{nlpd},{ece}",
    )
    with open('./ohsgp_freqspace/models'+str(args.seed_num)+'/training.log', "a+") as log_file:
    #with open('./training.log', "a+") as log_file:
        log_file.writelines(log)
        log.clear()
    days_plot = torch.linspace(torch.min(days_train), torch.max(days_train), 1000).to(device)
    mean, var = model.pred_f(days_plot.unsqueeze(1), w)

    deaths_pred_mean_plot = mean.detach().cpu().numpy()
    var_plot = var.detach().cpu().numpy()
    plt.figure(figsize=(12, 5))
    plt.plot(days_plot.detach().cpu().numpy(), deaths_pred_mean_plot.flatten(), label="Predicted Deaths Mean OSVGP", color='red')
    deaths_pred_lower_plot = deaths_pred_mean_plot - 2 * np.sqrt(var_plot)
    deaths_pred_upper_plot = deaths_pred_mean_plot + 2 * np.sqrt(var_plot)
    plt.fill_between(days_plot.detach().cpu().numpy(), deaths_pred_lower_plot.flatten(), deaths_pred_upper_plot.flatten(), color='blue', alpha=0.3, label="95% CI")

    # plot death counts
    plt.scatter(days_train.detach().cpu().numpy(), deaths_train.detach().cpu().numpy(), label="train", color='black', alpha=1, s=5)
    plt.scatter(days_test.detach().cpu().numpy(), deaths_test.detach().cpu().numpy(), label="test", color='blue', alpha=1, s=5)

    plt.xlabel("Day")
    plt.ylabel("Predicted Deaths")
    plt.title("Posterior Summary of Predicted Deaths")
    plt.legend()
    plt.savefig('./ohsgp_freqspace/models'+str(args.seed_num)+'/covid_task1.png', format='png')
    #prev_days_test_np = days_test.detach().cpu().numpy()
    #prev_deaths_test_np = deaths_test.detach().cpu().numpy()
    #prev_days_train_np = days_train.detach().cpu().numpy()
    #prev_deaths_train_np = deaths_train.detach().cpu().numpy()
    #np.savez(f'./ohsgp/models'+str(args.seed_num)+'/data1.npz', \
    #         prev_days_test=prev_days_test_np, prev_deaths_test=prev_deaths_test_np, \
    #            prev_days_train=prev_days_train_np, prev_deaths_train=prev_deaths_train_np)
    #torch.save(model.state_dict(), './ohsgp/models'+str(args.seed_num)+'/model1')
    return model, [days_test], [deaths_test], days_train, deaths_train, w, num_discrete_steps, end


#def train_nexttask(prev_model, prev_days_test, prev_deaths_test, prev_days_train, prev_deaths_train, \
#                   device, task_id, inducing_size, num_epochs, lr, step_size, w, prev_discrete_steps, prev_end):
def train_nexttask(prev_model, days_test_list, deaths_test_list, prev_days_train, prev_deaths_train, \
                   device, task_id, inducing_size, num_epochs, lr, step_size, w, prev_discrete_steps, prev_end):
    mv_old = prev_model.mv.clone()
    mv_old = mv_old.clone().detach().to(device=device)
    Lv_old = prev_model.Lv.clone()
    Lv_old = Lv_old.clone().detach().to(device=device)
    Z_old, _, Kuu_old, _, _ = prev_model.Kuu_se(w)
    Kuu_old *= torch.exp(prev_model.kernel.log_sf)
    num_inducing_old = prev_model.inducing_size

    kernel = prev_model.kernel
    likelihood = prev_model.likelihood

    dataset = COVIDDataset("data/covid/processed_timeseries_5tasks/task_"+str(task_id))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for batch in dataloader:
        deaths, days, Rt, Rt_lower, Rt_upper, num_days, f, EpidemicStart, population, filename, start_date, state_index, week_index = batch
        deaths, days, f, pop, EpidemicStart, N, stateindex, week_index = deaths.to(device), days.to(device), f.to(device), population.to(device), EpidemicStart.to(device), num_days.to(device), state_index.to(device), week_index.to(device)
        # move to float64
        deaths = deaths.double()
        days = days.double()
        f = f.double()
        pop = pop.double()
        EpidemicStart = EpidemicStart.double()
        N = N.double()
        break

    days = days.permute(1,0)[:int(N[0].item())]
    deaths = deaths.permute(1,0)[:int(N[0].item())]
    seed_everything(42)
    rand_ind = torch.randperm(int(N[0].item()))
    seed_everything(args.seed_num)
    test_size = int(N[0].item()* 0.2)
    days_test = days[rand_ind[:test_size]]
    days_train = days[rand_ind[test_size:]]
    deaths_test = deaths[rand_ind[:test_size]]
    deaths_train = deaths[rand_ind[test_size:]]


    start_next = prev_end
    end_next = torch.max(days_train).item() 
    num_discrete_steps = math.ceil((end_next-start_next) / step_size)
    end_next = start_next + num_discrete_steps * step_size

    model = HIPPOOSVGP(
        kernel=deepcopy(kernel),
        likelihood=deepcopy(likelihood),
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
        flag_update_kernel=False
    ).to(device)

    if task_id == 2:
        model_num = HIPPOOSVGP_NUM(
            kernel=deepcopy(kernel),
            likelihood=deepcopy(likelihood),
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
        ).to(device)
        with torch.no_grad():
            w_debug = w
            Z_num, Zold_num, _ = model_num.get_Z(w_debug)
            Z_ana, Zold_ana, _ = model.get_Z_analytic(w_debug)
            diag_num = torch.diag(Z_num.T @ Z_num)
            diag_ana = torch.diag(Z_ana.T @ Z_ana)
            corr_num = (Z_num.T @ Z_num) / (diag_num.sqrt().unsqueeze(1) * diag_num.sqrt().unsqueeze(0))
            corr_ana = (Z_ana.T @ Z_ana) / (diag_ana.sqrt().unsqueeze(1) * diag_ana.sqrt().unsqueeze(0))
            corr_diff = corr_ana - corr_num
            corr_mask = ~torch.eye(corr_diff.shape[0], dtype=torch.bool, device=corr_diff.device)
            corr_vals = corr_diff[corr_mask]
            print(
                f"[task2] corr_diff mean={corr_vals.mean().item():.6f}, "
                f"abs mean={corr_vals.abs().mean().item():.6f}, "
                f"min={corr_vals.min().item():.6f}, max={corr_vals.max().item():.6f}"
            )
            if Zold_num is not None and Zold_ana is not None:
                diag_old_num = torch.diag(Zold_num.T @ Zold_num)
                diag_old_ana = torch.diag(Zold_ana.T @ Zold_ana)
                corr_old_num = (Zold_num.T @ Zold_num) / (
                    diag_old_num.sqrt().unsqueeze(1) * diag_old_num.sqrt().unsqueeze(0)
                )
                corr_old_ana = (Zold_ana.T @ Zold_ana) / (
                    diag_old_ana.sqrt().unsqueeze(1) * diag_old_ana.sqrt().unsqueeze(0)
                )
                corr_old_diff = corr_old_ana - corr_old_num
                corr_old_mask = ~torch.eye(
                    corr_old_diff.shape[0], dtype=torch.bool, device=corr_old_diff.device
                )
                corr_old_vals = corr_old_diff[corr_old_mask]
                print(
                    f"[task2] corr_old_diff mean={corr_old_vals.mean().item():.6f}, "
                    f"abs mean={corr_old_vals.abs().mean().item():.6f}, "
                    f"min={corr_old_vals.min().item():.6f}, max={corr_old_vals.max().item():.6f}"
                )

    prev_discrete_steps += num_discrete_steps

    model.kernel.omega.requires_grad = False
    model.kernel.log_gamma_raw.requires_grad = False
    model.kernel.log_sf.requires_grad = False
    model.likelihood._nu.requires_grad = False
    model.mv = nn.Parameter(torch.tensor(prev_model.mv.clone().detach().cpu().numpy(), device=device))
    model.Lv = nn.Parameter(torch.tensor(prev_model.Lv.clone().detach().cpu().numpy(), device=device))
    
    # model.load_state_dict(torch.load('./ohsgp_'+str(inducing_size)+'z_dayu/models'+str(args.seed_num)+'/model'+str(task_id)))

    optimizer = Adam(model.parameters(), lr=lr)
    # Initialize the plot
    pbar = range(num_epochs)
    
    log = []
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        try:
            if epoch == 0:
                loss, ve_mean, kl = model.ELBO(days_train, deaths_train, w, recompute_k=True, cache_k=True)
            else:
                loss, ve_mean, kl = model.ELBO(days_train, deaths_train, w, recompute_k=False, cache_k=False)
            loss = -loss  # Negate the loss for minimization
        except torch.linalg.LinAlgError as e:
            print(f"Caught LinAlgError: {e}")
            print(f"Breaking training loop at epoch {epoch}")
            break
        loss.backward()
        optimizer.step()
        

        # Update plot every 100 epochs
        if epoch == 0 or epoch % 1000 == 999:
            model.eval()
            with torch.no_grad():
                log.append(f"epoch{epoch} elbo {loss.item()} ve_mean {ve_mean.item()} kl {kl.item()}\n")
    
    # compute test metric        
    #days_test = torch.concatenate([prev_days_test, days_test])
    days_test_list.append(days_test)
    #deaths_test = torch.concatenate([prev_deaths_test, deaths_test])
    deaths_test_list.append(deaths_test)
    metrics_path = './ohsgp_freqspace/models'+str(args.seed_num)+'/metrics.csv'
    count = 0
    for days_test in days_test_list:
        mean, var = model.pred_f(days_test, w)
        pred_mean = _predict_mean(model, mean)
        rmse, r2 = _rmse_r2(deaths_test_list[count], pred_mean)
        f_samples = mean + torch.sqrt(var) * torch.randn(1000, var.shape[0], var.shape[1]).to(device)  # [1000, Ntest, 1]

        nlpd = -torch.log(model.likelihood.log_prob(f_samples.view(-1,1), deaths_test_list[count].unsqueeze(0).tile(1000, 1, 1).view(-1,1)).view(1000,-1).exp().mean(dim=(0))).mean()
        log.append('task id: {}, test nlpd after task{}: {}\n'.format(count+1, task_id, nlpd))
        log.append('task id: {}, test rmse after task{}: {}\n'.format(count+1, task_id, rmse))
        log.append('task id: {}, test r2 after task{}: {}\n'.format(count+1, task_id, r2))
        print(f"[task{task_id} eval{count+1}] test rmse: {rmse}, r2: {r2}, nlpd: {nlpd}")
        
        f_samples = mean + torch.sqrt(var) * torch.randn(30, var.shape[0], var.shape[1]).to(device)  # [10, Ntest, 1]
        y_samples = model.likelihood.sample(f_samples.squeeze(-1), numy_perf=30).to(device)
        y_samples = y_samples.permute((1,0,2))
        y_samples = y_samples.reshape((y_samples.shape[0], -1))

        conf_level = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
        ece_list = []
        for conf in conf_level:
            deaths_pred_lower = torch.quantile(y_samples, 0.5-conf/2, dim=1)
            deaths_pred_upper = torch.quantile(y_samples, 0.5+conf/2, dim=1)
            ece_list.append(torch.abs(torch.mean((((((deaths_pred_lower - deaths_test_list[count].squeeze(1)) < 0)+0.) + (((deaths_pred_upper - deaths_test_list[count].squeeze(1)) > 0)+0.)) > 1)+0.) - conf).item())
        ece = np.mean(np.array(ece_list))
        log.append('task id: {}, test ece after task{}: {}\n'.format(count+1, task_id, ece))
        _append_metrics_row(
            metrics_path,
            "eval_task,after_task,rmse,r2,nlpd,ece",
            f"{count+1},{task_id},{rmse},{r2},{nlpd},{ece}",
        )
        with open('./ohsgp_freqspace/models'+str(args.seed_num)+'/training.log', "a+") as log_file:
        #with open('./training.log', "a+") as log_file:
            log_file.writelines(log)
            log.clear()
        count += 1
    

    days_train = torch.concatenate([prev_days_train, days_train])
    deaths_train = torch.concatenate([prev_deaths_train, deaths_train])
    days_plot = torch.linspace(torch.min(days_train), torch.max(days_train), 1000).to(device)
    mean, var = model.pred_f(days_plot.unsqueeze(1), w)

    deaths_pred_mean_plot = mean.detach().cpu().numpy()
    var_plot = var.detach().cpu().numpy()
    plt.figure(figsize=(12, 5))
    plt.plot(days_plot.detach().cpu().numpy(), deaths_pred_mean_plot.flatten(), label="Predicted Deaths Mean OSVGP", color='red')
    deaths_pred_lower_plot = deaths_pred_mean_plot - 2 * np.sqrt(var_plot)
    deaths_pred_upper_plot = deaths_pred_mean_plot + 2 * np.sqrt(var_plot)
    plt.fill_between(days_plot.detach().cpu().numpy(), deaths_pred_lower_plot.flatten(), deaths_pred_upper_plot.flatten(), color='blue', alpha=0.3, label="95% CI")

    # plot death counts
    plt.scatter(days_train.detach().cpu().numpy(), deaths_train.detach().cpu().numpy(), label="train", color='black', alpha=1, s=5)
    plt.scatter(days_test.detach().cpu().numpy(), deaths_test.detach().cpu().numpy(), label="test", color='blue', alpha=1, s=5)

    plt.xlabel("Day")
    plt.ylabel("Predicted Deaths")
    plt.title("Posterior Summary of Predicted Deaths")
    plt.legend()
    plt.savefig('./ohsgp_freqspace/models'+str(args.seed_num)+'/covid_task'+str(task_id)+'.png', format='png')
    #prev_days_test = days_test
    #prev_deaths_test = deaths_test
    prev_days_test = days_test_list
    prev_deaths_test = deaths_test_list
    prev_days_train = days_train
    prev_deaths_train = deaths_train
    
    #torch.save(model.state_dict(), './ohsgp/models'+str(args.seed_num)+'/model'+str(task_id))

    #prev_days_test_np = prev_days_test.detach().cpu().numpy()
    #prev_deaths_test_np = prev_deaths_test.detach().cpu().numpy()
    #prev_days_train_np = prev_days_train.detach().cpu().numpy()
    #prev_deaths_train_np = prev_deaths_train.detach().cpu().numpy()
    #np.savez(f'./ohsgp/models'+str(args.seed_num)+'/data'+str(task_id)+'.npz', \
    #         prev_days_test=prev_days_test_np, prev_deaths_test=prev_deaths_test_np, \
    #            prev_days_train=prev_days_train_np, prev_deaths_train=prev_deaths_train_np)
    #return model, prev_days_test, prev_deaths_test, prev_days_train, prev_deaths_train, w, prev_discrete_steps, end_next
    return model, prev_days_test, prev_deaths_test, prev_days_train, prev_deaths_train, w, prev_discrete_steps, end_next

if __name__ == '__main__':
    args = setup()
    seed_everything(args.seed_num)
    os.makedirs('./ohsgp_freqspace/models'+str(args.seed_num), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inducing_size = 15
    num_epochs = 5000
    lr = 1e-2
    step_size = 2
    hippo = HiPPO_LegS(inducing_size, device)
    rff_sample_size = args.num_components
    print(f"Using {rff_sample_size} frequency components")

    prev_model, prev_days_test, prev_deaths_test, prev_days_train, prev_deaths_train, w,\
         prev_discrete_steps, prev_end = train_task1(device, inducing_size, num_epochs, lr, step_size)

    for task_id in range(2, 6):
        prev_model, prev_days_test, prev_deaths_test, prev_days_train, prev_deaths_train, w, prev_discrete_steps, prev_end = train_nexttask(prev_model, prev_days_test, prev_deaths_test, prev_days_train, prev_deaths_train, \
                   device, task_id, inducing_size, num_epochs, lr, step_size, w, prev_discrete_steps, prev_end)
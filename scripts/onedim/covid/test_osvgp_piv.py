import numpy as np
import torch
import sys 
sys.path.append("../../")
from hipposvgp.svgp import OSVGP
from hipposvgp.likelihood import NegativeBinomial
from hipposvgp.kernel import SE_kernel
from hipposvgp.utils import pivoted_cholesky_init
from scripts.onedim.covid.data import COVIDDataset
import random
import matplotlib.pyplot as plt
from torch.optim import Adam
from copy import deepcopy
import torch.nn as nn
import os
import argparse


def setup():
    parser=argparse.ArgumentParser('Argument Parser')
    parser.add_argument('--seed_num',type=int,default=0)
    args=parser.parse_args()
    return args

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_task1(device, inducing_size, num_epochs, lr):
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

    kernel = SE_kernel(1, device=device)
    likelihood = NegativeBinomial()
    likelihood = likelihood.to(device)

    model = OSVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_size=inducing_size, 
        device=device, 
    ).to(device)
    Z = days_train[torch.randperm(days_train.shape[0])[:inducing_size]]
    model.Z = torch.nn.Parameter(Z.to(device))

    optimizer = Adam(model.parameters(), lr=lr)

    # Initialize the plot
    pbar = range(num_epochs)

    log = []
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        try:
            loss, ve_mean, kl = model.ELBO(days_train, deaths_train)
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

    log.append('log_ls: {}, log_sf: {}, nu: {}\n'.format(model.kernel.log_ls.item(), model.kernel.log_sf.item(), model.likelihood._nu.item()))
    # compute test metric        
    mean, var = model.pred_f(days_test)
    f_samples = mean + torch.sqrt(var) * torch.randn(100, var.shape[0], var.shape[1]).to(device)  # [100, Ntest, 1]

    nlpd = -torch.log(model.likelihood.log_prob(f_samples.view(-1,1), deaths_test.unsqueeze(0).tile(100, 1, 1).view(-1,1)).view(100,-1).exp().mean(dim=(0))).mean()
    log.append('test nlpd after task1: {}\n'.format(nlpd))
    
    f_samples = mean + torch.sqrt(var) * torch.randn(10, var.shape[0], var.shape[1]).to(device)  # [10, Ntest, 1]
    y_samples = model.likelihood.sample(f_samples.squeeze(-1), numy_perf=10).to(device)
    y_samples = y_samples.permute((1,0,2))
    y_samples = y_samples.reshape((y_samples.shape[0], -1))

    conf_level = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    ece_list = []
    for conf in conf_level:
        deaths_pred_lower = torch.quantile(y_samples, 0.5-conf/2, dim=1)
        deaths_pred_upper = torch.quantile(y_samples, 0.5+conf/2, dim=1)
        ece_list.append(torch.abs(torch.mean((((((deaths_pred_lower - deaths_test.squeeze(1)) < 0)+0.) + (((deaths_pred_upper - deaths_test.squeeze(1)) > 0)+0.)) > 1)+0.) - conf).item())
    log.append('test ece after task1: {}\n'.format(np.mean(np.array(ece_list))))
    os.makedirs('./osvgp_piv/models'+str(args.seed_num), exist_ok=True)
    with open('./osvgp_piv/models'+str(args.seed_num)+'/training.log', "a+") as log_file:
        log_file.writelines(log)
        log.clear()

    days_plot = torch.linspace(torch.min(days_train), torch.max(days_train), 1000).to(device)
    mean, var = model.pred_f(days_plot.unsqueeze(1))

    deaths_pred_mean_plot = mean.detach().cpu().numpy()
    plt.figure(figsize=(12, 5))
    plt.plot(days_plot.detach().cpu().numpy(), deaths_pred_mean_plot.flatten(), label="Predicted Deaths Mean OSVGP", color='red')
    deaths_pred_lower_plot = deaths_pred_mean_plot - 2* np.sqrt(var.detach().cpu().numpy())
    deaths_pred_upper_plot = deaths_pred_mean_plot + 2* np.sqrt(var.detach().cpu().numpy())
    plt.fill_between(days_plot.detach().cpu().numpy(), deaths_pred_lower_plot.flatten(), deaths_pred_upper_plot.flatten(), color='blue', alpha=0.3, label="95% CI")
    plt.scatter(model.Z.cpu().detach().numpy(), np.zeros_like(model.Z.cpu().detach().numpy()), marker='x', color='red')

    # plot death counts
    plt.scatter(days_train.detach().cpu().numpy(), deaths_train.detach().cpu().numpy(), label="train", color='black', alpha=1, s=5)
    plt.scatter(days_test.detach().cpu().numpy(), deaths_test.detach().cpu().numpy(), label="test", color='blue', alpha=1, s=5)

    plt.xlabel("Day")
    plt.ylabel("Predicted Deaths")
    plt.title("Posterior Summary of Predicted Deaths")
    plt.legend()
    plt.savefig('./osvgp_piv/models'+str(args.seed_num)+'/covid_task1.png', format='png')

    prev_days_test_np = days_test.detach().cpu().numpy()
    prev_deaths_test_np = deaths_test.detach().cpu().numpy()
    prev_days_train_np = days_train.detach().cpu().numpy()
    prev_deaths_train_np = deaths_train.detach().cpu().numpy()
    os.makedirs('./osvgp_piv/models'+str(args.seed_num), exist_ok=True)
    np.savez(f'./osvgp_piv/models'+str(args.seed_num)+'/data1.npz', \
             prev_days_test=prev_days_test_np, prev_deaths_test=prev_deaths_test_np, \
                prev_days_train=prev_days_train_np, prev_deaths_train=prev_deaths_train_np)
    os.makedirs('./osvgp_piv/models'+str(args.seed_num), exist_ok=True)
    torch.save(model.state_dict(), './osvgp_piv/models'+str(args.seed_num)+'/model1')
    return model, days_test, deaths_test, days_train, deaths_train


def train_nexttask(prev_model, prev_days_test, prev_deaths_test, prev_days_train, prev_deaths_train, \
                   device, task_id, inducing_size, num_epochs, lr):
    mv_old = prev_model.mv.clone()
    mv_old = mv_old.clone().detach().to(device=device)
    Lv_old = prev_model.Lv.clone()
    Lv_old = Lv_old.clone().detach().to(device=device)
    Kuu_old, _, _ = prev_model.Kuu_se()
    num_inducing_old = prev_model.inducing_size
    Z_old = prev_model.Z.clone()
    Z_old = Z_old.clone().detach().to(device=device)

    kernel = prev_model.kernel
    likelihood = prev_model.likelihood
    model = OSVGP(
        kernel=deepcopy(kernel),
        likelihood=deepcopy(likelihood),
        inducing_size=inducing_size, 
        device=device, 
        Z_old=Z_old,
        mv_old=mv_old,
        Lv_old=Lv_old,
        Kaa_old=Kuu_old,
        num_inducing_old=num_inducing_old,
    ).to(device)
    model.kernel.log_ls.requires_grad = False
    model.kernel.log_sf.requires_grad = False
    model.likelihood._nu.requires_grad = False
    model.mv = nn.Parameter(torch.tensor(prev_model.mv.clone().detach().cpu().numpy(), device=device))
    model.Lv = nn.Parameter(torch.tensor(prev_model.Lv.clone().detach().cpu().numpy(), device=device))

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

    Z_old = torch.as_tensor(prev_model.Z)
    M = Z_old.shape[0]
    old_Z = Z_old.clone()
    new_Z = days_train
    Z_input = torch.vstack((old_Z, new_Z))
    kernel_matrix = model.kernel(Z_input, Z_input)
    Z = pivoted_cholesky_init(Z_input, kernel_matrix, min(inducing_size, Z_input.shape[0]))
    model.Z = torch.nn.Parameter(Z)
    #model.Z.requires_grad = False

    optimizer = Adam(model.parameters(), lr=lr)
    # Initialize the plot
    pbar = range(num_epochs)
    
    log = []
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        try:
            loss, ve_mean, kl = model.ELBO(days_train, deaths_train)
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
    days_test = torch.concatenate([prev_days_test, days_test])
    deaths_test = torch.concatenate([prev_deaths_test, deaths_test])
    mean, var = model.pred_f(days_test)
    f_samples = mean + torch.sqrt(var) * torch.randn(100, var.shape[0], var.shape[1]).to(device)  # [100, Ntest, 1]

    nlpd = -torch.log(model.likelihood.log_prob(f_samples.view(-1,1), deaths_test.unsqueeze(0).tile(100, 1, 1).view(-1,1)).view(100,-1).exp().mean(dim=(0))).mean()
    log.append('test nlpd after task'+str(task_id)+': {}\n'.format(nlpd))
    
    f_samples = mean + torch.sqrt(var) * torch.randn(10, var.shape[0], var.shape[1]).to(device)  # [10, Ntest, 1]
    y_samples = model.likelihood.sample(f_samples.squeeze(-1), numy_perf=10).to(device)
    y_samples = y_samples.permute((1,0,2))
    y_samples = y_samples.reshape((y_samples.shape[0], -1))

    conf_level = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    ece_list = []
    for conf in conf_level:
        deaths_pred_lower = torch.quantile(y_samples, 0.5-conf/2, dim=1)
        deaths_pred_upper = torch.quantile(y_samples, 0.5+conf/2, dim=1)
        ece_list.append(torch.abs(torch.mean((((((deaths_pred_lower - deaths_test.squeeze(1)) < 0)+0.) + (((deaths_pred_upper - deaths_test.squeeze(1)) > 0)+0.)) > 1)+0.) - conf).item())
    log.append('test ece after task'+str(task_id)+': {}\n'.format(np.mean(np.array(ece_list))))
    os.makedirs('./osvgp_piv/models'+str(args.seed_num), exist_ok=True)
    with open('./osvgp_piv/models'+str(args.seed_num)+'/training.log', "a+") as log_file:
        log_file.writelines(log)
        log.clear()
    

    days_train = torch.concatenate([prev_days_train, days_train])
    deaths_train = torch.concatenate([prev_deaths_train, deaths_train])
    days_plot = torch.linspace(torch.min(days_train), torch.max(days_train), 1000).to(device)
    mean, var = model.pred_f(days_plot.unsqueeze(1))

    deaths_pred_mean_plot = mean.detach().cpu().numpy()
    plt.figure(figsize=(12, 5))
    plt.plot(days_plot.detach().cpu().numpy(), deaths_pred_mean_plot.flatten(), label="Predicted Deaths Mean OSVGP", color='red')
    deaths_pred_lower_plot = deaths_pred_mean_plot - 2* np.sqrt(var.detach().cpu().numpy())
    deaths_pred_upper_plot = deaths_pred_mean_plot + 2* np.sqrt(var.detach().cpu().numpy())
    plt.fill_between(days_plot.detach().cpu().numpy(), deaths_pred_lower_plot.flatten(), deaths_pred_upper_plot.flatten(), color='blue', alpha=0.3, label="95% CI")
    plt.scatter(model.Z.cpu().detach().numpy(), np.zeros_like(model.Z.cpu().detach().numpy()), marker='x', color='red')

    # plot death counts
    plt.scatter(days_train.detach().cpu().numpy(), deaths_train.detach().cpu().numpy(), label="train", color='black', alpha=1, s=5)
    plt.scatter(days_test.detach().cpu().numpy(), deaths_test.detach().cpu().numpy(), label="test", color='blue', alpha=1, s=5)

    plt.xlabel("Day")
    plt.ylabel("Predicted Deaths")
    plt.title("Posterior Summary of Predicted Deaths")
    plt.legend()
    os.makedirs('./osvgp_piv/models'+str(args.seed_num), exist_ok=True)
    plt.savefig('./osvgp_piv/models'+str(args.seed_num)+'/covid_task'+str(task_id)+'.png', format='png')

    prev_days_test = days_test
    prev_deaths_test = deaths_test
    prev_days_train = days_train
    prev_deaths_train = deaths_train
    
    torch.save(model.state_dict(), './osvgp_piv/models'+str(args.seed_num)+'/model'+str(task_id))

    prev_days_test_np = prev_days_test.detach().cpu().numpy()
    prev_deaths_test_np = prev_deaths_test.detach().cpu().numpy()
    prev_days_train_np = prev_days_train.detach().cpu().numpy()
    prev_deaths_train_np = prev_deaths_train.detach().cpu().numpy()
    os.makedirs('./osvgp_piv/models'+str(args.seed_num), exist_ok=True)
    np.savez(f'./osvgp_piv/models'+str(args.seed_num)+'/data'+str(task_id)+'.npz', \
             prev_days_test=prev_days_test_np, prev_deaths_test=prev_deaths_test_np, \
                prev_days_train=prev_days_train_np, prev_deaths_train=prev_deaths_train_np)
    return model, prev_days_test, prev_deaths_test, prev_days_train, prev_deaths_train

if __name__ == '__main__':
    args = setup()
    seed_everything(args.seed_num)
    os.makedirs('./osvgp_piv/models'+str(args.seed_num), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inducing_size = 30
    num_epochs = 5000
    lr = 1e-2
    prev_model, prev_days_test, prev_deaths_test, prev_days_train, prev_deaths_train = train_task1(device, inducing_size, num_epochs, lr)

    for task_id in range(2, 6):
        prev_model, prev_days_test, prev_deaths_test, prev_days_train, prev_deaths_train = train_nexttask(prev_model, prev_days_test, prev_deaths_test, prev_days_train, prev_deaths_train, \
                   device, task_id, inducing_size, num_epochs, lr)
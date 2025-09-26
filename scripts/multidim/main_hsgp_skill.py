import os
import numpy as np
import torch
import matplotlib.pyplot as plt  
from hipposvgp.multidim import HIPPOOSVGP, SE_kernel
from hipposvgp.likelihood import GaussianLikelihood
from hipposvgp.hippo import HiPPO_LegS
import torch.nn as nn
import time
import math
import argparse
import pandas as pd
from copy import deepcopy
import torch_optimizer
from tqdm import tqdm
from scripts.multidim.dataset import Powerplant, Elevators, Skillcraft
from hipposvgp.utils import sort_with_l2, sort_with_se_kernel, sort_with_dimension
import random


def setup():
    parser=argparse.ArgumentParser('Argument Parser')
    parser.add_argument('--data',type=str,default='skill')
    parser.add_argument('--kernel_type',type=str,default='se')
    parser.add_argument('--inducing_size',type=int,default=256)
    parser.add_argument('--rff_sample_size',type=int,default=4000)
    parser.add_argument('--subsampling_lag',type=int,default=2)
    parser.add_argument('--num_iters',type=int,default=2000)
    parser.add_argument('--run_num',type=int,default=1)
    parser.add_argument('--lr',type=float,default=5e-3)
    parser.add_argument('--flag_update_kernel',type=bool,default=False)
    args=parser.parse_args()
    return args

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    args=setup()
    seed_everything(seed=args.run_num) # 42, 106, 108
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hippo = HiPPO_LegS(args.inducing_size, device)
    #hippo = HiPPO_LegT(N=args.inducing_size, dt=10./9056, device=device)
    prev_discrete_steps = 0

    if args.data == 'powerplant':
        dataobj = Powerplant()
    elif args.data == 'skill':
        dataobj = Skillcraft()
        log_ls_forsort = torch.tensor(np.array([3.4783, -0.9032, -5.0846,  2.4429,  2.7161,  1.5580,  1.9396, -0.0452,
            -0.9826,  0.1504,  0.3997,  1.8558,  0.2807,  3.1336,  2.1943, -0.6936,
            -2.8923]), dtype=torch.float64, device=device)
    else:
        raise NotImplementedError

    train_set = dataobj.train_dataset
    train_x, train_y = train_set[:]

    train_x, train_y = sort_with_dimension(train_x, train_y)
    #train_x, train_y = sort_with_l2(train_x, train_y)
    train_x = train_x.to(device).to(torch.float64)
    train_y = train_y.to(device).to(torch.float64)

    train_x_tuple = train_x.chunk(10)
    train_y_tuple = train_y.chunk(10)

    train_x_list = []
    train_y_list = []
    test_x_list = []
    test_y_list = []
    for i in range(10):
        ind_shuffle = torch.randperm(train_x_tuple[i].shape[0])
        train_x_cur = train_x_tuple[i][ind_shuffle,:]
        train_y_cur = train_y_tuple[i][ind_shuffle,:]
        
        test_x_cur = train_x_cur[::10]
        test_y_cur = train_y_cur[::10]
        test_x_list.append(test_x_cur)
        test_y_list.append(test_y_cur)

        train_x_cur = torch.concatenate([train_x_cur[j::10] for j in range(1, 10)])
        train_y_cur = torch.concatenate([train_y_cur[j::10] for j in range(1, 10)])
        train_x_list.append(train_x_cur)
        train_y_list.append(train_y_cur)

    ini_x = train_x_list[0]
    ini_y = train_y_list[0]

    if args.kernel_type == 'se':
        input_dim = train_x.shape[1]
        kernel = SE_kernel(input_dim, device=device)
    else:
        raise NotImplementedError
    likelihood = GaussianLikelihood(0.1)

    #Z_interpolate, _ = sort_with_se_kernel(ini_x, ini_y, pivot=0, \
    #    log_ls=log_ls_forsort, subsampling_lag=args.subsampling_lag, flag_min=False)

    Z_interpolate, _ = sort_with_dimension(ini_x, ini_y)

    #Z_interpolate, _ = sort_with_l2(ini_x, ini_y)
    Z_interpolate = Z_interpolate[::args.subsampling_lag].to(torch.float64)

    model1 = HIPPOOSVGP(
        prev_discrete_steps=0,
        kernel=kernel,
        likelihood=likelihood,
        inducing_size=args.inducing_size,
        rff_sample_size=args.rff_sample_size, 
        hippo=hippo,
        num_inducing_old=args.inducing_size, 
        device=device, 
        Z_interpolate=Z_interpolate,
        flag_update_kernel=args.flag_update_kernel,
    ).to(device)
    prev_discrete_steps += Z_interpolate.shape[0]

    # full gp hyper
    
    model1.kernel.log_ls = nn.Parameter(torch.tensor(np.array([3.4783, -0.9032, -5.0846,  2.4429,  2.7161,  1.5580,  1.9396, -0.0452,
        -0.9826,  0.1504,  0.3997,  1.8558,  0.2807,  3.1336,  2.1943, -0.6936,
        -2.8923]), dtype=torch.float64, device=device))
    model1.kernel.log_sf = nn.Parameter(torch.tensor(np.array([1.5601]), dtype=torch.float64, device=device))
    model1.likelihood.log_variance = nn.Parameter(torch.tensor(np.array([-1.7071]), dtype=torch.float64, device=device))

    if not args.flag_update_kernel:# and t > 100:
        model1.kernel.log_ls.requires_grad = False
        model1.kernel.log_sf.requires_grad = False
        model1.likelihood.log_variance.requires_grad = False
    
    
    #optimizer = torch.optim.Adam(model1.parameters(), lr=args.lr)
    optimizer = torch_optimizer.Yogi(model1.parameters(), lr=args.lr)
    num_epochs = args.num_iters

    # Initialize the plot
    pbar = tqdm(range(num_epochs))
    
    if not args.flag_update_kernel:
        w = model1.kernel.sample_from_spectral(args.rff_sample_size)
        w = w.detach()

    for epoch in pbar:
        optimizer.zero_grad()
        if args.flag_update_kernel:
            loss, ve_mean, kl = model1.ELBO(ini_x, ini_y, None)
            retain_graph = True
        else:
            if epoch == 0:
                loss, ve_mean, kl = model1.ELBO(ini_x, ini_y, w, recompute_k=True, cache_k=True)  
            else:
                loss, ve_mean, kl = model1.ELBO(ini_x, ini_y, w, recompute_k=False, cache_k=False)
            retain_graph = False
        loss = -loss
        loss.backward(retain_graph=retain_graph)
        #loss.backward()
        optimizer.step()

        # Update plot every 100 epochs
        if epoch % 100 == 0:
            model1.eval()
            with torch.no_grad():
                if args.flag_update_kernel:
                    y_mean, f_var = model1.pred_f(ini_x, None)
                else:
                    y_mean, f_var = model1.pred_f(ini_x, w)
                rmse = torch.sqrt(torch.sum((y_mean - ini_y)**2) / y_mean.shape[0])
                print(f"Epoch {epoch} Train rmse current task: {rmse}")
                # print ve_mean and kl
                print(f"ve_mean {ve_mean.item()} kl {kl.item()}")
                print("log_variance: {}".format(model1.likelihood.log_variance.item()))
            model1.train()
    
    if args.flag_update_kernel:
        w = model1.kernel.sample_from_spectral(args.rff_sample_size)
        w = w.detach()
    
    y_mean, f_var = model1.pred_f(test_x_list[0], w)      
    y_var = f_var + torch.exp(model1.likelihood.log_variance) * torch.ones_like(f_var)   # [N, 1]
    rmse_test = torch.sqrt(torch.sum((y_mean - test_y_list[0])**2) / y_mean.shape[0])
    y_var = y_var.squeeze(1)
    chol_cov = y_var.pow(0.5)
    nlpd_test = torch.sum(torch.log(chol_cov)) + 0.5* y_mean.shape[0]* torch.log(torch.tensor(2* np.pi))
    nlpd_test += 0.5* torch.sum((test_y_list[0] - y_mean).pow(2).squeeze(1)/  y_var)
    nlpd_test /= y_mean.shape[0]
    print('t: {}, test rmse: {}, nlpd: {}'.format(0, rmse_test.item(), nlpd_test.item()))
    
    y_mean, f_var = model1.pred_f(train_x_list[0], w)      
    y_var = f_var + torch.exp(model1.likelihood.log_variance) * torch.ones_like(f_var)   # [N, 1]
    rmse_train = torch.sqrt(torch.sum((y_mean - train_y_list[0])**2) / y_mean.shape[0])
    y_var = y_var.squeeze(1)
    chol_cov = y_var.pow(0.5)
    nlpd_train = torch.sum(torch.log(chol_cov)) + 0.5* y_mean.shape[0]* torch.log(torch.tensor(2* np.pi))
    nlpd_train += 0.5* torch.sum((train_y_list[0] - y_mean).pow(2).squeeze(1)/  y_var)
    nlpd_train /= y_mean.shape[0]
    print('t: {}, train rmse: {}, nlpd: {}'.format(0, rmse_train.item(), nlpd_train.item()))
    
    os.makedirs('./models/run'+str(args.run_num), exist_ok=True)
    torch.save(model1.state_dict(), './models/run'+str(args.run_num)+'/hsgp_model0')

    prev_model = model1
    pivot = Z_interpolate[-1]
    
    for t in range(9):
        x = train_x_list[t+1]
        y = train_y_list[t+1]
        Z_old, _, Kuu_old, _, _ = prev_model.Kuu_se(w)
        Z_old = Z_old.detach()
        Kuu_old = (torch.exp(prev_model.kernel.log_sf)* Kuu_old).detach()
        mv_old = prev_model.mv.detach()
        Lv_old = prev_model.Lv.detach()

        if args.flag_update_kernel:
            #temp, _ = sort_with_se_kernel(x, y, pivot, log_ls_forsort, subsampling_lag=args.subsampling_lag)
            temp, _ = sort_with_dimension(x, y)

            #temp, _ = sort_with_l2(x, y)
            temp = temp[::args.subsampling_lag]
            prev_discrete_steps_ = prev_discrete_steps + temp.shape[0]
            Z_interpolate = torch.concatenate([Z_interpolate, temp])
        else:
            #Z_interpolate, _ = sort_with_se_kernel(x, y, pivot, log_ls_forsort, subsampling_lag=args.subsampling_lag)
            Z_interpolate, _ = sort_with_dimension(x, y)

            #Z_interpolate, _ = sort_with_l2(x, y)
            Z_interpolate = Z_interpolate[::args.subsampling_lag]
            prev_discrete_steps_ = prev_discrete_steps + Z_interpolate.shape[0]

        model = HIPPOOSVGP(kernel=deepcopy(kernel),
            likelihood=deepcopy(likelihood),
            prev_discrete_steps=prev_discrete_steps, 
            Z_interpolate=Z_interpolate,
            rff_sample_size=args.rff_sample_size,
            hippo=hippo,
            inducing_size=args.inducing_size,
            num_inducing_old=args.inducing_size,
            device=device, 
            Z_old=Z_old,
            mv_old=mv_old,
            Lv_old=Lv_old,
            Kaa_old=Kuu_old,
            flag_update_kernel=args.flag_update_kernel).to(device)
        prev_discrete_steps = prev_discrete_steps_
        
        log_ls = prev_model.kernel.log_ls
        log_sf = prev_model.kernel.log_sf
        log_variance = prev_model.likelihood.log_variance
        model.kernel.log_ls = nn.Parameter(torch.tensor(log_ls.detach().cpu().numpy(), dtype=torch.float64, device=device))
        model.kernel.log_sf = nn.Parameter(torch.tensor(log_sf.detach().cpu().numpy(), dtype=torch.float64, device=device))
        model.likelihood.log_variance = nn.Parameter(torch.tensor(log_variance.detach().cpu().numpy(), dtype=torch.float64, device=device))
        
        model.mv = nn.Parameter(torch.tensor(prev_model.mv.detach().cpu().numpy(), dtype=torch.float64, device=device))
        model.Lv = nn.Parameter(torch.tensor(prev_model.Lv.detach().cpu().numpy(), dtype=torch.float64, device=device))

        if not args.flag_update_kernel:# and t > 100:
            model.kernel.log_ls.requires_grad = False  
            model.kernel.log_sf.requires_grad = False 
            model.likelihood.log_variance.requires_grad = False 

        #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizer = torch_optimizer.Yogi(model.parameters(), lr=args.lr)
        num_epochs = args.num_iters

        # Initialize the plot
        pbar = range(num_epochs)
        for epoch in pbar:
            optimizer.zero_grad()

            if args.flag_update_kernel:
                loss, ve_mean, kl = model.ELBO(x, y, None, beta=1)
                retain_graph = True
            else:
                if epoch == 0:
                    loss, ve_mean, kl = model.ELBO(x, y, w, beta=1, recompute_k=True, cache_k=True)  
                else:
                    loss, ve_mean, kl = model.ELBO(x, y, w, beta=1, recompute_k=False, cache_k=False)
                retain_graph = False
            loss = -loss
            loss.backward(retain_graph=retain_graph)
            optimizer.step()
            """
            # Update plot every 100 epochs
            if epoch % 100 == 0:
                model.eval()
                with torch.no_grad():
                    y_mean, f_var = model.pred_f(x, w)
                    rmse = torch.sqrt(torch.sum((y_mean - y)**2) / y_mean.shape[0])
                    print(f"Epoch {epoch} Train rmse current task: {rmse}")
                    # print ve_mean and kl
                    print(f"ve_mean {ve_mean.item()} kl {kl.item()}")
                model.train()
            """
        if args.flag_update_kernel:
            w = model.kernel.sample_from_spectral(args.rff_sample_size)
            w = w.detach()
        
        res_rmse = []
        res_nlpd = []
        for k in range(t+2):
            y_mean, f_var = model.pred_f(test_x_list[k], w)      
            y_var = f_var + torch.exp(model.likelihood.log_variance) * torch.ones_like(f_var)   # [N, 1]
            rmse_test = torch.sqrt(torch.sum((y_mean - test_y_list[k])**2) / y_mean.shape[0])
            y_var = y_var.squeeze(1)
            chol_cov = y_var.pow(0.5)
            nlpd_test = torch.sum(torch.log(chol_cov)) + 0.5* y_mean.shape[0]* torch.log(torch.tensor(2* np.pi))
            nlpd_test += 0.5* torch.sum((test_y_list[k] - y_mean).pow(2).squeeze(1)/  y_var)
            nlpd_test /= y_mean.shape[0]
            res_rmse.append(rmse_test.item())
            res_nlpd.append(nlpd_test.item())
        print('t: {}, test rmse: {}, nlpd: {}'.format(t+1, res_rmse, res_nlpd))
        
        res_rmse = []
        res_nlpd = []
        for k in range(t+2):
            y_mean, f_var = model.pred_f(train_x_list[k], w)      
            y_var = f_var + torch.exp(model.likelihood.log_variance) * torch.ones_like(f_var)   # [N, 1]
            rmse_train = torch.sqrt(torch.sum((y_mean - train_y_list[k])**2) / y_mean.shape[0])
            y_var = y_var.squeeze(1)
            chol_cov = y_var.pow(0.5)
            nlpd_train = torch.sum(torch.log(chol_cov)) + 0.5* y_mean.shape[0]* torch.log(torch.tensor(2* np.pi))
            nlpd_train += 0.5* torch.sum((train_y_list[k] - y_mean).pow(2).squeeze(1)/  y_var)
            nlpd_train /= y_mean.shape[0]
            res_rmse.append(rmse_train.item())
            res_nlpd.append(nlpd_train.item())
        print('t: {}, train rmse: {}, nlpd: {}'.format(t+1,  res_rmse, res_nlpd))
        
        os.makedirs('./models/run'+str(args.run_num), exist_ok=True)
        torch.save(model.state_dict(), './models/run'+str(args.run_num)+'/hsgp_model'+str(t+1))

        prev_model = model
        pivot = Z_interpolate[-1]
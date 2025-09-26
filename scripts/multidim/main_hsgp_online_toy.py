import numpy as np
import torch
import matplotlib.pyplot as plt  
from hipposvgp.multidim import HIPPOOSVGP, SE_kernel
from hipposvgp.likelihood import Bernoulli
from hipposvgp.hippo import HiPPO_LegS
import torch.nn as nn
import argparse
import pandas as pd
from copy import deepcopy
from scripts.multidim.dataset import get_data_moon, get_data_banana
from hipposvgp.utils import pred_boundary, sort_with_l2, sort_with_l2_2, sort_with_se_kernel, sort_with_se_kernel2, sort_with_dimension
import torch_optimizer
from tqdm import tqdm
import random


def setup():
    parser=argparse.ArgumentParser('Argument Parser')
    parser.add_argument('--data',type=str,default='moon')
    parser.add_argument('--kernel_type',type=str,default='se')
    parser.add_argument('--inducing_size',type=int,default=20)
    parser.add_argument('--rff_sample_size',type=int,default=1000)
    parser.add_argument('--num_iters',type=int,default=2000)
    parser.add_argument('--subsampling_lag',type=int,default=1)
    parser.add_argument('--lr',type=float,default=5e-3)
    parser.add_argument('--flag_update_kernel',type=bool,default=False)
    args=parser.parse_args()
    return args

def seed_everything(seed=42):
    random.seed(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
  

if __name__ == "__main__":
    seed_everything(0)
    args=setup()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hippo = HiPPO_LegS(args.inducing_size, device)
    prev_discrete_steps = 0
    if args.kernel_type == 'se':
        kernel = SE_kernel(2, device=device)
    else:
        raise NotImplementedError
    likelihood = Bernoulli()

    if args.data == 'moon':
        X, y, Xtest, ytest = get_data_moon()
        X_train1 = X[0]
        X_train2 = X[1]
        X_train3 = X[2]
        #X_train4 = X[3]
        Y_train1 = y[0]
        Y_train2 = y[1]
        Y_train3 = y[2]
        #Y_train4 = y[3]
        X_test1 = Xtest[0]
        X_test2 = Xtest[1]
        X_test3 = Xtest[2]
        #X_test4 = Xtest[3]
        Y_test1 = ytest[0]
        Y_test2 = ytest[1]
        Y_test3 = ytest[2]
        #Y_test4 = ytest[3]
        x1 = torch.tensor(X_train1).to(device)
        y1 = torch.tensor(Y_train1, dtype=torch.float64).to(device)
        x2 = torch.tensor(X_train2).to(device)
        y2 = torch.tensor(Y_train2, dtype=torch.float64).to(device)
        x3 = torch.tensor(X_train3).to(device)
        y3 = torch.tensor(Y_train3, dtype=torch.float64).to(device)
        #x4 = torch.tensor(X_train4).to(device)
        #y4 = torch.tensor(Y_train4, dtype=torch.float64).to(device)
        x_min, x_max = -1.5, 2.5
        y_min, y_max = -1, 1.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
            np.arange(y_min, y_max, 0.05)) 
    elif args.data == 'banana':
        X, y, Xtest, ytest = get_data_banana(iid=False)
        X = torch.tensor(X)
        y = torch.tensor(y)
        batch_size = len(X) // 4
        x1 = X[:batch_size].to(device)
        y1 = y[:batch_size].to(device)
        x2 = X[batch_size:2*batch_size].to(device)
        y2 = y[batch_size:2*batch_size].to(device)
        x3 = X[2*batch_size:3*batch_size].to(device)
        y3 = y[2*batch_size:3*batch_size].to(device)
        x4 = X[3*batch_size:].to(device)
        y4 = y[3*batch_size:].to(device)
        X_test1 = Xtest[:batch_size]
        X_test2 = Xtest[batch_size:2*batch_size]
        X_test3 = Xtest[2*batch_size:3*batch_size]
        X_test4 = Xtest[3*batch_size:]
        Y_test1 = ytest[:batch_size]
        Y_test2 = ytest[batch_size:2*batch_size]
        Y_test3 = ytest[2*batch_size:3*batch_size]
        Y_test4 = ytest[3*batch_size:]
        x_min, x_max = -3, 3
        y_min, y_max = -2.5, 2.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
            np.arange(y_min, y_max, 0.05)) 
    else:
        raise NotImplementedError

    log_ls_forsort = torch.tensor([-0.6680, -0.1576], dtype=torch.float64, device=device).clone().detach()
    #Z_interpolate, _ = sort_with_se_kernel(x1, y1, pivot=0, log_ls=log_ls_forsort, subsampling_lag=args.subsampling_lag)
    #Z_interpolate, _ = sort_with_se_kernel2(x1, y1, pivot=0, log_ls=log_ls_forsort, subsampling_lag=args.subsampling_lag)
    #Z_interpolate, _ = sort_with_l2(x1, y1)
    #Z_interpolate, _ = sort_with_l2_2(x1, y1)
    #Z_interpolate = x1
    Z_interpolate = x1[torch.randperm(x1.shape[0])]
    #Z_interpolate = torch.flip(x1, (0,))
    #Z_interpolate, _ = sort_with_dimension(x1, y1, dim=1)
    #Z_interpolate, _ = sort_with_dimension(x1, y1, dim=0)
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

    log_ls = torch.tensor([-0.6680, -0.1576], dtype=torch.float64, device=device)
    log_sf = torch.tensor([3.0143], dtype=torch.float64, device=device)
    model1.kernel.log_ls = nn.Parameter(torch.tensor(log_ls.detach().cpu().numpy(), dtype=torch.float64, device=device))
    model1.kernel.log_sf = nn.Parameter(torch.tensor(log_sf.detach().cpu().numpy(), dtype=torch.float64, device=device))

    model1.kernel.log_ls.requires_grad = False
    model1.kernel.log_sf.requires_grad = False
    w = model1.kernel.sample_from_spectral(args.rff_sample_size)
    w = w.detach()

    #optimizer = torch.optim.Adam(model1.parameters(), lr=args.lr)
    optimizer = torch_optimizer.Yogi(model1.parameters(), args.lr)
    num_epochs = args.num_iters

    # Initialize the plot
    pbar = tqdm(range(num_epochs))

    for epoch in pbar:
        optimizer.zero_grad()
        #loss, ve_mean, kl = model1.ELBO(x1, y1, None)
        #loss = -loss
        #loss.backward()
        if args.flag_update_kernel:
            loss, ve_mean, kl = model1.ELBO(x1, y1, None)
            retain_graph = True
        else:
            if epoch == 0:
                loss, ve_mean, kl = model1.ELBO(x1, y1, w, recompute_k=True, cache_k=True)  
            else:
                loss, ve_mean, kl = model1.ELBO(x1, y1, w, recompute_k=False, cache_k=False)
            retain_graph = False
        loss = -loss
        loss.backward(retain_graph=retain_graph)
        optimizer.step()

        # Update plot every 100 epochs
        if epoch % 100 == 0:
            model1.eval()
            with torch.no_grad():
                y_pred, y_var = model1.pred_f(x1, None)
                y_pred = (torch.sigmoid(y_pred) > 0.5).double()

                acc = torch.isclose(y_pred, y1).float().mean()
                print(f"Epoch {epoch} Train acc current task: {acc}")
                # print ve_mean and kl
                print(f"ve_mean {ve_mean.item()} kl {kl.item()}")
            model1.train()
    
    #w = model1.kernel.sample_from_spectral(args.rff_sample_size)
    #w = w.detach()

    predicted_classes = pred_boundary(xx, yy, model1, w)
    fig =plt.figure(figsize=(9,3))
    plt.contourf(xx, yy, predicted_classes.view(xx.shape).cpu().numpy(), alpha=0.5)
    plt.scatter(x1[:,0].detach().cpu().numpy(), x1[:,1].detach().cpu().numpy(),\
        c=y1.detach().cpu().numpy(),  edgecolor='k',s=10,label='train')
    #plt.scatter(X_test1[:,0], X_test1[:,1], c=Y_test1, edgecolor='k', s=5,label='test')
    plt.title("Decision Boundary After Task no. 1")
    plt.savefig('./hsgp1_'+args.kernel_type+'_'+str(args.inducing_size)+'z.png', format='png', bbox_inches='tight')
    plt.close()

    if args.flag_update_kernel:
        w = model1.kernel.sample_from_spectral(args.rff_sample_size)
        w = w.detach()

    Z_old, _, Kuu_old, _, _ = model1.Kuu_se(w)
    Z_old = Z_old.detach()
    Kuu_old = (torch.exp(model1.kernel.log_sf)* Kuu_old).detach()
    mv_old = model1.mv.detach()
    Lv_old = model1.Lv.detach()
    
    pivot = Z_interpolate[-1]
    #log_ls_forsort = model1.kernel.log_ls.clone().detach()
    if args.flag_update_kernel:
        #temp, _ = sort_with_se_kernel(x2, y2, pivot=pivot, log_ls=log_ls_forsort, subsampling_lag=args.subsampling_lag)#
        #temp, _ = sort_with_se_kerne2(x2, y2, pivot=pivot, log_ls=log_ls_forsort, subsampling_lag=args.subsampling_lag)#
        #temp = x2
        temp = x2[torch.randperm(x2.shape[0])]
        #temp = torch.flip(x2, (0,))
        #temp, _ = sort_with_dimension(x2, y2, dim=1)
        Z_interpolate = torch.concatenate([Z_interpolate, temp])
        prev_discrete_steps_ = Z_interpolate.shape[0]
    else:
        #Z_interpolate, _ = sort_with_se_kernel(x2, y2, pivot=pivot, log_ls=log_ls_forsort, subsampling_lag=args.subsampling_lag)#
        #Z_interpolate, _ = sort_with_se_kernel2(x2, y2, pivot=pivot, log_ls=log_ls_forsort, subsampling_lag=args.subsampling_lag)#
        #Z_interpolate = x2
        Z_interpolate = x2[torch.randperm(x2.shape[0])]
        #Z_interpolate = torch.flip(x2, (0,))
        #Z_interpolate, _ = sort_with_dimension(x2, y2, dim=1)
        prev_discrete_steps_ = prev_discrete_steps + Z_interpolate.shape[0]

    model2 = HIPPOOSVGP(kernel=deepcopy(kernel),
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
    
    #log_ls = model1.kernel.log_ls
    #log_sf = model1.kernel.log_sf
    model2.kernel.log_ls = nn.Parameter(torch.tensor(log_ls.detach().cpu().numpy(), dtype=torch.float64, device=device))
    model2.kernel.log_sf = nn.Parameter(torch.tensor(log_sf.detach().cpu().numpy(), dtype=torch.float64, device=device))
    
    if not args.flag_update_kernel:
        model2.kernel.log_ls.requires_grad = False  
        model2.kernel.log_sf.requires_grad = False 
    
    #optimizer = torch.optim.Adam(model2.parameters(), lr=args.lr)
    optimizer = torch_optimizer.Yogi(model2.parameters(), lr=args.lr)
    num_epochs = args.num_iters

    # Initialize the plot
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        optimizer.zero_grad()

        if args.flag_update_kernel:
            loss, ve_mean, kl = model2.ELBO(x2, y2, None)
            retain_graph = True
        else:
            if epoch == 0:
                loss, ve_mean, kl = model2.ELBO(x2, y2, w, recompute_k=True, cache_k=True)  
            else:
                loss, ve_mean, kl = model2.ELBO(x2, y2, w, recompute_k=False, cache_k=False)
            retain_graph = False
        loss = -loss
        loss.backward(retain_graph=retain_graph)
        optimizer.step()

        # Update plot every 100 epochs
        if epoch % 100 == 0:
            model2.eval()
            with torch.no_grad():
                if not args.flag_update_kernel:
                    y_pred, y_var = model2.pred_f(x2, w)
                else:
                    y_pred, y_var = model2.pred_f(x2, None)
                y_pred = (torch.sigmoid(y_pred) > 0.5).double()

                acc = torch.isclose(y_pred, y2).float().mean()
                print(f"Epoch {epoch} Train acc current task: {acc}")
                # print ve_mean and kl
                print(f"ve_mean {ve_mean.item()} kl {kl.item()}")
            model2.train()

    if args.flag_update_kernel:
        w = model2.kernel.sample_from_spectral(args.rff_sample_size)
        w = w.detach()

    predicted_classes = pred_boundary(xx, yy, model2, w)
    fig =plt.figure(figsize=(9,3))
    plt.contourf(xx, yy, predicted_classes.view(xx.shape).cpu().numpy(), alpha=0.5)
    plt.scatter(x1[:,0].detach().cpu().numpy(), x1[:,1].detach().cpu().numpy(),\
        c=y1.detach().cpu().numpy(),  edgecolor='k',s=10,label='train')
    #plt.scatter(X_test1[:,0], X_test1[:,1], c=Y_test1, edgecolor='k', s=5,label='test')
    plt.scatter(x2[:,0].detach().cpu().numpy(), x2[:,1].detach().cpu().numpy(),\
        c=y2.detach().cpu().numpy(),  edgecolor='k',s=10,label='train')
    #plt.scatter(X_test2[:,0], X_test2[:,1], c=Y_test2, edgecolor='k', s=5,label='test')
    plt.title("Decision Boundary After Task no. 2")
    plt.savefig('./hsgp2_'+args.kernel_type+'_'+str(args.inducing_size)+'z.png', format='png', bbox_inches='tight')
    plt.close()

    if args.flag_update_kernel:
        w = model2.kernel.sample_from_spectral(10000)
        w = w.detach()
    
    Z_old, _, Kuu_old, _, _ = model2.Kuu_se(w)
    Z_old = Z_old.detach()
    Kuu_old = (torch.exp(model2.kernel.log_sf)* Kuu_old).detach()
    mv_old = model2.mv.detach()
    Lv_old = model2.Lv.detach()

    pivot = Z_interpolate[-1]
    log_ls_forsort = model2.kernel.log_ls.clone().detach()
    if args.flag_update_kernel:
        #temp, _ = sort_with_se_kernel(x3, y3, pivot=pivot, log_ls=log_ls_forsort, subsampling_lag=args.subsampling_lag)#
        #temp, _ = sort_with_se_kernel2(x3, y3, pivot=pivot, log_ls=log_ls_forsort, subsampling_lag=args.subsampling_lag)#
        #temp = x3
        temp = x3[torch.randperm(x3.shape[0])]
        #temp = torch.flip(x3, (0,))
        #temp, _ = sort_with_dimension(x3, y3, dim=1)
        Z_interpolate = torch.concatenate([Z_interpolate, temp])
        prev_discrete_steps_ = Z_interpolate.shape[0]
    else:
        #Z_interpolate, _ = sort_with_se_kernel(x3, y3, pivot=pivot, log_ls=log_ls_forsort, subsampling_lag=args.subsampling_lag)#
        #Z_interpolate, _ = sort_with_se_kernel2(x3, y3, pivot=pivot, log_ls=log_ls_forsort, subsampling_lag=args.subsampling_lag)#
        #Z_interpolate = x3
        Z_interpolate = x3[torch.randperm(x3.shape[0])]
        #Z_interpolate = torch.flip(x3, (0,))
        #Z_interpolate, _ = sort_with_dimension(x3, y3, dim=1)
        prev_discrete_steps_ = prev_discrete_steps + Z_interpolate.shape[0]

    model3 = HIPPOOSVGP(kernel=deepcopy(kernel),
        likelihood=deepcopy(likelihood),
        prev_discrete_steps=prev_discrete_steps,
        Z_interpolate=Z_interpolate,
        rff_sample_size=args.rff_sample_size,
        hippo=hippo,
        inducing_size=args.inducing_size,
        device=device, 
        Z_old=Z_old,
        mv_old=mv_old,
        Lv_old=Lv_old,
        Kaa_old=Kuu_old,
        num_inducing_old=args.inducing_size,
        flag_update_kernel=args.flag_update_kernel).to(device)
    prev_discrete_steps = prev_discrete_steps_

    log_ls = model2.kernel.log_ls
    log_sf = model2.kernel.log_sf
    model3.kernel.log_ls = nn.Parameter(torch.tensor(log_ls.detach().cpu().numpy(), dtype=torch.float64, device=device))
    model3.kernel.log_sf = nn.Parameter(torch.tensor(log_sf.detach().cpu().numpy(), dtype=torch.float64, device=device))
    if not args.flag_update_kernel:
        model3.kernel.log_ls.requires_grad = False  
        model3.kernel.log_sf.requires_grad = False 
    
    #optimizer = torch.optim.Adam(model3.parameters(), lr=args.lr)
    optimizer = torch_optimizer.Yogi(model3.parameters(), lr=args.lr)
    num_epochs = args.num_iters

    # Initialize the plot
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        optimizer.zero_grad()

        if args.flag_update_kernel:
            loss, ve_mean, kl = model3.ELBO(x3, y3, None)
            retain_graph = True
        else:
            if epoch == 0:
                loss, ve_mean, kl = model3.ELBO(x3, y3, w, recompute_k=True, cache_k=True)  
            else:
                loss, ve_mean, kl = model3.ELBO(x3, y3, w, recompute_k=False, cache_k=False)
            retain_graph = False
        loss = -loss
        loss.backward(retain_graph=retain_graph)
        optimizer.step()

        # Update plot every 100 epochs
        if epoch % 100 == 0:
            model3.eval()
            with torch.no_grad():
                if not args.flag_update_kernel:
                    y_pred, y_var = model3.pred_f(x3, w)
                else:
                    y_pred, y_var = model3.pred_f(x3, None)
                y_pred = (torch.sigmoid(y_pred) > 0.5).double()

                acc = torch.isclose(y_pred, y3).float().mean()
                print(f"Epoch {epoch} Train acc current task: {acc}")
                # print ve_mean and kl
                print(f"ve_mean {ve_mean.item()} kl {kl.item()}")
            model3.train()

    if args.flag_update_kernel:
        w = model3.kernel.sample_from_spectral(args.rff_sample_size)
        w = w.detach()

    predicted_classes = pred_boundary(xx, yy, model3, w)
    fig =plt.figure(figsize=(9,3))
    plt.contourf(xx, yy, predicted_classes.view(xx.shape).cpu().numpy(), alpha=0.5)
    plt.scatter(x1[:,0].detach().cpu().numpy(), x1[:,1].detach().cpu().numpy(),\
        c=y1.detach().cpu().numpy(),  edgecolor='k',s=10,label='train')
    #plt.scatter(X_test1[:,0], X_test1[:,1], c=Y_test1, edgecolor='k', s=5,label='test')
    plt.scatter(x2[:,0].detach().cpu().numpy(), x2[:,1].detach().cpu().numpy(),\
        c=y2.detach().cpu().numpy(),  edgecolor='k',s=10,label='train')
    #plt.scatter(X_test2[:,0], X_test2[:,1], c=Y_test2, edgecolor='k', s=5,label='test')
    plt.scatter(x3[:,0].detach().cpu().numpy(), x3[:,1].detach().cpu().numpy(),\
        c=y3.detach().cpu().numpy(),  edgecolor='k',s=10,label='train')
    #plt.scatter(X_test3[:,0], X_test3[:,1], c=Y_test3, edgecolor='k', s=5,label='test')
    plt.title("Decision Boundary After Task no. 3")
    plt.savefig('./hsgp3_'+args.kernel_type+'_'+str(args.inducing_size)+'z.png', format='png', bbox_inches='tight')
    plt.close()

    if args.flag_update_kernel:
        w = model3.kernel.sample_from_spectral(10000)
        w = w.detach()
    """
    Z_old, _, Kuu_old, _, _ = model3.Kuu_se(w)
    Z_old = Z_old.detach()
    Kuu_old = (torch.exp(model3.kernel.log_sf)* Kuu_old).detach()
    mv_old = model3.mv.detach()
    Lv_old = model3.Lv.detach()

    pivot = Z_interpolate[-1]
    log_ls_forsort = model3.kernel.log_ls.clone().detach()
    if args.flag_update_kernel:
        #temp, _ = sort_with_se_kernel(x4, y4, pivot=pivot, log_ls=log_ls_forsort, subsampling_lag=args.subsampling_lag)#
        #temp, _ = sort_with_se_kernel2(x4, y4, pivot=pivot, log_ls=log_ls_forsort, subsampling_lag=args.subsampling_lag)#
        #temp = x4
        #temp = x4[torch.randperm(x4.shape[0])]
        #temp = torch.flip(x4, (0,))
        temp, _ = sort_with_dimension(x4, y4, dim=1)
        Z_interpolate = torch.concatenate([Z_interpolate, temp])
        prev_discrete_steps_ = Z_interpolate.shape[0]
    else:
        #Z_interpolate, _ = sort_with_se_kernel(x4, y4, pivot=pivot, log_ls=log_ls_forsort, subsampling_lag=args.subsampling_lag)#
        #Z_interpolate, _ = sort_with_se_kernel2(x4, y4, pivot=pivot, log_ls=log_ls_forsort, subsampling_lag=args.subsampling_lag)#
        #Z_interpolate = x4
        #Z_interpolate = x4[torch.randperm(x4.shape[0])]
        #Z_interpolate = torch.flip(x4, (0,))
        Z_interpolate, _ = sort_with_dimension(x4, y4, dim=1)
        prev_discrete_steps_ = prev_discrete_steps + Z_interpolate.shape[0]

    model4 = HIPPOOSVGP(kernel=deepcopy(kernel),
        likelihood=deepcopy(likelihood),
        prev_discrete_steps=prev_discrete_steps,
        Z_interpolate=Z_interpolate,
        rff_sample_size=args.rff_sample_size,
        hippo=hippo,
        inducing_size=args.inducing_size,
        device=device, 
        Z_old=Z_old,
        mv_old=mv_old,
        Lv_old=Lv_old,
        Kaa_old=Kuu_old,
        num_inducing_old=args.inducing_size,
        flag_update_kernel=args.flag_update_kernel).to(device)
    prev_discrete_steps = prev_discrete_steps_
    
    log_ls = model3.kernel.log_ls
    log_sf = model3.kernel.log_sf
    model4.kernel.log_ls = nn.Parameter(torch.tensor(log_ls.detach().cpu().numpy(), dtype=torch.float64, device=device))
    model4.kernel.log_sf = nn.Parameter(torch.tensor(log_sf.detach().cpu().numpy(), dtype=torch.float64, device=device))
    if not args.flag_update_kernel:
        model4.kernel.log_ls.requires_grad = False  
        model4.kernel.log_sf.requires_grad = False

    #optimizer = torch.optim.Adam(model4.parameters(), lr=args.lr)
    optimizer = torch_optimizer.Yogi(model4.parameters(), lr=args.lr)
    num_epochs = args.num_iters

    # Initialize the plot
    pbar = tqdm(range(num_epochs))

    for epoch in pbar:
        optimizer.zero_grad()
        if args.flag_update_kernel:
            loss, ve_mean, kl = model4.ELBO(x4, y4, w)#None)
            retain_graph = True
        else:
            if epoch == 0:
                loss, ve_mean, kl = model4.ELBO(x4, y4, w, recompute_k=True, cache_k=True)  
            else:
                loss, ve_mean, kl = model4.ELBO(x4, y4, w, recompute_k=False, cache_k=False)
            retain_graph = False
        loss = -loss
        loss.backward(retain_graph=retain_graph)
        optimizer.step()

        # Update plot every 100 epochs
        if epoch % 100 == 0:
            model4.eval()
            with torch.no_grad():
                if not args.flag_update_kernel:
                    y_pred, y_var = model4.pred_f(x4, w)#None)
                else:
                    y_pred, y_var = model4.pred_f(x4, None)
                y_pred = (torch.sigmoid(y_pred) >= 0.5).double()
                acc = torch.isclose(y_pred, y4).float().mean()
                print(f"Epoch {epoch} Train acc current task: {acc}")
                # print ve_mean and kl
                print(f"ve_mean {ve_mean.item()} kl {kl.item()}")
            model4.train()

    if args.flag_update_kernel:
        w = model4.kernel.sample_from_spectral(args.rff_sample_size)
        w = w.detach()
    predicted_classes = pred_boundary(xx, yy, model4, w)
    fig =plt.figure(figsize=(9,3))
    plt.contourf(xx, yy, predicted_classes.view(xx.shape).cpu().numpy(), alpha=0.5)
    plt.scatter(x1[:,0].detach().cpu().numpy(), x1[:,1].detach().cpu().numpy(),\
        c=y1.detach().cpu().numpy(),  edgecolor='k',s=10,label='train')
    #plt.scatter(X_test1[:,0], X_test1[:,1], c=Y_test1, edgecolor='k', s=5,label='test')
    plt.scatter(x2[:,0].detach().cpu().numpy(), x2[:,1].detach().cpu().numpy(),\
        c=y2.detach().cpu().numpy(),  edgecolor='k',s=10,label='train')
    #plt.scatter(X_test2[:,0], X_test2[:,1], c=Y_test2, edgecolor='k', s=5,label='test')
    plt.scatter(x3[:,0].detach().cpu().numpy(), x3[:,1].detach().cpu().numpy(),\
        c=y3.detach().cpu().numpy(),  edgecolor='k',s=10,label='train')
    #plt.scatter(X_test3[:,0], X_test3[:,1], c=Y_test3, edgecolor='k', s=5,label='test')
    plt.scatter(x4[:,0].detach().cpu().numpy(), x4[:,1].detach().cpu().numpy(),\
        c=y4.detach().cpu().numpy(),  edgecolor='k',s=10,label='train')
    #plt.scatter(X_test4[:,0], X_test4[:,1], c=Y_test4, edgecolor='k', s=5,label='test')
    plt.title("Decision Boundary After Task no. 4")
    plt.savefig('./hsgp4_'+args.kernel_type+'_'+str(args.inducing_size)+'z.png', format='png', bbox_inches='tight')
    plt.close()
    """

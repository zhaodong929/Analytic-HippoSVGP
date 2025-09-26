import numpy as np
import torch
import matplotlib.pyplot as plt  
from hipposvgp.svgp import OSVGP
from hipposvgp.likelihood import Bernoulli
from hipposvgp.multidim import SE_kernel
import torch.nn as nn
import argparse
from copy import deepcopy
import torch_optimizer
from scripts.multidim.dataset import get_data_moon, get_data_banana
from hipposvgp.utils import pred_boundary
from tqdm import tqdm
import random


def setup():
    parser=argparse.ArgumentParser('Argument Parser')
    parser.add_argument('--data',type=str,default='moon')
    parser.add_argument('--kernel_type',type=str,default='se')
    parser.add_argument('--inducing_size',type=int,default=20)
    parser.add_argument('--num_iters',type=int,default=2000)
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
    seed_everything(7)
    args=setup()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    else:
        raise NotImplementedError

    model1 = OSVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_size=args.inducing_size,
        device=device,
        num_inducing_old=None,
        mv_old=None,
        Lv_old=None,
        Kaa_old=None,
        Z_old=None,
        dtype=torch.float64,
    ).to(device)
    model1.Z = torch.nn.Parameter(x1[torch.randperm(x1.shape[0])[:args.inducing_size], :])
    
    #optimizer = torch.optim.Adam(model1.parameters(), lr=args.lr)
    optimizer = torch_optimizer.Yogi(model1.parameters(), lr=args.lr)
    num_epochs = args.num_iters

    # Initialize the plot
    pbar = tqdm(range(num_epochs))

    for epoch in pbar:
        optimizer.zero_grad()
        loss, ve_mean, kl = model1.ELBO(x1, y1)  
        loss = -loss
        loss.backward()
        optimizer.step()

        # Update plot every 100 epochs
        if epoch % 100 == 0:
            model1.eval()
            with torch.no_grad():
                y_pred, y_var = model1.pred_f(x1)
                y_pred = (torch.sigmoid(y_pred) > 0.5).double()

                acc = torch.isclose(y_pred, y1).float().mean()
                print(f"Epoch {epoch} Train acc current task: {acc}")
                # print ve_mean and kl
                print(f"ve_mean {ve_mean.item()} kl {kl.item()}")
            model1.train()
    #print(model1.kernel.log_ls)
    #print(model1.kernel.log_sf)
    #exit()
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                        np.arange(y_min, y_max, 0.05)) 
      
    predicted_classes = pred_boundary(xx, yy, model1)
    fig =plt.figure(figsize=(9,3))
    plt.contourf(xx, yy, predicted_classes.view(xx.shape).cpu().numpy(), alpha=0.5)
    plt.scatter(x1[:,0].detach().cpu().numpy(), x1[:,1].detach().cpu().numpy(),\
        c=y1.detach().cpu().numpy(),  edgecolor='k',s=10,label='train')
    #plt.scatter(X_test1[:,0], X_test1[:,1], c=Y_test1, edgecolor='k', s=5,label='test')
    plt.scatter(model1.Z[:,0].detach().cpu().numpy(), model1.Z[:,1].detach().cpu().numpy(), color='red', s=20, label = 'induding')
    plt.title("Decision Boundary After Task no. 1")
    plt.xlim(x_min+0.05, x_max-0.05)
    plt.ylim(y_min+0.05, y_max-0.05)
    plt.savefig('./svgp1_'+args.kernel_type+'_'+str(args.inducing_size)+'z.png', format='png', bbox_inches='tight')
    plt.close()
    
    mv_old = model1.mv.clone().detach()
    Lv_old = deepcopy(model1.Lv).detach()
    Kuu_old, _, _ = model1.Kuu_se()
    Kuu_old = Kuu_old.detach()
    num_inducing_old = model1.inducing_size
    Z_old = torch.nn.Parameter(model1.Z.clone()).detach()

    model2 = OSVGP(kernel=deepcopy(kernel),
        likelihood=deepcopy(likelihood),
        inducing_size=args.inducing_size, 
        device=device, 
        Z_old=Z_old,
        mv_old=mv_old,
        Lv_old=Lv_old,
        Kaa_old=Kuu_old,
        num_inducing_old=num_inducing_old,
        ).to(device)
    
    M_old = int(args.inducing_size* 4/7)
    M_new = int(args.inducing_size - M_old)
    old_Z = model1.Z.clone()[torch.randperm(model1.Z.shape[0])[0:M_old], :]
    new_Z = x2[torch.randperm(x2.shape[0])[0:M_new], :]
    Z2 = torch.vstack((old_Z, new_Z))
    model2.Z = torch.nn.Parameter(Z2)
    
    log_ls = model1.kernel.log_ls
    log_sf = model1.kernel.log_sf
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
        loss, ve_mean, kl = model2.ELBO(x2, y2) 
        loss = -loss
        loss.backward()
        optimizer.step()

        # Update plot every 100 epochs
        if epoch % 100 == 0:
            model2.eval()
            with torch.no_grad():
                y_pred, y_var = model2.pred_f(x2)
                y_pred = (torch.sigmoid(y_pred) > 0.5).double()

                acc = torch.isclose(y_pred, y2).float().mean()
                print(f"Epoch {epoch} Train acc current task: {acc}")
                # print ve_mean and kl
                print(f"ve_mean {ve_mean.item()} kl {kl.item()}")
            model2.train()

    predicted_classes = pred_boundary(xx, yy, model2)
    fig =plt.figure(figsize=(9,3))
    plt.contourf(xx, yy, predicted_classes.view(xx.shape).cpu().numpy(), alpha=0.5)
    plt.scatter(x1[:,0].detach().cpu().numpy(), x1[:,1].detach().cpu().numpy(),\
        c=y1.detach().cpu().numpy(),  edgecolor='k',s=10,label='train')
    #plt.scatter(X_test1[:,0], X_test1[:,1], c=Y_test1, edgecolor='k', s=5,label='test')
    plt.scatter(x2[:,0].detach().cpu().numpy(), x2[:,1].detach().cpu().numpy(),\
        c=y2.detach().cpu().numpy(),  edgecolor='k',s=10,label='train')
    #plt.scatter(X_test2[:,0], X_test2[:,1], c=Y_test2, edgecolor='k', s=5,label='test')
    plt.scatter(model2.Z[:,0].detach().cpu().numpy(), model2.Z[:,1].detach().cpu().numpy(), color='red', s=20, label = 'induding')
    plt.title("Decision Boundary After Task no. 2")
    plt.xlim(x_min+0.05, x_max-0.05)
    plt.ylim(y_min+0.05, y_max-0.05)
    plt.savefig('./svgp2_'+args.kernel_type+'_'+str(args.inducing_size)+'z.png', format='png', bbox_inches='tight')
    plt.close()

    mv_old = model2.mv.clone()
    Lv_old = deepcopy(model2.Lv)
    Kuu_old, _, _ = model2.Kuu_se()
    num_inducing_old = model2.inducing_size
    Z_old = torch.nn.Parameter(model2.Z.clone()).detach()

    model3 = OSVGP(kernel=deepcopy(kernel),
        likelihood=deepcopy(likelihood),
        inducing_size=args.inducing_size, 
        device=device, 
        Z_old=Z_old,
        mv_old=mv_old,
        Lv_old=Lv_old,
        Kaa_old=Kuu_old,
        num_inducing_old=num_inducing_old,
        ).to(device)
    
    M_old = int(args.inducing_size* 0.7)
    M_new = int(args.inducing_size - M_old)
    old_Z = model2.Z.clone()[torch.randperm(model2.Z.shape[0])[0:M_old], :]
    new_Z = x3[torch.randperm(x3.shape[0])[0:M_new], :]
    Z2 = torch.vstack((old_Z, new_Z))
    model3.Z = torch.nn.Parameter(Z2)
    
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
        loss, ve_mean, kl = model3.ELBO(x3, y3)  
        loss = -loss
        loss.backward()
        optimizer.step()

        # Update plot every 100 epochs
        if epoch % 100 == 0:
            model3.eval()
            with torch.no_grad():
                y_pred, y_var = model3.pred_f(x3)
                y_pred = (torch.sigmoid(y_pred) > 0.5).double()

                acc = torch.isclose(y_pred, y3).float().mean()
                print(f"Epoch {epoch} Train acc current task: {acc}")
                # print ve_mean and kl
                print(f"ve_mean {ve_mean.item()} kl {kl.item()}")
            model3.train()

    predicted_classes = pred_boundary(xx, yy, model3)
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
    plt.scatter(model3.Z[:,0].detach().cpu().numpy(), model3.Z[:,1].detach().cpu().numpy(), color='red', s=20, label = 'induding')
    plt.title("Decision Boundary After Task no. 3")
    plt.xlim(x_min+0.05, x_max-0.05)
    plt.ylim(y_min+0.05, y_max-0.05)
    plt.savefig('./svgp3_'+args.kernel_type+'_'+str(args.inducing_size)+'z.png', format='png', bbox_inches='tight')
    plt.close()

    """
    mv_old = model3.mv.clone()
    Lv_old = deepcopy(model3.Lv)
    Kuu_old, _, _ = model3.Kuu_se()
    num_inducing_old = model3.inducing_size
    Z_old = torch.nn.Parameter(model3.Z.clone()).detach()

    model4 = OSVGP(kernel=deepcopy(kernel),
        likelihood=deepcopy(likelihood),
        inducing_size=args.inducing_size, 
        device=device, 
        Z_old=Z_old,
        mv_old=mv_old,
        Lv_old=Lv_old,
        Kaa_old=Kuu_old,
        num_inducing_old=num_inducing_old,
        ).to(device)
    
    M_old = int(args.inducing_size* 3/4)
    M_new = int(args.inducing_size - M_old)
    old_Z = model3.Z.clone()[torch.randperm(model3.Z.shape[0])[0:M_old], :]
    new_Z = x4[torch.randperm(x4.shape[0])[0:M_new], :]
    Z2 = torch.vstack((old_Z, new_Z))
    model4.Z = torch.nn.Parameter(Z2)
    
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
        loss, ve_mean, kl = model4.ELBO(x4, y4) 
        loss = -loss
        loss.backward()
        optimizer.step()

        # Update plot every 100 epochs
        if epoch % 100 == 0:
            model4.eval()
            with torch.no_grad():
                y_pred, y_var = model4.pred_f(x4)
                y_pred = (torch.sigmoid(y_pred) > 0.5).double()

                acc = torch.isclose(y_pred, y4).float().mean()
                print(f"Epoch {epoch} Train acc current task: {acc}")
                # print ve_mean and kl
                print(f"ve_mean {ve_mean.item()} kl {kl.item()}")
            model4.train()

    predicted_classes = pred_boundary(xx, yy, model4)
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
    plt.scatter(model4.Z[:,0].detach().cpu().numpy(), model4.Z[:,1].detach().cpu().numpy(), color='red', s=20, label = 'induding')
    plt.title("Decision Boundary After Task no. 4")
    plt.xlim(x_min+0.05, x_max-0.05)
    plt.ylim(y_min+0.05, y_max-0.05)
    plt.savefig('./svgp4_'+args.kernel_type+'_'+str(args.inducing_size)+'z.png', format='png', bbox_inches='tight')
    plt.close()
    """
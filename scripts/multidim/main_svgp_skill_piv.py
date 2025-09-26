import numpy as np
import torch
import matplotlib.pyplot as plt  
from hipposvgp.svgp import OSVGP
from hipposvgp.likelihood import GaussianLikelihood
from hipposvgp.multidim import SE_kernel
import torch.nn as nn
import argparse
from copy import deepcopy
import torch_optimizer
from tqdm import tqdm
from scripts.multidim.dataset import Powerplant
from hipposvgp.utils import sort_with_l2, sort_with_se_kernel, sort_with_dimension, pivoted_cholesky_init
import random


def setup():
    parser=argparse.ArgumentParser('Argument Parser')
    parser.add_argument('--data',type=str,default='skill')
    parser.add_argument('--kernel_type',type=str,default='se')
    parser.add_argument('--inducing_size',type=int,default=256)
    parser.add_argument('--num_iters',type=int,default=2000)
    parser.add_argument('--lr',type=float,default=5e-3)
    parser.add_argument('--run_num',type=int,default=1)
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

    if args.data == 'powerplant':
        dataobj = Powerplant()
    elif args.data == 'skill':
        dataobj = Skillcraft()
    elif args.data == 'elevator':
        dataobj = Elevators()
    else:
        raise NotImplementedError

    train_set = dataobj.train_dataset
    train_x, train_y = train_set[:]

    train_x, train_y = sort_with_dimension(train_x, train_y)
    #train_x, train_y = sort_with_l2(train_x, train_y)
    train_x = train_x.to(device)
    train_y = train_y.to(device)

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
    #model1.Z = torch.nn.Parameter(ini_x[torch.randperm(ini_x.shape[0])[:args.inducing_size], :])
    Z_input = ini_x
    kernel_matrix = model1.kernel(Z_input, Z_input)
    Z = pivoted_cholesky_init(Z_input, kernel_matrix, min(args.inducing_size, Z_input.shape[0]))
    model1.Z = torch.nn.Parameter(Z)
    model1.Z.requires_grad = False
   
    # fullGP kernel hyper
    model1.kernel.log_ls = nn.Parameter(torch.tensor(np.array([3.4783, -0.9032, -5.0846,  2.4429,  2.7161,  1.5580,  1.9396, -0.0452,
        -0.9826,  0.1504,  0.3997,  1.8558,  0.2807,  3.1336,  2.1943, -0.6936,
        -2.8923]), dtype=torch.float64, device=device))
    model1.kernel.log_sf = nn.Parameter(torch.tensor(np.array([1.5601]), dtype=torch.float64, device=device))
    model1.likelihood.log_variance = nn.Parameter(torch.tensor(np.array([-1.7071]), dtype=torch.float64, device=device))
   
    if not args.flag_update_kernel:
        model1.kernel.log_ls.requires_grad = False
        model1.kernel.log_sf.requires_grad = False
        model1.likelihood.log_variance.requires_grad = False
    
    
    #optimizer = torch.optim.Adam(model1.parameters(), lr=args.lr)
    optimizer = torch_optimizer.Yogi(model1.parameters(), args.lr)
    num_epochs = args.num_iters

    # Initialize the plot
    pbar = tqdm(range(num_epochs))

    for epoch in pbar:
        optimizer.zero_grad()
        loss, ve_mean, kl = model1.ELBO(ini_x, ini_y)  
        loss = -loss
        loss.backward()
        optimizer.step()

        # Update plot every 100 epochs
        if epoch % 100 == 0:
            model1.eval()
            with torch.no_grad():
                y_mean, f_var = model1.pred_f(ini_x)
                rmse = torch.sqrt(torch.sum((y_mean - ini_y)**2) / y_mean.shape[0])
                print(f"Epoch {epoch} Train rmse current task: {rmse}")
                # print ve_mean and kl
                print(f"ve_mean {ve_mean.item()} kl {kl.item()}")
                print("log_variance: {}".format(model1.likelihood.log_variance.item()))
            model1.train()

    
    y_mean, f_var = model1.pred_f(test_x_list[0])      
    y_var = f_var + torch.exp(model1.likelihood.log_variance) * torch.ones_like(f_var)   # [N, 1]
    rmse_test = torch.sqrt(torch.sum((y_mean - test_y_list[0])**2) / y_mean.shape[0])
    y_var = y_var.squeeze(1)
    chol_cov = y_var.pow(0.5)
    nlpd_test = torch.sum(torch.log(chol_cov)) + 0.5* y_mean.shape[0]* torch.log(torch.tensor(2* np.pi))
    nlpd_test += 0.5* torch.sum((test_y_list[0] - y_mean).pow(2).squeeze(1)/  y_var)
    nlpd_test /= y_mean.shape[0]
    print('t: {}, test rmse: {}, nlpd: {}'.format(0, rmse_test.item(), nlpd_test.item()))
    """
    y_mean, f_var = model1.pred_f(train_x)      
    y_var = f_var + torch.exp(model1.likelihood.log_variance) * torch.ones_like(f_var)   # [N, 1]
    rmse_train = torch.sqrt(torch.sum((y_mean - train_y)**2) / y_mean.shape[0])
    y_var = y_var.squeeze(1)
    chol_cov = y_var.pow(0.5)
    nlpd_train = torch.sum(torch.log(chol_cov)) + 0.5* y_mean.shape[0]* torch.log(torch.tensor(2* np.pi))
    nlpd_train += 0.5* torch.sum((train_y - y_mean).pow(2).squeeze(1)/  y_var)
    nlpd_train /= y_mean.shape[0]
    print('t: {}, train rmse: {}, nlpd: {}'.format(0, rmse_train.item(), nlpd_train.item()))
    """
    torch.save(model1.state_dict(), './models/svgp_model0')

    prev_model = model1

    for t in range(9):
        x = train_x_list[t+1]
        y = train_y_list[t+1]
        mv_old = prev_model.mv.clone().detach()
        Lv_old = deepcopy(prev_model.Lv).detach()
        Kuu_old, _, _ = prev_model.Kuu_se()
        Kuu_old = Kuu_old.detach()
        num_inducing_old = prev_model.inducing_size
        Z_old = torch.nn.Parameter(prev_model.Z.clone()).detach()

        model = OSVGP(kernel=deepcopy(kernel),
            likelihood=deepcopy(likelihood),
            inducing_size=args.inducing_size, 
            device=device, 
            Z_old=Z_old,
            mv_old=mv_old,
            Lv_old=Lv_old,
            Kaa_old=Kuu_old,
            num_inducing_old=num_inducing_old,
            ).to(device)
        
        old_Z = Z_old.clone()
        new_Z = x
        Z_input = torch.vstack((old_Z, new_Z))
        kernel_matrix = prev_model.kernel(Z_input, Z_input)
        Z2 = pivoted_cholesky_init(Z_input, kernel_matrix, min(args.inducing_size, Z_input.shape[0]))
        model.Z = torch.nn.Parameter(Z2)
        
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

        #optimizer = torch.optim.Adam(model2.parameters(), lr=args.lr)
        optimizer = torch_optimizer.Yogi(model.parameters(), lr=args.lr)
        num_epochs = args.num_iters

        # Initialize the plot
        #pbar = tqdm(range(num_epochs))
        pbar = range(num_epochs)
        for epoch in pbar:
            optimizer.zero_grad()
            loss, ve_mean, kl = model.ELBO(x, y) 
            loss = -loss
            loss.backward()
            optimizer.step()

            # Update plot every 100 epochs
            """
            if epoch % 100 == 0:
                model.eval()
                with torch.no_grad():
                    y_mean, f_var = model.pred_f(x)
                    rmse = torch.sqrt(torch.sum((y_mean - y)**2) / y_mean.shape[0])
                    print(f"Epoch {epoch} Train rmse current task: {rmse}")
                    # print ve_mean and kl
                    print(f"ve_mean {ve_mean.item()} kl {kl.item()}")
                model.train()
            """
        
        res_rmse = []
        res_nlpd = []
        for k in range(t+2):
            y_mean, f_var = model.pred_f(test_x_list[k])      
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
        """
        y_mean, f_var = model.pred_f(train_x)      
        y_var = f_var + torch.exp(model.likelihood.log_variance) * torch.ones_like(f_var)   # [N, 1]
        rmse_train = torch.sqrt(torch.sum((y_mean - train_y)**2) / y_mean.shape[0])
        y_var = y_var.squeeze(1)
        chol_cov = y_var.pow(0.5)
        nlpd_train = torch.sum(torch.log(chol_cov)) + 0.5* y_mean.shape[0]* torch.log(torch.tensor(2* np.pi))
        nlpd_train += 0.5* torch.sum((train_y - y_mean).pow(2).squeeze(1)/  y_var)
        nlpd_train /= y_mean.shape[0]
        print('t: {}, train rmse: {}, nlpd: {}'.format(t+1, rmse_train.item(), nlpd_train.item()))
        """
        torch.save(model.state_dict(), './models/svgp_model'+str(t+1))
        prev_model = model
  

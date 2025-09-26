import numpy as np
import torch 
from scripts.onedim.solar.baseline import SVGP, OSGPR_VFE, FullGP
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import argparse
import random
from hipposvgp.utils import pivoted_cholesky_init

def setup():
    parser=argparse.ArgumentParser('Argument Parser')
    parser.add_argument('--data',type=str,default='baseband')
    parser.add_argument('--kernel_type',type=str,default='se')
    parser.add_argument('--inducing_size',type=int,default=100)
    parser.add_argument('--num_iters',type=int,default=1000)
    parser.add_argument('--lr',type=int,default=0.01)
    parser.add_argument('--num_seed',type=int,default=0)
    parser.add_argument('--model_dir',type=str,default='./models')
    parser.add_argument('--model_full_dir',type=str,default='./models')
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
    seed_everything()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.data == 'solar':
        
        data_train = np.load('data/solar_data_train.npy')
        data_test = np.load('data/solar_data_test.npy')
        data = np.vstack([data_train, data_test])
        ind = np.argsort(data[:,0])
        data = data[ind]
        ind_test = np.random.choice(np.arange(data.shape[0]), size=91)
        ind_train = []
        for i in range(data.shape[0]):
            if i in ind_test:
                continue
            else:
                ind_train.append(i)
        data_test = data[ind_test]
        data_train = data[ind_train]

        ind = np.argsort(data_train[:,0])
        data_train = data_train[ind]
        x_test = torch.tensor(np.arange(-200., 200., 2.).reshape((-1,1))).to(device)
    else:
        data = np.load('data/baseband.npy') # [352672, 1]
        y_total = data[:18000]
        x_total = np.expand_dims((1+np.arange(18000))/ 1800, 1)
        data = np.concatenate([x_total, y_total], 1) 
        data_train = data[::2] 
        data_test = data[1::2]
        x_test = torch.tensor(np.arange(-0.02, 10.02, .002).reshape((-1,1))).to(device)

    x = torch.tensor(np.expand_dims(data_train[:,0], 1)).to(device)
    y = torch.tensor(np.expand_dims(data_train[:,1], 1)).to(device)
    x_t = np.expand_dims(data_test[:,0], 1)
    y_t = np.expand_dims(data_test[:,1], 1)

    x_list = []
    y_list = []
    split_size = x.shape[0] // 10
    split_size_t = x_t.shape[0] // 10
    for i in range(9):
        x_list.append(x[i*split_size:(i+1)*split_size])
        y_list.append(y[i*split_size:(i+1)*split_size])
    x_list.append(x[9*split_size:])
    y_list.append(y[9*split_size:])
    
    #fullGP = FullGP(X=x, y=y, device=device, kernel_type=args.kernel_type).to(device)
    #fullGP.load_state_dict(torch.load(args.model_full_dir+'/steps_nll2000'))
    # See below for the fullGP hyperparameters trained on the first 2 tasks for solar & audio:
    if args.data == 'solar':
        log_ls = 1.4108
        log_sf = -0.5334
        log_noise = -4.5
    else:
        log_ls = -3.8722
        log_sf = 0.3478
        log_noise = -4.

    
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    seed_everything(args.num_seed)

    
    model1 = SVGP(X=x_list[0], y=y_list[0], inducing_size=min(args.inducing_size, x_list[0].shape[0]),\
        device=device, kernel_type=args.kernel_type).to(device)
    model1.log_noise = nn.Parameter(torch.tensor(np.array([log_noise]), dtype=torch.float64, device=device))
    model1.Kernel.log_ls = nn.Parameter(torch.tensor(np.array([log_ls]), dtype=torch.float64, device=device))
    model1.Kernel.log_sf = nn.Parameter(torch.tensor(np.array([log_sf]), dtype=torch.float64, device=device))
    
    Z_input = x_list[0]
    kernel_matrix = model1.Kernel(Z_input, Z_input)
    Z = pivoted_cholesky_init(Z_input, kernel_matrix, min(args.inducing_size, Z_input.shape[0]))
    model1.inducing_size = Z.shape[0]
    model1.z = nn.Parameter(Z)
    
    model1.train(lr=args.lr, iters=args.num_iters)
    torch.save(model1.state_dict(), args.model_dir+'/model1_ind_' + str(args.inducing_size)+ '_steps_elbo'+str(args.num_iters))

    Z_old = torch.as_tensor(model1.z)
    M = Z_old.shape[0]
    old_Z = Z_old.clone()
    new_Z = x_list[1]
    Z_input = torch.vstack((old_Z, new_Z))
    kernel_matrix = model1.Kernel(Z_input, Z_input)
    Z = pivoted_cholesky_init(Z_input, kernel_matrix, min(args.inducing_size, Z_input.shape[0]))
    
    Z_old = Z_old.detach()
    mu_old, Su_old = model1.pred(Z_old, full_cov=True, flag_noise=False)
    mu_old = mu_old.detach()
    Su_old = Su_old.detach()
    Kaa_old = model1.Kernel(Z_old, Z_old).detach()

    for i in range(1, 10):
        cur_model = OSGPR_VFE(X=x_list[i], y=y_list[i], mu_old=mu_old, Su_old=Su_old, Kaa_old=Kaa_old, \
                        Z_old=Z_old, Z=Z, device=device, kernel_type=args.kernel_type).to(device)
        
        cur_model.log_noise = nn.Parameter(torch.tensor(np.array([log_noise]), dtype=torch.float64, device=device))
        cur_model.Kernel.log_ls = nn.Parameter(torch.tensor(np.array([log_ls]), dtype=torch.float64, device=device))
        cur_model.Kernel.log_sf = nn.Parameter(torch.tensor(np.array([log_sf]), dtype=torch.float64, device=device))
        
        cur_model.train(lr=args.lr, iters=args.num_iters)
        torch.save(cur_model.state_dict(), args.model_dir+ '/model'+str(i+1)+'_ind_' + str(args.inducing_size)+ '_steps_elbo'+str(args.num_iters))

        if i < 9:
            Z_old = torch.as_tensor(cur_model.z)
            M = Z_old.shape[0]
            old_Z = Z_old.clone()
            new_Z = x_list[i+1]
            Z_input = torch.vstack((old_Z, new_Z))
            kernel_matrix = cur_model.Kernel(Z_input, Z_input)
            Z = pivoted_cholesky_init(Z_input, kernel_matrix, min(args.inducing_size, Z_input.shape[0]))
            
            Z_old = Z_old.detach()
            mu_old, Su_old = cur_model.pred(Z_old, full_cov=True, flag_noise=False)
            mu_old = mu_old.detach()
            Su_old = Su_old.detach()
            Kaa_old = cur_model.Kernel(Z_old, Z_old).detach()

  
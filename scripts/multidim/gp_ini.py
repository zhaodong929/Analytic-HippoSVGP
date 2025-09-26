import numpy as np
import torch
import matplotlib.pyplot as plt  
from scripts.multidim.fullGP import FullGP
from hipposvgp.likelihood import GaussianLikelihood
from hipposvgp.kernel import SE_kernel
import argparse
from scripts.multidim.dataset import Powerplant, Skillcraft
import random

def setup():
    parser=argparse.ArgumentParser('Argument Parser')
    parser.add_argument('--data',type=str,default='skill')
    parser.add_argument('--kernel_type',type=str,default='se')
    parser.add_argument('--inducing_size',type=int,default=256)
    parser.add_argument('--num_iters',type=int,default=2000)
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
    seed_everything(seed=42)
    args=setup()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.data == 'powerplant':
        dataobj = Powerplant()
    elif args.data == 'skill':
        dataobj = Skillcraft()
    else:
        raise NotImplementedError

    train_set = dataobj.train_dataset
    train_x, train_y = train_set[:]
    ind_shuffle = torch.randperm(train_x.shape[0])
    train_x = train_x[ind_shuffle]
    train_y = train_y[ind_shuffle]

    train_x = train_x.to(device)
    train_y = train_y.to(device)

    train_x_tuple = train_x.chunk(10)
    train_y_tuple = train_y.chunk(10)

    train_x_list = []
    train_y_list = []
    test_x_list = []
    test_y_list = []
    for i in range(5):
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

    model1 = FullGP(X=ini_x, y=ini_y, device=device).to(device)
    model1.train()
    print(model1.Kernel.log_sf)
    print(model1.Kernel.log_ls)
    print(model1.log_noise)
    nll, rmse = model1.nll_rmse(X=torch.concatenate(test_x_list), y=torch.concatenate(test_y_list), flag_noise = True)
    print(nll)
    print(rmse)

    nll, rmse = model1.nll_rmse(X=torch.concatenate(train_x_list), y=torch.concatenate(train_y_list), flag_noise = True)
    print(nll)
    print(rmse)
    
import numpy as np
import torch
import matplotlib.pyplot as plt  
import torch.nn as nn
import argparse
from scripts.onedim.solar.vff import *
from scripts.onedim.solar.baseline import FullGP
import copy 


def setup():
    parser=argparse.ArgumentParser('Argument Parser')
    parser.add_argument('--data',type=str,default='solar')
    parser.add_argument('--inducing_size',type=int,default=200)
    parser.add_argument('--num_iters',type=int,default=2000)
    parser.add_argument('--model_dir',type=str,default='./models')
    parser.add_argument('--model_full_dir',type=str,default='./models')
    args=parser.parse_args()
    return args


if __name__ == "__main__":
    with torch.no_grad():
        args=setup()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if args.data == 'solar':
            data_train = np.load('data/solar_data_train.npy')
            data_test = np.load('data/solar_data_test.npy')
            x_test = torch.tensor(np.arange(-200., 200., 2.).reshape((-1,1))).to(device)
        else:
            data = np.load('./baseband.npy') # [352672, 1]
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
        for i in range(10):
            x_list.append(x[i*split_size:(i+1)*split_size])
            y_list.append(y[i*split_size:(i+1)*split_size])
        
        fullGP = FullGP(X=x, y=y, device=device, kernel_type='matern52').to(device)
        fullGP.load_state_dict(torch.load(args.model_full_dir+'/steps_nll'+str(args.num_iters)))
        
        log_ls = fullGP.Kernel.log_ls.item()
        log_sf = fullGP.Kernel.log_sf.item()
        log_noise = max(fullGP.log_noise.item(), -4.5) # clip log-noise-var

        vff = VFF_Matern52(a=torch.min(x_list[0]).item(), b=torch.max(x_list[9]).item(), \
            M=(args.inducing_size+2)//2, device=device).to(device)
        vff.log_sf = nn.Parameter(torch.tensor(np.array([log_sf]), dtype=torch.float64, device=device))
        vff.log_ls = nn.Parameter(torch.tensor(np.array([log_ls]), dtype=torch.float64, device=device))
        
        model1 = VFF_SVGP(X=x_list[0], y=y_list[0], vff=vff, device=device).to(device)
        model1.log_noise = nn.Parameter(torch.tensor(np.array([log_noise]), dtype=torch.float64, device=device))

        y_test_trained_mean_GP, y_test_trained_cov_GP = model1.pred(x_test)
        y_test_trained_mean_GP = y_test_trained_mean_GP.detach().cpu().numpy()
        y_test_trained_cov_GP = y_test_trained_cov_GP.detach().cpu().numpy()

        fig = plt.figure(figsize=(9,3))
        
        plt.plot(x_test.detach().cpu().numpy().flatten(), y_test_trained_mean_GP.flatten(), label = 'SGP1', color='orange')
        plt.fill_between(x_test.detach().cpu().numpy().flatten(), (y_test_trained_mean_GP.flatten()- 1.96* np.sqrt(np.abs(y_test_trained_cov_GP).flatten())), 
                        (y_test_trained_mean_GP.flatten()+ 1.96* np.sqrt(np.abs(y_test_trained_cov_GP).flatten())), alpha=.3, color='orange')
        if args.data == 'solar':
            plt.scatter(x.detach().cpu().numpy().flatten(), y.detach().cpu().numpy().flatten(), color = "blue", label = "train", marker = '.')
            plt.scatter(x_t.flatten(), y_t.flatten(), color = "black", label = "test", marker = '.')
        else:
            plt.scatter(x.detach().cpu().numpy().flatten(), y.detach().cpu().numpy().flatten(), color = "blue", label = "train", marker = '.', s=1)
            plt.scatter(x_t.flatten(), y_t.flatten(), color = "black", label = "test", marker = '.', s=1)
        
        plt.axvline(x = torch.max(x_list[0]).item(), color = 'grey', linestyle='--', label = 'splits')
        for i in range(1, 10):
            plt.axvline(x = torch.max(x_list[i]).item(), color = 'grey', linestyle='--')

        plt.legend(bbox_to_anchor=(1.01, 0.85),\
            loc='upper left', fontsize=18)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.savefig('./vff1_'+str(args.inducing_size)+'z.png', format='png', bbox_inches='tight')

        mu_old, Su_old, Kaa_old = model1.qb_opt()
        Kaa_old = Kaa_old.detach()
        mu_old = mu_old.detach()
        Su_old = Su_old.detach()

        nll, rmse = model1.nll_rmse(torch.tensor(x_t[x_t.flatten()<torch.max(x_list[0]).item()]).to(device),\
            torch.tensor(y_t[x_t.flatten()<torch.max(x_list[0]).item()]).to(device))
        print('SGP1, rmse = {}, nll = {}'.format(rmse, nll))

        old_vff = model1.vff
        for i in range(1, 10):
            cur_model = VFF_OSGPR_VFE(X=x_list[i], y=y_list[i], mu_old=mu_old, Su_old=Su_old, Kaa_old=Kaa_old, \
                vff=copy.deepcopy(vff), vff_old=copy.deepcopy(old_vff), device=device).to(device)
            cur_model.log_noise = nn.Parameter(torch.tensor(np.array([log_noise]), dtype=torch.float64, device=device))
            
            y_test_trained_mean_GP, y_test_trained_cov_GP = cur_model.pred(x_test)
            y_test_trained_mean_GP = y_test_trained_mean_GP.detach().cpu().numpy()
            y_test_trained_cov_GP = y_test_trained_cov_GP.detach().cpu().numpy()

            fig = plt.figure(figsize=(9,3))
            plt.plot(x_test.detach().cpu().numpy().flatten(), y_test_trained_mean_GP.flatten(), label = 'SGP'+str(i+1), color='orange')
            plt.fill_between(x_test.detach().cpu().numpy().flatten(), (y_test_trained_mean_GP.flatten()- 1.96* np.sqrt(np.abs(y_test_trained_cov_GP).flatten())), 
                            (y_test_trained_mean_GP.flatten()+ 1.96* np.sqrt(np.abs(y_test_trained_cov_GP).flatten())), alpha=.3, color='orange')
            if args.data == 'solar':
                plt.scatter(x.detach().cpu().numpy().flatten(), y.detach().cpu().numpy().flatten(), color = "blue", label = "train", marker = '.')
                plt.scatter(x_t.flatten(), y_t.flatten(), color = "black", label = "test", marker = '.')
            else:
                plt.scatter(x.detach().cpu().numpy().flatten(), y.detach().cpu().numpy().flatten(), color = "blue", label = "train", marker = '.', s=1)
                plt.scatter(x_t.flatten(), y_t.flatten(), color = "black", label = "test", marker = '.', s=1)
            
            plt.axvline(x = torch.max(x_list[0]).item(), color = 'grey', linestyle='--', label = 'splits')
            for j in range(1, 10):
                plt.axvline(x = torch.max(x_list[j]).item(), color = 'grey', linestyle='--')
            plt.legend(bbox_to_anchor=(1.01, 0.75), loc='upper left')
            plt.savefig('./vff'+str(i+1)+'_'+str(args.inducing_size)+'z.png', format='png', bbox_inches='tight')

            mu_old, Su_old, Kaa_old = cur_model.qb_opt()
            Kaa_old = Kaa_old.detach()
            mu_old = mu_old.detach()
            Su_old = Su_old.detach()

            nll, rmse = cur_model.nll_rmse(torch.tensor(x_t[x_t.flatten()<torch.max(x_list[i]).item()]).to(device), \
                torch.tensor(y_t[x_t.flatten()<torch.max(x_list[i]).item()]).to(device))
            print('SGP'+str(i+1)+', rmse = {}, nll = {}'.format(rmse, nll))
            
            old_vff = cur_model.vff

        print('OSGP10, log-noise: {}'.format(cur_model.log_noise.item()))
        print('OSGP10, log-ls: {}'.format(cur_model.vff.log_ls.item()))
        print('OSGP10, log-sf: {}'.format(cur_model.vff.log_sf.item()))

import numpy as np
import torch
from scripts.onedim.solar.baseline import FullGP
import os
import argparse


def setup():
    parser=argparse.ArgumentParser('Argument Parser')
    parser.add_argument('--data',type=str,default='solar')
    parser.add_argument('--kernel_type',type=str,default='se')
    parser.add_argument('--num_iters',type=int,default=2000)
    parser.add_argument('--model_dir',type=str,default='./models')
    args=parser.parse_args()
    return args

if __name__ == "__main__":
    args=setup()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.data == 'solar':
        data_train = np.load('data/solar_data_train.npy')
        data_test = np.load('data/solar_data_test.npy')
    else:
        data = np.load('data/baseband.npy') # [352672, 1]
        y_total = data[:18000]
        x_total = np.expand_dims((1+np.arange(18000))/ 1800, 1)
        data = np.concatenate([x_total, y_total], 1) # [18000, 2]
        data_train = data[::2] # [9000, 2]
        data_test = data[1::2]
    x = torch.tensor(np.expand_dims(data_train[:,0], 1)).to(device)
    y = torch.tensor(np.expand_dims(data_train[:,1], 1)).to(device)

    split_size = x.shape[0] // 5
    x1 = x[:split_size]
    y1 = y[:split_size]

    fullGP = FullGP(X=x1, y=y1, device=device, kernel_type=args.kernel_type)
    fullGP.to(device)
    fullGP.train(iters=args.num_iters)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    torch.save(fullGP.state_dict(), args.model_dir+'/steps_nll'+str(args.num_iters))

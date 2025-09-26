import torch
import torch.nn as nn
from hipposvgp.kernel import SE_kernel
import torch_optimizer
import numpy.random as npr
from tqdm import tqdm
import numpy as np

class FullGP(nn.Module):
    def __init__(self, X, y, device, kernel_type="se"):
        super(FullGP, self).__init__()
        self.X = torch.tensor(X)
        self.input_dim = X.shape[1]
        self.y = torch.tensor(y)
        self.log_noise = nn.Parameter(-2.3 + 0.* torch.tensor(npr.randn(1))) # log-noise-variance
        if kernel_type == "se":
            self.Kernel = SE_kernel(self.input_dim, device)
        elif kernel_type == "matern52":
            self.Kernel = Matern52_kernel(self.input_dim, device)
        elif kernel_type == "smk":
            self.Kernel = SMK_kernel(2, device)
        else:
            raise NotImplementedError
        self.jitter = 1e-6
        self.device = device
            
    def nll(self):
        Kff = self.Kernel(self.X, self.X)
        C = Kff + torch.exp(self.log_noise)* torch.eye(Kff.shape[0], device=self.device)
        chol_C = torch.linalg.cholesky(C + self.jitter* torch.mean(torch.diag(C)).detach()*\
                                        torch.eye(C.shape[0], device=self.device))
        nll = torch.sum(torch.log(torch.diag(chol_C))) + 0.5* self.X.shape[0] * torch.log(torch.tensor(2* np.pi))
        v = torch.linalg.solve_triangular(chol_C, self.y, upper=False)
        nll += 0.5* torch.sum(v.T @ v)
        return nll
        
    def train(self, lr = 0.01, iters = 2000):
        optimizer = torch_optimizer.Yogi(self.parameters(), lr=lr)
        #optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for i in tqdm(range(iters)):
            optimizer.zero_grad()
            loss = self.nll()
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print('epoch number: {}, nll: {}'.format(i+1, loss.item()))
                
    def pred(self, X, flag_noise = True):
        X = torch.tensor(X)
        Kxx = self.Kernel(X, X)
        Kxn = self.Kernel(X, self.X)
        Knn = self.Kernel(self.X, self.X)
        
        C = Knn + torch.exp(self.log_noise)* torch.eye(Knn.shape[0], device=self.device)
        chol_C = torch.linalg.cholesky(C + self.jitter* torch.mean(torch.diag(C)).detach()* \
                                       torch.eye(C.shape[0], device=self.device))
        v1 = torch.linalg.solve_triangular(chol_C, self.y,  upper=False)
        v2 = torch.linalg.solve_triangular(chol_C, Kxn.T, upper=False)
        
        mean = v2.T @ v1
        cov = Kxx - v2.T @ v2
        if flag_noise:
            cov += torch.exp(self.log_noise)* torch.eye(cov.shape[0], device=self.device) # add noise variance
        return mean, torch.diag(cov)

    def nll_rmse(self, X, y, flag_noise = True):
        mean, cov = self.pred(X, flag_noise = flag_noise)
        rmse = torch.sqrt(torch.mean((y - mean).pow(2)))
       
        chol_cov = cov.pow(0.5)
        nll = torch.sum(torch.log(chol_cov)) +  0.5* X.shape[0]* torch.log(torch.tensor(2* np.pi))
        nll += 0.5* torch.sum((y - mean).pow(2).squeeze(1)/  chol_cov.pow(2))
        
        nll /= y.shape[0]
        
        return nll, rmse
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import math

class SMK_kernel(nn.Module):
    def __init__(self, num_mixture, device='cpu'):
        super(SMK_kernel, self).__init__()
        self.num_mixture = num_mixture
        self.device = device
        self.mu_smk = nn.Parameter(0.1* torch.tensor(npr.randn(self.num_mixture, 1, 1)))
        self.log_var_smk = nn.Parameter(-10. + 1.* torch.tensor(npr.randn(self.num_mixture, 1, 1)))
        self.w_smk_raw = nn.Parameter(torch.tensor(npr.randn(self.num_mixture)))
    
    def forward(self, X1, X2):
        diff = X1.unsqueeze(1) - X2.unsqueeze(0)
        diff_sq = diff.pow(2)
        diff_sq = diff_sq.sum(-1)
        diff = diff.sum(-1)
        term1 = torch.exp(-2* torch.tensor(np.pi, dtype = torch.float32).pow(2) * diff_sq.unsqueeze(0) * \
            torch.exp(self.log_var_smk)) # [num_mixture, N, M]
        term2 = torch.cos(torch.tensor(2.* np.pi, dtype=torch.float32)* diff.unsqueeze(0)* \
            self.mu_smk) # [num_mixture, N, M]
        res = torch.sum(torch.softmax(self.w_smk_raw, 0).view(-1,1, 1) * \
            term1* term2, 0)
        return res
    
    def sample_from_spectral(self, num_rff):
        S = self.mu_smk.view(self.num_mixture, 1) + \
              torch.sqrt(torch.exp(self.log_var_smk)).view(self.num_mixture, 1) * \
                  torch.randn(self.num_mixture, num_rff, device=self.device) # [num_mixture, num_rff]
        return S, torch.softmax(self.w_smk_raw, 0) # [num_mixture]


class Matern52_kernel(nn.Module):
    def __init__(self, input_dim, device='cpu'):
        super(Matern52_kernel, self).__init__()
        self.input_dim = input_dim
        self.log_sf = nn.Parameter(-2. + 0.* torch.tensor(npr.randn(1), dtype=torch.float64))
        self.log_ls = nn.Parameter(1. + 0.* torch.tensor(npr.randn(self.input_dim), dtype=torch.float64))
        self.device = device
        
    def forward(self, x1, x2):
        length_scale = torch.exp(self.log_ls)
        dist_sq = torch.cdist(x1, x2, p=2).pow(2)
        dist = torch.sqrt(dist_sq) / length_scale
        sqrt5 = math.sqrt(5.)
        return torch.exp(self.log_sf)* (1. + sqrt5* dist + \
            (5./ 3.)* dist_sq/ length_scale.pow(2))* torch.exp(-sqrt5* dist)


class SE_kernel(nn.Module):
    def __init__(self, input_dim, device='cpu'):
        super(SE_kernel, self).__init__()
        self.input_dim = input_dim
        self.log_sf = nn.Parameter(-2. + 0.* torch.tensor(npr.randn(1), dtype=torch.float64))
        self.log_ls = nn.Parameter(1. + 0.* torch.tensor(npr.randn(self.input_dim), dtype=torch.float64))
        self.device = device

    def forward_slow(self, X1, X2):
        X1 = X1 * torch.exp(-self.log_ls)
        X2 = X2 * torch.exp(-self.log_ls)
        X1 = torch.unsqueeze(X1, 1)
        X2 = torch.unsqueeze(X2.T, 2)
        return (torch.exp(self.log_sf) * torch.exp(-0.5* torch.sum((X2-X1.T)**2, 0))).T
        
    def forward(self, X1, X2):
        # quicker way to compute SE kernel
        X1 = X1 * torch.exp(-self.log_ls)
        X2 = X2 * torch.exp(-self.log_ls)
        factor1 = torch.sum(X1.pow(2), -1)
        factor2 = torch.sum(X2.pow(2), -1)
        return torch.exp(self.log_sf) * torch.exp(-0.5* (factor1.unsqueeze(1) + factor2.unsqueeze(0) -2* X1 @ X2.permute(1,0)))
        
    def sample_from_spectral(self, num_rff):
        # For SE kernel, the spectral density is Gaussian(mean = 0, std = 1./ length-scale)
        # num_rff: the number of RFF samples
        return (1./ torch.exp(self.log_ls))* torch.randn(num_rff, device=self.device) # we ignore signal variance here, but will multiply it back later
"""    
    def forward_rff(self, X, num_rff):
        w = self.sample_from_spectral(num_rff).unsqueeze(0) 
        b = torch.rand(num_rff, device=self.device).unsqueeze(0)* 2* math.pi
        Z = (2** 0.5)*  torch.cos(w* X + b) 
        Z = Z.permute(1,0) 
        res = Z.unsqueeze(2) * Z.unsqueeze(1) 
        res = torch.mean(res, 0)
        return res
    
    def forward_rff_reduce_var(self, X, num_rff):
        w = self.sample_from_spectral(num_rff).unsqueeze(0)
        Z = (1/ num_rff)**0.5 * torch.cat([torch.sin(w* X), torch.cos(w* X)], 1)
        res = Z @ Z.T
        return res

if __name__ == "__main__":
    import math
    torch.manual_seed(42)
    X = torch.randn(100, 1)
    mykernel = SE_kernel(1)
    res = mykernel(X, X)
    res_rff = torch.exp(mykernel.log_sf) * mykernel.forward_rff(X, 1000)
    res_rff2 = torch.exp(mykernel.log_sf) * mykernel.forward_rff_reduce_var(X, 1000)
    print((res_rff-res).abs().mean())
    print((res_rff2-res).abs().mean())
"""


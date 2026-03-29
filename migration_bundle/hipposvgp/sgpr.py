import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
from tqdm import tqdm
from hipposvgp.kernel import SE_kernel, SMK_kernel
from hipposvgp.hippo import HiPPO_LegS
import math
from scipy import special as ss


class HIPPO_OSGPR_VFE(nn.Module):

    def __init__(self, X, y, mu_old, Su_old, Kaa_old, Z_old, end, start, prev_discrete_steps, \
                 rff_sample_size, num_discrete_steps, hippo, \
                      inducing_size, device, kernel_type='se', dtype=torch.float64, flag_fast=False):
        super(HIPPO_OSGPR_VFE, self).__init__()
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        if kernel_type == "se":
            self.Kernel = SE_kernel(1, device)
        elif kernel_type == "smk":
            self.Kernel = SMK_kernel(2, device)
        else:
            raise NotImplementedError

        self.num_data = self.X.shape[0]
        self.mu_old = torch.tensor(mu_old).detach()
        self.Su_old = torch.tensor(Su_old).detach()
        self.Kaa_old = torch.tensor(Kaa_old).detach()
        if Z_old != None:
            self.Z_old = torch.tensor(Z_old).detach()
        else:
            self.Z_old = None

        self.jitter = 1e-6 
        self.device= device
        self.log_noise = nn.Parameter(-3.+ 0.* torch.tensor(npr.randn(1)))

        self.rff_sample_size = rff_sample_size # the number of rff samples to estimate Kuu
        # hippo
        self.end = end
        self.start = start
        self.maxT = self.end - self.start
        self.num_discrete_steps = num_discrete_steps # so step-size = maxT / num_discrete_steps
        self.inducing_size = inducing_size
        self.hippo = hippo
        self.dtype = dtype
        self.flag_fast = flag_fast
        self.prev_discrete_steps = prev_discrete_steps

    def add_jitter(self, Kuu):
        return Kuu + self.jitter* torch.mean(torch.diag(Kuu)).detach()*\
            torch.eye(Kuu.shape[0], device=self.device)
    
    def get_Z(self, w):
        rff_sample_size= w.shape[1]
        
        time_stamps = self.start + (self.maxT/ self.num_discrete_steps) *\
              (1 + torch.arange(self.num_discrete_steps, dtype=self.dtype, \
            device=self.device)).unsqueeze(1) 
        
        h = (1/ rff_sample_size)**0.5 * torch.cat([torch.sin(w* time_stamps), \
                torch.cos(w* time_stamps)], 1) 
        Z = self.hippo(h, ini=self.Z_old, prev_discrete_steps=self.prev_discrete_steps,\
            fast=self.flag_fast)[-1] 
        return Z, time_stamps

    def Kuu_se(self, w):
        
        time_stamps = self.start + (self.maxT / self.num_discrete_steps) *\
              (1 + torch.arange(self.num_discrete_steps, dtype=self.dtype, \
            device=self.device)).unsqueeze(1) 

        rff_sample_size= w.shape[1]
        h = (1/ rff_sample_size)**0.5 * torch.cat([torch.sin(w* time_stamps), \
                torch.cos(w* time_stamps)], 1) #
        Z = self.hippo(h, ini=self.Z_old, prev_discrete_steps=self.prev_discrete_steps,\
             fast=self.flag_fast)[-1] # [num_rff, inducing_size]
        
        Kuu = Z.T @ Z
        Kuuold = Z.T @ self.Z_old
        Kuolduold = self.Z_old.T @ self.Z_old
        return Kuu, Kuuold, Kuolduold

    """
    def Kfu(self, X):
        time_stamps = self.start + (self.maxT / self.num_discrete_steps) * \
            (1 + torch.arange(self.num_discrete_steps, dtype=self.dtype,\
            device=self.device)).unsqueeze(1) 
        Ktf = self.Kernel(time_stamps, X) 
        res = self.hippo(Ktf, fast=self.flag_fast)[-1] 
        return res
    """

    def KfuKff_rff_se(self, w, X):
        time_stamps = self.start + (self.maxT / self.num_discrete_steps) *\
              (1 + torch.arange(self.num_discrete_steps, dtype=self.dtype, \
            device=self.device)).unsqueeze(1)  

        rff_sample_size= w.shape[1]
        h = (1/ rff_sample_size)**0.5 * torch.cat([torch.sin(w* time_stamps), \
                torch.cos(w* time_stamps)], 1)
        Z = self.hippo(h, ini=self.Z_old, prev_discrete_steps=self.prev_discrete_steps,\
            fast=self.flag_fast)[-1] # [num_rff, inducing_size]
        Z_f = (1/ rff_sample_size)**0.5 * torch.cat([torch.sin(w* X), \
                torch.cos(w* X)], 1) # [N, num_rff]
        Kfu = Z_f @ Z
        Kff = Z_f @ Z_f.T
        return Kff, Kfu

    def KuuKfuKff_rff_se(self, w, X):
        time_stamps = self.start + (self.maxT / self.num_discrete_steps) *\
              (1 + torch.arange(self.num_discrete_steps, dtype=self.dtype, \
            device=self.device)).unsqueeze(1)  

        rff_sample_size= w.shape[1]
        h = (1/ rff_sample_size)**0.5 * torch.cat([torch.sin(w* time_stamps), \
                torch.cos(w* time_stamps)], 1)
        Z = self.hippo(h, ini=self.Z_old, prev_discrete_steps=self.prev_discrete_steps,\
             fast=self.flag_fast)[-1] # [num_rff, inducing_size]
        Z_f = (1/ rff_sample_size)**0.5 * torch.cat([torch.sin(w* X), \
                torch.cos(w* X)], 1) # [N, num_rff]
        Kfu = Z_f @ Z
        Kff = Z_f @ Z_f.T
        Kuu = Z.T @ Z
        Kuuold = Z.T @ self.Z_old
        Kuolduold = self.Z_old.T @ self.Z_old
        return Kff, Kfu, Kuu, Kuuold, Kuolduold
    
    def _common_terms(self, w):
        Mb = self.inducing_size
        Ma = self.inducing_size

        sigma2 = torch.exp(self.log_noise)
        sigma = torch.sqrt(sigma2)

        Saa = self.Su_old
        ma = self.mu_old

        # a is old inducing points, b is new
        # f is training points
        # s is test points
        #Kbf = self.Kfu(self.X).T
        #Kff, Kfb = self.KfuKff_rff_se(w, self.X)
        Kff, Kfb, Kbb, Kba, Kaa_cur = self.KuuKfuKff_rff_se(w, self.X)
        Kff = torch.exp(self.Kernel.log_sf) * Kff
        Kfb = torch.exp(self.Kernel.log_sf) * Kfb 
        Kbf = Kfb.T
        #Kbb, Kba, Kaa_cur = self.Kuu_se(w)
        Kbb = torch.exp(self.Kernel.log_sf) * Kbb
        Kba = torch.exp(self.Kernel.log_sf) * Kba 
        Kaa_cur = torch.exp(self.Kernel.log_sf) * Kaa_cur
        Kbb = self.add_jitter(Kbb)
        Kaa_cur = self.add_jitter(Kaa_cur)
        Kaa = self.add_jitter(self.Kaa_old)

        err = self.y # prior-mean = 0

        Sainv_ma = torch.linalg.solve(Saa, ma)
        Sinv_y = self.y / sigma2
        c1 = Kbf@ Sinv_y
        c2 = Kba@ Sainv_ma
        c = c1 + c2

        Lb = torch.linalg.cholesky(Kbb)
        Lbinv_c = torch.linalg.solve_triangular(Lb, c, upper=False)
        Lbinv_Kba = torch.linalg.solve_triangular(Lb, Kba, upper=False)
        Lbinv_Kbf = torch.linalg.solve_triangular(Lb, Kbf, upper=False) / sigma
        d1 = Lbinv_Kbf @ Lbinv_Kbf.T  

        LSa = torch.linalg.cholesky(Saa)
        Kab_Lbinv = Lbinv_Kba.T
        LSainv_Kab_Lbinv = torch.linalg.solve_triangular(LSa, Kab_Lbinv, upper=False)
        d2 = LSainv_Kab_Lbinv.T @ LSainv_Kab_Lbinv
        
        La = torch.linalg.cholesky(Kaa)
        Lainv_Kab_Lbinv = torch.linalg.solve_triangular(La, Kab_Lbinv, upper=False)
        d3 = Lainv_Kab_Lbinv.T @ Lainv_Kab_Lbinv

        D = torch.eye(Mb, device=self.device) + d1 + d2 - d3
        D = self.add_jitter(D)
        LD = torch.linalg.cholesky(D)

        LDinv_Lbinv_c = torch.linalg.solve_triangular(LD, Lbinv_c, upper=False)

        return (Kff, Kbf, Kba, Kaa, Kaa_cur, La, Kbb, Lb, D, LD,
                Lbinv_Kba, LDinv_Lbinv_c, err, d1)

    def ELBO(self):
        Mb = self.inducing_size
        Ma = self.inducing_size
        sigma2 = torch.exp(self.log_noise)
        sigma = torch.sqrt(sigma2)
        N = self.num_data

        Saa = self.Su_old
        ma = self.mu_old

        # a is old inducing points, b is new
        # f is training points
        #Kfdiag = torch.diag(self.Kernel(self.X, self.X))
        w = self.Kernel.sample_from_spectral(self.rff_sample_size).unsqueeze(0)
        (Kff, Kbf, Kba, Kaa, Kaa_cur, La, Kbb, Lb, D, LD,
            Lbinv_Kba, LDinv_Lbinv_c, err, Qff) = self._common_terms(w)
        Kfdiag = torch.diag(Kff)

        LSa = torch.linalg.cholesky(Saa)
        Lainv_ma = torch.linalg.solve_triangular(LSa, ma, upper=False)

        # constant term
        bound = -0.5 * N * math.log(2 * math.pi)
        # quadratic term
        bound += -0.5 * torch.sum(err.pow(2)) / sigma2
        bound += -0.5 * torch.sum(Lainv_ma.pow(2))
        bound += 0.5 * torch.sum((LDinv_Lbinv_c).pow(2))
        # log det term
        bound += -0.5 * N * torch.sum(torch.log(sigma2))
        bound += - torch.sum(torch.log(torch.diag(LD)))

        # delta 1: trace term
        bound += -0.5 * torch.sum(Kfdiag) / sigma2
        bound += 0.5 * torch.sum(torch.diag(Qff))   

        # delta 2: a and b difference
        bound += torch.sum(torch.log(torch.diag(La)))
        bound += - torch.sum(torch.log(torch.diag(LSa)))

        Kaadiff = Kaa_cur - Lbinv_Kba.T @ Lbinv_Kba
        Sainv_Kaadiff = torch.linalg.solve(Saa, Kaadiff)
        Kainv_Kaadiff = torch.linalg.solve(Kaa, Kaadiff)

        bound += -0.5 * torch.sum(
            torch.diag(Sainv_Kaadiff) - torch.diag(Kainv_Kaadiff))

        return bound

    def train(self, lr = 0.01, iters = 5000):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for i in tqdm(range(iters)):
            optimizer.zero_grad()
            loss = -self.ELBO()
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print('epoch number: {}, ELBO: {}'.format(i+1, loss.item()))

    def pred(self, Xnew, w, full_cov=False, flag_noise = True):
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. 
        """

        # a is old inducing points, b is new
        # f is training points
        # s is test points
        #Kbs = self.Kernel(self.z, Xnew)

        (Kff, Kbf, Kba, Kaa, Kaa_cur, La, Kbb, Lb, D, LD,
            Lbinv_Kba, LDinv_Lbinv_c, err, Qff) = self._common_terms(w)
        Kss, Ksb = self.KfuKff_rff_se(w, Xnew)
        Kss = torch.exp(self.Kernel.log_sf) * Kss
        Ksb = torch.exp(self.Kernel.log_sf) * Ksb
        Kbs = Ksb.T

        Lbinv_Kbs = torch.linalg.solve_triangular(Lb, Kbs, upper=False)
        LDinv_Lbinv_Kbs = torch.linalg.solve_triangular(LD, Lbinv_Kbs, upper=False)
        mean = LDinv_Lbinv_Kbs.T @ LDinv_Lbinv_c

        if full_cov:
            Kss = self.add_jitter(Kss) 
            var1 = Kss
            var2 = - Lbinv_Kbs.T @ Lbinv_Kbs 
            var3 = LDinv_Lbinv_Kbs.T @ LDinv_Lbinv_Kbs
            var = var1 + var2 + var3
            if flag_noise:
                var += torch.exp(self.log_noise)* torch.eye(var.shape[0], device=self.device) # add noise variance
        else:
            var1 = torch.diag(self.Kernel(Xnew, Xnew))
            var2 = -torch.sum(Lbinv_Kbs.pow(2), 0)
            var3 = torch.sum(LDinv_Lbinv_Kbs.pow(2), 0)
            var = var1 + var2 + var3
            if flag_noise:
                var += torch.exp(self.log_noise)* torch.ones_like(var) # add noise variance

        return mean, var

    def nll_rmse(self, X, y, w, flag_noise = True):
        mean, cov = self.pred(X, w=w, full_cov=False, flag_noise = True)
        chol_cov = cov.pow(0.5)
        nll = torch.sum(torch.log(chol_cov)) + 0.5* X.shape[0]* torch.log(torch.tensor(2* np.pi))
        nll += 0.5* torch.sum((y - mean).pow(2).squeeze(1)/  cov)
        nll /= y.shape[0]
        rmse = torch.sqrt(torch.mean((y - mean)**2))
        return nll, rmse

    
    def qb_opt(self, w, full_cov=True):
        (Kff, Kbf, Kba, Kaa, Kaa_cur, La, Kbb, Lb, D, LD,
            Lbinv_Kba, LDinv_Lbinv_c, err, Qff) = self._common_terms(w)
        Kss = Kbb.clone()
        Kbs = Kbb.clone()

        Lbinv_Kbs = torch.linalg.solve_triangular(Lb, Kbs, upper=False)
        LDinv_Lbinv_Kbs = torch.linalg.solve_triangular(LD, Lbinv_Kbs, upper=False)
        mean = LDinv_Lbinv_Kbs.T @ LDinv_Lbinv_c

        if full_cov:
            #Kss = self.add_jitter(Kss) 
            var1 = Kss
            var2 = - Lbinv_Kbs.T @ Lbinv_Kbs  
            var3 = LDinv_Lbinv_Kbs.T @ LDinv_Lbinv_Kbs
            var = var1 + var2 + var3
        else:
            var1 = torch.diag(Kss)
            var2 = -torch.sum(Lbinv_Kbs.pow(2), 0)
            var3 = torch.sum(LDinv_Lbinv_Kbs.pow(2), 0)
            var = var1 + var2 + var3

        return mean, var, Kbb
    

class HIPPO_SVGP(nn.Module):
    def __init__(self, X, y, maxT, inducing_size, rff_sample_size, num_discrete_steps, device, start, hippo, \
                 kernel_type="se", dtype=torch.float64, flag_fast=False):
        super(HIPPO_SVGP, self).__init__()
        self.X = torch.tensor(X)
        self.input_dim = X.shape[1]
        self.y = torch.tensor(y)
        self.jitter = 1e-6       
        self.log_noise = nn.Parameter(-3.+ 0. * torch.tensor(npr.randn(1)))
        self.kernel_type = kernel_type
        if kernel_type == "se":
            self.Kernel = SE_kernel(self.input_dim, device)
        elif kernel_type == "smk":
            self.Kernel = SMK_kernel(2, device)
        else:
            raise NotImplementedError
        self.dtype = dtype
        # inducing points
        self.inducing_size = inducing_size # the number of legendre polynomial bases 

        self.rff_sample_size = rff_sample_size # the number of rff samples to estimate Kuu
        # hippo
        self.maxT = maxT
        self.num_discrete_steps = num_discrete_steps # so step-size = maxT / num_discrete_steps
        self.hippo = hippo
        self.start = start
        self.device = device
        self.flag_fast = flag_fast
    
    def get_Z(self, w):
        rff_sample_size= w.shape[1]
        time_stamps = self.start + (self.maxT / self.num_discrete_steps) *\
              (1 + torch.arange(self.num_discrete_steps, dtype=self.dtype, \
            device=self.device)).unsqueeze(1) 
        h = (1/ rff_sample_size)**0.5 * torch.cat([torch.sin(w* time_stamps), \
                torch.cos(w* time_stamps)], 1) 
        Z = self.hippo(h, prev_discrete_steps=0, fast=self.flag_fast)[-1] # [num_rff, inducing_size]
        return Z, time_stamps
    
    def Kuu_se(self, w, b=None, reducevar=True):
        time_stamps = self.start + (self.maxT / self.num_discrete_steps) *\
              (1 + torch.arange(self.num_discrete_steps, dtype=self.dtype, \
            device=self.device)).unsqueeze(1) 
        rff_sample_size= w.shape[1]
        if reducevar:
            h = (1/ rff_sample_size)**0.5 * torch.cat([torch.sin(w* time_stamps), \
                    torch.cos(w* time_stamps)], 1) #
            Z = self.hippo(h, prev_discrete_steps=0, fast=self.flag_fast)[-1] # [num_rff, inducing_size]
            Kuu = Z.T @ Z
        else:
            h = (2** 0.5)*  torch.cos(w* time_stamps + b) 
            Z = self.hippo(h, prev_discrete_steps=0, fast=self.flag_fast)[-1] # [num_rff, inducing_size]
            res = Z.unsqueeze(2) * Z.unsqueeze(1) 
            Kuu = torch.mean(res, 0)
        return Kuu
    
    def KfuKff_rff_se(self, w, X, b=None, reducevar=True):
        time_stamps = self.start + (self.maxT / self.num_discrete_steps) *\
              (1 + torch.arange(self.num_discrete_steps, dtype=self.dtype, \
            device=self.device)).unsqueeze(1)  
        rff_sample_size= w.shape[1]
        if reducevar:
            h = (1/ rff_sample_size)**0.5 * torch.cat([torch.sin(w* time_stamps), \
                    torch.cos(w* time_stamps)], 1)
            Z = self.hippo(h, prev_discrete_steps=0, fast=self.flag_fast)[-1]  # [num_rff, inducing_size]
            Z_f = (1/ rff_sample_size)**0.5 * torch.cat([torch.sin(w* X), \
                    torch.cos(w* X)], 1) # [N, num_rff]
            Kfu = Z_f @ Z
            Kff = Z_f @ Z_f.T
        else:
            h = (2** 0.5)*  torch.cos(w* time_stamps + b) 
            Z = self.hippo(h, prev_discrete_steps=0, fast=self.flag_fast)[-1]  # [num_rff, inducing_size]
            Z_f = (2** 0.5)*  torch.cos(w* X + b) 
            Z_f = Z_f.permute(1,0) 
            Kfu = Z_f.unsqueeze(2) * Z.unsqueeze(1) 
            Kfu = torch.mean(Kfu, 0)
            Kff = Z_f.unsqueeze(2) * Z_f.unsqueeze(1)
            Kff = torch.mean(Kff, 0)
        return Kff, Kfu
    
    def Kuu_smk(self, S, w_smk):
        S = S.unsqueeze(0) 
        time_stamps = self.start + (self.maxT / self.num_discrete_steps) * \
            (1 + torch.arange(self.num_discrete_steps, dtype=self.dtype, \
            device=self.device)).view(-1, 1, 1) 
        h = torch.cos(2* math.pi* S* time_stamps) 
        Z = self.hippo(h, prev_discrete_steps=0,fast=self.flag_fast)[-1] 
        Z = (torch.sqrt(w_smk).view(-1,1,1) * Z).permute(1, 2, 0) 
        res = Z @ Z.permute(0,2,1)

        h2 = torch.sin(2* math.pi* S* time_stamps) 
        Z2 = self.hippo(h2, prev_discrete_steps=0,fast=self.flag_fast)[-1] 
        Z2 = (torch.sqrt(w_smk).view(-1,1,1) * Z2).permute(1, 2, 0) 
        res2 = Z2 @ Z2.permute(0,2,1)
        Kuu = torch.mean(res+res2, 0)
        return Kuu
    
    def KfuKff_rff_smk(self, S, w_smk, X):
        S = S.unsqueeze(0) 
        time_stamps = self.start + (self.maxT / self.num_discrete_steps) *\
              (1 + torch.arange(self.num_discrete_steps, dtype=self.dtype, \
            device=self.device)).view(-1,1,1) 
        h = torch.cos(2* math.pi* S* time_stamps) 
        Z = self.hippo(h, prev_discrete_steps=0,fast=self.flag_fast)[-1] #
        Z = (torch.sqrt(w_smk).view(-1,1,1) * Z).permute(1, 2, 0)

        Z_f = torch.cos(2* math.pi* S* X.view(-1,1,1)) 
        Z_f = (torch.sqrt(w_smk).view(1,-1,1) * Z_f).permute(2, 0, 1) 
        
        Kfu = Z_f @ Z.permute(0,2,1) 

        h2 = torch.sin(2* math.pi* S* time_stamps) 
        Z2 = self.hippo(h2, prev_discrete_steps=0,fast=self.flag_fast)[-1] 
        Z2 = (torch.sqrt(w_smk).view(-1,1,1) * Z2).permute(1, 2, 0) 

        Z_f2 = torch.sin(2* math.pi* S* X.view(-1,1,1)) 
        Z_f2 = (torch.sqrt(w_smk).view(1,-1,1) * Z_f2).permute(2, 0, 1) 
        
        Kfu2 = Z_f2 @ Z2.permute(0,2,1) 

        Kfu = torch.mean(Kfu + Kfu2, 0)

        Kff = Z_f @ Z_f.permute(0,2,1) + Z_f2 @ Z_f2.permute(0,2,1)
        Kff = torch.mean(Kff, 0)
        return Kff, Kfu
    
    def Kfu(self, X):
        time_stamps = self.start + (self.maxT / self.num_discrete_steps) * \
            (1 + torch.arange(self.num_discrete_steps, dtype=self.dtype,\
            device=self.device)).unsqueeze(1) 
        Ktf = self.Kernel(time_stamps, X) 
        res = self.hippo(Ktf, prev_discrete_steps=0, fast=self.flag_fast)[-1] 
        return res
    
    def ELBO(self, reducevar=True):
        if self.kernel_type == "se":
            w = self.Kernel.sample_from_spectral(self.rff_sample_size).unsqueeze(0)
            if not reducevar:
                b_ = torch.rand(self.rff_sample_size, device=self.device).unsqueeze(0)* 2* math.pi  
            else:
                b_ = None
            Kuu = self.Kuu_se(w, b_, reducevar) 
            Kuu = torch.exp(self.Kernel.log_sf) * Kuu 
        elif self.kernel_type == "smk":
            S, w_smk = self.Kernel.sample_from_spectral(self.rff_sample_size) 
            Kuu = self.Kuu_smk(S, w_smk) 
        else:
            raise NotImplementedError
        
        Kfu = self.Kfu(self.X)
        Kff = self.Kernel(self.X, self.X)
        """
        Kff_rff, Kfu_rff = self.KfuKff_rff_se(w, self.X, b_, reducevar)
        Kfu = torch.exp(self.Kernel.log_sf) * Kfu_rff 
        Kff = torch.exp(self.Kernel.log_sf) * Kff_rff
        """
        N_input = self.X.shape[0]
        const = -0.5* N_input * torch.log(torch.tensor(2* np.pi))      
        
        # To compute the log-determinant term, we need to use Sylvester's determinant identity
        M_aux = Kuu + torch.exp(-self.log_noise)* (Kfu.T @ Kfu) 
        chol_M_aux = torch.linalg.cholesky(M_aux + self.jitter* torch.mean(torch.diag(M_aux)).detach()* \
                                           torch.eye(self.inducing_size, device=self.device))
        chol_Kuu = torch.linalg.cholesky(Kuu+ self.jitter* torch.mean(torch.diag(Kuu)).detach()* \
                                         torch.eye(self.inducing_size, device=self.device))
        log_det = -0.5* (N_input * self.log_noise) - torch.sum(torch.log(torch.diag(chol_M_aux))) \
                                              + torch.sum(torch.log(torch.diag(chol_Kuu)))
        
        # To compute the quadratic term, we need to use Woodbury's matrix identity
        quadratic = -0.5* torch.exp(-self.log_noise)* self.y.T @ self.y 
        v = torch.linalg.solve_triangular(chol_M_aux, Kfu.T @ self.y, upper=False)
        quadratic += 0.5* torch.exp(-2* self.log_noise)* v.T @ v 
        
        # Compute the trace term
        b = torch.linalg.solve_triangular(chol_Kuu, Kfu.T, upper=False)
        trace = -0.5* torch.exp(-self.log_noise)* (torch.sum(torch.diag(Kff))- torch.sum(b.pow(2))) # this term is unstable with RFF approximation during training
        
        if trace.item() > 0:
            # rewrite Kff and Kfu using rff to compute trace to make sure it (-trace) is positive definite
            if self.kernel_type == "se":
                Kff_rff, Kfu_rff = self.KfuKff_rff_se(w, self.X, b_, reducevar)
                Kfu_rff = torch.exp(self.Kernel.log_sf) * Kfu_rff 
                Kff_rff = torch.exp(self.Kernel.log_sf) * Kff_rff
            elif self.kernel_type == "smk":
                Kff_rff, Kfu_rff = self.KfuKff_rff_smk(S, w_smk, self.X)
            else:
                raise NotImplementedError
            b_rff = torch.linalg.solve_triangular(chol_Kuu, Kfu_rff.T, upper=False)
            trace_rff = -0.5* torch.exp(-self.log_noise)* (torch.sum(torch.diag(Kff_rff))- torch.sum(b_rff.pow(2)))
            return const + log_det + quadratic + trace_rff
        else:
            return const + log_det + quadratic + trace
        
    def train(self, lr = 0.01, iters = 5000):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for i in tqdm(range(iters)):
            optimizer.zero_grad()
            loss = -self.ELBO()
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print('epoch number: {}, ELBO: {}'.format(i+1, loss.item()))
            """
                if self.kernel_type == "se":
                    torch.save(self.state_dict(), './models_se_baseband/ind_'+str(self.inducing_size)+'_steps_elbo'+str(i+1))
                elif self.kernel_type == "smk":
                    torch.save(self.state_dict(), './models_smk/ind_'+str(self.inducing_size)+'_steps_elbo'+str(i+1))
                else:
                    raise NotImplementedError
            """
            

    def train_nll(self, lr = 0.01, iters = 1000):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for i in tqdm(range(iters)):
            optimizer.zero_grad()
            mean, cov = self.pred(self.X, False, True)
            chol_cov = torch.abs(cov).pow(0.5)
            f_sample = mean + chol_cov* torch.randn_like(mean)
            loss = torch.mean(0.5* (self.y - f_sample).pow(2)/ torch.exp(self.log_noise)) \
                + self.log_noise - math.log(0.5)
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print('epoch number: {}, NLL: {}'.format(i+1, loss.item()))
            """
                if self.kernel_type == "se":
                    torch.save(self.state_dict(), './models_se/ind_'+str(self.inducing_size)+'_steps_nll'+str(i+1))
                elif self.kernel_type == "smk":
                    torch.save(self.state_dict(), './models_smk/ind_'+str(self.inducing_size)+'_steps_nll'+str(i+1))
                else:
                    raise NotImplementedError
            """
    
    def pred(self, X, w, flag_noise = True, flag_diag = True, reducevar=True):
        X = torch.tensor(X)
        if self.kernel_type == "se":
            #w = self.Kernel.sample_from_spectral(num_rff).unsqueeze(0) 
            if not reducevar:
                b_ = torch.rand(num_rff, device=self.device).unsqueeze(0)* 2* math.pi 
            else:
                b_ = None
            Kuu = self.Kuu_se(w, b_, reducevar)
            Kuu = torch.exp(self.Kernel.log_sf) * Kuu 
        elif self.kernel_type == "smk":
            S, w_smk = self.Kernel.sample_from_spectral(num_rff) 
            Kuu = self.Kuu_smk(S, w_smk) 
        else:
            raise NotImplementedError

        Kff = self.Kernel(X, X)
        Kfu = self.Kfu(X)
        Kfu_train = self.Kfu(self.X)
        
        chol_Kuu = torch.linalg.cholesky(Kuu+ self.jitter* torch.mean(torch.diag(Kuu)).detach()*\
                                         torch.eye(self.inducing_size, device=self.device))
        M_aux = Kuu + torch.exp(-self.log_noise)* (Kfu_train.T @ Kfu_train) 
        chol_M_aux = torch.linalg.cholesky(M_aux + self.jitter* torch.mean(torch.diag(M_aux)).detach()* \
                                           torch.eye(self.inducing_size, device=self.device))
        
        v1 = torch.linalg.solve_triangular(chol_M_aux, Kfu.T, upper=False)
        v2 = torch.linalg.solve_triangular(chol_M_aux, Kfu_train.T @ self.y, upper=False)
        mean = torch.exp(-self.log_noise)* v1.T @ v2
        
        v3 = torch.linalg.solve_triangular(chol_Kuu, Kfu.T, upper=False)
        nystrome_diff = torch.sum(torch.diag(Kff))- torch.sum(v3.pow(2)) # this term is unstable with RFF approximation during training
        if nystrome_diff.item() < 0:
            # rewrite Kff and Kfu using rff to compute trace to make sure it (-trace) is positive definite
            if self.kernel_type == "se":
                Kff_rff, Kfu_rff = self.KfuKff_rff_se(w, X, b_, reducevar)
                Kfu_rff = torch.exp(self.Kernel.log_sf) * Kfu_rff 
                Kff_rff = torch.exp(self.Kernel.log_sf) * Kff_rff
            elif self.kernel_type == "smk":
                Kff_rff, Kfu_rff = self.KfuKff_rff_smk(S, w_smk, X)
            else:
                raise NotImplementedError
            v3_rff = torch.linalg.solve_triangular(chol_Kuu, Kfu_rff.T, upper=False)
            cov = Kff_rff- v3_rff.T @ v3_rff + v1.T @ v1
        else:
            cov = Kff- v3.T @ v3 + v1.T @ v1
        
        if flag_noise:
            cov += torch.exp(self.log_noise)* torch.eye(cov.shape[0], device=self.device) 
        if flag_diag:
            return mean, torch.diag(cov)
        else:
            return mean, cov

    def pred_approx(self, X, num_rff, flag_noise = True, reducevar=True):
        length = X.shape[0]
        vals = np.linspace(0.0, 1.0, length)
        q = np.arange(self.inducing_size, dtype=np.float64)
        T = np.sqrt(np.diag(2 * q + 1))
        B = np.diag(T)[:, None]
        B = B.squeeze(-1)
        eval_matrix = torch.tensor((B[:, None] * ss.eval_legendre(np.arange(self.inducing_size)[:, None],\
                                                                   2 * vals - 1)).T).to(self.device)

        X = torch.tensor(X)
        if self.kernel_type == "se":
            w = self.Kernel.sample_from_spectral(num_rff).unsqueeze(0) 
            if not reducevar:
                b_ = torch.rand(num_rff, device=self.device).unsqueeze(0)* 2* math.pi 
            else:
                b_ = None
            Kuu = self.Kuu_se(w, b_, reducevar)
            Kuu = torch.exp(self.Kernel.log_sf) * Kuu 
        elif self.kernel_type == "smk":
            S, w_smk = self.Kernel.sample_from_spectral(snum_rff) 
            Kuu = self.Kuu_smk(S, w_smk) 
        else:
            raise NotImplementedError 

        Kfu_train = self.Kfu(self.X)

        M_aux = Kuu + torch.exp(-self.log_noise)* (Kfu_train.T @ Kfu_train) 
        chol_M_aux = torch.linalg.cholesky(M_aux + self.jitter* torch.mean(torch.diag(M_aux)).detach()* \
                                           torch.eye(self.inducing_size, device=self.device))
        v1 = torch.linalg.solve_triangular(chol_M_aux, Kuu, upper=False)
        v2 = torch.linalg.solve_triangular(chol_M_aux, Kfu_train.T @ self.y, upper=False)

        meanU = torch.exp(-self.log_noise) * v1.T @ v2 
        covU = v1.T @ v1

        mean_approx = eval_matrix @ meanU
        cov_approx = eval_matrix.pow(2) @ torch.diag(covU).view(-1, 1) +\
              2 * torch.sum(torch.tril(eval_matrix.unsqueeze(-1)* eval_matrix.unsqueeze(1)* covU, diagonal = -1),\
                             (1, 2)).unsqueeze(1)
        
        if flag_noise:
            cov_approx += torch.exp(self.log_noise)* torch.ones_like(cov_approx) 
        return mean_approx, cov_approx
    
    def nll_rmse(self, X, y, w, flag_noise = True, reducevar=True):
        mean, cov = self.pred(X, w, flag_noise = flag_noise, flag_diag = True, reducevar=reducevar)
        rmse = torch.sqrt(torch.mean((y - mean).pow(2)))

        chol_cov = cov.pow(0.5)
        nll = torch.sum(torch.log(chol_cov)) + 0.5* X.shape[0] * torch.log(torch.tensor(2* np.pi))
        nll += 0.5* torch.sum((y - mean).pow(2).squeeze(1)/  chol_cov.pow(2))
        nll /= y.shape[0]
        
        return nll, rmse


    def nll_rmse_approx(self, X, y, num_rff, flag_noise = True, reducevar=True):
        mean, cov = self.pred_approx(X, num_rff, flag_noise = flag_noise, reducevar=reducevar)    
        rmse = torch.sqrt(torch.mean((y - mean).pow(2)))
        chol_cov = cov.pow(0.5)
        
        nll = torch.sum(torch.log(chol_cov)) + 0.5* X.shape[0] * torch.log(torch.tensor(2* np.pi))
        nll += 0.5* torch.sum((y - mean).pow(2)/  chol_cov.pow(2))
        nll /= y.shape[0]
        
        return nll, rmse
    
    def qb_opt(self, w):
        if self.kernel_type == "se":
            Kuu = self.Kuu_se(w, b=None, reducevar=True)
            Kuu = torch.exp(self.Kernel.log_sf) * Kuu 
        else:
            raise NotImplementedError
        Kuu += self.jitter* torch.mean(torch.diag(Kuu)).detach()* \
                torch.eye(self.inducing_size, device=self.device)

        Kfu = Kuu.clone()
        Kfu_train = self.Kfu(self.X)
        
        M_aux = Kuu + torch.exp(-self.log_noise)* (Kfu_train.T @ Kfu_train) 
        chol_M_aux = torch.linalg.cholesky(M_aux + self.jitter* torch.mean(torch.diag(M_aux)).detach()* \
                                           torch.eye(self.inducing_size, device=self.device))
        
        v1 = torch.linalg.solve_triangular(chol_M_aux, Kfu.T, upper=False)
        v2 = torch.linalg.solve_triangular(chol_M_aux, Kfu_train.T @ self.y, upper=False)
        mean = torch.exp(-self.log_noise)* v1.T @ v2
        cov = v1.T @ v1
        return mean, cov, Kuu

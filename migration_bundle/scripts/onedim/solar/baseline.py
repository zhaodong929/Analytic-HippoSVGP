import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
from tqdm import tqdm
from scripts.onedim.solar.kernel import *
import math


class OSGPR_VFE(nn.Module):
    """
    Online Sparse Variational GP regression.
    
    Streaming Gaussian process approximations
    Thang D. Bui, Cuong V. Nguyen, Richard E. Turner
    NIPS 2017
    code adapted from https://github.com/thangbui/streaming_sparse_gp/blob/master/code/osgpr.py
    """

    def __init__(self, X, y, mu_old, Su_old, Kaa_old, Z_old, Z, device, kernel_type='se', jitter=1e-6):
        super(OSGPR_VFE, self).__init__()
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        if kernel_type == "se":
            self.Kernel = SE_kernel(1, device)
        elif kernel_type == "smk":
            self.Kernel = SMK_kernel(2, device)
        else:
            raise NotImplementedError

        self.num_data = self.X.shape[0]
        self.z = nn.Parameter(torch.tensor(Z)) # current inducing locations, trainable
        self.mu_old = torch.tensor(mu_old).detach()
        self.M_old = Z_old.shape[0] # number of old inudicng points
        self.Su_old = torch.tensor(Su_old).detach()
        self.Kaa_old = torch.tensor(Kaa_old).detach()
        self.Z_old = torch.tensor(Z_old).detach()

        self.jitter = jitter
        self.device= device
        self.log_noise = nn.Parameter(-3.+ 0.* torch.tensor(npr.randn(1)))

    def add_jitter(self, Kuu):
        return Kuu + self.jitter* torch.mean(torch.diag(Kuu)).detach()*\
            torch.eye(Kuu.shape[0], device=self.device)
    
    def _common_terms(self):
        Mb = self.z.shape[0]
        Ma = self.M_old

        sigma2 = torch.exp(self.log_noise)
        sigma = torch.sqrt(sigma2)

        Saa = self.Su_old
        ma = self.mu_old

        # a is old inducing points, b is new
        # f is training points
        # s is test points
        Kbf = self.Kernel(self.z, self.X)
        Kbb = self.add_jitter(self.Kernel(self.z, self.z))
        Kba = self.Kernel(self.z, self.Z_old)
        Kaa_cur = self.add_jitter(self.Kernel(self.Z_old, self.Z_old))
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

        return (Kbf, Kba, Kaa, Kaa_cur, La, Kbb, Lb, D, LD,
                Lbinv_Kba, LDinv_Lbinv_c, err, d1)

    def ELBO(self):
        Mb = self.z.shape[0]
        Ma = self.M_old
        sigma2 = torch.exp(self.log_noise)
        sigma = torch.sqrt(sigma2)
        N = self.num_data

        Saa = self.Su_old
        ma = self.mu_old

        # a is old inducing points, b is new
        # f is training points
        Kfdiag = torch.diag(self.Kernel(self.X, self.X))
        (Kbf, Kba, Kaa, Kaa_cur, La, Kbb, Lb, D, LD,
            Lbinv_Kba, LDinv_Lbinv_c, err, Qff) = self._common_terms()

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
        #optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer = torch.optim.Adam([self.z], lr=lr)
        for i in tqdm(range(iters)):
            optimizer.zero_grad()
            loss = -self.ELBO()
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print('epoch number: {}, ELBO: {}'.format(i+1, loss.item()))

    def pred(self, Xnew, full_cov=False, flag_noise = True):
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. 
        """

        # a is old inducing points, b is new
        # f is training points
        # s is test points
        Kbs = self.Kernel(self.z, Xnew)
        (Kbf, Kba, Kaa, Kaa_cur, La, Kbb, Lb, D, LD,
            Lbinv_Kba, LDinv_Lbinv_c, err, Qff) = self._common_terms()

        Lbinv_Kbs = torch.linalg.solve_triangular(Lb, Kbs, upper=False)
        LDinv_Lbinv_Kbs = torch.linalg.solve_triangular(LD, Lbinv_Kbs, upper=False)
        mean = LDinv_Lbinv_Kbs.T @ LDinv_Lbinv_c

        if full_cov:
            Kss = self.add_jitter(self.Kernel(Xnew, Xnew)) 
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
    
    def nll_rmse(self, X, y, flag_noise = True):
        mean, cov = self.pred(X, full_cov=False, flag_noise = flag_noise)
        chol_cov = cov.pow(0.5)
        nll = torch.sum(torch.log(chol_cov)) + 0.5* X.shape[0]* torch.log(torch.tensor(2* np.pi))
        nll += 0.5* torch.sum((y - mean).pow(2).squeeze(1)/  cov)
        nll /= y.shape[0]

        rmse = torch.sqrt(torch.mean((y - mean)**2))
        return nll, rmse

    

class SVGP(nn.Module):
    def __init__(self, X, y, inducing_size, device, kernel_type="se"):
        super(SVGP, self).__init__()
        self.X = torch.tensor(X)
        self.input_dim = X.shape[1]
        self.y = torch.tensor(y)
        self.jitter = 1e-6        
        self.log_noise = nn.Parameter(-3.+ 0.* torch.tensor(npr.randn(1)))
        self.device=device
        if kernel_type == "se":
            self.Kernel = SE_kernel(self.input_dim, device)
        elif kernel_type == "smk":
            self.Kernel = SMK_kernel(2, device)
        else:
            raise NotImplementedError
        # inducing points
        self.inducing_size = inducing_size
        if inducing_size < X.shape[0]:
            self.z = nn.Parameter(torch.tensor(X[npr.randint(0, X.shape[0], inducing_size), :]))
        else:
            self.z = nn.Parameter(X.clone())
            
    def ELBO(self):
        Kuu = self.Kernel(self.z, self.z)
        Kfu = self.Kernel(self.X, self.z)
        Kff = self.Kernel(self.X, self.X)
        N_input = self.X.shape[0]
        const = -0.5* N_input * torch.log(torch.tensor(2* np.pi))      
        
        # To compute the log-determinant term, we need to use Sylvester's determinant identity
        M_aux = Kuu + torch.exp(-self.log_noise)* (Kfu.T @ Kfu) 
        chol_M_aux = torch.linalg.cholesky(M_aux + self.jitter* torch.mean(torch.diag(M_aux)).detach()*\
                                            torch.eye(self.inducing_size, device=self.device))
        chol_Kuu = torch.linalg.cholesky(Kuu+ self.jitter* torch.mean(torch.diag(Kuu)).detach()*\
                                          torch.eye(self.inducing_size, device=self.device))
        log_det = -0.5* (N_input * self.log_noise) - torch.sum(torch.log(torch.diag(chol_M_aux))) \
                                              + torch.sum(torch.log(torch.diag(chol_Kuu)))
        
        # To compute the quadratic term, we need to use Woodbury's matrix identity
        quadratic = -0.5* torch.exp(-self.log_noise)* self.y.T @ self.y 
        v = torch.linalg.solve_triangular(chol_M_aux, Kfu.T @ self.y, upper=False)
        quadratic += 0.5* torch.exp(-2* self.log_noise)* v.T @ v 
        
        # Compute the trace term
        b = torch.linalg.solve_triangular(chol_Kuu, Kfu.T, upper=False)
        trace = -0.5* torch.exp(-self.log_noise)* (torch.sum(torch.diag(Kff))- torch.sum(b**2))
        
        return const + log_det + quadratic + trace
    
    def train(self, lr = 0.01, iters = 5000):
        #optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer = torch.optim.Adam([self.z], lr=lr)
        for i in tqdm(range(iters)):
            optimizer.zero_grad()
            loss = -self.ELBO()
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print('epoch number: {}, ELBO: {}'.format(i+1, loss.item()))
    
    def pred(self, X, full_cov=False, flag_noise = True):
        X = torch.tensor(X)
        Kuu = self.Kernel(self.z, self.z)
        Kfu = self.Kernel(X, self.z)
        Kff = self.Kernel(X, X)
        Kfu_train = self.Kernel(self.X, self.z)
        
        chol_Kuu = torch.linalg.cholesky(Kuu+ self.jitter* torch.mean(torch.diag(Kuu)).detach()*\
                                         torch.eye(self.inducing_size, device=self.device))
        M_aux = Kuu + torch.exp(-self.log_noise)* (Kfu_train.T @ Kfu_train) 
        chol_M_aux = torch.linalg.cholesky(M_aux + self.jitter* torch.mean(torch.diag(M_aux)).detach()* \
                                           torch.eye(self.inducing_size, device=self.device))
        
        v1 = torch.linalg.solve_triangular(chol_M_aux, Kfu.T, upper=False)
        v2 = torch.linalg.solve_triangular(chol_M_aux, self.Kernel(self.X, self.z).T @ self.y, upper=False)
        mean = torch.exp(-self.log_noise)* v1.T @ v2
        
        v3 = torch.linalg.solve_triangular(chol_Kuu, Kfu.T, upper=False)
        cov = Kff - v3.T @ v3 + v1.T @ v1
        if flag_noise:
            cov += torch.exp(self.log_noise)* torch.eye(cov.shape[0], device=self.device) # add noise variance
        if full_cov:
            return mean, cov
        else:
            return mean, torch.diag(cov)
    
    def nll_rmse(self, X, y, flag_noise = True):
        mean, cov = self.pred(X, full_cov=False, flag_noise=flag_noise)
        rmse = torch.sqrt(torch.mean((y - mean).pow(2)))

        chol_cov = cov.pow(0.5)
        nll = torch.sum(torch.log(chol_cov)) +  0.5* X.shape[0]* torch.log(torch.tensor(2* np.pi))
        nll += 0.5* torch.sum((y - mean).pow(2).squeeze(1)/  chol_cov.pow(2))
        nll /= y.shape[0]
        return nll, rmse


class FullGP(nn.Module):
    def __init__(self, X, y, device, kernel_type="se"):
        super(FullGP, self).__init__()
        self.X = torch.tensor(X)
        self.input_dim = X.shape[1]
        self.y = torch.tensor(y)
        self.log_noise = nn.Parameter(-3.+ 0.* torch.tensor(npr.randn(1))) # log-noise-variance
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
        
    def train(self, lr = 0.01, iters = 5000):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
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

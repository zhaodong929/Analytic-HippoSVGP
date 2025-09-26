"""Torch Implementation for kernel matrices of VFF with Matern52 kernel"""
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import math
from tqdm import tqdm

float_type = torch.float64

def BlockDiagMat(A, B):
    tl_shape = (A.shape[0], B.shape[1])
    br_shape = (B.shape[0], A.shape[1])
    top = torch.concat([A, torch.zeros(tl_shape, dtype=float_type, device=A.device)], axis=1)
    bottom = torch.concat([torch.zeros(br_shape, dtype=float_type, device=B.device), B], axis=1)
    return torch.concat([top, bottom], axis=0)

def LowRankMat(d, W):
    return torch.diag(d) + W @ W.T

def Rank1Mat(d, v):
    V = v.unsqueeze(1)
    return torch.diag(d) + V @ V.T

class VFF_Matern52(nn.Module):
    def __init__(self, a, b, M, device='cpu', dtype=torch.float64):
        super(VFF_Matern52, self).__init__()
        self.a = a
        self.b = b
        self.M = M
        self.ms = torch.arange(M, device=device)
        self.dtype = dtype
        self.device = device
        self.log_sf = nn.Parameter(-2. + 0.* torch.tensor(npr.randn(1), dtype=dtype))
        self.log_ls = nn.Parameter(1. + 0.* torch.tensor(npr.randn(1), dtype=dtype))
    
    def Kernel(self, x1, x2):
        length_scale = torch.exp(self.log_ls)
        dist_sq = torch.cdist(x1, x2, p=2).pow(2)
        dist = torch.sqrt(dist_sq) / length_scale
        sqrt5 = math.sqrt(5.)
        return torch.exp(self.log_sf)* (1. + sqrt5* dist + \
            (5./ 3.)* dist_sq/ length_scale.pow(2))* torch.exp(-sqrt5* dist)

    def make_Kuu(self):
        omegas = 2. * np.pi * self.ms / (self.b-self.a)
        lengthscale = torch.exp(self.log_ls)
        variance = torch.exp(self.log_sf)
        # cos part:
        lamb = np.sqrt(5.0) / lengthscale
        sixteen_or_32 = torch.where(omegas == 0, 16., 32.)
        v1 = (3 * torch.square(omegas / lamb) - 1) / torch.sqrt(8 * variance)
        v2 = torch.ones(v1.shape, dtype=self.dtype, device=self.device) / torch.sqrt(variance)
        W_cos = torch.concat([v1.unsqueeze(1), v2.unsqueeze(1)], axis=1)
        d_cos = 3 * (self.b - self.a) / sixteen_or_32 / lamb.pow(5)/ variance \
            * (torch.square(lamb) + torch.square(omegas)).pow(3)

        # sin part
        omegas = omegas[omegas != 0]  # don't compute omega=0
        v_sin = np.sqrt(3.) * omegas / lamb / torch.sqrt(variance)
        d_sin = 3 * (self.b - self.a) / 32. / lamb.pow(5) / variance\
            * (torch.square(lamb) + torch.square(omegas)).pow(3)
        return BlockDiagMat(LowRankMat(d_cos, W_cos), Rank1Mat(d_sin, v_sin))

    def make_Kuf(self, X):
        omegas = 2. * np.pi * self.ms / (self.b-self.a)
        Kuf_cos = (torch.cos(omegas * (X - self.a))).T
        omegas = omegas[omegas != 0]  # don't compute zeros freq.
        Kuf_sin = (torch.sin(omegas * (X - self.a))).T
        return torch.concat([Kuf_cos, Kuf_sin], axis=0)


class VFF_OSGPR_VFE(nn.Module):
    """
    Online Sparse Variational GP regression.
    
    Streaming Gaussian process approximations
    Thang D. Bui, Cuong V. Nguyen, Richard E. Turner
    NIPS 2017
    code adapted from https://github.com/thangbui/streaming_sparse_gp/blob/master/code/osgpr.py
    """

    def __init__(self, X, y, mu_old, Su_old, Kaa_old, vff, vff_old, device, jitter=1e-6):
        super(VFF_OSGPR_VFE, self).__init__()
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        self.vff = vff
        self.vff_old = vff_old

        self.num_data = self.X.shape[0]
        self.mu_old = torch.tensor(mu_old).detach()
        self.M_old = 2* self.vff_old.M- 1
        self.Su_old = torch.tensor(Su_old).detach()
        self.Kaa_old = torch.tensor(Kaa_old).detach()

        self.jitter = jitter
        self.device= device
        self.log_noise = nn.Parameter(-3.+ 0.* torch.tensor(npr.randn(1)))

    def add_jitter(self, Kuu):
        return Kuu + self.jitter* torch.mean(torch.diag(Kuu)).detach()*\
            torch.eye(Kuu.shape[0], device=self.device)
    
    def _common_terms(self):
        Mb = 2* self.vff.M- 1
        Ma = self.M_old

        sigma2 = torch.exp(self.log_noise)
        sigma = torch.sqrt(sigma2)

        Saa = self.Su_old
        ma = self.mu_old

        # a is old inducing points, b is new
        # f is training points
        # s is test points
        Kbf = self.vff.make_Kuf(self.X)
        Kbb = self.add_jitter(self.vff.make_Kuu())
        Kba = Kbb.clone()
        Kaa_cur = Kbb.clone()
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
        Mb = 2* self.vff.M- 1
        Ma = self.M_old
        sigma2 = torch.exp(self.log_noise)
        sigma = torch.sqrt(sigma2)
        N = self.num_data

        Saa = self.Su_old
        ma = self.mu_old

        # a is old inducing points, b is new
        # f is training points
        Kfdiag = torch.diag(self.vff.Kernel(self.X, self.X))
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
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
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
        Kbs = self.vff.make_Kuf(Xnew)
        (Kbf, Kba, Kaa, Kaa_cur, La, Kbb, Lb, D, LD,
            Lbinv_Kba, LDinv_Lbinv_c, err, Qff) = self._common_terms()

        Lbinv_Kbs = torch.linalg.solve_triangular(Lb, Kbs, upper=False)
        LDinv_Lbinv_Kbs = torch.linalg.solve_triangular(LD, Lbinv_Kbs, upper=False)
        mean = LDinv_Lbinv_Kbs.T @ LDinv_Lbinv_c

        if full_cov:
            Kss = self.add_jitter(self.vff.Kernel(Xnew, Xnew)) 
            var1 = Kss
            var2 = - Lbinv_Kbs.T @ Lbinv_Kbs
            var3 = LDinv_Lbinv_Kbs.T @ LDinv_Lbinv_Kbs
            var = var1 + var2 + var3
            if flag_noise:
                var += torch.exp(self.log_noise)* torch.eye(var.shape[0], device=self.device) # add noise variance
        else:
            var1 = torch.diag(self.vff.Kernel(Xnew, Xnew))
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
    
    def qb_opt(self, full_cov=True):
        (Kbf, Kba, Kaa, Kaa_cur, La, Kbb, Lb, D, LD,
            Lbinv_Kba, LDinv_Lbinv_c, err, Qff) = self._common_terms()
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

    
class VFF_SVGP(nn.Module):
    def __init__(self, X, y, vff, device):
        super(VFF_SVGP, self).__init__()
        self.X = torch.tensor(X)
        self.input_dim = X.shape[1]
        self.y = torch.tensor(y)
        self.jitter = 1e-6        
        self.log_noise = nn.Parameter(-3.+ 0.* torch.tensor(npr.randn(1)))
        self.device=device
        self.vff = vff
        # inducing points
        self.inducing_size = 2* vff.M - 1
            
    def ELBO(self):
        Kuu = self.vff.make_Kuu()
        Kfu = self.vff.make_Kuf(self.X).T
        Kff = self.vff.Kernel(self.X, self.X)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for i in tqdm(range(iters)):
            optimizer.zero_grad()
            loss = -self.ELBO()
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print('epoch number: {}, ELBO: {}'.format(i+1, loss.item()))
    
    def pred(self, X, full_cov=False, flag_noise = True):
        X = torch.tensor(X)
        Kuu = self.vff.make_Kuu()
        Kfu = self.vff.make_Kuf(X).T
        Kff = self.vff.Kernel(X, X)

        Kfu_train = self.vff.make_Kuf(self.X).T
        
        chol_Kuu = torch.linalg.cholesky(Kuu+ self.jitter* torch.mean(torch.diag(Kuu)).detach()*\
                                         torch.eye(self.inducing_size, device=self.device))
        M_aux = Kuu + torch.exp(-self.log_noise)* (Kfu_train.T @ Kfu_train) 
        chol_M_aux = torch.linalg.cholesky(M_aux + self.jitter* torch.mean(torch.diag(M_aux)).detach()* \
                                           torch.eye(self.inducing_size, device=self.device))
        
        v1 = torch.linalg.solve_triangular(chol_M_aux, Kfu.T, upper=False)
        v2 = torch.linalg.solve_triangular(chol_M_aux, Kfu_train.T @ self.y, upper=False)
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
        rmse = torch.sqrt(torch.mean((y - mean)**2))
        chol_cov = cov.pow(0.5) 
        nll = torch.sum(torch.log(chol_cov)) +  0.5* X.shape[0]* torch.log(torch.tensor(2* np.pi))
        nll += 0.5* torch.sum((y - mean).pow(2).squeeze(1)/  chol_cov.pow(2))
        nll /= y.shape[0]
        return nll, rmse
    
    def qb_opt(self):
        Kuu = self.vff.make_Kuu()
        Kuu += self.jitter* torch.mean(torch.diag(Kuu)).detach()* \
                torch.eye(self.inducing_size, device=self.device)

        Kfu = Kuu.clone()
        Kfu_train = self.vff.make_Kuf(self.X).T
        
        M_aux = Kuu + torch.exp(-self.log_noise)* (Kfu_train.T @ Kfu_train) 
        chol_M_aux = torch.linalg.cholesky(M_aux + self.jitter* torch.mean(torch.diag(M_aux)).detach()* \
                                           torch.eye(self.inducing_size, device=self.device))
        
        v1 = torch.linalg.solve_triangular(chol_M_aux, Kfu.T, upper=False)
        v2 = torch.linalg.solve_triangular(chol_M_aux, Kfu_train.T @ self.y, upper=False)
        mean = torch.exp(-self.log_noise)* v1.T @ v2
        cov = v1.T @ v1
        return mean, cov, Kuu


"""
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VFF_Matern52(a=0, b=10, M=50, device=device).to(device)
    Kuu = model.make_Kuu()
    print(Kuu.shape)
    print(torch.mean((Kuu == Kuu.T)+0.).item() == 1)

    X = torch.randn(20, 1).to(device)
    Kuf = model.make_Kuf(X)
    print(Kuf.shape)
"""
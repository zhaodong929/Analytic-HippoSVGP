from hipposvgp.kernel import SE_kernel, SMK_kernel
from hipposvgp.hippo import HiPPO_LegS
from hipposvgp.utils import PositiveDefiniteParameter
import torch
import torch.nn as nn
import numpy.random as npr
import math
import numpy as np

import torch
import torch.nn as nn
import numpy as np

class HIPPOOSVGP(nn.Module):
    def __init__(
        self,
        kernel,
        likelihood,
        end,
        start,
        rff_sample_size,
        num_discrete_steps,
        hippo,
        inducing_size,
        device,
        prev_discrete_steps=0,
        num_inducing_old=None,
        mv_old=None,
        Lv_old=None,
        Kaa_old=None,
        Z_old=None,
        dtype=torch.float64,
        flag_fast=False,
        flag_update_kernel=True,
    ):
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.flag_fast = flag_fast

        # Kernel selection
        self.kernel = kernel

        # Likelihood
        self.likelihood = likelihood

        # Old posterior parameters
        self.num_inducing_old = (num_inducing_old
            if num_inducing_old is not None
            else None
        )

        self.mv_old = (
            torch.tensor(mv_old, dtype=dtype, device=device).detach()
            if mv_old is not None
            else None
        )
        self.Lv_old = (
            torch.tensor(Lv_old, dtype=dtype, device=device).detach()
            if Lv_old is not None
            else None
        )
        self.Kaa_old = (
            torch.tensor(Kaa_old, dtype=dtype, device=device).detach()
            if Kaa_old is not None
            else None
        )
        self.Z_old = (
            torch.tensor(Z_old, dtype=dtype, device=device).detach()
            if Z_old is not None
            else None
        )

        # initialize the variational parameters
        # initialise a as zeros and L as an identity matrix
        self.inducing_size = inducing_size
        # Initialize mv and Lv on CPU first
        mv = torch.zeros(self.inducing_size).double()
        Lv = torch.eye(self.inducing_size).double()

        # Wrap them with nn.Parameter
        self.mv = nn.Parameter(mv, requires_grad=True)
        # self.Lv = PositiveDefiniteParameter(inducing_size)
        self.Lv = nn.Parameter(Lv, requires_grad=True)

        # Move the parameters to the specified device
        # self.mv = self.mv.to(self.device)
        # self.Lv = self.Lv.to(self.device)

        # Random Fourier Feature (RFF) sample size
        self.rff_sample_size = rff_sample_size

        # HIPPO parameters
        self.end = end
        self.start = start
        self.maxT = end - start
        self.num_discrete_steps = (
            num_discrete_steps  # Step size = maxT / num_discrete_steps
        )
        self.prev_discrete_steps = int(prev_discrete_steps)
        self.hippo = hippo
        
        self.flag_update_kernel = flag_update_kernel

        # Stability
        self.jitter = 1e-6

    def add_jitter(self, Kuu):
        return Kuu + self.jitter * torch.mean(torch.diag(Kuu)).detach() * torch.eye(
            Kuu.shape[0], device=self.device
        )

    def get_time_stamps(self):
        time_stamps = self.start + (self.maxT / self.num_discrete_steps) * (
            1
            + torch.arange(
                self.num_discrete_steps, dtype=self.dtype, device=self.device
            )
        ).unsqueeze(1)
        return time_stamps

    def apply_hippo(self, h):
        if not self.flag_update_kernel and self.Z_old!= None:
            Z = self.hippo(
                h,
                ini=self.Z_old,
                prev_discrete_steps=self.prev_discrete_steps,
                fast=self.flag_fast,
            )[-1]  # [num_rff, inducing_size]
            Z_old_newk = None
        else:
            res = self.hippo(h, ini=None, prev_discrete_steps=0,\
                 fast=self.flag_fast)
            Z = res[-1]
            Z_old_newk = res[self.prev_discrete_steps-1] 
        return Z, Z_old_newk

    def get_Z(self, w):
        """Get the RFF features term Z"""
        time_stamps = self.get_time_stamps()
        rff_sample_size = w.shape[1]

        h = (1 / rff_sample_size) ** 0.5 * torch.cat(
            [torch.sin(w * time_stamps), torch.cos(w * time_stamps)], 1
        )
        Z, Z_old_newk = self.apply_hippo(h)
        return Z, Z_old_newk, time_stamps

    def Kuu_se(self, w):
        Z, Z_old_newk, _ = self.get_Z(w)
        Kuu = Z.T @ Z

        Kuuold = None
        Kuolduold = None
        if self.flag_update_kernel:
            if self.Kaa_old is not None:
                Kuuold = Z.T @ Z_old_newk
                Kuolduold = self.Kaa_old
        else:
            if self.Z_old is not None:
                Kuuold = Z.T @ self.Z_old
                Kuolduold = self.Kaa_old
        return Z, Z_old_newk, Kuu, Kuuold, Kuolduold

    def KfuKff_rff_se(self, w, X):
        Z, _, _ = self.get_Z(w)
        rff_sample_size = w.shape[1]
        Z_f = (1 / rff_sample_size) ** 0.5 * torch.cat(
            [torch.sin(w * X), torch.cos(w * X)], 1
        )  # [n, num_rff]
        
        Kfu = Z_f @ Z
        Kff = Z_f @ Z_f.T
        return Kff, Kfu

    def KuuKfuKff_rff_se(self, w, X):
        Z, Z_old_newk, _ = self.get_Z(w)
        rff_sample_size = w.shape[1]
        Z_f = (1 / rff_sample_size) ** 0.5 * torch.cat(
            [torch.sin(w * X), torch.cos(w * X)], 1
        )  # [n, num_rff] 
        Kfu = Z_f @ Z # [n, inducing_size]
        Kff = Z_f @ Z_f.T # [n, n]
        Kuu = Z.T @ Z # [inducing_size, inducing_size]
        Kuuold = None
        Kuolduold = None
        kuolduold_newk = None
        if self.flag_update_kernel:
            if self.Kaa_old is not None:
                Kuuold = Z.T @ Z_old_newk
                Kuolduold = self.Kaa_old
                kuolduold_newk = Z_old_newk.T @ Z_old_newk
        else:
            if self.Z_old is not None:
                Kuuold = Z.T @ self.Z_old
                Kuolduold = self.Kaa_old
                kuolduold_newk = self.Kaa_old / self.kernel.log_sf
        return Kff, Kfu, Kuu, Kuuold, Kuolduold, kuolduold_newk

    def quold_new(self, Kuu, Kuu_old, Ku_old_old_newk, full_cov=True):
         Kff_new = Ku_old_old_newk
         Kfu_new = Kuu_old.T
 
         Luu = torch.linalg.cholesky(self.add_jitter(Kuu))# + 1e-6 * torch.eye(Kuu.shape[0], device=Kuu.device))
         # 2) get Kuu^{-1}-related terms
         #    Luu is cholesky(Kuu)
         #    so e.g.  torch.cholesky_solve(B, Luu) => Kuu^{-1} B
         # 3) unwhiten the posterior parameters in "u"-space:
         mu_u = Luu @ self.mv           # mean of u
         mu_u = mu_u.unsqueeze(1)       # [M, 1]
 
         L_u  = Luu @ self.Lv           # factor for Cov[u] = L_u L_u^T
 
         # 4) predictive mean
         alpha = torch.cholesky_solve(mu_u, Luu)  # => Kuu^-1 mu_u
         mean  = Kfu_new @ alpha                  # => [N, 1]
 
         # 5) predictive var
         Kuu_inv_Kfu = torch.cholesky_solve(Kfu_new.T, Luu)
         var = Kff_new - Kfu_new @ Kuu_inv_Kfu  # "prior conditional" part
 
         if full_cov:
             temp = torch.cholesky_solve(L_u, Luu)  # => Kuu^-1 L_u
             Z    = Kfu_new @ temp                 # => [N, r]
             var += Z @ Z.T                        # => [N, N]
         else:
             temp = torch.cholesky_solve(L_u, Luu)  # => [M, r]
             var_diag_extra = torch.sum((Kfu_new @ temp)**2, dim=1)
             var = torch.diag(Kff_new) - torch.sum(Kfu_new * Kuu_inv_Kfu.T, dim=1)
             var += var_diag_extra
             var = var.unsqueeze(1)  # [N, 1]
         return mean, var

    def correction_term(self, Kuu, Kuu_old, Ku_old_old, Ku_old_old_newk):
        """
        if Kuu is None or Kuu_old is None or Ku_old_old is None:
            _, _, Kuu, Kuu_old, Ku_old_old = self.KuuKfuKff_rff_se(w, X)
            Kuu *= torch.exp(self.kernel.log_sf)
            Kuu_old *= torch.exp(self.kernel.log_sf)
            Ku_old_old *= torch.exp(self.kernel.log_sf)
        Ku_old_u = Kuu_old.T
        Luu = torch.linalg.cholesky(self.add_jitter(Kuu))
        
        mv_new = self.mv.unsqueeze(-1)
        Lv_new = self.Lv
        m_new = Luu @ mv_new
        S_new = Luu @ Lv_new @ Lv_new.T @ Luu.T
        """
        if self.mv_old is None or self.Lv_old is None:
            return torch.tensor(0.0, device=self.device)
        mv_old = self.mv_old.unsqueeze(-1)
        Lv_old = self.Lv_old
        Luu_old = torch.linalg.cholesky(self.add_jitter(Ku_old_old))
        m_old = Luu_old @ mv_old
        S_old = Luu_old @ Lv_old @ Lv_old.T @ Luu_old.T
        
        #mu = Ku_old_u @ torch.cholesky_solve(m_new, Luu)
        #Sigma = compute_Sigma(Ku_old_old, Ku_old_u, Kuu_old, Kuu, Luu, S_new)
        mu, Sigma = self.quold_new(Kuu, Kuu_old, Ku_old_old_newk, full_cov=True)
        Smm = Sigma + mu @ mu.T
        
        LSa = torch.linalg.cholesky(self.add_jitter(S_old))
        LKa = torch.linalg.cholesky(self.add_jitter(Ku_old_old))
        
        obj = torch.sum(torch.log(torch.diagonal(LKa))) - torch.sum(torch.log(torch.diagonal(LSa)))
        Sainv_ma = torch.cholesky_solve(m_old, LSa).squeeze(-1)
        obj -= 0.5 * torch.sum(m_old.squeeze(-1) * Sainv_ma)
        obj += torch.sum(mu.squeeze(-1) * Sainv_ma)
        
        Sainv_Smm = torch.cholesky_solve(Smm, LSa)
        Kainv_Smm = torch.cholesky_solve(Smm, LKa)
        obj -= 0.5 * (torch.trace(Sainv_Smm) - torch.trace(Kainv_Smm))
        
        return obj

    def ELBO(self, X, y, w, recompute_k=True, cache_k=False):
        # When updating kernel - recompute_k=True, cache_k=False
        # For fixed kernel hyperparameters 
        #    - recompute_k=True, cache_k=True (the first time computing elbo)
        #    - recompute_k=False, cache_k=False (after first time computing elbo)
        # 1. Sample RFF
        if w == None:
            w = self.kernel.sample_from_spectral(self.rff_sample_size).unsqueeze(0)
        # 2. Compute kernel matrices
        if recompute_k:
            Kff, Kfu, Kuu, Kuuold, Kuolduold, kuolduold_newk = self.KuuKfuKff_rff_se(w, X)
            Kff *= torch.exp(self.kernel.log_sf)
            Kfu *= torch.exp(self.kernel.log_sf)
            Kuu *= torch.exp(self.kernel.log_sf)
            if cache_k:
                self.Kff = Kff
                self.Kfu = Kfu
                self.Kuu = Kuu
                self.kuolduold_newk = kuolduold_newk
            if Kuuold != None:
                Kuuold *= torch.exp(self.kernel.log_sf)
                if cache_k:
                    self.Kuuold = Kuuold
            else:
                if cache_k:
                    self.Kuuold = None
            if kuolduold_newk != None:
                kuolduold_newk *= torch.exp(self.kernel.log_sf)
                if cache_k:
                    self.kuolduold_newk = kuolduold_newk
            else:
                if cache_k:
                    self.kuolduold_newk = None
            if cache_k:
                if Kuolduold != None:
                    self.Kuolduold = Kuolduold
                else:
                    self.Kuolduold = None

        # 3. Cholesky of Kuu
        if not recompute_k: # for fix kernel, we just use the cached kernel matrix
            Kff = self.Kff
            Kfu = self.Kfu
            Kuu = self.Kuu
            if self.Kuuold != None:
                Kuuold = self.Kuuold
            if self.Kuolduold != None:
                Kuolduold = self.Kuolduold
            if self.kuolduold_newk != None:
                kuolduold_newk = self.kuolduold_newk

        # 3. Cholesky of Kuu
        #jitter  = 1e-6
        L_Kuu   = torch.linalg.cholesky(self.add_jitter(Kuu))# + jitter*torch.eye(Kuu.shape[0], device=Kuu.device))

        # 4. Variational parameters in whitened space
        mv = self.mv  # shape (M,)
        Lv = self.Lv  # shape (M, M)

        # 5. KL[q(v)||p(v)] in whitened space

        #    Sv = LvLv^T
        Sv             = Lv @ Lv.T
        trace_term     = torch.trace(Sv)
        quadratic_term = torch.sum(mv**2)        # mv^T mv
        if torch.allclose(Lv, torch.tril(Lv)):
            log_det_Sv = 2.0 * torch.sum(torch.log(torch.diagonal(Lv)))
        else:
            log_det_Sv = torch.logdet(Sv)
        M   = Kuu.shape[0]
        kl  = 0.5*(trace_term + quadratic_term - M - log_det_Sv)

        # --> This is correct for KL[q(v) || p(v)=N(0,I)].

        # 6. Mean of q(f)
        mu_u = L_Kuu @ mv.unsqueeze(-1)  # Unwhiten: m_u = L_Kuu * m_v
        L_Kuu_inv_mu_u = torch.cholesky_solve(mu_u, L_Kuu)  # Compute (Kuu^-1) * mu_u
        mean_f = Kfu @ L_Kuu_inv_mu_u  # Final mean

        # L_Kuu_inv_mv = torch.cholesky_solve(mv.unsqueeze(-1), L_Kuu)
        # # => solves L_Kuu L_Kuu^T x = mv, so x = (Kuu^-1) mv
        # mean_f = Kfu @ L_Kuu_inv_mv
        # => Kfu * (Kuu^-1 mv) = Kfu Kuu^-1 L_Kuu mv (since L_Kuu mv = u)
        # So mean_f = Kfu (Kuu^-1 (E[u])). With E[u]=L_Kuu mv, that part is correct.

        # 7. Covariance of q(f)
        L_Kuu_inv_Kfu_T = torch.cholesky_solve(Kfu.T, L_Kuu)
        # => Kuu^-1 * Kfu^T
        Qff = Kfu @ L_Kuu_inv_Kfu_T
        # => Kfu Kuu^-1 Kfu^T
        L_u = L_Kuu @ Lv
        temp = torch.cholesky_solve(L_u, L_Kuu)
        Z = Kfu @ temp
        Cov_f = Kff - Qff + Z @ Z.T

        # => Kff - Kfu Kuu^-1 Kfu^T + Kfu Kuu^-1 (Lv Lv^T) (Kuu^-1) Kfu^T
        # which is missing the factors L_Kuu and L_Kuu^T in the middle of Lv Lv^T if q(u) = N(L_Kuu mv, L_Kuu LvLv^T L_Kuu^T).

        # --> For whitened q(v), the correct "extra" should be:
        #       Kfu Kuu^-1 [L_Kuu Lv Lv^T L_Kuu^T] Kuu^-1 Kfu^T
        #    i.e. Kfu L_Kuu^-T (Lv Lv^T) L_Kuu^-1 Kfu^T
        #    not Kfu Kuu^-1 (Lv Lv^T) Kuu^-1 Kfu^T.

        # add jitter
        Cov_f = self.add_jitter(Cov_f)# + jitter * torch.eye(Cov_f.shape[0], device=Cov_f.device)

        # 8. Sample f ~ q(f)
        L_Cov_f = torch.linalg.cholesky(Cov_f)
        eps     = torch.randn_like(mean_f)
        f_samples = mean_f + L_Cov_f @ eps

        # 9. Evaluate variational expectations
        variational_expectations = self.likelihood.log_prob(f_samples, y)
        ve_sum  = torch.sum(variational_expectations, dim=-1)
        ve_mean = torch.sum(ve_sum)

        # ve = self.likelihood.variational_expectations(mean_f, Cov_f.diag(), y)
        # ve_sum  = torch.sum(ve, dim=-1)
        # ve_mean = torch.mean(ve_sum)

        # 10. ELBO
        elbo = ve_mean - kl

        # 11. Optional correction
        if self.Z_old is not None:
            corr = self.correction_term(Kuu, Kuuold, Kuolduold, kuolduold_newk)
            elbo = elbo + corr

        return elbo, ve_mean, kl

    def pred_f(self, Xnew, w, full_cov=False, flag_noise=True):
        """
        Compute the mean and variance of the latent function at new points Xnew.

        Args:
            Xnew (torch.Tensor): New input points [N, D].
            full_cov (bool): If True, return the full covariance matrix. If False, return only the diagonal.
            flag_noise (bool): If True, include observation noise in the variance.

        Returns:
            mean (torch.Tensor): Predictive mean at Xnew [N, 1].
            var (torch.Tensor): Predictive variance at Xnew [N, 1] or [N, N] depending on full_cov.
        """

        # 1. Compute required kernel matrices
        if w == None:
            w = self.kernel.sample_from_spectral(self.rff_sample_size).unsqueeze(0)
        _, _, Kuu, _, _ = self.Kuu_se(w)
        Kuu *= torch.exp(self.kernel.log_sf)
        Kff_new, Kfu_new = self.KfuKff_rff_se(w, Xnew)[:2]
        Kff_new *= torch.exp(self.kernel.log_sf)
        Kfu_new *= torch.exp(self.kernel.log_sf)
        Luu = torch.linalg.cholesky(self.add_jitter(Kuu))# + 1e-6 * torch.eye(Kuu.shape[0], device=Kuu.device))
        # 2) get Kuu^{-1}-related terms
        #    Luu is cholesky(Kuu)
        #    so e.g.  torch.cholesky_solve(B, Luu) => Kuu^{-1} B
        # 3) unwhiten the posterior parameters in "u"-space:
        mu_u = Luu @ self.mv           # mean of u
        mu_u = mu_u.unsqueeze(1)       # [M, 1]

        L_u  = Luu @ self.Lv           # factor for Cov[u] = L_u L_u^T

        # 4) predictive mean
        alpha = torch.cholesky_solve(mu_u, Luu)  # => Kuu^-1 mu_u
        mean  = Kfu_new @ alpha                  # => [N, 1]

        # 5) predictive var
        Kuu_inv_Kfu = torch.cholesky_solve(Kfu_new.T, Luu)
        var = Kff_new - Kfu_new @ Kuu_inv_Kfu  # "prior conditional" part

        if full_cov:
            temp = torch.cholesky_solve(L_u, Luu)  # => Kuu^-1 L_u
            Z    = Kfu_new @ temp                 # => [N, r]
            var += Z @ Z.T                        # => [N, N]
        else:
            temp = torch.cholesky_solve(L_u, Luu)  # => [M, r]
            var_diag_extra = torch.sum((Kfu_new @ temp)**2, dim=1)
            var = torch.diag(Kff_new) - torch.sum(Kfu_new * Kuu_inv_Kfu.T, dim=1)
            var += var_diag_extra
            var = var.unsqueeze(1)  # [N, 1]
        return mean, var

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    import numpy as np
    import torch
    import math
    from hipposvgpv2.likelihood import GaussianLikelihood, NegativeBinomial
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kernel_type = "se"
    rff_sample_size = 1000 # we need a large number of rff samples for stable estimation of Kuu...
    inducing_size = 150
    step_size = 2.45

    # load data
    data_train = np.load('data/solar_data_train.npy')
    data_test = np.load('data/solar_data_test.npy')
    x_test = torch.tensor(np.arange(-200., 200., 2.).reshape((-1,1))).to(device)

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

    hippo = HiPPO_LegS(inducing_size, device)
    prev_discrete_steps = 0

    maxT = torch.max(x_list[0]).item()-torch.min(x_list[0]).item()
    start = torch.min(x_list[0]).item() 
    num_discrete_steps = math.ceil(maxT / step_size)
    end = start + num_discrete_steps * step_size
    maxT = end - start

    i = 0
    kernel = SE_kernel(1)
    # likelihood = NegativeBinomial()
    likelihood = GaussianLikelihood(0.1)
    model = HIPPOOSVGP(
        kernel=kernel,
        likelihood=likelihood,
        start=start, 
        end=end, 
        rff_sample_size=rff_sample_size, 
        hippo=hippo,
        num_discrete_steps=num_discrete_steps, 
        inducing_size=inducing_size, 
        device=device, 
    ).to(device)
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.optim import Adam
    from torch.optim.lr_scheduler import StepLR
    from tqdm import trange

    # Assuming 'model', 'x_list', 'y_list', 'x_test', and 'y_t' are predefined
    optimizer = Adam(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    num_epochs = 1000

    # Initialize the plot
    pbar = trange(num_epochs)

    for epoch in pbar:
        optimizer.zero_grad()
        loss = -model.ELBO(x_list[0], y_list[0])  # Assuming x_list[0] and y_list[0] are your training data
        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.set_postfix({'Loss': loss.item()})

        # Update plot every 100 epochs
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                y_pred, y_var = model.pred_f(x_t)
                y_pred = y_pred.cpu().numpy()
                rmse = np.sqrt(np.mean((y_pred - y_t)**2))
                print(f"Epoch {epoch} Test RMSE: {rmse}")

import torch
import torch.nn as nn
from hipposvgp.utils import PositiveDefiniteParameter
from functools import partial

class OSVGP(nn.Module):
    def __init__(
        self,
        kernel,
        likelihood,
        inducing_size,
        device,
        num_inducing_old=None,
        mv_old=None,
        Lv_old=None,
        Kaa_old=None,
        Z_old=None,
        dtype=torch.float64,
    ):
        super().__init__()

        self.device = device
        self.dtype = dtype

        # Kernel selection
        self.kernel = kernel

        # Likelihood
        self.likelihood = likelihood

        # Old posterior parameters
        self.Z_old = Z_old
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
        self.Z_old = Z_old

        # initialize the variational parameters
        # initialise a as zeros and L as an identity matrix
        self.inducing_size = inducing_size
        # trainable inducing points Z
        self.Z = torch.zeros(self.inducing_size).double()[:, None]
        self.Z = nn.Parameter(self.Z, requires_grad=True)
        # Initialize mv and Lv on CPU first

        mv = torch.zeros(self.inducing_size).double()
        Lv = torch.eye(self.inducing_size).double()

        # Wrap them with nn.Parameter
        self.mv = nn.Parameter(mv, requires_grad=True)
        # self.Lv = PositiveDefiniteParameter(inducing_size)
        self.Lv = nn.Parameter(Lv, requires_grad=True)

        # Stability
        self.jitter = 1e-6

    def add_jitter(self, Kuu):
        return Kuu + self.jitter * torch.mean(torch.diag(Kuu)).detach() * torch.eye(
            Kuu.shape[0], device=self.device
        )

    def Kuu_se(self):
        Kuu = self.kernel(self.Z, self.Z)
        if self.Z_old is not None:
            Kuuold = self.kernel(self.Z_old, self.Z)
            #Kuolduold = self.kernel(self.Z_old, self.Z_old)
            Kuolduold = self.Kaa_old
        else:
            Kuuold = None
            Kuolduold = None
        return Kuu, Kuuold, Kuolduold

    def KfuKff_se(self, X):
        Kfu = self.kernel(X, self.Z)
        Kff = self.kernel(X, X)
        return Kff, Kfu

    def KuuKfuKff_se(self, X):
        Kfu = self.kernel(X, self.Z)
        Kff = self.kernel(X, X)
        Kuu = self.kernel(self.Z, self.Z)
        if self.Z_old is not None:
            Kuuold = self.kernel(self.Z, self.Z_old)
            #Kuolduold = self.kernel(self.Z_old, self.Z_old)
            Kuolduold = self.Kaa_old
        else:
            Kuuold = None
            Kuolduold = None
        return Kff, Kfu, Kuu, Kuuold, Kuolduold

    # def correction_term(self, X,
    #                     Kuu=None,
    #                     Kuu_old=None,
    #                     Ku_old_old=None):
    #     """
    #     Computes the correction term using whitened variational parameters.

    #     Args:
    #         X: Input data tensor of shape (N, d).

    #     Returns:
    #         correction: Scalar correction term.
    #     """
    #     # Compute kernel matrices
    #     # If not provided, compute:
    #     if Kuu is None or Kuu_old is None or Ku_old_old is None:
    #         _, _, Kuu, Kuu_old, Ku_old_old = self.KuuKfuKff_se(X)
    #     # Kff, Kfu, Kuu, Kuu_old, Kuu_old_old = self.KuuKfuKff_rff_se(w, X)
    #     Ku_old_u = Kuu_old.T
    #     # Cholesky decomposition of Kuu
    #     Luu = torch.linalg.cholesky(Kuu + 1e-6 * torch.eye(Kuu.shape[0], device=Kuu.device))
        
    #     # Retrieve new whitened variational parameters
    #     mv_new = self.mv      # New variational mean (M,)
    #     Lv_new = self.Lv      # New Cholesky factor of the covariance (M, M)
    #     mv_old = self.mv_old  # Old variational mean (M,)
    #     Lv_old = self.Lv_old  # Old Cholesky factor of the covariance (M, M)

    #     # Transform mv and Lv to the original function space
    #     m_new = Luu @ mv_new
    #     m_new = m_new.unsqueeze(-1)
    #     S_new = Luu @ Lv_new @ Lv_new.T @ Luu.T
    #     Luu_old = torch.linalg.cholesky(Ku_old_old + 1e-6 * torch.eye(Ku_old_old.shape[0], device=Ku_old_old.device))
    #     m_old = Luu_old @ mv_old
    #     m_old = m_old.unsqueeze(-1)
    #     S_old = Luu_old @ Lv_old @ Lv_old.T @ Luu_old.T

    #     # Compute q(f_new) = N(mu, Sigma)
    #     mu = Ku_old_u @ torch.cholesky_solve(m_new, Luu)
    #     # Compute Sigma
    #     Sigma = compute_Sigma(Ku_old_old, Ku_old_u, Kuu_old, Kuu, Luu, S_new)

    #     # Compute Smm
    #     Smm = Sigma + mu @ mu.T

    #     # Add jitter for numerical stability
    #     Ku_old_old = self.add_jitter(Ku_old_old)
    #     # Compute Cholesky decompositions
    #     LSa = torch.linalg.cholesky(S_old)
    #     LKa = torch.linalg.cholesky(Ku_old_old)

    #     # Compute log determinants
    #     obj = torch.sum(torch.log(torch.diagonal(LKa)))
    #     obj -= torch.sum(torch.log(torch.diagonal(LSa)))

    #     # Solve linear systems using Cholesky decomposition
    #     Sainv_ma = torch.cholesky_solve(m_old, LSa).squeeze(-1)
    #     obj -= 0.5 * torch.sum(m_old * Sainv_ma)
    #     obj += torch.sum(mu * Sainv_ma)

    #     Sainv_Smm = torch.cholesky_solve(Smm, LSa)
    #     Kainv_Smm = torch.cholesky_solve(Smm, LKa)

    #     obj -= 0.5 * torch.sum(torch.diagonal(Sainv_Smm) - torch.diagonal(Kainv_Smm))

    #     return obj

    def correction_term(self, X, Kuu=None, Kuu_old=None, Ku_old_old=None):
        if Kuu is None or Kuu_old is None or Ku_old_old is None:
            _, _, Kuu, Kuu_old, Ku_old_old = self.KuuKfuKff_se(X)
        """
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
        mu, Sigma = self.pred_f(self.Z_old, full_cov=True, flag_noise=False)
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

    def ELBO(self, X, y):
        # 2. Compute kernel matrices
        Kff, Kfu, Kuu, Kuuold, Kuolduold = self.KuuKfuKff_se(X)

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
            corr = self.correction_term(X, Kuu, Kuuold, Kuolduold)
            elbo = elbo + corr

        return elbo, ve_mean, kl

    def pred_f(self, Xnew, full_cov=False, flag_noise=True):
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
        Kuu, _, _ = self.Kuu_se()
        Kff_new, Kfu_new = self.KfuKff_se(Xnew)[:2]
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


# def compute_Sigma(K_uolduold, K_uoldu, K_uuold, Kuu, Luu, L_var):
#     """
#     Computes 
#       Sigma = K_uolduold 
#               - K_uoldu @ K_uu^-1 (K_uu - L_var L_var^T) K_uu^-1 @ K_uuold
    
#     using Cholesky-based linear solves instead of explicitly inverting K_uu.
    
#     Arguments:
#     ----------
#     K_uolduold : torch.Tensor (N x N)
#         K_{uold, uold}
#     K_uoldu     : torch.Tensor (N x M)
#         K_{uold, u}
#     K_uuold     : torch.Tensor (M x N)
#         K_{u, uold}
#     K_uu        : torch.Tensor (M x M)
#         K_{u, u}
#     L_var       : torch.Tensor (M x M)
#         L_var in the expression (K_uu - L_var @ L_var^T)
        
#     Returns:
#     --------
#     Sigma : torch.Tensor (N x N)
#         Result of the matrix expression above.
#     """
    
#     # 2. Solve "tmp1 = K_uu^-1 @ K_uuold"
#     #    Using cholesky_solve lets us do K_uu^-1 matmul some tensor
#     #    without forming K_uu^-1 explicitly.
#     tmp1 = torch.cholesky_solve(K_uuold, Luu, upper=False) 
#     # shape of tmp1: (M x N)
    
#     # 3. Compute "tmp2 = (K_uu - L_var @ L_var.T) @ tmp1"
#     K_uu_minus_Lvar = Kuu - L_var @ L_var.T
#     tmp2 = K_uu_minus_Lvar @ tmp1  # shape: (M x N)
    
#     # 4. Solve again for "tmp3 = K_uu^-1 @ tmp2"
#     tmp3 = torch.cholesky_solve(tmp2, Luu, upper=False) 
#     # shape of tmp3: (M x N)
    
#     # 5. Compute the final Sigma
#     #    Sigma = K_uolduold - K_uoldu @ tmp3
#     Sigma = K_uolduold - K_uoldu @ tmp3  # shape: (N x N)
    
#     return Sigma


class OSVGPRenewal(OSVGP):
    def __init__(
        self,
        num_regions,
        kernel,
        likelihood,
        inducing_size,
        device,
        num_inducing_old=None,
        mv_old=None,
        Lv_old=None,
        Kaa_old=None,
        Z_old=None,
        dtype=torch.float64,
    ):
        super().__init__(
            kernel=kernel,
            likelihood=likelihood,
            inducing_size=inducing_size,
            device=device,
            num_inducing_old=num_inducing_old,
            mv_old=mv_old,
            Lv_old=Lv_old,
            Kaa_old=Kaa_old,
            Z_old=Z_old,
            dtype=dtype
        ) 
        
        # override the mv and Lv to have 1 per region
        self.num_regions = num_regions
        mv = torch.zeros(num_regions, self.inducing_size).double()
        Lv = torch.eye(self.inducing_size).double()
        # Expand then clone to create independent copies for each region
        Lv = Lv.unsqueeze(0).expand(num_regions, -1, -1).clone()  # [num_regions, M, M]

        # Wrap them with nn.Parameter
        self.mv = nn.Parameter(mv, requires_grad=True)
        self.Lv = nn.Parameter(Lv, requires_grad=True)

    def add_jitter(self, K):
        """
        Adds a constant 'jitter' to the diagonal of a kernel matrix K for numerical stability.
        Works for either a 2D (single matrix) or 3D (batched) tensor.
        """
        if K.ndim == 2:
            # K is [M, M]
            M = K.shape[0]
            return K + self.jitter * torch.eye(M, device=K.device)

        elif K.ndim == 3:
            # K is [B, M, M]
            B, M, _ = K.shape
            I = torch.eye(M, device=K.device).unsqueeze(0).expand(B, M, M)
            return K + self.jitter * I

        else:
            raise ValueError(f"add_jitter expected 2D or 3D input, got shape {K.shape}")

    def pre_ELBO(self, Kff_batched, Kfu_batched, Kuu, N_batched, stateindex_batched):
        """
        Kff_batched:  [B, N_max, N_max]
        Kfu_batched:  [B, N_max, M]
        Kuu:          [M, M]      (shared across batch)
        N_batched:    [B]
        stateindex_batched: [B]
        Returns: f_samples [B, N_max], kl [B]
        """

        # 1) Batched gather of mv and Lv
        R = self.mv.shape[0]  # number of total "regions"
        B = stateindex_batched.shape[0]
        M = Kuu.shape[0]

        # Build one-hot
        one_hot = torch.nn.functional.one_hot(stateindex_batched, num_classes=R).double()
        mv_batched = one_hot @ self.mv             # [B, M]
        Lv_batched = (one_hot.unsqueeze(-1).unsqueeze(-1) * self.Lv).sum(dim=1)   # [B, M, M]

        # 2) Cholesky of Kuu (shared across batch)
        L_Kuu = torch.linalg.cholesky(Kuu + 1e-6 * torch.eye(M, device=Kuu.device))

        # 3) KL term for each batch item (whitened)
        Sv_batched = Lv_batched @ Lv_batched.transpose(-2, -1)  # [B, M, M]
        trace_term = torch.einsum("bii->b", Sv_batched)         # trace
        quadratic_term = (mv_batched**2).sum(dim=1)            # sum of squares

        # logdet
        # If you know each Lv is lower-triangular, you can do:
        diag_Lv = Lv_batched.diagonal(dim1=-2, dim2=-1)  # [B, M]
        log_det_Sv = 2.0 * torch.log(diag_Lv).sum(dim=1)  # [B]

        M_ = Sv_batched.shape[-1]
        kl_batched = 0.5 * (trace_term + quadratic_term - M_ - log_det_Sv)  # [B]

        # 4) mean of q(f)
        # We'll do a batched solve. We can expand L_Kuu so that it is [B, M, M] with broadcasting
        L_Kuu_expanded = L_Kuu.unsqueeze(0).expand(B, -1, -1)  # [B, M, M]
        mv_batched_ = mv_batched.unsqueeze(-1)                 # [B, M, 1]
        mu_u_batched = torch.bmm(L_Kuu_expanded, mv_batched_)  # [B, M, 1]

        # cholesky_solve for each batch
        L_Kuu_inv_mu_u_batched = torch.cholesky_solve(mu_u_batched, L_Kuu_expanded)  # [B, M, 1]
        mean_f_batched = torch.bmm(Kfu_batched, L_Kuu_inv_mu_u_batched).squeeze(-1)  # [B, N_max]

        # 5) Covariance of q(f)
        L_Kuu_inv_Kfu_T_batched = torch.cholesky_solve(Kfu_batched.transpose(-2, -1), L_Kuu_expanded)  # [B, M, N_max]
        Qff_batched = torch.bmm(Kfu_batched, L_Kuu_inv_Kfu_T_batched)  # [B, N_max, N_max]

        L_u_batched = torch.bmm(L_Kuu_expanded, Lv_batched)    # [B, M, M]
        temp_batched = torch.cholesky_solve(L_u_batched, L_Kuu_expanded)  # [B, M, M]
        Z_batched = torch.bmm(Kfu_batched, temp_batched)       # [B, N_max, M]
        Cov_f_batched = Kff_batched - Qff_batched + torch.bmm(Z_batched, Z_batched.transpose(-2, -1))
        
        # 6) Build mask
        B, N_max, _ = Kff_batched.shape
        indices = torch.arange(N_max, device=N_batched.device).view(1, -1).expand(B, N_max)  # [B, N_max]
        # each row is 0..N_max-1, shape [B, N_max]
        mask = (indices < N_batched.unsqueeze(-1))  # [B, N_max]
        mask_2d = mask.unsqueeze(-1) & mask.unsqueeze(1)      # shape [B, N_max, N_max], broadcasting

        Cov_f_batched = Cov_f_batched * mask_2d.double()
        Cov_f_batched = Cov_f_batched + torch.eye(N_max, device=Cov_f_batched.device).unsqueeze(0) * 1e-6

        # 7) Sample f ~ q(f)
        # We'll do a batched Cholesky now:
        L_Cov_f_batched = torch.linalg.cholesky(Cov_f_batched)  # [B, N_max, N_max]
        eps = torch.randn_like(mean_f_batched)                  # [B, N_max]
        # we need to do a batched matmul of L_Cov_f_batched with eps
        eps_3d = eps.unsqueeze(-1)                              # [B, N_max, 1]
        f_samps_3d = torch.bmm(L_Cov_f_batched, eps_3d).squeeze(-1)  # [B, N_max]
        f_samples = mean_f_batched + f_samps_3d                 # [B, N_max]
        f_samples = f_samples * mask.double()                   # zero out invalid indices

        return f_samples, kl_batched


    def ELBO(self, deaths, days, f, pop, EpidemicStart, N, stateindex):
        X = days
        # 2. Compute kernel matrices
        Kff, Kfu, Kuu, Kuuold, Kuolduold = self.KuuKfuKff_se(X)
        f_samples, kl = self.pre_ELBO(Kff, Kfu, Kuu, N, stateindex)
        # f_samples = torch.clamp(f_samples, -5.0, 5.0)

        # 9. Evaluate variational expectations
        variational_expectations = self.likelihood(deaths, f_samples.squeeze(-1), f, pop, EpidemicStart, N, stateindex)
        ve_sum  = torch.sum(variational_expectations, dim=-1)
        ve_mean = torch.mean(ve_sum)

        # ve = self.likelihood.variational_expectations(mean_f, Cov_f.diag(), y)
        # ve_sum  = torch.sum(ve, dim=-1)
        # ve_mean = torch.mean(ve_sum)

        # 10. ELBO
        elbo = ve_mean - kl
        # elbo = ve_mean 

        # 11. Optional correction
        if self.Z_old is not None:
            corr = self.correction_term(X, N, stateindex, Kuu, Kuuold, Kuolduold)
            elbo = elbo + corr

        return elbo.mean(), ve_mean.mean(), kl.mean()
    
    def correction_term(self, X, N, stateindex, Kuu=None, Kuu_old=None, Ku_old_old=None):
        # If no kernels are provided, compute them.
        if Kuu is None or Kuu_old is None or Ku_old_old is None:
            # Expecting self.KuuKfuKff_se(X) to return 5 outputs;
            # here Kuu, Kuu_old, and Ku_old_old become batched tensors over all regions,
            # with shape [R_total, M, M]
            _, _, Kuu, Kuu_old, Ku_old_old = self.KuuKfuKff_se(X)

        # --- Select the kernel for the old variational parameters ---
        # Convert 1-based stateindex to 0-based.
        stateindex_correct = (stateindex - 1).to(torch.long)

        # If we don't have stored old variational parameters, return zero correction.
        if self.mv_old is None or self.Lv_old is None:
            return torch.tensor(0.0, device=self.device)

        # --- Select the old variational parameters using stateindex ---
        # Assume self.mv_old: [R_total, M] and self.Lv_old: [R_total, M, M]
        mv_old = self.mv_old[stateindex_correct].unsqueeze(-1)  # -> [R, M, 1]
        Lv_old = self.Lv_old[stateindex_correct]                  # -> [R, M, M]

        # Compute the batched Cholesky of the selected old kernel matrix.
        Luu_old = torch.linalg.cholesky(self.add_jitter(Ku_old_old))  # -> [R, M, M]

        # Compute the old mean in the inducing space: m_old = Luu_old @ mv_old.
        m_old = Luu_old @ mv_old  # [R, M, 1]

        # Compute the old covariance:
        # S_old = Luu_old @ Lv_old @ Lv_old^T @ Luu_old^T, computed in two steps.
        temp = torch.bmm(Lv_old, Lv_old.transpose(-2, -1))  # [R, M, M]
        S_old = Luu_old @ temp                                # [R, M, M]
        S_old = S_old @ Luu_old.transpose(-2, -1)             # [R, M, M]

        # --- Obtain predictions from the model ---
        # Assume self.Z_old now has shape [R, M, D] for the same R selected by stateindex.
        # expand Z_old to [R, M, D] 
        Z_old = self.Z_old[None].expand(stateindex.shape[0], -1, -1)  # [R, M, D]
        mu, Sigma = self.pred_f(Z_old, N, stateindex, full_cov=True, flag_noise=False)
        # Second moment: Smm = Sigma + mu @ mu^T in batched fashion.
        mu_muT = torch.bmm(mu, mu.transpose(-2, -1))  # [R, M, M]
        Smm = Sigma + mu_muT                          # [R, M, M]

        # --- Compute batched Cholesky factors for S_old and Ku_old_old ---
        M_size = S_old.shape[-1]
        # For numerical stability, add a tiny jitter to S_old.
        LSa = torch.linalg.cholesky(S_old + 1e-6 * torch.eye(M_size, device=S_old.device))  # [R, M, M]
        LKa = torch.linalg.cholesky(self.add_jitter(Ku_old_old))  # [R, M, M]

        # --- Compute log-determinants for each region ---
        logdet_LKa = torch.log(torch.diagonal(LKa, dim1=-2, dim2=-1)).sum(-1)  # shape: [R]
        logdet_LSa = torch.log(torch.diagonal(LSa, dim1=-2, dim2=-1)).sum(-1)  # shape: [R]
        obj = (logdet_LKa - logdet_LSa).sum()  # sum over regions → scalar

        # --- Compute quadratic and cross terms ---
        Sainv_ma = torch.cholesky_solve(m_old, LSa)  # [R, M, 1]
        Sainv_ma = Sainv_ma.squeeze(-1)               # now [R, M]

        # Subtract half the quadratic term: 0.5 * sum(m_old^T Sainv_ma) over all regions.
        obj = obj - 0.5 * (m_old.squeeze(-1) * Sainv_ma).sum()
        # Add the cross term: sum(mu^T Sainv_ma)
        obj = obj + (mu.squeeze(-1) * Sainv_ma).sum()

        # --- Solve for the inverses of Smm with respect to LSa and LKa ---
        Sainv_Smm = torch.cholesky_solve(Smm, LSa)  # [R, M, M]
        Kainv_Smm = torch.cholesky_solve(Smm, LKa)    # [R, M, M]

        # Compute the trace differences per region.
        trace_Sainv_Smm = torch.diagonal(Sainv_Smm, dim1=-2, dim2=-1).sum(-1)  # [R]
        trace_Kainv_Smm = torch.diagonal(Kainv_Smm, dim1=-2, dim2=-1).sum(-1)  # [R]
        obj -= 0.5 * (trace_Sainv_Smm - trace_Kainv_Smm).sum()

        return obj



    def pred_f(self, Xnew, N, state_index, full_cov=False, flag_noise=True):
        """
        Compute the predictive mean and variance of the latent function at new points Xnew 
        for each region in a batched manner.

        Assumptions:
        - Xnew: [R, N_max, D] where R is number of regions and N_max is the maximum number of test points.
        - N: [R] vector of valid test points for each region (each entry is an int).
        - state_index: [R] vector of indices into the model's variational parameters;
                        these indicate which row in self.mv ([R_total, M]) and self.Lv ([R_total, M, M])
                        to use for each region.
        - self.Kuu_se(): returns Kuu (shared kernel matrix for inducing points) of shape [M, M].
        - self.KfuKff_se(Xnew): when Xnew is batched with shape [R, N_max, D] should return a tuple
                        where the first element is Kff of shape [R, N_max, N_max] and the second is
                        Kfu of shape [R, N_max, M].
        - flag_noise: if True you might add additional noise (not shown here—but one could add it later).

        Returns:
        - mean: Predictive mean of shape [R, N_max, 1] (with invalid entries masked to 0).
        - var:  Predictive variance, either [R, N_max, 1] for full_cov=False or [R, N_max, N_max] if full_cov=True.
        """
        R, N_max, D = Xnew.shape

        # Ensure N and state_index are integer tensors on the correct device.
        N = N.to(dtype=torch.long)

        # 1. Compute kernel matrices that are shared or batched over regions.
        # Kuu is shared.
        Kuu, _, _ = self.Kuu_se()  # [M, M]

        # Compute kernel matrices for each region.
        # Here we assume self.KfuKff_se is batched over the first dimension.
        Kff_new, Kfu_new = self.KfuKff_se(Xnew)
        # Kff_new: [R, N_max, N_max],  Kfu_new: [R, N_max, M]

        # 2. Compute the Cholesky decomposition of Kuu (shared among regions).
        Luu = torch.linalg.cholesky(self.add_jitter(Kuu))  # [M, M]
        # Expand Luu to have batch dimension R for later batched solves.
        Luu_expanded = Luu.unsqueeze(0).expand(R, -1, -1)  # [R, M, M]

        # 3. Select variational parameters for each region.
        #   self.mv: [R_total, M], self.Lv: [R_total, M, M].
        # Build one-hot selection to pick out the correct parameters.
        R_total = self.mv.shape[0]
        device = self.mv.device
        one_hot = torch.nn.functional.one_hot(state_index, num_classes=R_total).double().to(device)  # [R, R_total]
        mv_pred = one_hot @ self.mv      # [R, M]
        Lv_pred = (one_hot.unsqueeze(-1).unsqueeze(-1) * self.Lv).sum(dim=1)  # [R, M, M]

        # 4. Unwhiten the variational parameters for each region.
        # Posterior mean for u:
        # mv_pred is [R, M] --> unsqueeze to [R, M, 1] then multiply.
        mu_u = torch.bmm(Luu_expanded, mv_pred.unsqueeze(-1))  # [R, M, 1]
        # And compute L_u = Luu @ Lv_pred for each region.
        L_u = torch.bmm(Luu_expanded, Lv_pred)  # [R, M, M]

        # 5. Compute predictive mean.
        # Solve for alpha = Kuu^{-1} * mu_u in batched fashion.
        alpha = torch.cholesky_solve(mu_u, Luu_expanded)  # [R, M, 1]
        # Now, mean = Kfu_new @ alpha, where Kfu_new is [R, N_max, M].
        mean = torch.bmm(Kfu_new, alpha)  # [R, N_max, 1]

        # 6. Compute predictive variance.
        # First, compute the "prior conditional" variance term.
        # Solve for Kuu^{-1} Kfu_new^T in batch:
        # Kfu_new.transpose(1,2): [R, M, N_max]
        Kuu_inv_Kfu = torch.cholesky_solve(Kfu_new.transpose(1,2), Luu_expanded)  # [R, M, N_max]
        var = Kff_new - torch.bmm(Kfu_new, Kuu_inv_Kfu)  # [R, N_max, N_max]

        if full_cov:
            # Compute extra term:
            temp = torch.cholesky_solve(L_u, Luu_expanded)  # [R, M, M]
            Z = torch.bmm(Kfu_new, temp)  # [R, N_max, M]
            var = var + torch.bmm(Z, Z.transpose(1, 2))  # [R, N_max, N_max]
        else:
            temp = torch.cholesky_solve(L_u, Luu_expanded)  # [R, M, M]
            Z = torch.bmm(Kfu_new, temp)  # [R, N_max, M]
            var_diag_extra = torch.sum(Z**2, dim=2)  # [R, N_max]
            # Diagonal part of the prior conditional
            # (note: torch.diagonal with dim arguments returns shape [R, N_max])
            Kff_diag = Kff_new.diagonal(dim1=1, dim2=2)  # [R, N_max]
            # Compute the diagonal of the term Kfu_new @ Kuu_inv_Kfu:
            second_term = torch.sum(Kfu_new * Kuu_inv_Kfu.transpose(1,2), dim=2)  # [R, N_max]
            var_diag = Kff_diag - second_term + var_diag_extra  # [R, N_max]
            var = var_diag.unsqueeze(-1)  # [R, N_max, 1]

        # 7. Mask the outputs so that only the first N[r] values per region are valid.
        # Build a mask: for each region r, mask[r] is [N_max] with True for indices < N[r].
        device = N.device
        arange = torch.arange(N_max, device=device).unsqueeze(0).expand(R, N_max)  # [R, N_max]
        mask = (arange < N.unsqueeze(1))  # [R, N_max], boolean

        # For the mean and (if not full_cov) the variance, we multiply the valid entries.
        mean = mean * mask.unsqueeze(-1).to(mean.dtype)
        if full_cov:
            # For full covariance, mask both rows and columns.
            mask2d = mask.unsqueeze(1) * mask.unsqueeze(2)  # [R, N_max, N_max]
            var = var * mask2d.to(var.dtype)
        else:
            var = var * mask.unsqueeze(-1).to(var.dtype)

        return mean, var

from hipposvgp.kernel import SE_kernel, SMK_kernel, FreqSpaceKernel
from hipposvgp.hippo import HiPPO_LegS
from hipposvgp.utils import PositiveDefiniteParameter
import torch
import torch.nn as nn
import numpy.random as npr
import math
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
        use_analytic_Z=True, 
        **kwargs 
    ):
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.flag_fast = flag_fast
        self.use_analytic_Z = use_analytic_Z 

        # Kernel selection
        self.kernel = kernel

        # Likelihood
        self.likelihood = likelihood

        # Old posterior parameters
        self.num_inducing_old = (num_inducing_old
            if num_inducing_old is not None
            else None
        )

        # [Fix 2] Fix UserWarning: Use clone().detach()
        self.mv_old = (
            mv_old.clone().detach().to(dtype=dtype, device=device)
            if mv_old is not None
            else None
        )
        self.Lv_old = (
            Lv_old.clone().detach().to(dtype=dtype, device=device)
            if Lv_old is not None
            else None
        )
        self.Kaa_old = (
            Kaa_old.clone().detach().to(dtype=dtype, device=device)
            if Kaa_old is not None
            else None
        )
        self.Z_old = (
            Z_old.clone().detach().to(dtype=dtype, device=device)
            if Z_old is not None
            else None
        )

        # initialize the variational parameters
        self.inducing_size = inducing_size
        # Initialize mv and Lv on CPU first
        mv = torch.zeros(self.inducing_size).double()
        Lv = torch.eye(self.inducing_size).double()

        # Wrap them with nn.Parameter
        self.mv = nn.Parameter(mv, requires_grad=True)
        self.Lv = nn.Parameter(Lv, requires_grad=True)

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
        # add Jitter  
        # return Kuu + self.jitter * torch.eye(Kuu.shape[0], device=self.device)
        return Kuu + self.jitter * torch.mean(torch.diag(Kuu)).detach() * torch.eye(
            Kuu.shape[0], device=self.device
        )

    # [Fix 4] add Robust Cholesky to prevent LinAlgError
    def robust_cholesky(self, matrix, base_jitter=1e-6, retries=10):
        jitter = base_jitter
    
        for i in range(retries):
            try:
                factor = 10 ** i if i > 0 else 1.0
                current_jitter = jitter * factor
                
                mat_reg = matrix + current_jitter * torch.eye(matrix.shape[0], device=self.device, dtype=self.dtype)
                L = torch.linalg.cholesky(mat_reg)
                return L
            except torch.linalg.LinAlgError:
                continue
        raise torch.linalg.LinAlgError(f"Robust Cholesky failed after {retries} retries.")

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

    @staticmethod
    def spherical_bessel_j(lmax, x):
        """Compute spherical Bessel functions j_0...j_lmax for scalar or tensor x."""
        import scipy.special
        
        # --- Scipy wrapper with custom backward pass for gradients ---
        class BesselJFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, nu, inp):
                ctx.save_for_backward(nu, inp)
                
                # Move to CPU/NumPy for Scipy execution
                nu_np = nu.detach().cpu().numpy()
                inp_np = inp.detach().cpu().numpy()
                
                res = scipy.special.jv(nu_np, inp_np)
                
                # Convert back to Tensor on original device
                return torch.from_numpy(res).to(device=inp.device, dtype=inp.dtype)

            @staticmethod
            def backward(ctx, grad_output):
                nu, inp = ctx.saved_tensors
                
                nu_np = nu.detach().cpu().numpy()
                inp_np = inp.detach().cpu().numpy()
                
                # Derivative formula: dJ_v/dx = 0.5 * (J_{v-1} - J_{v+1})
                j_minus = scipy.special.jv(nu_np - 1, inp_np)
                j_plus = scipy.special.jv(nu_np + 1, inp_np)
                grad_inp_np = 0.5 * (j_minus - j_plus)
                
                grad_inp = torch.from_numpy(grad_inp_np).to(device=inp.device, dtype=inp.dtype)
                
                # Return gradients (None for 'nu', calculated grad for 'inp')
                return None, grad_inp * grad_output

        # ----------------------------------------
        
        # 1. Handle x=0 to avoid division by zero
        mask_nonzero = (x.abs() > 1e-8)
        x_safe = torch.where(mask_nonzero, x, torch.ones_like(x)) 

        # 2. Prepare Order nu = n + 0.5
        L = torch.arange(lmax + 1, device=x.device, dtype=x.dtype).view(-1, 1, 1)
        nu = L + 0.5

        # 3. Calculate prefactor sqrt(pi / 2|x|)
        x_abs = x_safe.abs()
        prefactor = torch.sqrt(math.pi / (2 * x_abs))
        
        # 4. Compute J_nu(|x|) using the custom Scipy wrapper
        bessel_val = BesselJFunction.apply(nu, x_abs)
        
        # Combine terms; for negative x use parity j_l(-x)=(-1)^l j_l(x)
        parity = torch.where((L % 2) == 0, 1.0, -1.0)
        x_nonneg = (x_safe >= 0).unsqueeze(0)
        sign_factor = torch.where(x_nonneg, torch.ones_like(x_abs).unsqueeze(0), parity)
        j = prefactor * bessel_val * sign_factor

        # 5. Correct boundary condition at x=0: j_0(0)=1, j_n(0)=0
        j_at_0 = torch.zeros((lmax+1,) + x.shape, device=x.device, dtype=x.dtype)
        j_at_0[0] = 1.0
        
        j = torch.where(mask_nonzero, j, j_at_0)
        
        return j
    
    # @staticmethod
    # def spherical_bessel_j(lmax, x):
    #     """
    #     Compute spherical Bessel functions j_0...j_lmax for scalar or tensor x on GPU.
    #     Uses Miller's Algorithm (Backward Recurrence) for numerical stability.
    #     """
        
    #     class SphericalBesselJFunction(torch.autograd.Function):
    #         @staticmethod
    #         def forward(ctx, lmax, x):
    #             # 1. Handle x=0 singularity
    #             # If x < epsilon, we treat it as 0.
    #             # j_0(0) = 1, j_n(0) = 0 for n > 0
    #             epsilon = 1e-16
    #             mask_nonzero = x.abs() > epsilon
    #             x_safe = torch.where(mask_nonzero, x, torch.full_like(x, epsilon))
                
    #             # 2. Setup Backward Recurrence (Miller's Method)
    #             # We need to compute j_0 ... j_lmax.
    #             # We start recurrence from a sufficiently large N_start > lmax.
    #             # Heuristic: N_start ~ lmax + C + x (if x is large).
    #             # Here we use a safe buffer. For double precision, +50 to +100 is usually safe.
    #             # We take max(x) to ensure stability for the whole batch.
                
    #             # Note: We compute gradients manually, so we don't need to save graph for this loop.
    #             with torch.no_grad():
    #                 max_x = x_safe.abs().max().item()
    #                 N_start = int(lmax + max_x + 60) # Heuristic starting point
                    
    #                 # Allocate buffer for recurrence: [N_start + 2, *x.shape]
    #                 # We compute "unscaled" values F_n
    #                 # F_{N_start+1} = 0
    #                 # F_{N_start} = 1e-30 (small number to prevent overflow, arbitrary scale)
                    
    #                 scale_val = 1e-30
    #                 f_next = torch.zeros_like(x_safe)
    #                 f_curr = torch.full_like(x_safe, scale_val)
                    
    #                 # Storage for the requested range 0...lmax
    #                 # We only need to store F_0 ... F_lmax.
    #                 # But we generate them backwards.
                    
    #                 # To normalize, we need F_0. We will store it in the loop.
    #                 # We'll use a dictionary or list to temporarily hold needed indices if memory is tight,
    #                 # but for lmax=150, storing a [151, batch] tensor is fine.
                    
    #                 # Let's create a tensor to hold j_0 to j_lmax
    #                 res_F = torch.zeros((lmax + 1,) + x.shape, device=x.device, dtype=x.dtype)
                    
    #                 # Recurrence: F_{n-1} = ((2n+1)/x) * F_n - F_{n+1}
    #                 for n in range(N_start, -1, -1):
    #                     # Standard Backward Recurrence relation
    #                     # f_prev = (2*n + 3) / x * f_curr - f_next
    #                     # Wait, definition: j_{n+1} = (2n+1)/x j_n - j_{n-1}
    #                     # => j_{n-1} = (2n+1)/x j_n - j_{n+1}
                        
    #                     f_prev = (2 * n + 3) / x_safe * f_curr - f_next
                        
    #                     # Store if within range
    #                     if n <= lmax:
    #                         res_F[n] = f_prev
                        
    #                     # Shift
    #                     f_next = f_curr
    #                     f_curr = f_prev
                    
    #                 # Now f_prev is F_{-1} (which we don't strictly need)
    #                 # The loop ended at n=0. Inside loop, at n=0, we computed f_prev for n-1=-1?
    #                 # No, let's trace carefully.
    #                 # We want j_0...j_lmax.
    #                 # Inside loop, we calculate term for index n using (n+1) and (n+2).
    #                 # Actually, let's use the index `n` as the one we are computing.
    #                 # Formula: j_n = (2n+3)/x * j_{n+1} - j_{n+2}
                    
    #                 # Let's restart logic slightly for clarity:
    #                 # We start with F_{N+1}=0, F_{N}=tiny.
    #                 # We iterate downwards to calculate F_{N-1}, ..., F_0.
                    
    #                 f_plus2 = torch.zeros_like(x_safe)       # F_{N+1}
    #                 f_plus1 = torch.full_like(x_safe, 1e-30) # F_N
                    
    #                 for n in range(N_start - 1, -1, -1):
    #                     val = (2 * n + 3) / x_safe * f_plus1 - f_plus2
    #                     if n <= lmax:
    #                         res_F[n] = val
    #                     f_plus2 = f_plus1
    #                     f_plus1 = val
                        
    #                 # Now res_F holds unscaled [F_0, ..., F_lmax]
    #                 # f_plus1 holds F_0. (Loop ended at n=0, calculating F_0).
                    
    #                 # 3. Normalization
    #                 # We know j_0(x) = sin(x)/x
    #                 # So scale = j_0(x) / F_0
    #                 true_j0 = torch.sin(x_safe) / x_safe
    #                 normalization = true_j0 / f_plus1
                    
    #                 # Apply normalization
    #                 # res_F: [lmax+1, batch]
    #                 # normalization: [batch]
    #                 j_vals = res_F * normalization.unsqueeze(0)
                
    #             # 4. Handle boundary x=0
    #             # j_0(0)=1, others=0
    #             j_zero = torch.zeros_like(j_vals)
    #             j_zero[0] = 1.0
                
    #             result = torch.where(mask_nonzero.unsqueeze(0), j_vals, j_zero)
                
    #             ctx.save_for_backward(result, x_safe, mask_nonzero)
    #             ctx.lmax = lmax
                
    #             return result

    #         @staticmethod
    #         def backward(ctx, grad_output):
    #             # grad_output shape: [lmax+1, batch]
    #             j_vals, x_safe, mask_nonzero = ctx.saved_tensors
    #             lmax = ctx.lmax
                
    #             # Derivative of spherical Bessel function:
    #             # d/dx j_n(x) = j_{n-1}(x) - (n+1)/x * j_n(x)
    #             # For n=0: d/dx j_0(x) = -j_1(x)
                
    #             # We need j_{n-1}. For n=0, it's defined via relation or special case.
    #             # Actually simpler relation: j_0' = -j_1
                
    #             grad_x = torch.zeros_like(x_safe)
                
    #             # Loop over n to accumulate gradients
    #             # grad_x = sum_n ( dL/d_jn * d_jn/dx )
                
    #             # We have j_vals for 0...lmax.
                
    #             # Handle n=0:
    #             # term = grad_output[0] * (-j_1)
    #             if lmax >= 1:
    #                 grad_x += grad_output[0] * (-j_vals[1])
                
    #             # Handle n=1...lmax
    #             # d/dx j_n = j_{n-1} - (n+1)/x * j_n
                
    #             # Vectorized preparation
    #             # L_range: [1, 2, ..., lmax]
    #             L_range = torch.arange(1, lmax + 1, device=x_safe.device, dtype=x_safe.dtype).view(-1, 1)
                
    #             # j_n for n=1..lmax
    #             j_n = j_vals[1:]
    #             # j_{n-1} for n=1..lmax -> j_vals[0:-1]
    #             j_prev = j_vals[:-1]
    #             # grad_output for n=1..lmax
    #             grad_n = grad_output[1:]
                
    #             # Derivative term
    #             derivs = j_prev - (L_range + 1) / x_safe.unsqueeze(0) * j_n
                
    #             # Accumulate
    #             grad_x += torch.sum(grad_n * derivs, dim=0)
                
    #             # Handle x=0 gradients (usually 0 due to parity, except odd/even terms?)
    #             # For simple even/odd functions like j_n, derivative at 0 is 0 for even n?
    #             # j_0(x) ~ 1 - x^2/6 -> deriv is 0
    #             # j_1(x) ~ x/3 -> deriv is 1/3
    #             # For safety, mask gradient at 0 to 0 (or handle carefully).
    #             # Given usage in RFF, x is r*w.
    #             grad_x = torch.where(mask_nonzero, grad_x, torch.zeros_like(grad_x))
                
    #             return None, grad_x

    #     return SphericalBesselJFunction.apply(lmax, x)
    

    def _get_rff_scale(self, num_rff):
        """RFF特徴量のスケーリング係数を取得

        FreqSpaceKernelの場合は学習済み重みのsqrt、
        それ以外は均等重み (1/num_rff)^0.5 を返す。

        Returns:
            scale: [num_rff] or scalar
        """
        if hasattr(self.kernel, 'get_weights'):
            gamma = self.kernel.get_weights()  # [num_components]
            return torch.sqrt(gamma)  # [num_components]
        else:
            return (1 / num_rff) ** 0.5

    def _compute_Z_at_t(self, w_eff, t_index, phase_start):
        # 1. Scaling Factor (重み付き or 均等)
        num_rff = w_eff.shape[-1]
        scale = self._get_rff_scale(num_rff)  # [R] or scalar

        kappa = 0.5 * w_eff * t_index  # [1, num_rff]
        kappa = kappa
        L = torch.arange(self.inducing_size, dtype=self.dtype, device=self.device)

        # 2. Compute Bessel
        j = self.spherical_bessel_j(self.inducing_size-1, kappa)
        if len(j.shape) == 3: j = j.squeeze(1) # [M, R]

        # 3. Prefactor sqrt(2l+1)
        prefactor = torch.sqrt(2 * L + 1)[:, None]

        # 4. Phase shift terms
        phase = kappa + L[:, None] * math.pi / 2
        cos_term = torch.cos(phase)
        sin_term = torch.sin(phase)

        # 5. Combine
        Z_cos = prefactor * j * cos_term
        Z_sin = prefactor * j * sin_term


        # 6. Stack and Scale (scaleが[R]の場合はbroadcast)
        # Z_sin, Z_cos: [M, R] -> cat -> [M, 2R]
        # scale: [R] -> [2R] にするため repeat
        if isinstance(scale, torch.Tensor) and scale.dim() > 0:
            scale_full = torch.cat([scale, scale])  # [2R]
            # [M, 2R].T -> [2R, M], scale_full[:, None] -> [2R, 1] for broadcasting
            Z = scale_full[:, None] * torch.cat([Z_sin, Z_cos], dim=1).T  # [2R, M]
        else:
            Z = scale * torch.cat([Z_sin, Z_cos], dim=1).T

        return Z

    def get_Z_analytic(self, w):
        time_stamps = self.get_time_stamps()
        
        step_size = self.maxT / self.num_discrete_steps
        w_eff = w * step_size
        start_t = torch.as_tensor(
            self.start + 0.5 * step_size, dtype=self.dtype, device=self.device
        )
        phase_start = w * start_t  # [1, R]
        t_index = torch.as_tensor(
            self.prev_discrete_steps + self.num_discrete_steps,
            dtype=self.dtype,
            device=self.device,
        )
        
        Z = self._compute_Z_at_t(w_eff, t_index, phase_start)
        
        Z_old_newk = None
        if self.prev_discrete_steps > 0:
            idx_old = self.prev_discrete_steps - 1
            if 0 <= idx_old < len(time_stamps):
                t_index_old = torch.as_tensor(
                    self.prev_discrete_steps, dtype=self.dtype, device=self.device
                )
                Z_old_newk = self._compute_Z_at_t(w_eff, t_index_old, phase_start)
        
        return Z, Z_old_newk, time_stamps

    def Kuu_se(self, w):
        Z, Z_old_newk, _ = self.get_Z_analytic(w)
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

    def _compute_Z_f(self, w, X):
        """データ点のRFF特徴量を計算 (重み付き対応)

        Args:
            w: [1, R] or [R] 周波数
            X: [N, D] データ点

        Returns:
            Z_f: [N, 2R] RFF特徴量
        """
        num_rff = w.shape[-1]
        scale = self._get_rff_scale(num_rff)  # [R] or scalar

        sin_term = torch.sin(w * X)  # [N, R]
        cos_term = torch.cos(w * X)  # [N, R]

        if isinstance(scale, torch.Tensor) and scale.dim() > 0:
            # 重み付き: scale is [R]
            Z_f = torch.cat([scale * sin_term, scale * cos_term], 1)  # [N, 2R]
        else:
            # 均等重み
            Z_f = scale * torch.cat([sin_term, cos_term], 1)  # [N, 2R]

        return Z_f

    def KfuKff_rff_se(self, w, X):
        Z, _, _ = self.get_Z_analytic(w)
        Z_f = self._compute_Z_f(w, X)  # [N, 2R]

        Kfu = Z_f @ Z
        Kff = Z_f @ Z_f.T
        return Kff, Kfu

    def KuuKfuKff_rff_se(self, w, X):
        Z, Z_old_newk, _ = self.get_Z_analytic(w)
        Z_f = self._compute_Z_f(w, X)  # [N, 2R] 
        Kfu = Z_f @ Z 
        Kff = Z_f @ Z_f.T 
        Kuu = Z.T @ Z 
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
 
         # [Fix 5] Use robust_cholesky
         Luu = self.robust_cholesky(Kuu)

         mu_u = Luu @ self.mv            
         mu_u = mu_u.unsqueeze(1)        
 
         L_u  = Luu @ self.Lv            
 
         alpha = torch.cholesky_solve(mu_u, Luu) 
         mean  = Kfu_new @ alpha                 
 
         Kuu_inv_Kfu = torch.cholesky_solve(Kfu_new.T, Luu)
         var = Kff_new - Kfu_new @ Kuu_inv_Kfu  
 
         if full_cov:
             temp = torch.cholesky_solve(L_u, Luu) 
             Z    = Kfu_new @ temp                 
             var += Z @ Z.T                        
         else:
             temp = torch.cholesky_solve(L_u, Luu) 
             var_diag_extra = torch.sum((Kfu_new @ temp)**2, dim=1)
             var = torch.diag(Kff_new) - torch.sum(Kfu_new * Kuu_inv_Kfu.T, dim=1)
             var += var_diag_extra
             var = var.unsqueeze(1) 
         return mean, var

    def correction_term(self, Kuu, Kuu_old, Ku_old_old, Ku_old_old_newk):
        if self.mv_old is None or self.Lv_old is None:
            return torch.tensor(0.0, device=self.device)
        mv_old = self.mv_old.unsqueeze(-1)
        Lv_old = self.Lv_old
        
        # [Fix 5] Use robust_cholesky
        Luu_old = self.robust_cholesky(Ku_old_old)
        
        m_old = Luu_old @ mv_old
        S_old = Luu_old @ Lv_old @ Lv_old.T @ Luu_old.T
        
        mu, Sigma = self.quold_new(Kuu, Kuu_old, Ku_old_old_newk, full_cov=True)
        Smm = Sigma + mu @ mu.T
        
        # [Fix 5] Use robust_cholesky
        LSa = self.robust_cholesky(S_old)
        LKa = self.robust_cholesky(Ku_old_old)
        
        obj = torch.sum(torch.log(torch.diagonal(LKa))) - torch.sum(torch.log(torch.diagonal(LSa)))
        Sainv_ma = torch.cholesky_solve(m_old, LSa).squeeze(-1)
        obj -= 0.5 * torch.sum(m_old.squeeze(-1) * Sainv_ma)
        obj += torch.sum(mu.squeeze(-1) * Sainv_ma)
        
        Sainv_Smm = torch.cholesky_solve(Smm, LSa)
        Kainv_Smm = torch.cholesky_solve(Smm, LKa)
        obj -= 0.5 * (torch.trace(Sainv_Smm) - torch.trace(Kainv_Smm))
        
        return obj

    def ELBO(self, X, y, w, recompute_k=True, cache_k=False):
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

        if not recompute_k:
            Kff = self.Kff
            Kfu = self.Kfu
            Kuu = self.Kuu
            if self.Kuuold != None:
                Kuuold = self.Kuuold
            if self.Kuolduold != None:
                Kuolduold = self.Kuolduold
            if self.kuolduold_newk != None:
                kuolduold_newk = self.kuolduold_newk

        # [Fix 5] Use robust_cholesky
        L_Kuu = self.robust_cholesky(Kuu)

        mv = self.mv 
        Lv = self.Lv 

        Sv = Lv @ Lv.T
        trace_term = torch.trace(Sv)
        quadratic_term = torch.sum(mv**2) 
        if torch.allclose(Lv, torch.tril(Lv)):
            log_det_Sv = 2.0 * torch.sum(torch.log(torch.diagonal(Lv)))
        else:
            log_det_Sv = torch.logdet(Sv)
        M = Kuu.shape[0]
        kl = 0.5*(trace_term + quadratic_term - M - log_det_Sv)

        mu_u = L_Kuu @ mv.unsqueeze(-1) 
        L_Kuu_inv_mu_u = torch.cholesky_solve(mu_u, L_Kuu) 
        mean_f = Kfu @ L_Kuu_inv_mu_u 

        L_Kuu_inv_Kfu_T = torch.cholesky_solve(Kfu.T, L_Kuu)
        Qff = Kfu @ L_Kuu_inv_Kfu_T
        L_u = L_Kuu @ Lv
        temp = torch.cholesky_solve(L_u, L_Kuu)
        Z = Kfu @ temp
        Cov_f = Kff - Qff + Z @ Z.T
        
        # add jitter
        Cov_f = self.add_jitter(Cov_f)

        L_Cov_f = torch.linalg.cholesky(Cov_f)
        eps     = torch.randn_like(mean_f)
        f_samples = mean_f + L_Cov_f @ eps

        variational_expectations = self.likelihood.log_prob(f_samples, y)
        ve_sum  = torch.sum(variational_expectations, dim=-1)
        ve_mean = torch.sum(ve_sum)

        elbo = ve_mean - kl

        if self.Z_old is not None:
            corr = self.correction_term(Kuu, Kuuold, Kuolduold, kuolduold_newk)
            elbo = elbo + corr

        return elbo, ve_mean, kl

    def pred_f(self, Xnew, w, full_cov=False, flag_noise=True):
        if w == None:
            w = self.kernel.sample_from_spectral(self.rff_sample_size).unsqueeze(0)
        _, _, Kuu, _, _ = self.Kuu_se(w)
        Kuu *= torch.exp(self.kernel.log_sf)
        Kff_new, Kfu_new = self.KfuKff_rff_se(w, Xnew)[:2]
        Kff_new *= torch.exp(self.kernel.log_sf)
        Kfu_new *= torch.exp(self.kernel.log_sf)
        
        # [Fix 5] Use robust_cholesky
        Luu = self.robust_cholesky(Kuu)
        
        mu_u = Luu @ self.mv 
        mu_u = mu_u.unsqueeze(1) 

        L_u  = Luu @ self.Lv 

        alpha = torch.cholesky_solve(mu_u, Luu) 
        mean  = Kfu_new @ alpha 

        Kuu_inv_Kfu = torch.cholesky_solve(Kfu_new.T, Luu)
        var = Kff_new - Kfu_new @ Kuu_inv_Kfu 

        if full_cov:
            temp = torch.cholesky_solve(L_u, Luu) 
            Z    = Kfu_new @ temp 
            var += Z @ Z.T 
        else:
            temp = torch.cholesky_solve(L_u, Luu) 
            var_diag_extra = torch.sum((Kfu_new @ temp)**2, dim=1)
            var = torch.diag(Kff_new) - torch.sum(Kfu_new * Kuu_inv_Kfu.T, dim=1)
            var += var_diag_extra
            var = var.unsqueeze(1) 
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

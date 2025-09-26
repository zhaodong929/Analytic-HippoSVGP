from abc import abstractmethod
import torch
import torch.nn as nn
from functools import partial
import functorch
import numpy as np

def compute_median_lengthscale_init(gp_input):
    dist_squared = torch.cdist(gp_input, gp_input) ** 2
    dist_squared = torch.triu(dist_squared, diagonal=1)
    return torch.sqrt(0.5 * torch.median(dist_squared[dist_squared>0.0]))

def softplus_inv(x):
    return x + torch.log(-torch.expm1(-x))

def squared_euclidean_distance(x1, x2):
    x1_norm_squared = torch.sum(x1**2, axis=-1, keepdim=True)
    x2_norm_squared = torch.sum(x2**2, axis=-1, keepdim=True)
    x1_dot_x2 = torch.matmul(x1, x2.transpose(-2, -1))
    dist = x1_norm_squared + x2_norm_squared.transpose(-2, -1) - 2 * x1_dot_x2
    return dist

def scaled_squared_euclidean_distance(x1, x2, scale):
    x1_scaled = x1 / scale
    x2_scaled = x2 / scale
    return squared_euclidean_distance(x1_scaled, x2_scaled)

def compute_Sigma(K_uolduold, K_uoldu, K_uuold, Kuu, Luu, Snew):
    """
    Computes 
      Sigma = K_uolduold 
              - K_uoldu @ K_uu^-1 (K_uu - L_var L_var^T) K_uu^-1 @ K_uuold
    
    using Cholesky-based linear solves instead of explicitly inverting K_uu.
    
    Arguments:
    ----------
    K_uolduold : torch.Tensor (N x N)
        K_{uold, uold}
    K_uoldu     : torch.Tensor (N x M)
        K_{uold, u}
    K_uuold     : torch.Tensor (M x N)
        K_{u, uold}
    K_uu        : torch.Tensor (M x M)
        K_{u, u}
    L_var       : torch.Tensor (M x M)
        L_var in the expression (K_uu - L_var @ L_var^T)
        
    Returns:
    --------
    Sigma : torch.Tensor (N x N)
        Result of the matrix expression above.
    """
    
    # 2. Solve "tmp1 = K_uu^-1 @ K_uuold"
    #    Using cholesky_solve lets us do K_uu^-1 matmul some tensor
    #    without forming K_uu^-1 explicitly.
    tmp1 = torch.cholesky_solve(K_uuold, Luu, upper=False) 
    # shape of tmp1: (M x N)
    
    # 3. Compute "tmp2 = (K_uu - L_var @ L_var.T) @ tmp1"
    K_uu_minus_Lvar = Kuu - Snew
    tmp2 = K_uu_minus_Lvar @ tmp1  # shape: (M x N)
    
    # 4. Solve again for "tmp3 = K_uu^-1 @ tmp2"
    tmp3 = torch.cholesky_solve(tmp2, Luu, upper=False) 
    # shape of tmp3: (M x N)
    
    # 5. Compute the final Sigma
    #    Sigma = K_uolduold - K_uoldu @ tmp3
    Sigma = K_uolduold - K_uoldu @ tmp3  # shape: (N x N)
    
    return Sigma

class Kernel(nn.Module):

    def __call__(self, x1, x2=None, diag=False):
        return self.K(x1, x2, diag)
    
    @abstractmethod
    def K(self, x1, x2=None, diag=False):
        raise NotImplementedError
    
    @abstractmethod
    def reinit_adapt_params(self, lengthscale_init, device):
        raise NotImplementedError
    

class Stationary(Kernel):

    def __init__(self, init_variance, init_lengthscale, fixed_variance, fixed_lengthscale):
        super().__init__()
         
        if not isinstance(init_variance, torch.Tensor):
            init_variance = torch.tensor(init_variance)
        self.init_raw_variance = softplus_inv(init_variance)
        if fixed_variance:
            self.raw_variance = self.init_raw_variance
        else:
            self.raw_variance = nn.Parameter(self.init_raw_variance)
        
        if not isinstance(init_lengthscale, torch.Tensor):
            init_lengthscale = torch.tensor(init_lengthscale)
        self.init_raw_lengthscale = softplus_inv(init_lengthscale)
        if fixed_lengthscale:
            self.raw_lengthscale = self.init_raw_lengthscale
        else:
            self.raw_lengthscale = nn.Parameter(self.init_raw_lengthscale) 

        self.fixed_variance = fixed_variance
        self.fixed_lengthscale = fixed_lengthscale     

    @property
    def variance(self):
        return torch.nn.functional.softplus(self.raw_variance)
    
    @property
    def lengthscale(self):
        return torch.nn.functional.softplus(self.raw_lengthscale)
    
    def K(self, x1, x2=None, diag=False):
        if diag:
            assert x2 is None or torch.equal(x1, x2)
            return self.variance * torch.ones([x1.shape[0]]).to(self.variance.device)
        else:
            if x2 is None:
                x2 = x1
            r2 = scaled_squared_euclidean_distance(x1, x2, self.lengthscale)
            return self.variance * self.K_r2(r2)
    
    def K_r2(self, r2):
        r = torch.sqrt(torch.maximum(r2, torch.tensor(1e-36)))
        return self.K_r(r)
    
    @abstractmethod
    def K_r(self, r):
        raise NotImplementedError
    
    def reinit_adapt_params(self, lengthscale_init, device):
        if not self.fixed_variance:
            self.raw_variance.data = self.init_raw_variance.to(device)
        if not self.fixed_lengthscale:
            if lengthscale_init is not None:
                self.raw_lengthscale.data = lengthscale_init.to(device)
            else:
                self.raw_lengthscale.data = self.init_raw_variance.to(device)
            
            lengthscale_prior_scale = 0.25
            lengthscale_prior_loc = torch.log(self.lengthscale.detach()) + lengthscale_prior_scale**2
            self.lengthscale_prior = torch.distributions.log_normal.LogNormal(lengthscale_prior_loc, lengthscale_prior_scale)

            variance_prior_scale = 0.25
            variance_prior_loc = torch.log(self.variance.detach()) + variance_prior_scale**2
            self.variance_prior = torch.distributions.log_normal.LogNormal(variance_prior_loc, variance_prior_scale)

    

class Matern52(Stationary):

    def __init__(self, init_variance, init_lengthscale, fixed_variance, fixed_lengthscale):
        super().__init__(init_variance, init_lengthscale, fixed_variance, fixed_lengthscale)

    def K_r(self, r):
        sqrt5 = torch.sqrt(torch.tensor(5.0))
        return (1.0 + sqrt5 * r + 5.0 * r**2 / 3.0) * torch.exp(-sqrt5 * r)
    

class Matern32(Stationary):

    def __init__(self, init_variance, init_lengthscale, fixed_variance, fixed_lengthscale, markovian=False):
        super().__init__(init_variance, init_lengthscale, fixed_variance, fixed_lengthscale)
        self.markovian = markovian
        if markovian:
            assert init_variance.shape[0] > 0, "Matern32 only supports >0-dimensional variances"
            assert (
                init_lengthscale.shape[0] > 0
            ), "Matern32 only supports >0-dimensional lengthscales"            
            self.num_latent = self.raw_lengthscale.shape[0]
    @property
    def state_dim(self):
        return 2
    
    def K_r(self, r):
        sqrt3 = torch.sqrt(torch.tensor(3.0))
        return (1.0 + sqrt3 * r) * torch.exp(-sqrt3 * r)
    
    def stationary_covariance(self):
        # Pinf = torch.tile(torch.eye(2), (self.num_latent, 1, 1)).to(self.variance.device)
        # Pinf *= self.variance[:, None, None]
        # Pinf[:, 1, 1] *= 3.0 / self.lengthscale**2.0
        # return Pinf
        return self.stationary_covariance_(self.variance, self.lengthscale)
    
    @partial(functorch.vmap, in_dims=(None, 0, 0), out_dims=0)
    def stationary_covariance_(self, variance, lengthscale):
        Pinf = torch.eye(2).to(variance.device)
        Pinf = Pinf * variance
        Pinf[1, 1] = Pinf[1,1] * 3.0 / lengthscale**2.0
        return Pinf

    def measurement_model(self):
        H = torch.Tensor([[1.0, 0.0]]).to(self.variance.device)
        # no need to tile, we can use nested vmap's to do this
        return H

    @partial(functorch.vmap, in_dims=(None, 0))
    def state_transition(self, dt):
        lam = np.sqrt(3.0) / self.lengthscale
        A = torch.tile(torch.ones((2,2)), (self.num_latent, 1, 1)).to(self.variance.device)
        A[:, 0, 0] = lam
        A[:, 0, 1] = 1.0
        A[:, 1, 0] = -lam**2.0
        A[:, 1, 1] = -lam
        A = A * dt
        A += torch.eye(2).to(self.variance.device)
        A *= torch.exp(-dt * lam)[:, None, None]
        return A
        # return self.state_transition_(dt, self.lengthscale, self.variance)
    
    # @partial(functorch.vmap, in_dims=(None, None, 0, 0), out_dims=0)
    # def state_transition_(self, dt, lengthscale, variance):
    #     lam = np.sqrt(3.0) / lengthscale
    #     A = torch.ones((2,2)).to(variance.device)
    #     print(A.shape, lam.shape)
    #     A[0, 0] = lam
    #     A[0, 1] = 1.0
    #     A[1, 0] = -lam**2.0
    #     A[1, 1] = -lam
    #     A = A * dt
    #     A += torch.eye(2).to(variance.device)
    #     A *= torch.exp(-dt * lam)
    #     return A
    
class RBF(Stationary):

    def __init__(self, init_variance, init_lengthscale, fixed_variance, fixed_lengthscale):
        super().__init__(init_variance, init_lengthscale, fixed_variance, fixed_lengthscale)

    def K_r(self, r):
        return torch.exp(-r**2 / 2.0)
    

"""
Code modified from Wenlin Chen & Harrison Zhu's implementation of SVGPVAE
"""

import torch
import torch.nn as nn
from abc import abstractmethod
from functools import partial
import numpy as np
from hipposvgpv2.svgpvae_utils import gauss_cross_entropy


class GPVAEBaseModel(nn.Module):
    """Base class for GP-VAE models."""
    
    def __init__(self, kernel, likelihood, encoder, decoder, jitter=1e-6):
        super().__init__()
        self.kernel = kernel
        self.likelihood = likelihood
        self.encoder = encoder
        self.decoder = decoder
        self.dim_f = None  # To be set by subclasses
        self.jitter = jitter
    
    def add_jitter(self, Kuu):
        return Kuu + self.jitter * torch.mean(torch.diag(Kuu)).detach() * torch.eye(
            Kuu.shape[0], device=Kuu.device
        )

    @abstractmethod
    def elbo(self, x_train, y_train, mask):
        """Compute ELBO for the GPVAE model."""
        raise NotImplementedError

    @abstractmethod
    def evaluate_nll(self, x_train, y_train, x_test, y_test, mask, num_samples):
        """Evaluate negative log likelihood."""
        raise NotImplementedError

    @abstractmethod
    def pred_f(self, x_train, y_train, x_test, mask):
        """Predict latent function values."""
        raise NotImplementedError

    @partial(
        torch.vmap,
        in_dims=(None, 0, 0, 0, 0),
        out_dims=(0, 0),
        randomness="different",
    )
    def pred_f_vmap(self, x_train, y_train, x_test, mask):
        return self.pred_f(x_train, y_train, x_test, mask)

    @partial(
        torch.vmap,
        in_dims=(None, 0, 0, 0, 0),
        out_dims=(0, 0, 0, 0),
        randomness="different",
    )
    def pred_y_vmap(self, x_train, y_train, x_test, mask, num_samples):
        """Vectorized prediction of observations at test points."""
        mean_f, var_f = self.pred_f(x_train, y_train, x_test, mask)
        mean_y, var_y = self.pred_y_mc(mean_f, var_f, num_samples)
        return mean_y, var_y, mean_f, var_f

    def pred_y(self, x_train, y_train, x_test, mask, num_samples):
        """Predict observations at test points."""
        mean_f, var_f = self.pred_f(x_train, y_train, x_test, mask)
        mean_y, var_y = self.pred_y_mc(mean_f, var_f, num_samples)
        return mean_y, var_y, mean_f, var_f

    @partial(torch.vmap, in_dims=(None, 0, 0, None), out_dims=(0, 0), randomness="different")
    def pred_y_mc(self, mean_f, var_f, num_samples):
        # mean_f.shape = (dim_f,)
        # var_f.shape = (dim_f,)

        # scale_f.shape = (dim_f,)
        scale_f = torch.sqrt(var_f)
        # samples_f.shape = (num_samples, dim_f)
        samples_f = (
            mean_f + torch.randn(num_samples, self.dim_f).to(mean_f.device) * scale_f
        )
        # conditional_mean_y.shape = (num_samples, dim_y)
        # conditional_var_y.shape = (,) for GaussianDecoder and (num_samples, dim_y) for BernoulliDecoder
        samples_f_decode = self.decoder(samples_f)
        conditional_mean_y, conditional_var_y = self.likelihood.conditional_moments(
            samples_f_decode
        )
        # mean_y.shape = (dim_y,)
        mean_y = torch.mean(conditional_mean_y, dim=0)
        # squared_conditional_mean_y.shape = (num_samples, dim_y)
        squared_conditional_mean_y = conditional_mean_y**2
        # mean_y_squared.shape = (dim_y,)
        mean_y_squared = torch.mean(
            conditional_var_y + squared_conditional_mean_y, dim=0
        )
        # squared_mean_y.shape = (dim_y, )
        squared_mean_y = mean_y**2
        # var_y.shape = (dim_y, )
        var_y = mean_y_squared - squared_mean_y
        return mean_y, var_y



class SVGPVAEMultipleKernels(GPVAEBaseModel):
    """SVGPVAE implementation with multiple kernels."""
    
    def __init__(
        self,
        kernel,
        likelihood,
        encoder,
        decoder,
        inducing_points,
        N_train,
        mu_hat_old = None,
        A_hat_old = None,
        encoder_old=None,
        inducing_points_old=None,
        num_input_axes=2,
        beta=1.0,
        beta_corr=1.0,
        mode="missing"
    ):
        kernel = nn.ModuleList(kernel)
        super().__init__(kernel, likelihood, encoder, decoder)



        self.beta = beta
        self.beta_corr=beta_corr
        self.N_train = N_train
        self.inducing_points = inducing_points
        # make inducing points trainable 
        self.num_inducing_points = inducing_points.shape[0]
        self.inducing_points = nn.Parameter(self.inducing_points, requires_grad=True)
        self.num_input_axes = num_input_axes
        self.mode = mode
        self.dim_f = len(kernel)

        # online params
        if mu_hat_old is not None:
            self.mu_hat_old = mu_hat_old.detach()
        else:
            self.mu_hat_old = None
        if A_hat_old is not None:
            self.A_hat_old = A_hat_old.detach()
        else:
            self.A_hat_old = None


        self.encoder_old = encoder_old if encoder_old is not None else None

        if inducing_points_old is not None:
            self.inducing_points_old = inducing_points_old.detach()
            for param in self.encoder_old.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
        else:
            self.inducing_points_old = None
            
        # Setup permutation and summation axes
        self._setup_axes()

    def _setup_axes(self):
        """Initialize permutation and summation axes."""
        self.perm_plus = list(range(self.num_input_axes + 1))
        self.perm_plus[-1] -= 1
        self.perm_plus[-2] += 1
        
        self.perm = list(range(self.num_input_axes))
        self.perm[-1] -= 1
        self.perm[-2] += 1
        
        if self.num_input_axes == 2:
            self.sum_axes = [0, 1]
            self.sum_time_axes = [0]
        elif self.num_input_axes == 4:
            self.sum_axes = [0, 1]
            self.sum_time_axes = [0, 1, 2]
        else:
            self.sum_axes = list(range(self.num_input_axes))
            self.sum_time_axes = [0]

    def pred_f(self, x_train, y_train, x_test, mask):
        qnet_mu, qnet_var = self.encoder(y_train)
        # qnet_mu, qnet_var = self.encoder_old(y_train)
        qnet_var = torch.clamp(qnet_var, 1e-3, 10)

        # applying masking to "latent dataset"
        if self.mode == "missing":
            mask_frame = mask
        else:
            mask_frame = 1
        qnet_mu = qnet_mu * mask_frame
        Kmms = torch.stack(
            [
                self.add_jitter(self.kernel[i](self.inducing_points))
                for i in range(self.dim_f)
            ]
        )
        Knms = torch.stack(
            [self.kernel[i](x_train, self.inducing_points) for i in range(self.dim_f)]
        )
        Kxxs = torch.stack(
            [self.kernel[i](x_test, diag=True) for i in range(self.dim_f)]
        ).to(self.inducing_points.device)
        Kxms = torch.stack(
            [self.kernel[i](x_test, self.inducing_points) for i in range(self.dim_f)]
        )

        # obtain qS
        p_m, p_v, _, _ = self.approximate_posterior_params(
            mask_frame, qnet_mu, qnet_var, Kmms, Kxxs, Kxms, Knms
        )
        return p_m, p_v
    

    @partial(
        torch.vmap,
        in_dims=(None, None, -1, -1, 0, 0, 0, 0),
        out_dims=(-1, -1, -1, -1),
    )
    def approximate_posterior_params(
        self, mask, y_train, noise, Kmm, Kxx, Kxm, Knm
    ):
        """
        Computes parameters of q_S: i.e. \mu_b^l and A_b^l.
        return: posterior mean at index points,
                posterior covariance matrix (diag) at index points
        """
        y_train = y_train.unsqueeze(-1)
        noise = noise.unsqueeze(-1)

        # compute K_mm=k(U, U) kernel matrix
        # add jitter of 1e-5 for stability
        Kmx = torch.permute(Kxm, dims=self.perm)  # (m, x)
        Kmn = torch.permute(Knm, dims=self.perm)  # (m, N)
        precision = torch.nan_to_num(
            torch.reciprocal(noise), nan=0.0, posinf=0.0, neginf=0.0
        )
        # computes the \Sigma_b^l matrix in q_S
        sigma_l = Kmm + torch.matmul(Kmn, torch.multiply(Knm, precision * mask))
        # invert it so that we can calculate \mu_b^l and A_b^l soon
        sigma_l_jittered = self.add_jitter(sigma_l)

        # This is product of the final 2 terms in \mu_b^l
        noise_inv_y = precision * y_train * mask

        tmp_variable = torch.linalg.solve(sigma_l_jittered, Kmn @ noise_inv_y)

        # TODO: figure this out. These are required for H1
        mean_vector = Kxm @ tmp_variable
        Kxm_Sigma_l_Kmx = torch.matmul(Kxm, torch.linalg.solve(sigma_l_jittered, Kmx))
        B_vector = Kxx + torch.diagonal(
            -torch.matmul(Kxm, torch.linalg.solve(Kmm, Kmx)) + Kxm_Sigma_l_Kmx,
            dim1=-2,
            dim2=-1,
        )

        # exact \mu_b^l and A_b^l
        mu_hat = Kmm @ tmp_variable
        A_hat = torch.matmul(Kmm, torch.linalg.solve(sigma_l_jittered, Kmm))

        # These are the q(u)=N(mu_hat, A_hat) parameters
        # mu_hat is torch.Size([num_inducing, 1, num_latent])
        # A_hat is torch.Size([num_inducing, num_inducing, num_latent])

        return mean_vector[..., 0], B_vector, mu_hat, A_hat
    


    @partial(
        torch.vmap,
        in_dims=(None, None, -1, -1, 0, 0),
        out_dims=(-1, -1),
    )
    def approximate_posterior_inducing_params(
        self, mask, y_train, noise, Kmm, Knm
    ):
        """
        Computes parameters of q_S: i.e. \mu_b^l and A_b^l.
        return: posterior mean at index points,
                posterior covariance matrix (diag) at index points
        """
    
        y_train = y_train.unsqueeze(-1)
        noise = noise.unsqueeze(-1)

        # compute K_mm=k(U, U) kernel matrix
        # add jitter of 1e-5 for stability
        Kmn = torch.permute(Knm, dims=self.perm)  # (m, N)
        precision = torch.nan_to_num(
            torch.reciprocal(noise), nan=0.0, posinf=0.0, neginf=0.0
        )
        # computes the \Sigma_b^l matrix in q_S
        sigma_l = Kmm + torch.matmul(Kmn, torch.multiply(Knm, precision * mask))
        # invert it so that we can calculate \mu_b^l and A_b^l soon
        sigma_l_jittered = self.add_jitter(sigma_l)

        # This is product of the final 2 terms in \mu_b^l
        noise_inv_y = precision * y_train * mask

        tmp_variable = torch.linalg.solve(sigma_l_jittered, Kmn @ noise_inv_y)

        # exact \mu_b^l and A_b^l
        mu_hat = Kmm @ tmp_variable
        A_hat = torch.matmul(Kmm, torch.linalg.solve(sigma_l_jittered, Kmm))

        # These are the q(u)=N(mu_hat, A_hat) parameters
        # mu_hat is torch.Size([num_inducing, 1, num_latent])
        # A_hat is torch.Size([num_inducing, num_inducing, num_latent])

        return mu_hat, A_hat

    @partial(
        torch.vmap,
        in_dims=(None, None, None, None, -1, -1, -1, -1, 0, 0, 0),
        out_dims=(0, 0),
    )
    def variational_loss(
        self, seq_length_train, mask, x, y, mu_hat, A_hat, noise, Kmm, Knn, Knm
    ):
        """
        Computes L_H (equation H3) for the data in the current batch.
        :return: sum_term, KL_term (variational loss = sum_term + KL_term)  (1,)
        """
        y = y.unsqueeze(-1)
        noise = noise.unsqueeze(-1)

        # kernel matrices
        # K_mm, K_nn, K_nm in the notebook
        Kmm_inv = torch.linalg.inv(Kmm)  # (m,m)

        # K_nm \cdot K_mm_inv \cdot m,  (b, N, 1)]
        # This corresponds to the mean in H3.1. mu_hat here is mu_b
        mean_vector = Knm @ Kmm_inv @ mu_hat
        # This is basically A_b
        # (b, m, m)
        S_matrix = A_hat
        # KL term. This corresponds to H3.4 These are standard KL divergences between
        # multivariate Gaussians https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
        # Prior: N(0, Kmm)
        # KL(N(\mu_b, A_b) || N(0, Kmm))
        Kmm_chol = torch.linalg.cholesky(Kmm)
        S_chol = torch.linalg.cholesky(
            self.add_jitter(A_hat)
        )
        Kmm_log_det = 2 * torch.sum(
            torch.log(torch.diagonal(Kmm_chol, dim1=-2, dim2=-1)), dim=-1
        )
        S_log_det = 2 * torch.sum(
            torch.log(torch.diagonal(S_chol, dim1=-2, dim2=-1)), dim=-1
        )  # (b)

        KL_term = 0.5 * (
            Kmm_log_det
            - S_log_det  # (b)
            - self.num_inducing_points
            + torch.diagonal(Kmm_inv @ A_hat, dim1=-2, dim2=-1).sum(-1)  # (b)
            + torch.sum(mu_hat * Kmm_inv @ mu_hat, dim=self.sum_axes)  # (b)
        )

        # diag(K_tilde), (b, )
        # Precision is the \tilde{\sigma}_i^{-2} term in H3.1
        precision = torch.nan_to_num(
            torch.reciprocal(noise), nan=0.0, posinf=0.0, neginf=0.0
        )

        # This computes the \tilde{k}_ii term in H3.2
        correction_term = torch.diagonal(
            Knm @ Kmm_inv @ torch.permute(Knm, dims=self.perm), dim1=-2, dim2=-1
        )
        if Knn.ndim == 1:
            Knn = Knn.unsqueeze(-1)
            correction_term = correction_term.unsqueeze(-1)
        # H3.2 term (b, T, 1)
        K_tilde_terms = precision * (Knn - correction_term)

        # This section pertains to computing H3.3
        # First we compute the Lambda_i matrix
        # First the k_i \cdot k_i^T, (b, m, m) term
        # if batching sequences, we have b x T x m x m
        Knm = Knm.unsqueeze(-1)
        lambda_mat = torch.matmul(
            Knm,
            torch.permute(Knm, dims=self.perm_plus),
        )

        # Now to get the term inside the trace in H3.3, multiple both sides of k_i k_i^T
        # by K_mm^{-1}
        # K_mm_inv \cdot k_i \cdot k_i^T \cdot K_mm_inv, (b, m, m)
        if self.num_input_axes == 2:
            mult = torch.einsum("tij, jk -> tik", lambda_mat, Kmm_inv)
            lambda_mat = torch.einsum("ij,tjk -> tik", Kmm_inv, mult)
            mult = torch.einsum("ij, tjk->tik", S_matrix, lambda_mat)
            # (b, t)
            mult = torch.einsum("tii", lambda_mat)[..., None]
            trace_terms = precision * mult
        else:
            lambda_mat = torch.matmul(Kmm_inv, torch.matmul(lambda_mat, Kmm_inv))
            # Trace terms, (b,)
            trace_terms = precision * torch.diagonal(
                torch.matmul(S_matrix, lambda_mat), dim1=-2, dim2=-1
            ).sum(-1)

        L_3_sum_term = -0.5 * (
            torch.sum(K_tilde_terms * mask)  # H3.2
            + torch.sum(trace_terms * mask)  # H3.3
            + torch.sum(torch.log(noise) * mask)  # part of H3.1
            + seq_length_train
            * torch.log(torch.tensor(2 * np.pi).to(noise.device))  # part of H3.1
            + torch.sum(precision * (y - mean_vector) ** 2 * mask)  # main part of H3.1
        )
        # return the sum part of H3 and H3.4 separately
        return L_3_sum_term, KL_term

    @partial(
        torch.vmap,
        in_dims=(None, 0, 0, 0, 0, -1, -1),
        out_dims=(0,),
        randomness="different",
    )
    def elbo(self, x_train, y_train, mask, seq_length_train, mu_hat_old_batch, A_hat_old_batch):
        # encode to get the \tilde{y} and \tilde{\sigma}
        qnet_mu, qnet_var = self.encoder(y_train)
        # qnet_mu, qnet_var = self.encoder_old(y_train)

        # no idea, some trick that Jazbec et al. (2021) used to make training stable
        qnet_var = torch.clamp(qnet_var, 1e-3, 10)

        # applying masking to "latent dataset"
        if self.mode == "missing":
            mask_frame = mask
        else:
            mask_frame = 1
        qnet_mu = qnet_mu * mask_frame
        # obtain qS=N(\mu_b^l, A_b^l)
        Kmms = torch.stack(
            [
                self.add_jitter(self.kernel[i](self.inducing_points))
                for i in range(self.dim_f)
            ]
        )
        Knns = torch.stack(
            [self.kernel[i](x_train, diag=True) for i in range(self.dim_f)]
        )
        Knms = torch.stack(
            [self.kernel[i](x_train, self.inducing_points) for i in range(self.dim_f)]
        )

        p_m, p_v, mu_hat, A_hat = self.approximate_posterior_params(
            mask_frame,
            qnet_mu,
            qnet_var,
            Kmms,
            Knns,
            Knms,
            Knms,
        )
        # Compute H3 for each l=1,..,L
        inside_elbo_recon, inside_elbo_kl = self.variational_loss(
            seq_length_train,
            mask_frame,
            x_train,
            qnet_mu,
            mu_hat,
            A_hat,
            qnet_var,
            Kmms,
            Knns,
            Knms,
        )
        # reduce sum to get the final H3
        inside_elbo_recon = torch.sum(inside_elbo_recon, dim=0)
        inside_elbo_kl = torch.sum(inside_elbo_kl, dim=0)
        inside_elbo = inside_elbo_recon - inside_elbo_kl

        # cross entropy term i.e. H2
        ce_term = gauss_cross_entropy(p_m, p_v, qnet_mu, qnet_var)
        ce_term *= mask_frame
        # print(ce_term[0].numpy()[:,0])
        ce_term = torch.sum(ce_term, dim=self.sum_axes)

        # Put H2 and H3 together
        KL_term = -ce_term + inside_elbo

        # Sample and compute reconstruction loss
        epsilon = torch.randn(p_m.shape, device=p_m.device)
        samples_f = p_m + epsilon * torch.sqrt(p_v)
        samples_f_decode = self.decoder(samples_f)
        log_prob = self.likelihood.log_prob(y_train, samples_f_decode) * mask_frame
        log_prob = log_prob.sum(-1)
        recon_loss = torch.sum(log_prob, dim=self.sum_time_axes)

        # H1 + H2 + H3 all put together
        elbo = recon_loss + (self.beta / self.dim_f) * KL_term
        """
        if self.inducing_points_old is not None:
            Ku_old_old = torch.stack([self.kernel[i](self.inducing_points_old) for i in range(self.dim_f)])
            Kuu_old = torch.stack([self.kernel[i](self.inducing_points, self.inducing_points_old) for i in range(self.dim_f)])

            # # caching network only
            # qnet_mu_old, qnet_var_old = self.encoder_old(y_train)
            # # qnet_mu_old, qnet_var_old = self.encoder(y_train)
            # Knms_old = torch.stack([self.kernel[i](x_train, self.inducing_points_old) for i in range(self.dim_f)])
            # mu_hat_old, A_hat_old = self.approximate_posterior_inducing_params(
            #     mask,
            #     qnet_mu_old,
            #     qnet_var_old,
            #     Ku_old_old,
            #     Knms_old,
            # )
            # elbo_correction = self.correction_term(mu_hat, A_hat, mu_hat_old, A_hat_old, Kmms, Kuu_old, Ku_old_old)
            
            #use caching
            elbo_correction = self.correction_term(mu_hat, A_hat, mu_hat_old_batch, A_hat_old_batch, Kmms, Kuu_old, Ku_old_old)

            elbo_correction = torch.sum(elbo_correction, dim=0)
            # print(elbo_correction, elbo)
            elbo = elbo + self.beta_corr* elbo_correction
        """
        return elbo

    @partial(
        torch.vmap,
        in_dims=(None, -1, -1, -1, -1, 0, 0, 0),
        out_dims=(0,),
        randomness="different",
    )
    def correction_term(self, m_new, S_new, m_old, S_old, Kuu, Kuu_old, Ku_old_old):
        Ku_old_u = Kuu_old.T
        Luu = torch.linalg.cholesky(self.add_jitter(Kuu))
        
        # torch.Size([50, 1]) torch.Size([50, 50])
        mu = Ku_old_u @ torch.cholesky_solve(m_new, Luu)
        Sigma = compute_Sigma(Ku_old_old, Ku_old_u, Kuu_old, Kuu, Luu, S_new)
        Smm = Sigma + mu @ mu.T
        
        LSa = torch.linalg.cholesky(self.add_jitter(S_old))
        LKa = torch.linalg.cholesky(self.add_jitter(Ku_old_old))
        
        obj = torch.sum(torch.log(torch.diagonal(LKa))) - torch.sum(torch.log(torch.diagonal(LSa)))
        Sainv_ma = torch.cholesky_solve(m_old, LSa).squeeze(-1)
        obj = obj - 0.5 * torch.sum(m_old.squeeze(-1) * Sainv_ma)
        obj = obj + torch.sum(mu.squeeze(-1) * Sainv_ma)
        
        Sainv_Smm = torch.cholesky_solve(Smm, LSa)
        Kainv_Smm = torch.cholesky_solve(Smm, LKa)
        obj = obj - 0.5 * (torch.trace(Sainv_Smm) - torch.trace(Kainv_Smm))
        
        return obj


class SVGPVAEMultipleKernelsMultipleInducing(GPVAEBaseModel):
    """SVGPVAE implementation with multiple kernels."""
    
    def __init__(
        self,
        kernel,
        likelihood,
        encoder,
        decoder,
        inducing_points,
        N_train,
        mu_hat_old = None,
        A_hat_old = None,
        encoder_old=None,
        inducing_points_old=None,
        num_input_axes=2,
        beta=1.0,
        beta_corr=1.0,
        mode="missing"
    ):
        kernel = nn.ModuleList(kernel)
        super().__init__(kernel, likelihood, encoder, decoder)

        self.beta = beta
        self.beta_corr=beta_corr
        self.N_train = N_train
        self.inducing_points = inducing_points # shape (num_latent, num_inducing, 1)
        # make inducing points trainable 
        self.num_inducing_points = inducing_points.shape[1]
        self.inducing_points = nn.Parameter(self.inducing_points, requires_grad=True)
        self.num_input_axes = num_input_axes
        self.mode = mode
        self.dim_f = len(kernel)

        # online params
        if mu_hat_old is not None:
            self.mu_hat_old = mu_hat_old.detach()
        else:
            self.mu_hat_old = None
        if A_hat_old is not None:
            self.A_hat_old = A_hat_old.detach()
        else:
            self.A_hat_old = None


        self.encoder_old = encoder_old if encoder_old is not None else None

        if inducing_points_old is not None:
            self.inducing_points_old = inducing_points_old.detach()
            for param in self.encoder_old.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
        else:
            self.inducing_points_old = None
            
        # Setup permutation and summation axes
        self._setup_axes()

    def _setup_axes(self):
        """Initialize permutation and summation axes."""
        self.perm_plus = list(range(self.num_input_axes + 1))
        self.perm_plus[-1] -= 1
        self.perm_plus[-2] += 1
        
        self.perm = list(range(self.num_input_axes))
        self.perm[-1] -= 1
        self.perm[-2] += 1
        
        if self.num_input_axes == 2:
            self.sum_axes = [0, 1]
            self.sum_time_axes = [0]
        elif self.num_input_axes == 4:
            self.sum_axes = [0, 1]
            self.sum_time_axes = [0, 1, 2]
        else:
            self.sum_axes = list(range(self.num_input_axes))
            self.sum_time_axes = [0]

    def pred_f(self, x_train, y_train, x_test, mask):
        qnet_mu, qnet_var = self.encoder(y_train)
        # qnet_mu, qnet_var = self.encoder_old(y_train)
        qnet_var = torch.clamp(qnet_var, 1e-3, 10)

        # applying masking to "latent dataset"
        if self.mode == "missing":
            mask_frame = mask
        else:
            mask_frame = 1
        qnet_mu = qnet_mu * mask_frame
        Kmms = torch.stack(
            [
                self.add_jitter(self.kernel[i](self.inducing_points[i]))
                for i in range(self.dim_f)
            ]
        )
        Knms = torch.stack(
            [self.kernel[i](x_train, self.inducing_points[i]) for i in range(self.dim_f)]
        )
        Kxxs = torch.stack(
            [self.kernel[i](x_test, diag=True) for i in range(self.dim_f)]
        ).to(self.inducing_points.device)
        Kxms = torch.stack(
            [self.kernel[i](x_test, self.inducing_points[i]) for i in range(self.dim_f)]
        )

        # obtain qS
        p_m, p_v, _, _ = self.approximate_posterior_params(
            mask_frame, qnet_mu, qnet_var, Kmms, Kxxs, Kxms, Knms
        )
        return p_m, p_v
    

    @partial(
        torch.vmap,
        in_dims=(None, None, -1, -1, 0, 0, 0, 0),
        out_dims=(-1, -1, -1, -1),
    )
    def approximate_posterior_params(
        self, mask, y_train, noise, Kmm, Kxx, Kxm, Knm
    ):
        """
        Computes parameters of q_S: i.e. \mu_b^l and A_b^l.
        return: posterior mean at index points,
                posterior covariance matrix (diag) at index points
        """
        y_train = y_train.unsqueeze(-1)
        noise = noise.unsqueeze(-1)

        # compute K_mm=k(U, U) kernel matrix
        # add jitter of 1e-5 for stability
        Kmx = torch.permute(Kxm, dims=self.perm)  # (m, x)
        Kmn = torch.permute(Knm, dims=self.perm)  # (m, N)
        precision = torch.nan_to_num(
            torch.reciprocal(noise), nan=0.0, posinf=0.0, neginf=0.0
        )
        # computes the \Sigma_b^l matrix in q_S
        sigma_l = Kmm + torch.matmul(Kmn, torch.multiply(Knm, precision * mask))
        # invert it so that we can calculate \mu_b^l and A_b^l soon
        sigma_l_jittered = self.add_jitter(sigma_l)

        # This is product of the final 2 terms in \mu_b^l
        noise_inv_y = precision * y_train * mask

        tmp_variable = torch.linalg.solve(sigma_l_jittered, Kmn @ noise_inv_y)

        # TODO: figure this out. These are required for H1
        mean_vector = Kxm @ tmp_variable
        Kxm_Sigma_l_Kmx = torch.matmul(Kxm, torch.linalg.solve(sigma_l_jittered, Kmx))
        B_vector = Kxx + torch.diagonal(
            -torch.matmul(Kxm, torch.linalg.solve(Kmm, Kmx)) + Kxm_Sigma_l_Kmx,
            dim1=-2,
            dim2=-1,
        )

        # exact \mu_b^l and A_b^l
        mu_hat = Kmm @ tmp_variable
        A_hat = torch.matmul(Kmm, torch.linalg.solve(sigma_l_jittered, Kmm))

        # These are the q(u)=N(mu_hat, A_hat) parameters
        # mu_hat is torch.Size([num_inducing, 1, num_latent])
        # A_hat is torch.Size([num_inducing, num_inducing, num_latent])

        return mean_vector[..., 0], B_vector, mu_hat, A_hat
    


    @partial(
        torch.vmap,
        in_dims=(None, None, -1, -1, 0, 0),
        out_dims=(-1, -1),
    )
    def approximate_posterior_inducing_params(
        self, mask, y_train, noise, Kmm, Knm
    ):
        """
        Computes parameters of q_S: i.e. \mu_b^l and A_b^l.
        return: posterior mean at index points,
                posterior covariance matrix (diag) at index points
        """
    
        y_train = y_train.unsqueeze(-1)
        noise = noise.unsqueeze(-1)

        # compute K_mm=k(U, U) kernel matrix
        # add jitter of 1e-5 for stability
        Kmn = torch.permute(Knm, dims=self.perm)  # (m, N)
        precision = torch.nan_to_num(
            torch.reciprocal(noise), nan=0.0, posinf=0.0, neginf=0.0
        )
        # computes the \Sigma_b^l matrix in q_S
        sigma_l = Kmm + torch.matmul(Kmn, torch.multiply(Knm, precision * mask))
        # invert it so that we can calculate \mu_b^l and A_b^l soon
        sigma_l_jittered = self.add_jitter(sigma_l)

        # This is product of the final 2 terms in \mu_b^l
        noise_inv_y = precision * y_train * mask

        tmp_variable = torch.linalg.solve(sigma_l_jittered, Kmn @ noise_inv_y)

        # exact \mu_b^l and A_b^l
        mu_hat = Kmm @ tmp_variable
        A_hat = torch.matmul(Kmm, torch.linalg.solve(sigma_l_jittered, Kmm))

        # These are the q(u)=N(mu_hat, A_hat) parameters
        # mu_hat is torch.Size([num_inducing, 1, num_latent])
        # A_hat is torch.Size([num_inducing, num_inducing, num_latent])

        return mu_hat, A_hat

    @partial(
        torch.vmap,
        in_dims=(None, None, None, None, -1, -1, -1, -1, 0, 0, 0),
        out_dims=(0, 0),
    )
    def variational_loss(
        self, seq_length_train, mask, x, y, mu_hat, A_hat, noise, Kmm, Knn, Knm
    ):
        """
        Computes L_H (equation H3) for the data in the current batch.
        :return: sum_term, KL_term (variational loss = sum_term + KL_term)  (1,)
        """
        y = y.unsqueeze(-1)
        noise = noise.unsqueeze(-1)

        # kernel matrices
        # K_mm, K_nn, K_nm in the notebook
        Kmm_inv = torch.linalg.inv(Kmm)  # (m,m)

        # K_nm \cdot K_mm_inv \cdot m,  (b, N, 1)]
        # This corresponds to the mean in H3.1. mu_hat here is mu_b
        mean_vector = Knm @ Kmm_inv @ mu_hat
        # This is basically A_b
        # (b, m, m)
        S_matrix = A_hat
        # KL term. This corresponds to H3.4 These are standard KL divergences between
        # multivariate Gaussians https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
        # Prior: N(0, Kmm)
        # KL(N(\mu_b, A_b) || N(0, Kmm))
        Kmm_chol = torch.linalg.cholesky(Kmm)
        S_chol = torch.linalg.cholesky(
            self.add_jitter(A_hat)
        )
        Kmm_log_det = 2 * torch.sum(
            torch.log(torch.diagonal(Kmm_chol, dim1=-2, dim2=-1)), dim=-1
        )
        S_log_det = 2 * torch.sum(
            torch.log(torch.diagonal(S_chol, dim1=-2, dim2=-1)), dim=-1
        )  # (b)

        KL_term = 0.5 * (
            Kmm_log_det
            - S_log_det  # (b)
            - self.num_inducing_points
            + torch.diagonal(Kmm_inv @ A_hat, dim1=-2, dim2=-1).sum(-1)  # (b)
            + torch.sum(mu_hat * Kmm_inv @ mu_hat, dim=self.sum_axes)  # (b)
        )

        # diag(K_tilde), (b, )
        # Precision is the \tilde{\sigma}_i^{-2} term in H3.1
        precision = torch.nan_to_num(
            torch.reciprocal(noise), nan=0.0, posinf=0.0, neginf=0.0
        )

        # This computes the \tilde{k}_ii term in H3.2
        correction_term = torch.diagonal(
            Knm @ Kmm_inv @ torch.permute(Knm, dims=self.perm), dim1=-2, dim2=-1
        )
        if Knn.ndim == 1:
            Knn = Knn.unsqueeze(-1)
            correction_term = correction_term.unsqueeze(-1)
        # H3.2 term (b, T, 1)
        K_tilde_terms = precision * (Knn - correction_term)

        # This section pertains to computing H3.3
        # First we compute the Lambda_i matrix
        # First the k_i \cdot k_i^T, (b, m, m) term
        # if batching sequences, we have b x T x m x m
        Knm = Knm.unsqueeze(-1)
        lambda_mat = torch.matmul(
            Knm,
            torch.permute(Knm, dims=self.perm_plus),
        )

        # Now to get the term inside the trace in H3.3, multiple both sides of k_i k_i^T
        # by K_mm^{-1}
        # K_mm_inv \cdot k_i \cdot k_i^T \cdot K_mm_inv, (b, m, m)
        if self.num_input_axes == 2:
            mult = torch.einsum("tij, jk -> tik", lambda_mat, Kmm_inv)
            lambda_mat = torch.einsum("ij,tjk -> tik", Kmm_inv, mult)
            mult = torch.einsum("ij, tjk->tik", S_matrix, lambda_mat)
            # (b, t)
            mult = torch.einsum("tii", lambda_mat)[..., None]
            trace_terms = precision * mult
        else:
            lambda_mat = torch.matmul(Kmm_inv, torch.matmul(lambda_mat, Kmm_inv))
            # Trace terms, (b,)
            trace_terms = precision * torch.diagonal(
                torch.matmul(S_matrix, lambda_mat), dim1=-2, dim2=-1
            ).sum(-1)

        L_3_sum_term = -0.5 * (
            torch.sum(K_tilde_terms * mask)  # H3.2
            + torch.sum(trace_terms * mask)  # H3.3
            + torch.sum(torch.log(noise) * mask)  # part of H3.1
            + seq_length_train
            * torch.log(torch.tensor(2 * np.pi).to(noise.device))  # part of H3.1
            + torch.sum(precision * (y - mean_vector) ** 2 * mask)  # main part of H3.1
        )
        # return the sum part of H3 and H3.4 separately
        return L_3_sum_term, KL_term

    @partial(
        torch.vmap,
        in_dims=(None, 0, 0, 0, 0, -1, -1),
        out_dims=(0,),
        randomness="different",
    )
    def elbo(self, x_train, y_train, mask, seq_length_train, mu_hat_old_batch, A_hat_old_batch):
        # encode to get the \tilde{y} and \tilde{\sigma}
        qnet_mu, qnet_var = self.encoder(y_train)
        # qnet_mu, qnet_var = self.encoder_old(y_train)

        # no idea, some trick that Jazbec et al. (2021) used to make training stable
        qnet_var = torch.clamp(qnet_var, 1e-3, 10)

        # applying masking to "latent dataset"
        if self.mode == "missing":
            mask_frame = mask
        else:
            mask_frame = 1
        qnet_mu = qnet_mu * mask_frame
        # obtain qS=N(\mu_b^l, A_b^l)
        Kmms = torch.stack(
            [
                self.add_jitter(self.kernel[i](self.inducing_points[i]))
                for i in range(self.dim_f)
            ]
        )
        Knns = torch.stack(
            [self.kernel[i](x_train, diag=True) for i in range(self.dim_f)]
        )
        Knms = torch.stack(
            [self.kernel[i](x_train, self.inducing_points[i]) for i in range(self.dim_f)]
        )

        p_m, p_v, mu_hat, A_hat = self.approximate_posterior_params(
            mask_frame,
            qnet_mu,
            qnet_var,
            Kmms,
            Knns,
            Knms,
            Knms,
        )
        # Compute H3 for each l=1,..,L
        inside_elbo_recon, inside_elbo_kl = self.variational_loss(
            seq_length_train,
            mask_frame,
            x_train,
            qnet_mu,
            mu_hat,
            A_hat,
            qnet_var,
            Kmms,
            Knns,
            Knms,
        )
        # reduce sum to get the final H3
        inside_elbo_recon = torch.sum(inside_elbo_recon, dim=0)
        inside_elbo_kl = torch.sum(inside_elbo_kl, dim=0)
        inside_elbo = inside_elbo_recon - inside_elbo_kl

        # cross entropy term i.e. H2
        ce_term = gauss_cross_entropy(p_m, p_v, qnet_mu, qnet_var)
        ce_term *= mask_frame
        # print(ce_term[0].numpy()[:,0])
        ce_term = torch.sum(ce_term, dim=self.sum_axes)

        # Put H2 and H3 together
        KL_term = -ce_term + inside_elbo

        # Sample and compute reconstruction loss
        epsilon = torch.randn(p_m.shape, device=p_m.device)
        samples_f = p_m + epsilon * torch.sqrt(p_v)
        samples_f_decode = self.decoder(samples_f)
        log_prob = self.likelihood.log_prob(y_train, samples_f_decode) * mask_frame
        log_prob = log_prob.sum(-1)
        recon_loss = torch.sum(log_prob, dim=self.sum_time_axes)

        # H1 + H2 + H3 all put together
        elbo = recon_loss + (self.beta / self.dim_f) * KL_term
        """
        if self.inducing_points_old is not None:
            Ku_old_old = torch.stack([self.kernel[i](self.inducing_points_old) for i in range(self.dim_f)])
            Kuu_old = torch.stack([self.kernel[i](self.inducing_points, self.inducing_points_old) for i in range(self.dim_f)])

            # # caching network only
            # qnet_mu_old, qnet_var_old = self.encoder_old(y_train)
            # # qnet_mu_old, qnet_var_old = self.encoder(y_train)
            # Knms_old = torch.stack([self.kernel[i](x_train, self.inducing_points_old) for i in range(self.dim_f)])
            # mu_hat_old, A_hat_old = self.approximate_posterior_inducing_params(
            #     mask,
            #     qnet_mu_old,
            #     qnet_var_old,
            #     Ku_old_old,
            #     Knms_old,
            # )
            # elbo_correction = self.correction_term(mu_hat, A_hat, mu_hat_old, A_hat_old, Kmms, Kuu_old, Ku_old_old)
            
            #use caching
            elbo_correction = self.correction_term(mu_hat, A_hat, mu_hat_old_batch, A_hat_old_batch, Kmms, Kuu_old, Ku_old_old)

            elbo_correction = torch.sum(elbo_correction, dim=0)
            # print(elbo_correction, elbo)
            elbo = elbo + self.beta_corr* elbo_correction
        """
        return elbo

    @partial(
        torch.vmap,
        in_dims=(None, -1, -1, -1, -1, 0, 0, 0),
        out_dims=(0,),
        randomness="different",
    )
    def correction_term(self, m_new, S_new, m_old, S_old, Kuu, Kuu_old, Ku_old_old):
        Ku_old_u = Kuu_old.T
        Luu = torch.linalg.cholesky(self.add_jitter(Kuu))
        
        # torch.Size([50, 1]) torch.Size([50, 50])
        mu = Ku_old_u @ torch.cholesky_solve(m_new, Luu)
        Sigma = compute_Sigma(Ku_old_old, Ku_old_u, Kuu_old, Kuu, Luu, S_new)
        Smm = Sigma + mu @ mu.T
        
        LSa = torch.linalg.cholesky(self.add_jitter(S_old))
        LKa = torch.linalg.cholesky(self.add_jitter(Ku_old_old))
        
        obj = torch.sum(torch.log(torch.diagonal(LKa))) - torch.sum(torch.log(torch.diagonal(LSa)))
        Sainv_ma = torch.cholesky_solve(m_old, LSa).squeeze(-1)
        obj = obj - 0.5 * torch.sum(m_old.squeeze(-1) * Sainv_ma)
        obj = obj + torch.sum(mu.squeeze(-1) * Sainv_ma)
        
        Sainv_Smm = torch.cholesky_solve(Smm, LSa)
        Kainv_Smm = torch.cholesky_solve(Smm, LKa)
        obj = obj - 0.5 * (torch.trace(Sainv_Smm) - torch.trace(Kainv_Smm))
        
        return obj

class SVGPVAEMultipleKernels_EWC(SVGPVAEMultipleKernels):
    """
    SVGPVAE model with Elastic Weight Consolidation (EWC) to regularize encoder and decoder.
    """

    def __init__(self, *args, lambda_ewc=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_ewc = lambda_ewc  # Regularization strength

        # Store optimal params and Fisher Information for EWC
        self.optimal_params = None
        self.fisher_info = None

    def apply_ewc_regularization(self):
        """
        Applies EWC penalty to `encoder` and `decoder` parameters.
        """
        if self.fisher_info is None or self.optimal_params is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        ewc_loss = 0.0
        for name, param in self.named_parameters():
            if name in self.fisher_info:
                ewc_loss += (self.fisher_info[name] * (param - self.optimal_params[name])**2).sum()

        return self.lambda_ewc * ewc_loss

    def elbo(self, x_train, y_train, mask, seq_length_train, mu_hat_old_batch, A_hat_old_batch, add_ewc=True):
        """
        Computes the ELBO with EWC regularization.
        """
        # Compute ELBO from parent class
        elbo_loss = super().elbo(x_train, y_train, mask, seq_length_train, mu_hat_old_batch, A_hat_old_batch)

        # Apply EWC penalty
        if add_ewc:
            ewc_loss = self.apply_ewc_regularization()
            return elbo_loss - ewc_loss  # EWC acts as a penalty
        else:
            return elbo_loss

class SVGPVAEMultipleKernelsMultipleInducing_EWC(SVGPVAEMultipleKernelsMultipleInducing):
    """
    SVGPVAE model with Elastic Weight Consolidation (EWC) to regularize encoder and decoder.
    """

    def __init__(self, *args, lambda_ewc=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_ewc = lambda_ewc  # Regularization strength

        # Store optimal params and Fisher Information for EWC
        self.optimal_params = None
        self.fisher_info = None

    def apply_ewc_regularization(self):
        """
        Applies EWC penalty to `encoder` and `decoder` parameters.
        """
        if self.fisher_info is None or self.optimal_params is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        ewc_loss = 0.0
        for name, param in self.named_parameters():
            if name in self.fisher_info:
                ewc_loss += (self.fisher_info[name] * (param - self.optimal_params[name])**2).sum()

        return self.lambda_ewc * ewc_loss

    def elbo(self, x_train, y_train, mask, seq_length_train, mu_hat_old_batch, A_hat_old_batch, add_ewc=True):
        """
        Computes the ELBO with EWC regularization.
        """
        # Compute ELBO from parent class
        elbo_loss = super().elbo(x_train, y_train, mask, seq_length_train, mu_hat_old_batch, A_hat_old_batch)

        # Apply EWC penalty
        if add_ewc:
            ewc_loss = self.apply_ewc_regularization()
            return elbo_loss - ewc_loss  # EWC acts as a penalty
        else:
            return elbo_loss

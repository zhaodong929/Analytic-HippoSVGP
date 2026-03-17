import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import trange
import os

# =====================================
#  0. Configure save path
# =====================================
SAVE_DIR = "./svgp_results"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"Created directory: {SAVE_DIR}")
else:
    print(f"Saving results to existing directory: {SAVE_DIR}")


# =====================================
#  1. Multi-dimensional RBF Kernel (shared across outputs)
# =====================================
class MultiDimRBFKernel(nn.Module):
    """
    RBF Kernel for multi-dimensional inputs, with ARD over input dims.
    Shared across all output dimensions.
    """
    def __init__(self, input_dim, lengthscales=None, variance=1.0, ard=True):
        super().__init__()
        self.input_dim = input_dim
        self.ard = ard

        if lengthscales is None:
            lengthscales = torch.ones(input_dim)

        if ard:
            self.log_lengthscales = nn.Parameter(
                torch.log(lengthscales * torch.ones(input_dim))
            )
        else:
            self.log_lengthscales = nn.Parameter(
                torch.log(torch.tensor(lengthscales.mean().item()))
            )

        self.log_variance = nn.Parameter(
            torch.log(torch.tensor(variance, dtype=torch.float32))
        )

    def forward(self, X1, X2):
        ell = torch.exp(self.log_lengthscales).to(X1.device)
        var = torch.exp(self.log_variance).to(X1.device)

        X1_scaled = X1 / ell
        X2_scaled = X2 / ell
        dist_sq = (X1_scaled[:, None, :] - X2_scaled[None, :, :]).pow(2).sum(-1)
        return var * torch.exp(-0.5 * dist_sq)


# =====================================
#  2. Likelihood abstraction (agnostic to likelihood type)
# =====================================
class Likelihood(nn.Module):
    """Base class for likelihoods."""
    def expected_log_prob(self, y, f_mean, f_var):
        """
        y, f_mean, f_var: shape (N, P)
        Return elementwise E_q[log p(y | f)] with same shape.
        """
        raise NotImplementedError

    def predict_mean_var(self, f_mean, f_var):
        """Map latent f -> observation y (default: identity)."""
        return f_mean, f_var


class GaussianLikelihood(Likelihood):
    """
    Independent Gaussian likelihood per output dimension.
    """
    def __init__(self, output_dim, noise_std=0.1):
        super().__init__()
        if isinstance(noise_std, (float, int)):
            init = torch.full((output_dim,), float(noise_std), dtype=torch.float32)
        else:
            noise_std = torch.tensor(noise_std, dtype=torch.float32)
            assert noise_std.shape[0] == output_dim
            init = noise_std
        self.log_noise = nn.Parameter(torch.log(init))  # (P,)

    def expected_log_prob(self, y, f_mean, f_var):
        # y, f_mean, f_var: (N, P)
        sigma2 = torch.exp(2 * self.log_noise)  # (P,)
        total_var = f_var + sigma2              # (N, P) via broadcasting
        return -0.5 * torch.log(2 * math.pi * total_var) \
               - 0.5 * (y - f_mean) ** 2 / total_var

    def predict_mean_var(self, f_mean, f_var):
        sigma2 = torch.exp(2 * self.log_noise)  # (P,)
        return f_mean, f_var + sigma2


# =====================================
#  3. Multi-output SVGP (shared kernel, whitened, vectorized over P)
# =====================================
class IndependentMultiOutputSVGP(nn.Module):
    """
    Independent multi-output SVGP with:
    - one shared kernel over inputs,
    - P independent outputs (different variational & noise params),
    - optional whitening of inducing variables.
    """

    def __init__(self, Z, kernel, likelihood, output_dim=1, whiten=True):
        super().__init__()
        self.Z = nn.Parameter(Z.clone().detach())   # (M, D)
        self.kernel = kernel
        self.likelihood = likelihood
        self.output_dim = output_dim
        self.whiten = whiten

        M = Z.shape[0]
        P = output_dim

        # Variational mean and diagonal covariance for all outputs
        self.m = nn.Parameter(torch.randn(M, P) * 0.1)     # (M, P)
        self.log_diag_S = nn.Parameter(torch.zeros(M, P))  # (M, P)

    # ---------- KL[q(u)||p(u)] ----------
    def compute_kl(self):
        """
        Vectorized over output dimension P.
        If whiten=True, prior is N(0, I) and KL has closed form.
        Otherwise p(u) = N(0, Kzz) with shared Kzz.
        """
        M, P = self.m.shape
        S_diag = torch.exp(self.log_diag_S)

        if self.whiten:
            # q(v) = N(m, diag(S_diag)), p(v) = N(0, I)
            term1 = (self.m ** 2).sum()      # sum over M,P of m^2
            term2 = S_diag.sum()             # Tr(S)
            logdet_S = self.log_diag_S.sum()
            kl = 0.5 * (term1 + term2 - M * P - logdet_S)
            return kl

        # Non-whitened: p(u) = N(0, Kzz)
        Kzz = self.kernel(self.Z, self.Z) + 1e-4 * torch.eye(M, device=self.Z.device)
        Lz = torch.linalg.cholesky(Kzz)
        Kinv = torch.cholesky_solve(torch.eye(M, device=self.Z.device), Lz)

        # m^T K^{-1} m for all outputs at once
        alpha = Kinv @ self.m                      # (M, P)
        term1 = (self.m * alpha).sum()

        # Tr(K^{-1} S) with diagonal S
        Kinv_diag = Kinv.diagonal().unsqueeze(1)   # (M, 1)
        term2 = (Kinv_diag * S_diag).sum()

        logdet_K = 2 * torch.log(torch.diag(Lz)).sum()
        logdet_S = self.log_diag_S.sum()

        kl = 0.5 * (term1 + term2 - M * P + logdet_K - logdet_S)
        return kl

    # ---------- helper: whitened -> unwhitened u ----------
    def _transform_whitened_to_u(self, Kzz, Lz):
        """
        Map whitening variables v to u:
        u = Lz v,  S_u = Lz diag(S_v) Lz^T.
        Done vectorized over P using shared Kzz.
        """
        S_v_diag = torch.exp(self.log_diag_S)   # (M, P)

        if self.whiten:
            # m_u: (M, P)
            m_u = Lz @ self.m
            # S_u_diag = diag(Lz diag(S_v) Lz^T) for each output
            Lz_sq = Lz ** 2                    # (M, M)
            S_u_diag = Lz_sq @ S_v_diag        # (M, P)
        else:
            m_u = self.m
            S_u_diag = S_v_diag

        return m_u, S_u_diag

    # ---------- Expected log likelihood ----------
    def expected_log_likelihood(self, X, Y):
        """
        X: (N, D)
        Y: (N, P)
        """
        M = self.Z.shape[0]
        Kzz = self.kernel(self.Z, self.Z) + 1e-4 * torch.eye(M, device=self.Z.device)
        Lz = torch.linalg.cholesky(Kzz)
        Kxz = self.kernel(X, self.Z)

        # A = Kxz Kzz^{-1}  (N, M)
        A = torch.cholesky_solve(Kxz.T, Lz).T

        # Map whitened parameters to u-space
        m_u, S_diag_u = self._transform_whitened_to_u(Kzz, Lz)  # (M,P), (M,P)

        # Mean of f: (N, P)
        f_mean = A @ m_u

        # Variance of f: diag(Kxx - A Kzz A^T + A S_u A^T)
        Kxx_diag = self.kernel(X, X).diagonal()                 # (N,)
        base = Kxx_diag - torch.sum(A * (Kzz @ A.T).T, dim=1)   # (N,)

        A_sq = A ** 2                                           # (N, M)
        S_term = A_sq @ S_diag_u                                # (N, P)

        f_var = base.unsqueeze(1) + S_term
        f_var = torch.clamp(f_var, min=1e-6)

        log_lik_pointwise = self.likelihood.expected_log_prob(Y, f_mean, f_var)
        return log_lik_pointwise.sum()

    # ---------- ELBO ----------
    def elbo(self, X, Y, N_total=None):
        kl = self.compute_kl()
        log_likelihood = self.expected_log_likelihood(X, Y)

        if N_total is not None:
            scale = N_total / X.shape[0]
        else:
            scale = 1.0

        return scale * log_likelihood - kl

    # ---------- Prediction (vectorized over P, no for p in range) ----------
    def predict(self, Xnew):
        M = self.Z.shape[0]
        Kzz = self.kernel(self.Z, self.Z) + 1e-4 * torch.eye(M, device=self.Z.device)
        Lz = torch.linalg.cholesky(Kzz)
        Kxz = self.kernel(Xnew, self.Z)

        A = torch.cholesky_solve(Kxz.T, Lz).T                    # (N, M)
        m_u, S_diag_u = self._transform_whitened_to_u(Kzz, Lz)   # (M,P), (M,P)

        f_mean = A @ m_u                                         # (N, P)

        Kxx_diag = self.kernel(Xnew, Xnew).diagonal()            # (N,)
        base = Kxx_diag - torch.sum(A * (Kzz @ A.T).T, dim=1)    # (N,)

        A_sq = A ** 2
        S_term = A_sq @ S_diag_u                                 # (N, P)

        f_var = base.unsqueeze(1) + S_term
        f_var = torch.clamp(f_var, min=1e-6)

        mean_y, var_y = self.likelihood.predict_mean_var(f_mean, f_var)
        return mean_y, var_y


# =====================================
#  4. 2D Example: Franke's Function
# =====================================
def franke_function(x, y):
    term1 = 0.75 * torch.exp(-(9*x-2)**2/4 - (9*y-2)**2/4)
    term2 = 0.75 * torch.exp(-(9*x+1)**2/49 - (9*y+1)/10)
    term3 = 0.5 * torch.exp(-(9*x-7)**2/4 - (9*y-3)**2/4)
    term4 = -0.2 * torch.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def run_2d_multioutput_regression():

    torch.manual_seed(0)

    # Generate 2D data
    n_points = 400
    x = torch.rand(n_points) * 2 - 1  # [-1, 1]
    y = torch.rand(n_points) * 2 - 1  # [-1, 1]
    X = torch.stack([x, y], dim=1)

    # Franke's function with noise
    f = franke_function(x, y) + 0.1 * torch.randn(n_points)

    # 2D outputs
    y1 = f + 0.1 * torch.randn(n_points)
    y2 = 0.5 * f + torch.cos(2 * math.pi * x) + 0.1 * torch.randn(n_points)
    Y = torch.stack([y1, y2], dim=1)      # (N,2)

    print(f"Training 2D multi-output SVGP with {n_points} points, output_dim=2")
    print(f"Input dimension: {X.shape[1]}")

    # Inducing points in 2D space
    n_inducing = 64
    Z_x = torch.linspace(-1, 1, int(np.sqrt(n_inducing)))
    Z_y = torch.linspace(-1, 1, int(np.sqrt(n_inducing)))
    Z_grid = torch.stack(torch.meshgrid(Z_x, Z_y, indexing='ij'), dim=-1).reshape(-1, 2)
    Z = Z_grid

    # Shared multi-dimensional kernel + Gaussian likelihood
    kernel = MultiDimRBFKernel(input_dim=2, ard=True)
    likelihood = GaussianLikelihood(output_dim=2, noise_std=0.1)
    model = IndependentMultiOutputSVGP(Z, kernel=kernel, likelihood=likelihood,
                                       output_dim=2, whiten=True)

    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    # DataLoader + tqdm (和 1D 版本保持一致风格)
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=100, shuffle=True)
    data_iter = iter(loader)

    N_total = X.shape[0]
    losses = []

    for step in trange(2000, desc="Training 2D SVGP"):
        try:
            Xb, Yb = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            Xb, Yb = next(data_iter)

        opt.zero_grad()
        elbo = model.elbo(Xb, Yb, N_total=N_total)
        loss = -elbo
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        losses.append(loss.item())

        if step % 200 == 0:
            print(f"Step {step}: ELBO = {elbo.item():.4f}, Loss = {loss.item():.4f}")

    # ================= Prediction on grid =================
    n_test = 30
    x_test = torch.linspace(-1, 1, n_test)
    y_test = torch.linspace(-1, 1, n_test)
    X_grid, Y_grid = torch.meshgrid(x_test, y_test, indexing='ij')
    X_test = torch.stack([X_grid.reshape(-1), Y_grid.reshape(-1)], dim=-1)  # (n_test^2, 2)

    with torch.no_grad():
        mu_test, var_test = model.predict(X_test)   # (900, 2)
    std_test = var_test.sqrt()

    f_grid = franke_function(X_grid, Y_grid)  # (30,30)
    y1_true_grid = f_grid
    y2_true_grid = 0.5 * f_grid + torch.cos(2 * math.pi * X_grid)

    mu1 = mu_test[:, 0].reshape(n_test, n_test)
    mu2 = mu_test[:, 1].reshape(n_test, n_test)
    std1 = std_test[:, 0].reshape(n_test, n_test)
    std2 = std_test[:, 1].reshape(n_test, n_test)

    # ================= Visualization: 2 rows × 3 columns =================
    fig = plt.figure(figsize=(18, 10))

    def set_axes_limits(ax):
        ax.set_xlabel('X1'); ax.set_ylabel('X2')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    # ---------- First line: Output 1 ----------
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.plot_surface(X_grid.numpy(), Y_grid.numpy(), y1_true_grid.numpy(),
                     cmap='viridis', alpha=0.8)
    # 明确：黑色叉号 = 训练数据
    ax1.scatter(X[:, 0].numpy(), X[:, 1].numpy(), y1.numpy(),
                c='k', s=20, alpha=0.6, marker='x', label='Training data')
    ax1.set_title('Output 1: True + Training')
    ax1.set_xlabel('X1'); ax1.set_ylabel('X2'); ax1.set_zlabel('y1')
    ax1.legend()

    ax2 = fig.add_subplot(232, projection='3d')
    ax2.plot_surface(X_grid.numpy(), Y_grid.numpy(), mu1.numpy(),
                     cmap='viridis', alpha=0.8)
    # 红色三角形 = 诱导点（不是测试数据）
    ax2.scatter(model.Z[:, 0].detach().numpy(), model.Z[:, 1].detach().numpy(),
                np.zeros_like(model.Z[:, 0].detach().numpy()),
                c='red', s=50, marker='^', label='Inducing points')
    ax2.set_title('Output 1: SVGP mean')
    set_axes_limits(ax2)
    ax2.set_xlabel('X1'); ax2.set_ylabel('X2'); ax2.set_zlabel('μ1')
    ax2.legend()

    ax3 = fig.add_subplot(233, projection='3d')
    ax3.plot_surface(X_grid.numpy(), Y_grid.numpy(), (mu1 + 2*std1).numpy(),
                     color='blue', alpha=0.3, label='+2σ')
    ax3.plot_surface(X_grid.numpy(), Y_grid.numpy(), (mu1 - 2*std1).numpy(),
                     color='red', alpha=0.3, label='-2σ')
    ax3.set_title('Output 1: Uncertainty (±2σ)')
    ax3.set_xlabel('X1'); ax3.set_ylabel('X2'); ax3.set_zlabel('y1')

    # ---------- Second line: Output 2 ----------
    ax4 = fig.add_subplot(234, projection='3d')
    ax4.plot_surface(X_grid.numpy(), Y_grid.numpy(), y2_true_grid.numpy(),
                     cmap='viridis', alpha=0.8)
    ax4.scatter(X[:, 0].numpy(), X[:, 1].numpy(), y2.numpy(),
                c='k', s=20, alpha=0.6, marker='x', label='Training data')
    ax4.set_title('Output 2: True + Training')
    ax4.set_xlabel('X1'); ax4.set_ylabel('X2'); ax4.set_zlabel('y2')
    ax4.legend()

    ax5 = fig.add_subplot(235, projection='3d')
    ax5.plot_surface(X_grid.numpy(), Y_grid.numpy(), mu2.numpy(),
                     cmap='viridis', alpha=0.8)
    ax5.scatter(model.Z[:, 0].detach().numpy(), model.Z[:, 1].detach().numpy(),
                np.zeros_like(model.Z[:, 0].detach().numpy()),
                c='red', s=50, marker='^', label='Inducing points')
    ax5.set_title('Output 2: SVGP mean')
    set_axes_limits(ax5)
    ax5.set_xlabel('X1'); ax5.set_ylabel('X2'); ax5.set_zlabel('μ2')
    ax5.legend()

    ax6 = fig.add_subplot(236, projection='3d')
    ax6.plot_surface(X_grid.numpy(), Y_grid.numpy(), (mu2 + 2*std2).numpy(),
                     color='blue', alpha=0.3, label='+2σ')
    ax6.plot_surface(X_grid.numpy(), Y_grid.numpy(), (mu2 - 2*std2).numpy(),
                     color='red', alpha=0.3, label='-2σ')
    ax6.set_title('Output 2: Uncertainty (±2σ)')
    ax6.set_xlabel('X1'); ax6.set_ylabel('X2'); ax6.set_zlabel('y2')

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "2d_prediction_v2.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {save_path}")
    plt.show()

    # --- Loss curves ---
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Iteration')
    plt.ylabel('Loss (-ELBO)')
    plt.title('2D Training Loss (Full)')
    save_path_loss = os.path.join(SAVE_DIR, "2d_loss_full_v2.png")
    plt.savefig(save_path_loss, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {save_path_loss}")
    plt.show()

    plt.figure(figsize=(8, 4))
    start_idx = 200
    if len(losses) > start_idx:
        plt.plot(range(start_idx, len(losses)), losses[start_idx:])
    else:
        plt.plot(losses)
    plt.title("2D Training Loss (Zoomed, skipping first 200 steps)")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (-ELBO)")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    save_path_loss_zoom = os.path.join(SAVE_DIR, "2d_loss_zoomed_v2.png")
    plt.savefig(save_path_loss_zoom, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {save_path_loss_zoom}")
    plt.show()

    # Print final hyperparameters
    print("\nFinal kernel & likelihood parameters:")
    var = torch.exp(model.kernel.log_variance).item()
    print(f"  Kernel variance: {var:.3f}")
    if model.kernel.ard:
        ls = torch.exp(model.kernel.log_lengthscales).detach().numpy()
        print(f"  Lengthscales: {ls}")
        print(f"  ARD importance: {1.0 / ls}")
    noise = torch.exp(model.likelihood.log_noise).detach().numpy()
    print(f"  Noise std per output: {noise}")

    return model


# =====================================
#  5. Higher-dimensional Example (3D)
# =====================================
def run_3d_multioutput_regression(num_steps=1500, output_dim=2):

    torch.manual_seed(0)

    n_points = 800
    X = torch.rand(n_points, 3) * 2 - 1            # (N,3)

    def target_function(x):
        return torch.sin(2 * x[:, 0]) + torch.cos(3 * x[:, 1]) + 0.5 * x[:, 2]

    f = target_function(X) + 0.1 * torch.randn(n_points)

    y1 = f + 0.1 * torch.randn(n_points)
    y2 = 0.7 * f + torch.sin(3 * X[:, 0]) + 0.1 * torch.randn(n_points)
    Y = torch.stack([y1, y2], dim=1)        # (N,2)

    print(f"Training 3D Multi-Output SVGP: N={n_points}, output_dim={output_dim}")

    coords = torch.linspace(-1, 1, 6)   # 6^3 = 216 inducing points
    Z = torch.stack(torch.meshgrid(coords, coords, coords), dim=-1).reshape(-1, 3)

    kernel = MultiDimRBFKernel(input_dim=3, ard=True)
    likelihood = GaussianLikelihood(output_dim=output_dim, noise_std=0.1)
    model = IndependentMultiOutputSVGP(Z, kernel=kernel, likelihood=likelihood,
                                       output_dim=output_dim, whiten=True)

    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    batch_size = 200
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data_iter = iter(loader)

    N_total = X.shape[0]
    losses = []

    for step in trange(num_steps, desc="Training 3D SVGP"):
        try:
            Xb, Yb = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            Xb, Yb = next(data_iter)

        opt.zero_grad()
        elbo = model.elbo(Xb, Yb, N_total=N_total)
        loss = -elbo
        loss.backward()
        opt.step()

        losses.append(loss.item())

        if step % 200 == 0:
            print(f"Step {step}: ELBO={elbo.item():.4f}, Loss={loss.item():.4f}")

    # ================= Prediction Grid ====================
    grid = 25
    x_ = torch.linspace(-1, 1, grid)
    y_ = torch.linspace(-1, 1, grid)
    z_ = torch.linspace(-1, 1, grid)

    Xg, Yg, Zg = torch.meshgrid(x_, y_, z_, indexing="ij")
    X_test = torch.stack([Xg.reshape(-1), Yg.reshape(-1), Zg.reshape(-1)], dim=1)

    with torch.no_grad():
        mu, var = model.predict(X_test)

    mu = mu.reshape(grid, grid, grid, output_dim)
    std = var.sqrt().reshape(grid, grid, grid, output_dim)

    # ================= Visualization ====================
    fig = plt.figure(figsize=(18, 10))

    for p in range(output_dim):
        base_idx = p * 3

        ax1 = fig.add_subplot(2, 3, base_idx + 1, projection="3d")
        F_true = target_function(torch.stack([
            Xg[:, :, grid//2].reshape(-1),
            Yg[:, :, grid//2].reshape(-1),
            Zg[:, :, grid//2].reshape(-1)
        ], dim=1)).reshape(grid, grid)

        ax1.plot_surface(Xg[:, :, 0].numpy(), Yg[:, :, 0].numpy(),
                         F_true.numpy(), cmap="viridis", alpha=0.8)
        ax1.set_title(f"Output {p+1}: True surface (z=0 slice)")
        ax1.set_xlim(-1, 1); ax1.set_ylim(-1, 1)

        ax2 = fig.add_subplot(2, 3, base_idx + 2, projection="3d")
        Mu_slice = mu[:, :, grid//2, p]
        ax2.plot_surface(Xg[:, :, 0].numpy(), Yg[:, :, 0].numpy(),
                         Mu_slice.numpy(), cmap="viridis", alpha=0.8)
        ax2.set_title(f"Output {p+1}: SVGP mean (z=0 slice)")
        ax2.set_xlim(-1, 1); ax2.set_ylim(-1, 1)

        ax3 = fig.add_subplot(2, 3, base_idx + 3, projection="3d")
        Std_slice = std[:, :, grid//2, p]
        ax3.plot_surface(Xg[:, :, 0].numpy(), Yg[:, :, 0].numpy(),
                         (Mu_slice + 2*Std_slice).numpy(), color='blue', alpha=0.3)
        ax3.plot_surface(Xg[:, :, 0].numpy(), Yg[:, :, 0].numpy(),
                         (Mu_slice - 2*Std_slice).numpy(), color='red', alpha=0.3)
        ax3.set_title(f"Output {p+1}: Uncertainty ±2σ (z=0 slice)")
        ax3.set_xlim(-1, 1); ax3.set_ylim(-1, 1)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "3d_prediction_v2.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {save_path}")
    plt.show()

    # --- Loss ---
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title("3D Training Loss (-ELBO)")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    save_path_loss = os.path.join(SAVE_DIR, "3d_loss_full_v2.png")
    plt.savefig(save_path_loss, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {save_path_loss}")
    plt.show()

    plt.figure(figsize=(8, 4))
    start_idx = 200
    if len(losses) > start_idx:
        plt.plot(range(start_idx, len(losses)), losses[start_idx:])
    else:
        plt.plot(losses)
    plt.title("3D Training Loss (Zoomed, skipping first 200 steps)")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (-ELBO)")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    save_path_loss_zoom = os.path.join(SAVE_DIR, "3d_loss_zoomed_v2.png")
    plt.savefig(save_path_loss_zoom, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {save_path_loss_zoom}")
    plt.show()

    return model


if __name__ == "__main__":
    print("=== 2D Regression Example ===")
    model_2d = run_2d_multioutput_regression()

    print("=== 3D Multi-Output Regression ===")
    model_3d = run_3d_multioutput_regression(
        num_steps=8000,
        output_dim=2,
    )

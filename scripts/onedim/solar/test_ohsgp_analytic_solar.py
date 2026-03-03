import argparse
import math
import os
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from hipposvgp.hippo import HiPPO_LegS
from hipposvgp.sgpr import HIPPO_OSGPR_VFE, HIPPO_SVGP
from scripts.onedim.solar.baseline import FullGP

ANALYTIC_FIXED_CFG = {
    "globalstart_wt_mode": "w",
    "analytic_psd_project": True,
    "analytic_robust_cholesky": True,
    "analytic_symmetrize": True,
    "analytic_base_jitter": 1e-6,
}


def setup():
    parser = argparse.ArgumentParser("OHSGPR Analytic (fixed-kernel style)")
    parser.add_argument("--seed_num", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device. 'auto' uses cuda if available, otherwise cpu.",
    )
    parser.add_argument("--data", type=str, default="solar")
    parser.add_argument("--kernel_type", type=str, default="se")
    parser.add_argument("--inducing_size", type=int, default=100)
    parser.add_argument("--rff_sample_size", type=int, default=1000)
    parser.add_argument("--rff_sample_size_qb", type=int, default=5000)
    parser.add_argument("--rff_sample_size_pred", type=int, default=5000)
    parser.add_argument("--step_size", type=float, default=2.45)
    parser.add_argument("--model_full_dir", type=str, default="./models")
    parser.add_argument("--fullgp_ckpt", type=str, default="steps_nll2000")
    parser.add_argument(
        "--hyper_source",
        type=str,
        default="load_fullgp",
        choices=["load_fullgp", "manual"],
    )
    parser.add_argument("--log_ls", type=float, default=1.4108)
    parser.add_argument("--log_sf", type=float, default=-0.5334)
    parser.add_argument("--log_noise", type=float, default=-4.5)
    parser.add_argument(
        "--split_replace",
        type=int,
        default=1,
        choices=[0, 1],
        help="1 keeps ODE script behavior (sampling with replacement); 0 is without replacement.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./ohsgp_analytic/solar_dataset_compare/analytic version",
    )
    parser.add_argument(
        "--analytic_z_type",
        type=str,
        default="bessel",
        choices=["bessel"],
        help="Use spherical-Bessel analytic feature construction.",
    )
    parser.add_argument(
        "--phase_origin_mode",
        type=str,
        default="global_start",
        choices=["global_start", "zero"],
        help="Phase origin used in analytic spherical-Bessel construction.",
    )
    parser.add_argument(
        "--spectral_sample_device",
        type=str,
        default="cpu",
        choices=["cpu", "match"],
        help=(
            "Device used to sample spectral frequencies w. "
            "'cpu' improves cross-device reproducibility; 'match' keeps legacy behavior."
        ),
    )
    return parser.parse_args()


def _spherical_bessel_j(lmax, x):
    """Pure-torch spherical Bessel j_0..j_lmax via stable Miller recurrence."""
    eps = 1e-8
    x_abs = x.abs()
    mask_nonzero = x_abs > eps
    x_safe = torch.where(mask_nonzero, x_abs, torch.full_like(x_abs, eps))

    max_x = float(x_abs.max().detach().cpu())
    n_start = int(lmax + max_x + 50)

    out_shape = (lmax + 1,) + tuple(x.shape)
    f = torch.zeros(out_shape, device=x.device, dtype=x.dtype)
    log_s = torch.zeros(out_shape, device=x.device, dtype=x.dtype)

    f_np1 = torch.zeros_like(x_safe)
    f_n = torch.ones_like(x_safe)
    cur_log_s = torch.zeros_like(x_safe)

    if x.dtype == torch.float64:
        large_thr = 1e100
        small_thr = 1e-100
    else:
        large_thr = 1e15
        small_thr = 1e-15

    for n in range(n_start, 0, -1):
        f_nm1 = ((2 * n + 1) / x_safe) * f_n - f_np1

        scale = torch.maximum(torch.maximum(f_nm1.abs(), f_n.abs()), f_np1.abs())
        scale = torch.where(
            scale > large_thr,
            scale,
            torch.where(scale < small_thr, torch.full_like(scale, small_thr), torch.ones_like(scale)),
        )

        f_nm1 = f_nm1 / scale
        f_n = f_n / scale
        f_np1 = f_np1 / scale
        cur_log_s = cur_log_s + torch.log(scale)

        if n - 1 <= lmax:
            f[n - 1] = f_nm1
            log_s[n - 1] = cur_log_s
        f_np1 = f_n
        f_n = f_nm1

    true_j0 = torch.sin(x_safe) / x_safe
    scale0 = true_j0 / (f[0] + 1e-300)
    corr = torch.exp(torch.clamp(log_s - log_s[0].unsqueeze(0), min=-700.0, max=700.0))
    j = f * corr * scale0.unsqueeze(0)

    parity = torch.where(
        (torch.arange(lmax + 1, device=x.device) % 2) == 0,
        torch.ones(lmax + 1, device=x.device, dtype=x.dtype),
        -torch.ones(lmax + 1, device=x.device, dtype=x.dtype),
    ).view(-1, *([1] * x.ndim))
    sign_is_pos = (x >= 0).unsqueeze(0)
    j = torch.where(sign_is_pos, j, j * parity)

    j_at_0 = torch.zeros_like(j)
    j_at_0[0] = 1.0
    j = torch.where(mask_nonzero.unsqueeze(0), j, j_at_0)
    return j


def _analytic_get_Z_bessel(model, w):
    rff_sample_size = w.shape[1]
    step_size = model.maxT / model.num_discrete_steps
    w_eff = w * step_size
    t_index = int(getattr(model, "prev_discrete_steps", 0)) + int(model.num_discrete_steps)
    kappa = 0.5 * w_eff * t_index
    phase_origin = torch.as_tensor(
        float(getattr(model, "phase_origin", 0.0)),
        dtype=model.dtype,
        device=model.device,
    )
    wt_mode = str(getattr(model, "globalstart_wt_mode", "w"))
    if wt_mode == "none":
        phase_origin_term = torch.zeros_like(w)
    elif wt_mode == "w_eff":
        phase_origin_term = w_eff * phase_origin
    else:
        phase_origin_term = w * phase_origin

    L = torch.arange(model.inducing_size, dtype=model.dtype, device=model.device)
    j = _spherical_bessel_j(model.inducing_size - 1, kappa).squeeze(1)
    pref = torch.sqrt(2.0 * L + 1.0)[:, None]
    phase = phase_origin_term + kappa + L[:, None] * math.pi / 2.0
    z_sin = pref * j * torch.sin(phase)
    z_cos = pref * j * torch.cos(phase)
    scale = (1.0 / rff_sample_size) ** 0.5
    Z = scale * torch.cat([z_sin, z_cos], dim=1).T

    time_stamps = model.start + (model.maxT / model.num_discrete_steps) * (
        1
        + torch.arange(model.num_discrete_steps, dtype=model.dtype, device=model.device)
    ).unsqueeze(1)
    return Z, time_stamps


def _analytic_get_Z(model, w):
    return _analytic_get_Z_bessel(model, w)


def _analytic_feature(w, X):
    rff_sample_size = w.shape[1]
    return (1.0 / rff_sample_size) ** 0.5 * torch.cat([torch.sin(w * X), torch.cos(w * X)], 1)


def _sample_w(model, num_rff, sample_device_mode="cpu"):
    """Sample spectral frequencies with optional CPU anchoring for reproducibility."""
    if sample_device_mode == "cpu" and model.kernel_type == "se":
        log_ls_cpu = model.Kernel.log_ls.detach().to(device="cpu", dtype=model.dtype)
        # Match legacy SE sampling path: torch.randn default dtype, then scale.
        w_cpu = (1.0 / torch.exp(log_ls_cpu)) * torch.randn(num_rff, device="cpu")
        return w_cpu.to(device=model.device, dtype=model.dtype).unsqueeze(0)
    return model.Kernel.sample_from_spectral(num_rff).unsqueeze(0)


class HIPPO_SVGP_ANALYTIC(HIPPO_SVGP):
    def __init__(self, *args, enable_analytic_runtime_guard=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._enable_analytic_runtime_guard = enable_analytic_runtime_guard
        self._forbidden_call_counts = {
            "kernel_forward": 0,
            "hippo_forward": 0,
            "exact_kfu": 0,
        }
        self._active_w = None
        self._original_kernel_forward = self.Kernel.forward
        self._original_hippo_forward = self.hippo.forward
        if self._enable_analytic_runtime_guard:
            self.Kernel.forward = self._guarded_kernel_forward
            self.hippo.forward = self._guarded_hippo_forward

    def _runtime_stack(self):
        return "".join(traceback.format_stack(limit=20))

    def _raise_forbidden_runtime_path(self, path_name):
        raise RuntimeError(
            "[HIPPO_SVGP_ANALYTIC] Forbidden exact/ODE path accessed: "
            f"{path_name}. Analytic backend must not call hippo()/odeint()/exact Kernel.\n"
            f"Call stack:\n{self._runtime_stack()}"
        )

    def _guarded_kernel_forward(self, *args, **kwargs):
        self._forbidden_call_counts["kernel_forward"] += 1
        self._raise_forbidden_runtime_path("Kernel.forward (exact kernel)")

    def _guarded_hippo_forward(self, *args, **kwargs):
        self._forbidden_call_counts["hippo_forward"] += 1
        self._raise_forbidden_runtime_path("hippo.forward (HiPPO/ODE path)")

    def get_forbidden_call_counts(self):
        return dict(self._forbidden_call_counts)

    def _as_input_tensor(self, X):
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        if X.ndim == 1:
            X = X.unsqueeze(-1)
        return X

    def _as_w_tensor(self, w):
        w = torch.as_tensor(w, dtype=self.dtype, device=self.device)
        if w.ndim == 1:
            w = w.unsqueeze(0)
        return w

    def get_Z(self, w):
        return _analytic_get_Z(self, w)

    def Kuu_se(self, w, b=None, reducevar=True):
        if not reducevar:
            self._raise_forbidden_runtime_path("Kuu_se(reducevar=False) -> super().Kuu_se")
        Z, _ = self.get_Z(w)
        return Z.T @ Z

    def KfuKff_rff_se(self, w, X, b=None, reducevar=True):
        if not reducevar:
            self._raise_forbidden_runtime_path(
                "KfuKff_rff_se(reducevar=False) -> super().KfuKff_rff_se"
            )
        Z, _ = self.get_Z(w)
        Z_f = _analytic_feature(w, X)
        Kfu = Z_f @ Z
        Kff = Z_f @ Z_f.T
        return Kff, Kfu

    def Kfu(self, X, w=None, b=None, reducevar=True):
        if self.kernel_type != "se":
            self._forbidden_call_counts["exact_kfu"] += 1
            self._raise_forbidden_runtime_path("Kfu(non-se) -> exact/ODE backend")
        if not reducevar:
            self._forbidden_call_counts["exact_kfu"] += 1
            self._raise_forbidden_runtime_path("Kfu(reducevar=False) -> non-analytic path")
        if w is None:
            if self._active_w is None:
                self._forbidden_call_counts["exact_kfu"] += 1
                raise RuntimeError(
                    "[HIPPO_SVGP_ANALYTIC] Kfu requires analytic spectral samples `w` "
                    "(either pass `w` directly or call from pred/ELBO/qb_opt)."
                )
            w = self._active_w
        X = self._as_input_tensor(X)
        _, Kfu = self.KfuKff_rff_se(self._as_w_tensor(w), X, b=b, reducevar=True)
        return torch.exp(self.Kernel.log_sf) * Kfu

    def ELBO(self, reducevar=True):
        if self.kernel_type != "se":
            raise NotImplementedError
        if not reducevar:
            self._raise_forbidden_runtime_path("ELBO(reducevar=False) -> non-analytic path")

        w = self._as_w_tensor(self.Kernel.sample_from_spectral(self.rff_sample_size))
        self._active_w = w
        try:
            Kuu = torch.exp(self.Kernel.log_sf) * self.Kuu_se(w, b=None, reducevar=True)
            Kfu = self.Kfu(self.X, w=w, b=None, reducevar=True)
            Kff, _ = self.KfuKff_rff_se(w, self.X, b=None, reducevar=True)
            Kff = torch.exp(self.Kernel.log_sf) * Kff
        finally:
            self._active_w = None

        N_input = self.X.shape[0]
        const = -0.5 * N_input * torch.log(
            torch.tensor(2 * np.pi, dtype=self.dtype, device=self.device)
        )

        eye = torch.eye(self.inducing_size, device=self.device, dtype=self.dtype)
        M_aux = Kuu + torch.exp(-self.log_noise) * (Kfu.T @ Kfu)
        chol_M_aux = torch.linalg.cholesky(
            M_aux + self.jitter * torch.mean(torch.diag(M_aux)).detach() * eye
        )
        chol_Kuu = torch.linalg.cholesky(
            Kuu + self.jitter * torch.mean(torch.diag(Kuu)).detach() * eye
        )
        log_det = (
            -0.5 * (N_input * self.log_noise)
            - torch.sum(torch.log(torch.diag(chol_M_aux)))
            + torch.sum(torch.log(torch.diag(chol_Kuu)))
        )

        quadratic = -0.5 * torch.exp(-self.log_noise) * self.y.T @ self.y
        v = torch.linalg.solve_triangular(chol_M_aux, Kfu.T @ self.y, upper=False)
        quadratic += 0.5 * torch.exp(-2 * self.log_noise) * v.T @ v

        b = torch.linalg.solve_triangular(chol_Kuu, Kfu.T, upper=False)
        trace_inner = torch.sum(torch.diag(Kff)) - torch.sum(b.pow(2))
        trace_inner = torch.clamp(trace_inner, min=0.0)
        trace = -0.5 * torch.exp(-self.log_noise) * trace_inner
        return const + log_det + quadratic + trace

    def pred(self, X, w, flag_noise=True, flag_diag=True, reducevar=True):
        X = self._as_input_tensor(X)
        if self.kernel_type != "se":
            raise NotImplementedError
        if not reducevar:
            self._raise_forbidden_runtime_path("pred(reducevar=False) -> non-analytic path")

        w = self._as_w_tensor(w)
        self._active_w = w
        try:
            Kuu = torch.exp(self.Kernel.log_sf) * self.Kuu_se(w, b=None, reducevar=True)
            Kff, Kfu = self.KfuKff_rff_se(w, X, b=None, reducevar=True)
            Kff = torch.exp(self.Kernel.log_sf) * Kff
            Kfu = torch.exp(self.Kernel.log_sf) * Kfu
            Kfu_train = self.Kfu(self.X, w=w, b=None, reducevar=True)
        finally:
            self._active_w = None

        eye_u = torch.eye(self.inducing_size, device=self.device, dtype=self.dtype)
        chol_Kuu = torch.linalg.cholesky(
            Kuu + self.jitter * torch.mean(torch.diag(Kuu)).detach() * eye_u
        )
        M_aux = Kuu + torch.exp(-self.log_noise) * (Kfu_train.T @ Kfu_train)
        chol_M_aux = torch.linalg.cholesky(
            M_aux + self.jitter * torch.mean(torch.diag(M_aux)).detach() * eye_u
        )

        v1 = torch.linalg.solve_triangular(chol_M_aux, Kfu.T, upper=False)
        v2 = torch.linalg.solve_triangular(chol_M_aux, Kfu_train.T @ self.y, upper=False)
        mean = torch.exp(-self.log_noise) * v1.T @ v2

        v3 = torch.linalg.solve_triangular(chol_Kuu, Kfu.T, upper=False)
        cov = Kff - v3.T @ v3 + v1.T @ v1
        cov = 0.5 * (cov + cov.T)

        if flag_noise:
            cov += torch.exp(self.log_noise) * torch.eye(cov.shape[0], device=self.device, dtype=self.dtype)
        if flag_diag:
            return mean, torch.diag(cov)
        return mean, cov

    def qb_opt(self, w):
        if self.kernel_type != "se":
            raise NotImplementedError

        w = self._as_w_tensor(w)
        Kuu = torch.exp(self.Kernel.log_sf) * self.Kuu_se(w, b=None, reducevar=True)
        Kuu += self.jitter * torch.mean(torch.diag(Kuu)).detach() * torch.eye(
            self.inducing_size, device=self.device, dtype=self.dtype
        )

        Kfu = Kuu.clone()
        self._active_w = w
        try:
            Kfu_train = self.Kfu(self.X, w=w, b=None, reducevar=True)
        finally:
            self._active_w = None

        M_aux = Kuu + torch.exp(-self.log_noise) * (Kfu_train.T @ Kfu_train)
        chol_M_aux = torch.linalg.cholesky(
            M_aux
            + self.jitter
            * torch.mean(torch.diag(M_aux)).detach()
            * torch.eye(self.inducing_size, device=self.device, dtype=self.dtype)
        )

        v1 = torch.linalg.solve_triangular(chol_M_aux, Kfu.T, upper=False)
        v2 = torch.linalg.solve_triangular(chol_M_aux, Kfu_train.T @ self.y, upper=False)
        mean = torch.exp(-self.log_noise) * v1.T @ v2
        cov = v1.T @ v1
        return mean, cov, Kuu


class HIPPO_OSGPR_VFE_ANALYTIC(HIPPO_OSGPR_VFE):
    def _cfg_bool(self, name, default):
        return bool(getattr(self, name, default))

    def _robust_cholesky(self, matrix, base_jitter=1e-6, retries=8):
        if not self._cfg_bool("analytic_robust_cholesky", True):
            return torch.linalg.cholesky(matrix)
        base_jitter = float(getattr(self, "analytic_base_jitter", base_jitter))
        eye = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
        diag_mean = torch.mean(torch.diag(matrix)).detach()
        if not torch.isfinite(diag_mean) or diag_mean <= 0:
            diag_mean = torch.tensor(1.0, dtype=matrix.dtype, device=matrix.device)
        for i in range(retries):
            factor = 10.0 ** i
            try:
                return torch.linalg.cholesky(matrix + base_jitter * factor * diag_mean * eye)
            except torch.linalg.LinAlgError:
                continue
        return torch.linalg.cholesky(matrix + 1e-3 * diag_mean * eye)

    def _project_psd(self, matrix, floor_scale=1e-8):
        if self._cfg_bool("analytic_symmetrize", True):
            sym = 0.5 * (matrix + matrix.T)
        else:
            sym = matrix
        if not self._cfg_bool("analytic_psd_project", True):
            return sym
        evals, evecs = torch.linalg.eigh(sym)
        diag_abs_mean = torch.mean(torch.abs(torch.diag(sym))).detach()
        floor = floor_scale * diag_abs_mean
        if not torch.isfinite(floor) or floor <= 0:
            floor = torch.tensor(floor_scale, dtype=matrix.dtype, device=matrix.device)
        evals = torch.clamp(evals, min=floor)
        return (evecs * evals.unsqueeze(0)) @ evecs.T

    def get_Z(self, w):
        return _analytic_get_Z(self, w)

    def Kuu_se(self, w):
        Z, _ = self.get_Z(w)
        Kuu = Z.T @ Z
        Kuuold = Z.T @ self.Z_old
        Kuolduold = self.Z_old.T @ self.Z_old
        return Kuu, Kuuold, Kuolduold

    def KfuKff_rff_se(self, w, X):
        Z, _ = self.get_Z(w)
        Z_f = _analytic_feature(w, X)
        Kfu = Z_f @ Z
        Kff = Z_f @ Z_f.T
        return Kff, Kfu

    def KuuKfuKff_rff_se(self, w, X):
        Z, _ = self.get_Z(w)
        Z_f = _analytic_feature(w, X)
        Kfu = Z_f @ Z
        Kff = Z_f @ Z_f.T
        Kuu = Z.T @ Z
        Kuuold = Z.T @ self.Z_old
        Kuolduold = self.Z_old.T @ self.Z_old
        return Kff, Kfu, Kuu, Kuuold, Kuolduold

    def _common_terms(self, w):
        Mb = self.inducing_size
        sigma2 = torch.exp(self.log_noise)
        sigma = torch.sqrt(sigma2)

        Saa = self.add_jitter(self._project_psd(self.Su_old))
        ma = self.mu_old

        Kff, Kfb, Kbb, Kba, Kaa_cur = self.KuuKfuKff_rff_se(w, self.X)
        Kff = torch.exp(self.Kernel.log_sf) * Kff
        Kfb = torch.exp(self.Kernel.log_sf) * Kfb
        Kbf = Kfb.T
        Kbb = torch.exp(self.Kernel.log_sf) * Kbb
        Kba = torch.exp(self.Kernel.log_sf) * Kba
        Kaa_cur = torch.exp(self.Kernel.log_sf) * Kaa_cur
        if self._cfg_bool("analytic_symmetrize", True):
            Kbb = self.add_jitter(0.5 * (Kbb + Kbb.T))
            Kaa_cur = self.add_jitter(0.5 * (Kaa_cur + Kaa_cur.T))
        else:
            Kbb = self.add_jitter(Kbb)
            Kaa_cur = self.add_jitter(Kaa_cur)
        Kaa = self.add_jitter(self._project_psd(self.Kaa_old))

        err = self.y

        Sainv_ma = torch.linalg.solve(Saa, ma)
        Sinv_y = self.y / sigma2
        c1 = Kbf @ Sinv_y
        c2 = Kba @ Sainv_ma
        c = c1 + c2

        Lb = self._robust_cholesky(Kbb)
        Lbinv_c = torch.linalg.solve_triangular(Lb, c, upper=False)
        Lbinv_Kba = torch.linalg.solve_triangular(Lb, Kba, upper=False)
        Lbinv_Kbf = torch.linalg.solve_triangular(Lb, Kbf, upper=False) / sigma
        d1 = Lbinv_Kbf @ Lbinv_Kbf.T

        LSa = self._robust_cholesky(Saa)
        Kab_Lbinv = Lbinv_Kba.T
        LSainv_Kab_Lbinv = torch.linalg.solve_triangular(LSa, Kab_Lbinv, upper=False)
        d2 = LSainv_Kab_Lbinv.T @ LSainv_Kab_Lbinv

        La = self._robust_cholesky(Kaa)
        Lainv_Kab_Lbinv = torch.linalg.solve_triangular(La, Kab_Lbinv, upper=False)
        d3 = Lainv_Kab_Lbinv.T @ Lainv_Kab_Lbinv

        D = torch.eye(Mb, device=self.device, dtype=self.dtype) + d1 + d2 - d3
        if self._cfg_bool("analytic_symmetrize", True):
            D = 0.5 * (D + D.T)
        D = self.add_jitter(D)
        LD = self._robust_cholesky(D)

        LDinv_Lbinv_c = torch.linalg.solve_triangular(LD, Lbinv_c, upper=False)

        return (
            Kff,
            Kbf,
            Kba,
            Kaa,
            Kaa_cur,
            La,
            Kbb,
            Lb,
            D,
            LD,
            Lbinv_Kba,
            LDinv_Lbinv_c,
            err,
            d1,
        )


def resolve_fixed_hypers(args, x, y, device):
    if args.hyper_source == "manual":
        return args.log_ls, args.log_sf, args.log_noise, "manual"

    ckpt_path = os.path.join(args.model_full_dir, args.fullgp_ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"FullGP checkpoint not found: {ckpt_path}")

    full_gp = FullGP(X=x, y=y, device=device, kernel_type=args.kernel_type).to(device)
    full_gp.X = full_gp.X.to(device=device, dtype=torch.float64)
    full_gp.y = full_gp.y.to(device=device, dtype=torch.float64)
    full_gp.load_state_dict(torch.load(ckpt_path, map_location=device))
    return (
        full_gp.Kernel.log_ls.item(),
        full_gp.Kernel.log_sf.item(),
        full_gp.log_noise.item(),
        f"fullgp_loaded:{ckpt_path}",
    )


if __name__ == "__main__":
    args = setup()
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("`--device cuda` requested but CUDA is not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] using {device}")

    np.random.seed(args.seed_num)
    torch.manual_seed(args.seed_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed_num)

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    with open(metrics_path, "w") as f:
        f.write("task,rmse,nll\n")

    with torch.no_grad():
        if args.data == "solar":
            data_train = np.load("data/solar_data_train.npy")
            data_test = np.load("data/solar_data_test.npy")
            data = np.vstack([data_train, data_test])
            data = data[np.argsort(data[:, 0])]
            ind_test = np.random.choice(
                np.arange(data.shape[0]), size=91, replace=bool(args.split_replace)
            )
            ind_train = []
            for i in range(data.shape[0]):
                if i in ind_test:
                    continue
                ind_train.append(i)
            data_test = data[ind_test]
            data_train = data[ind_train]
            data_train = data_train[np.argsort(data_train[:, 0])]
            x_test = torch.tensor(np.arange(-200.0, 200.0, 2.0).reshape((-1, 1))).to(device)
        else:
            data = np.load("./baseband.npy")
            y_total = data[:18000]
            x_total = np.expand_dims((1 + np.arange(18000)) / 1800, 1)
            data = np.concatenate([x_total, y_total], 1)
            data_train = data[::2]
            data_test = data[1::2]
            x_test = torch.tensor(np.arange(-0.02, 10.02, 0.002).reshape((-1, 1))).to(device)

        x = torch.tensor(np.expand_dims(data_train[:, 0], 1)).to(device)
        y = torch.tensor(np.expand_dims(data_train[:, 1], 1)).to(device)
        x_t = np.expand_dims(data_test[:, 0], 1)
        y_t = np.expand_dims(data_test[:, 1], 1)

        x_list = []
        y_list = []
        split_size = x.shape[0] // 10
        for i in range(9):
            x_list.append(x[i * split_size : (i + 1) * split_size])
            y_list.append(y[i * split_size : (i + 1) * split_size])
        x_list.append(x[9 * split_size :])
        y_list.append(y[9 * split_size :])

        log_ls, log_sf, log_noise, hyper_source = resolve_fixed_hypers(
            args,
            x,
            y,
            device,
        )
        print(
            "[Hyper] source={}, log_ls={:.4f}, log_sf={:.4f}, log_noise={:.4f}".format(
                hyper_source, log_ls, log_sf, log_noise
            )
        )
        print(
            "[Analytic] z_type={}, split_replace={}, seed={}".format(
                args.analytic_z_type, args.split_replace, args.seed_num
            )
        )
        if args.phase_origin_mode == "global_start":
            phase_origin = torch.min(x_list[0]).item()
        else:
            phase_origin = 0.0
        print(
            "[Analytic] phase_origin_mode={}, phase_origin={:.6f}".format(
                args.phase_origin_mode, phase_origin
            )
        )

        hippo = HiPPO_LegS(args.inducing_size, device)
        prev_discrete_steps = 0.0

        start_t = time.time()
        maxT = torch.max(x_list[0]).item() - torch.min(x_list[0]).item()
        start = torch.min(x_list[0]).item()
        num_discrete_steps = math.ceil(maxT / args.step_size)
        end = start + num_discrete_steps * args.step_size
        maxT = end - start

        model1 = HIPPO_SVGP_ANALYTIC(
            X=x_list[0],
            y=y_list[0],
            maxT=maxT,
            inducing_size=args.inducing_size,
            rff_sample_size=args.rff_sample_size,
            num_discrete_steps=num_discrete_steps,
            start=start,
            device=device,
            hippo=hippo,
            kernel_type=args.kernel_type,
        ).to(device)
        model1.analytic_z_type = args.analytic_z_type
        model1.phase_origin = phase_origin
        model1.globalstart_wt_mode = ANALYTIC_FIXED_CFG["globalstart_wt_mode"]
        prev_discrete_steps += num_discrete_steps

        model1.log_noise = nn.Parameter(
            torch.tensor(np.array([log_noise]), dtype=torch.float64, device=device),
            requires_grad=False,
        )
        model1.Kernel.log_ls = nn.Parameter(
            torch.tensor(np.array([log_ls]), dtype=torch.float64, device=device),
            requires_grad=False,
        )
        model1.Kernel.log_sf = nn.Parameter(
            torch.tensor(np.array([log_sf]), dtype=torch.float64, device=device),
            requires_grad=False,
        )

        w = _sample_w(
            model1,
            args.rff_sample_size_qb,
            sample_device_mode=args.spectral_sample_device,
        )

        y_mean, y_cov = model1.pred(x_test, w=w)
        y_mean = y_mean.detach().cpu().numpy()
        y_cov = y_cov.detach().cpu().numpy()

        plt.figure(figsize=(9, 3))
        plt.plot(x_test.detach().cpu().numpy().flatten(), y_mean.flatten(), label="SGP1", color="orange")
        plt.fill_between(
            x_test.detach().cpu().numpy().flatten(),
            (y_mean.flatten() - 1.96 * np.sqrt(np.abs(y_cov).flatten())),
            (y_mean.flatten() + 1.96 * np.sqrt(np.abs(y_cov).flatten())),
            alpha=0.3,
            color="orange",
        )
        if args.data == "solar":
            plt.scatter(
                x.detach().cpu().numpy().flatten(),
                y.detach().cpu().numpy().flatten(),
                color="blue",
                label="train",
                marker=".",
            )
            plt.scatter(x_t.flatten(), y_t.flatten(), color="black", label="test", marker=".")
        else:
            plt.scatter(
                x.detach().cpu().numpy().flatten(),
                y.detach().cpu().numpy().flatten(),
                color="blue",
                label="train",
                marker=".",
                s=1,
            )
            plt.scatter(x_t.flatten(), y_t.flatten(), color="black", label="test", marker=".", s=1)

        plt.axvline(x=torch.max(x_list[0]).item(), color="grey", linestyle="--", label="splits")
        for i in range(1, 10):
            plt.axvline(x=torch.max(x_list[i]).item(), color="grey", linestyle="--")
        plt.legend(bbox_to_anchor=(1.01, 0.75), loc="upper left")
        plt.savefig(
            os.path.join(
                args.out_dir,
                "hsgp1_analytic_{}_{}z.pdf".format(args.kernel_type, args.inducing_size),
            ),
            format="pdf",
            bbox_inches="tight",
        )
        plt.close()

        mu_old, Su_old, Kaa_old = model1.qb_opt(w)
        Kaa_old = Kaa_old.detach()
        mu_old = mu_old.detach()
        Su_old = Su_old.detach()
        Z_old, _ = model1.get_Z(w)
        Z_old = Z_old.detach()

        nll, rmse = model1.nll_rmse(
            torch.tensor(x_t[x_t.flatten() < torch.max(x_list[0]).item()]).to(device),
            torch.tensor(y_t[x_t.flatten() < torch.max(x_list[0]).item()]).to(device),
            w=w,
        )
        print("HGP1, rmse = {:.4f}, nll = {:.4f}".format(rmse, nll))
        with open(metrics_path, "a") as f:
            f.write("1,{:.8f},{:.8f}\n".format(rmse.item(), nll.item()))

        for i in range(1, 10):
            start = end
            end = torch.max(x_list[i]).item()
            num_discrete_steps = math.ceil((end - start) / args.step_size)
            end = start + num_discrete_steps * args.step_size

            cur_model = HIPPO_OSGPR_VFE_ANALYTIC(
                X=x_list[i],
                y=y_list[i],
                prev_discrete_steps=prev_discrete_steps,
                mu_old=mu_old,
                Su_old=Su_old,
                Kaa_old=Kaa_old,
                Z_old=Z_old,
                end=end,
                start=start,
                rff_sample_size=args.rff_sample_size,
                hippo=hippo,
                num_discrete_steps=num_discrete_steps,
                inducing_size=args.inducing_size,
                device=device,
                kernel_type=args.kernel_type,
            ).to(device)
            cur_model.analytic_z_type = args.analytic_z_type
            cur_model.phase_origin = phase_origin
            cur_model.globalstart_wt_mode = ANALYTIC_FIXED_CFG["globalstart_wt_mode"]
            cur_model.analytic_psd_project = ANALYTIC_FIXED_CFG["analytic_psd_project"]
            cur_model.analytic_robust_cholesky = ANALYTIC_FIXED_CFG["analytic_robust_cholesky"]
            cur_model.analytic_symmetrize = ANALYTIC_FIXED_CFG["analytic_symmetrize"]
            cur_model.analytic_base_jitter = ANALYTIC_FIXED_CFG["analytic_base_jitter"]
            prev_discrete_steps += num_discrete_steps

            cur_model.log_noise = nn.Parameter(
                torch.tensor(np.array([log_noise]), dtype=torch.float64, device=device),
                requires_grad=False,
            )
            cur_model.Kernel.log_ls = nn.Parameter(
                torch.tensor(np.array([log_ls]), dtype=torch.float64, device=device),
                requires_grad=False,
            )
            cur_model.Kernel.log_sf = nn.Parameter(
                torch.tensor(np.array([log_sf]), dtype=torch.float64, device=device),
                requires_grad=False,
            )

            y_mean, y_cov = cur_model.pred(x_test, w=w)
            y_mean = y_mean.detach().cpu().numpy()
            y_cov = y_cov.detach().cpu().numpy()

            plt.figure(figsize=(9, 3))
            plt.plot(
                x_test.detach().cpu().numpy().flatten(),
                y_mean.flatten(),
                label="SGP{}".format(i + 1),
                color="orange",
            )
            plt.fill_between(
                x_test.detach().cpu().numpy().flatten(),
                (y_mean.flatten() - 1.96 * np.sqrt(np.abs(y_cov).flatten())),
                (y_mean.flatten() + 1.96 * np.sqrt(np.abs(y_cov).flatten())),
                alpha=0.3,
                color="orange",
            )
            if args.data == "solar":
                plt.scatter(
                    x.detach().cpu().numpy().flatten(),
                    y.detach().cpu().numpy().flatten(),
                    color="blue",
                    label="train",
                    marker=".",
                )
                plt.scatter(x_t.flatten(), y_t.flatten(), color="black", label="test", marker=".")
            else:
                plt.scatter(
                    x.detach().cpu().numpy().flatten(),
                    y.detach().cpu().numpy().flatten(),
                    color="blue",
                    label="train",
                    marker=".",
                    s=1,
                )
                plt.scatter(x_t.flatten(), y_t.flatten(), color="black", label="test", marker=".", s=1)

            plt.axvline(x=torch.max(x_list[0]).item(), color="grey", linestyle="--", label="splits")
            for j in range(1, 10):
                plt.axvline(x=torch.max(x_list[j]).item(), color="grey", linestyle="--")
            plt.legend(bbox_to_anchor=(1.01, 0.75), loc="upper left")
            plt.savefig(
                os.path.join(
                    args.out_dir,
                    "hsgp{}_analytic_{}_{}z.pdf".format(i + 1, args.kernel_type, args.inducing_size),
                ),
                format="pdf",
                bbox_inches="tight",
            )
            plt.close()

            mu_old, Su_old, Kaa_old = cur_model.qb_opt(w)
            Kaa_old = Kaa_old.detach()
            mu_old = mu_old.detach()
            Su_old = Su_old.detach()
            Z_old, _ = cur_model.get_Z(w)
            Z_old = Z_old.detach()

            nll, rmse = cur_model.nll_rmse(
                torch.tensor(x_t[x_t.flatten() < torch.max(x_list[i]).item()]).to(device),
                torch.tensor(y_t[x_t.flatten() < torch.max(x_list[i]).item()]).to(device),
                w=w,
            )
            print("HGP{}, rmse = {:.4f}, nll = {:.4f}".format(i + 1, rmse, nll))
            with open(metrics_path, "a") as f:
                f.write("{},{:.8f},{:.8f}\n".format(i + 1, rmse.item(), nll.item()))

        end_t = time.time()
        print("M={}, time={:.3f}s".format(args.inducing_size, end_t - start_t))

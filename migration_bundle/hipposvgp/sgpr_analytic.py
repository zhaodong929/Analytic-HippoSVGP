import traceback

import numpy as np
import torch

from hipposvgp.analytic_backend import analytic_feature, analytic_get_z_bessel
from hipposvgp.sgpr import HIPPO_OSGPR_VFE, HIPPO_SVGP


ANALYTIC_FIXED_CFG = {
    "globalstart_wt_mode": "w",
    "analytic_psd_project": True,
    "analytic_robust_cholesky": True,
    "analytic_symmetrize": True,
    "analytic_base_jitter": 1e-6,
}


def sample_w(model, num_rff, sample_device_mode="cpu"):
    """Sample spectral frequencies with optional CPU anchoring for reproducibility."""
    if sample_device_mode == "cpu" and model.kernel_type == "se":
        log_ls_cpu = model.Kernel.log_ls.detach().to(device="cpu", dtype=model.dtype)
        w_cpu = (1.0 / torch.exp(log_ls_cpu)) * torch.randn(num_rff, device="cpu")
        return w_cpu.to(device=model.device, dtype=model.dtype).unsqueeze(0)
    return model.Kernel.sample_from_spectral(num_rff).unsqueeze(0)


def apply_analytic_fixed_config(model, phase_origin=0.0):
    model.phase_origin = phase_origin
    model.globalstart_wt_mode = ANALYTIC_FIXED_CFG["globalstart_wt_mode"]
    model.analytic_psd_project = ANALYTIC_FIXED_CFG["analytic_psd_project"]
    model.analytic_robust_cholesky = ANALYTIC_FIXED_CFG["analytic_robust_cholesky"]
    model.analytic_symmetrize = ANALYTIC_FIXED_CFG["analytic_symmetrize"]
    model.analytic_base_jitter = ANALYTIC_FIXED_CFG["analytic_base_jitter"]
    return model


def _analytic_get_Z(model, w):
    phase_origin = float(getattr(model, "phase_origin", 0.0))
    wt_mode = str(getattr(model, "globalstart_wt_mode", "w"))
    return analytic_get_z_bessel(
        inducing_size=model.inducing_size,
        w=w,
        start=model.start,
        max_t=model.maxT,
        num_discrete_steps=model.num_discrete_steps,
        prev_discrete_steps=getattr(model, "prev_discrete_steps", 0),
        dtype=model.dtype,
        device=model.device,
        phase_origin=phase_origin,
        globalstart_wt_mode=wt_mode,
    )


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
            f"{path_name}. Analytic backend must not call hippo()/exact Kernel.\n"
            f"Call stack:\n{self._runtime_stack()}"
        )

    def _guarded_kernel_forward(self, *args, **kwargs):
        self._forbidden_call_counts["kernel_forward"] += 1
        self._raise_forbidden_runtime_path("Kernel.forward (exact kernel)")

    def _guarded_hippo_forward(self, *args, **kwargs):
        self._forbidden_call_counts["hippo_forward"] += 1
        self._raise_forbidden_runtime_path("hippo.forward (HiPPO path)")

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
            self._raise_forbidden_runtime_path("Kuu_se(reducevar=False)")
        Z, _ = self.get_Z(w)
        return Z.T @ Z

    def KfuKff_rff_se(self, w, X, b=None, reducevar=True):
        if not reducevar:
            self._raise_forbidden_runtime_path("KfuKff_rff_se(reducevar=False)")
        Z, _ = self.get_Z(w)
        Z_f = analytic_feature(w, X)
        Kfu = Z_f @ Z
        Kff = Z_f @ Z_f.T
        return Kff, Kfu

    def Kfu(self, X, w=None, b=None, reducevar=True):
        if self.kernel_type != "se":
            self._forbidden_call_counts["exact_kfu"] += 1
            self._raise_forbidden_runtime_path("Kfu(non-se)")
        if not reducevar:
            self._forbidden_call_counts["exact_kfu"] += 1
            self._raise_forbidden_runtime_path("Kfu(reducevar=False)")
        if w is None:
            if self._active_w is None:
                self._forbidden_call_counts["exact_kfu"] += 1
                raise RuntimeError(
                    "[HIPPO_SVGP_ANALYTIC] Kfu requires analytic spectral samples `w`."
                )
            w = self._active_w
        X = self._as_input_tensor(X)
        _, Kfu = self.KfuKff_rff_se(self._as_w_tensor(w), X, b=b, reducevar=True)
        return torch.exp(self.Kernel.log_sf) * Kfu

    def ELBO(self, reducevar=True):
        if self.kernel_type != "se":
            raise NotImplementedError
        if not reducevar:
            self._raise_forbidden_runtime_path("ELBO(reducevar=False)")

        w = self._as_w_tensor(self.Kernel.sample_from_spectral(self.rff_sample_size))
        self._active_w = w
        try:
            Kuu = torch.exp(self.Kernel.log_sf) * self.Kuu_se(w, b=None, reducevar=True)
            Kfu = self.Kfu(self.X, w=w, b=None, reducevar=True)
            Kff, _ = self.KfuKff_rff_se(w, self.X, b=None, reducevar=True)
            Kff = torch.exp(self.Kernel.log_sf) * Kff
        finally:
            self._active_w = None

        n_input = self.X.shape[0]
        const = -0.5 * n_input * torch.log(
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
            -0.5 * (n_input * self.log_noise)
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
            self._raise_forbidden_runtime_path("pred(reducevar=False)")

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
            cov += torch.exp(self.log_noise) * torch.eye(
                cov.shape[0], device=self.device, dtype=self.dtype
            )
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
        Z_f = analytic_feature(w, X)
        Kfu = Z_f @ Z
        Kff = Z_f @ Z_f.T
        return Kff, Kfu

    def KuuKfuKff_rff_se(self, w, X):
        Z, _ = self.get_Z(w)
        Z_f = analytic_feature(w, X)
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

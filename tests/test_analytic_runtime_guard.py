import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

from hipposvgp.hippo import HiPPO_LegS


def _load_analytic_class():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    module_path = (
        repo_root
        / "scripts"
        / "onedim"
        / "solar"
        / "test_ohsgp_analytic_solar.py"
    )
    spec = importlib.util.spec_from_file_location("test_ohsgp_analytic_solar", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.HIPPO_SVGP_ANALYTIC


HIPPO_SVGP_ANALYTIC = _load_analytic_class()


def _build_toy_analytic_model():
    torch.manual_seed(0)
    np.random.seed(0)

    x = np.linspace(0.0, 5.0, 12, dtype=np.float64).reshape(-1, 1)
    y = np.sin(x)

    inducing_size = 6
    hippo = HiPPO_LegS(N=inducing_size, device="cpu", max_length=256)
    model = HIPPO_SVGP_ANALYTIC(
        X=x,
        y=y,
        maxT=5.0,
        inducing_size=inducing_size,
        rff_sample_size=64,
        num_discrete_steps=12,
        device="cpu",
        start=0.0,
        hippo=hippo,
        kernel_type="se",
    )
    model.phase_origin = 0.0
    return model


def test_analytic_kfu_matches_analytic_feature_projection():
    model = _build_toy_analytic_model()
    w = model.Kernel.sample_from_spectral(model.rff_sample_size).unsqueeze(0)
    x_eval = model.X[:5]

    kff, kfu_raw = model.KfuKff_rff_se(w, x_eval, b=None, reducevar=True)
    kfu = model.Kfu(x_eval, w=w, b=None, reducevar=True)

    assert kff.shape == (x_eval.shape[0], x_eval.shape[0])
    torch.testing.assert_close(kfu, torch.exp(model.Kernel.log_sf) * kfu_raw)


def test_analytic_pred_elbo_qb_opt_do_not_touch_exact_or_hippo_path():
    model = _build_toy_analytic_model()
    w = model.Kernel.sample_from_spectral(model.rff_sample_size).unsqueeze(0)

    pred_mean, pred_var = model.pred(model.X.numpy(), w=w, flag_noise=True, flag_diag=True)
    elbo = model.ELBO()
    qb_mean, qb_cov, qb_kuu = model.qb_opt(w)

    assert pred_mean.shape == (model.X.shape[0], 1)
    assert pred_var.shape == (model.X.shape[0],)
    assert torch.isfinite(elbo).all()
    assert qb_mean.shape == (model.inducing_size, 1)
    assert qb_cov.shape == (model.inducing_size, model.inducing_size)
    assert qb_kuu.shape == (model.inducing_size, model.inducing_size)

    counts = model.get_forbidden_call_counts()
    assert counts["kernel_forward"] == 0
    assert counts["hippo_forward"] == 0
    assert counts["exact_kfu"] == 0


def test_analytic_kuu_is_cholesky_factorizable_with_jitter():
    model = _build_toy_analytic_model()
    w = model.Kernel.sample_from_spectral(model.rff_sample_size).unsqueeze(0)

    kuu = torch.exp(model.Kernel.log_sf) * model.Kuu_se(w, b=None, reducevar=True)
    jittered = kuu + model.jitter * torch.mean(torch.diag(kuu)).detach() * torch.eye(
        model.inducing_size, dtype=model.dtype, device=model.device
    )
    chol = torch.linalg.cholesky(jittered)

    assert chol.shape == (model.inducing_size, model.inducing_size)
    assert torch.isfinite(chol).all()
    assert kuu.shape == (model.inducing_size, model.inducing_size)

import math

import torch


def stable_spherical_bessel_j(lmax, x):
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
            torch.where(
                scale < small_thr,
                torch.full_like(scale, small_thr),
                torch.ones_like(scale),
            ),
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


def compute_time_stamps(start, max_t, num_discrete_steps, dtype, device):
    return start + (max_t / num_discrete_steps) * (
        1 + torch.arange(num_discrete_steps, dtype=dtype, device=device)
    ).unsqueeze(1)


def _expand_scale(scale, dtype, device):
    if isinstance(scale, torch.Tensor):
        return scale.to(device=device, dtype=dtype)
    return torch.as_tensor(scale, device=device, dtype=dtype)


def analytic_feature(w, x, scale=None):
    num_rff = w.shape[-1]
    if scale is None:
        scale = (1.0 / num_rff) ** 0.5
    scale = _expand_scale(scale, dtype=w.dtype, device=w.device)

    features = torch.cat([torch.sin(w * x), torch.cos(w * x)], dim=1)
    if isinstance(scale, torch.Tensor) and scale.ndim > 0:
        scale_full = torch.cat([scale, scale], dim=0)
        return features * scale_full.unsqueeze(0)
    return scale * features


def analytic_get_z_bessel(
    *,
    inducing_size,
    w,
    start,
    max_t,
    num_discrete_steps,
    prev_discrete_steps,
    dtype,
    device,
    phase_origin=0.0,
    globalstart_wt_mode="w",
    scale=None,
):
    num_rff = w.shape[-1]
    if scale is None:
        scale = (1.0 / num_rff) ** 0.5
    scale = _expand_scale(scale, dtype=dtype, device=device)

    step_size = max_t / num_discrete_steps
    w_eff = w * step_size
    t_index = int(prev_discrete_steps) + int(num_discrete_steps)
    kappa = 0.5 * w_eff * t_index

    phase_origin = torch.as_tensor(float(phase_origin), dtype=dtype, device=device)
    if globalstart_wt_mode == "none":
        phase_origin_term = torch.zeros_like(w)
    elif globalstart_wt_mode == "w_eff":
        phase_origin_term = w_eff * phase_origin
    else:
        phase_origin_term = w * phase_origin

    L = torch.arange(inducing_size, dtype=dtype, device=device)
    j = stable_spherical_bessel_j(inducing_size - 1, kappa).squeeze(1)
    pref = torch.sqrt(2.0 * L + 1.0)[:, None]
    phase = phase_origin_term + kappa + L[:, None] * math.pi / 2.0
    z_sin = pref * j * torch.sin(phase)
    z_cos = pref * j * torch.cos(phase)

    z = torch.cat([z_sin, z_cos], dim=1).T
    if isinstance(scale, torch.Tensor) and scale.ndim > 0:
        scale_full = torch.cat([scale, scale], dim=0)
        z = scale_full[:, None] * z
    else:
        z = scale * z

    time_stamps = compute_time_stamps(start, max_t, num_discrete_steps, dtype, device)
    return z, time_stamps

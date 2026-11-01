# Copyright (c) 2026 The GradSplat Authors.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Activation of the raw Gaussian parameters.

Every input pixel carries one Gaussian with 23 raw channels (spherical harmonics of degree 1):
    opacity (1) | scale (3) | quaternion xyzw (4) | SH coefficients (12) | center offset (3)
"""
import torch
import torch.nn.functional as F

SH_C0 = 0.28209479177387814


def n_sh(sh_degree):
    return (sh_degree + 1) ** 2


def gauss_channels(sh_degree):
    return 1 + 3 + 4 + 3 * n_sh(sh_degree) + 3


def split_params(gp, sh_degree):
    ns = n_sh(sh_degree)
    i = 0
    op = gp[..., i:i + 1]; i += 1
    sc = gp[..., i:i + 3]; i += 3
    qt = gp[..., i:i + 4]; i += 4
    sh = gp[..., i:i + 3 * ns]; i += 3 * ns
    dm = gp[..., i:i + 3]; i += 3
    return dict(opacity=op, scale=sc, quat=qt, sh=sh, dmean=dm, n_sh=ns)


def rgb_to_sh_dc(rgb):
    return (rgb - 0.5) / SH_C0


def sh_dc_to_rgb(sh_dc):
    return SH_C0 * sh_dc + 0.5


def apply_rot_delta(quats, omega):
    """Compose the axis-angle increment omega (..., 3) with the quaternions (..., 4, xyzw)."""
    th = omega.norm(dim=-1, keepdim=True)
    half = 0.5 * th
    k = torch.where(th > 1e-6, torch.sin(half) / th.clamp_min(1e-6), 0.5 * torch.ones_like(th))
    dq = torch.cat([omega * k, torch.cos(half)], dim=-1)
    x1, y1, z1, w1 = dq.unbind(-1)
    x2, y2, z2, w2 = quats.unbind(-1)
    return F.normalize(torch.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], dim=-1), dim=-1)


def activate_shape(raw, amp=4.0, rot_amp=1.0):
    """Raw shape output (..., 6) -> zero-sum log-anisotropy delta (..., 3) and rotation increment omega (..., 3)."""
    a = amp * torch.tanh(raw[..., :3])
    a = a - a.mean(dim=-1, keepdim=True)
    w = rot_amp * torch.tanh(raw[..., 3:])
    return a, w


def assemble(gauss_params, world_means_base, rgb, sh_degree, scale_max=0.3, dmean_scale=0.05,
             scale_mult=0.001, aniso=None, rot_delta=None):
    """Activated Gaussians in the world frame.

    gauss_params (..., H, W, C) raw channels, world_means_base (..., H, W, 3) unprojected pixels,
    rgb (..., 3, H, W) input images in [0, 1]. aniso / rot_delta are the shape correction (None: the base shape).
    Returns means (..., H, W, 3), scales (..., H, W, 3), quats (..., H, W, 4, xyzw), opacities (..., H, W)
    and sh (..., H, W, n_sh, 3)."""
    p = split_params(gauss_params, sh_degree)
    opac = p["opacity"].squeeze(-1).sigmoid()
    scales = scale_mult * F.softplus(p["scale"])
    if aniso is not None:
        scales = scales * torch.exp(aniso)
    scales = scales.clamp_max(scale_max)
    quats = F.normalize(p["quat"], dim=-1)
    if rot_delta is not None:
        quats = apply_rot_delta(quats, rot_delta)
    ns = p["n_sh"]
    sh = p["sh"].reshape(*p["sh"].shape[:-1], ns, 3)
    rgb_hw = rgb.movedim(-3, -1)
    anchor = rgb_to_sh_dc(rgb_hw)[..., None, :]
    sh_dc = sh[..., 0:1, :] + anchor
    sh = torch.cat([sh_dc, sh[..., 1:, :]], dim=-2)
    dmean = dmean_scale * torch.tanh(p["dmean"])
    means = world_means_base + dmean
    return dict(means=means, scales=scales, quats=quats, opacities=opac, sh=sh)


def flatten_gaussians(g):
    """(B, V, H, W, ...) -> (B, V*H*W, ...), the layout of the renderer."""
    B = g["means"].shape[0]

    def fl(x, d):
        return x.reshape(B, -1, *x.shape[-d:]) if d > 0 else x.reshape(B, -1)
    return dict(
        means=fl(g["means"], 1),
        scales=fl(g["scales"], 1),
        quats=fl(g["quats"], 1),
        opacities=fl(g["opacities"], 0),
        sh=fl(g["sh"], 2),
    )


def select(gaussians, b=0):
    """The Gaussians of one scene of a batch, as (N, ...) tensors."""
    return {k: v[b] for k, v in gaussians.items()}

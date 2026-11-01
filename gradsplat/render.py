# Copyright (c) 2026 The GradSplat Authors.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Rasterization with gsplat."""
import torch

try:
    from gsplat import rasterization as _rasterization
    _GSPLAT_ERROR = None
except Exception as e:                                   # noqa: BLE001
    _rasterization = None
    _GSPLAT_ERROR = e


def _denorm_K(K_norm, H, W):
    K = K_norm.clone()
    K[..., 0, :] = K[..., 0, :] * W
    K[..., 1, :] = K[..., 1, :] * H
    return K


def gsplat_available():
    return _rasterization is not None


def render(gaussians, c2w, K_norm, H, W, background=None, eps2d=None):
    """Render Gaussians into cameras.

    gaussians: dict of means (B, N, 3), scales (B, N, 3), quats (B, N, 4, xyzw), opacities (B, N), sh (B, N, K, 3).
    c2w (B, V, 4, 4) camera-to-world, K_norm (B, V, 3, 3) intrinsics normalized by W and H.
    Returns colors (B, V, 3, H, W) clamped to [0, 1] and depths (B, V, H, W).
    """
    if _rasterization is None:
        raise RuntimeError(
            "GradSplat renders with gsplat, which could not be imported "
            f"({type(_GSPLAT_ERROR).__name__}: {_GSPLAT_ERROR}). Install it with `pip install gsplat` "
            "(see the README); the in-forward refinement needs the rasterizer as well.")
    if not c2w.is_cuda:
        raise RuntimeError("gsplat needs a CUDA device")
    B, V = c2w.shape[:2]
    if background is None:
        background = torch.zeros(3, device=c2w.device, dtype=c2w.dtype)
    K_pix = _denorm_K(K_norm, H, W)
    sh_degree = int(round(gaussians["sh"].shape[-2] ** 0.5)) - 1
    colors, depths = [], []
    for b in range(B):
        w2c = torch.linalg.inv(c2w[b].float())
        out, _, _ = _rasterization(
            gaussians["means"][b], gaussians["quats"][b], gaussians["scales"][b], gaussians["opacities"][b],
            gaussians["sh"][b].contiguous(), w2c, K_pix[b], W, H, sh_degree=sh_degree, render_mode="RGB+D",
            packed=False, backgrounds=background[None].expand(V, 3), rasterize_mode="classic",
            **({} if eps2d is None else {"eps2d": float(eps2d)}),
        )
        colors.append(out[..., :3].clamp(0, 1).permute(0, 3, 1, 2))
        depths.append(out[..., 3])
    return torch.stack(colors), torch.stack(depths)

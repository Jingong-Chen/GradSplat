# Copyright (c) 2026 The GradSplat Authors.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Camera conventions.

Cameras are camera-to-world matrices (OpenCV axes: x right, y down, z forward) and pixel
intrinsics. Depth is z-depth along the optical axis, and pixel centers sit at integer + 0.5.
"""
import torch

from ._vggt_omega.utils.pose_enc import encoding_to_camera


def pose_enc_to_c2w_K(pose_enc, image_hw):
    """Camera-head output (..., V, 9) -> camera-to-world (..., V, 4, 4) and pixel intrinsics (..., V, 3, 3).

    The decoded extrinsics are world-to-camera and are inverted."""
    H, W = image_hw
    extr, K = encoding_to_camera(pose_enc, (H, W))
    *lead, _, _ = extr.shape
    w2c = torch.eye(4, device=extr.device, dtype=extr.dtype).expand(*lead, 4, 4).clone()
    w2c[..., :3, :4] = extr
    c2w = torch.linalg.inv(w2c)
    return c2w, K


def pixel_grid(H, W, device, dtype=torch.float32):
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    return u + 0.5, v + 0.5


def unproject_zdepth(depth, K_pix, c2w):
    """depth (..., H, W), K_pix (..., 3, 3), c2w (..., 4, 4) -> world points (..., H, W, 3)."""
    *lead, H, W = depth.shape
    u, v = pixel_grid(H, W, depth.device, depth.dtype)
    fx = K_pix[..., 0, 0][..., None, None]
    fy = K_pix[..., 1, 1][..., None, None]
    cx = K_pix[..., 0, 2][..., None, None]
    cy = K_pix[..., 1, 2][..., None, None]
    x = (u - cx) / fx
    y = (v - cy) / fy
    ones = torch.ones_like(x)
    dirs = torch.stack([x, y, ones], dim=-1)
    p_cam = dirs * depth[..., None]
    p_h = torch.cat([p_cam, torch.ones_like(p_cam[..., :1])], dim=-1)
    return torch.einsum("...ij,...hwj->...hwi", c2w, p_h)[..., :3]


def baseline_scale(c2w_pred, c2w_given, eps=1e-6):
    """Ratio of the given to the predicted distance between the first two cameras, (B,)."""
    def base(c2w):
        rel = torch.linalg.inv(c2w[:, 0]) @ c2w[:, 1]
        return rel[:, :3, 3].norm(dim=-1)
    return base(c2w_given) / base(c2w_pred).clamp_min(eps)


def normalize_K(K_pix, H, W):
    """Pixel intrinsics -> intrinsics normalized by the image width and height."""
    K = K_pix.clone()
    K[..., 0, :] = K[..., 0, :] / W
    K[..., 1, :] = K[..., 1, :] / H
    return K


def relative_to_first(c2w):
    """Express cameras (..., V, 4, 4) in the frame of the first one."""
    ref = torch.linalg.inv(c2w[..., :1, :, :])
    return ref @ c2w

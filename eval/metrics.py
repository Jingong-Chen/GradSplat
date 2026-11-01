# Copyright (c) 2026 The GradSplat Authors.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""The metrics of the paper, computed identically for every method there.

PSNR is computed per target view and averaged over the targets of a scene. SSIM uses an 11 x 11 uniform window
and is averaged over all pixels and targets of a scene. LPIPS uses the VGG network of the lpips package on images
in [-1, 1]. The value of a protocol is the mean over its scenes.
"""
import torch
import torch.nn.functional as F


def psnr_per_view(pred, gt):
    """pred, gt (..., 3, H, W) in [0, 1] -> PSNR of every view (...)."""
    mse = ((pred - gt) ** 2).mean(dim=(-1, -2, -3))
    return -10 * torch.log10(mse.clamp_min(1e-9))


def ssim(x, y, C1=0.01 ** 2, C2=0.03 ** 2):
    """Mean SSIM of views x, y (V, 3, H, W) with an 11 x 11 uniform window."""
    k = 11
    pad = k // 2
    w = x.new_ones(x.shape[1], 1, k, k) / (k * k)

    def g(t):
        return F.conv2d(t, w, padding=pad, groups=t.shape[1])
    mx, my = g(x), g(y)
    sxx, syy, sxy = g(x * x) - mx * mx, g(y * y) - my * my, g(x * y) - mx * my
    return float((((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx * mx + my * my + C1) * (sxx + syy + C2))).mean())


class LPIPS:
    def __init__(self, device):
        import lpips
        self.net = lpips.LPIPS(net="vgg", verbose=False).to(device)
        for p in self.net.parameters():
            p.requires_grad_(False)

    def __call__(self, pred, gt):
        """Mean LPIPS of views pred, gt (V, 3, H, W) in [0, 1]."""
        return float(self.net(pred * 2 - 1, gt * 2 - 1).mean())

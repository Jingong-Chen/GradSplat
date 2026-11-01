# Copyright (c) 2026 The GradSplat Authors.
# Portions adapted from VGGT-Omega (Copyright (c) Meta Platforms, Inc. and affiliates,
# FAIR Noncommercial Research License).
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Dense head with a Gaussian branch, and the shape head of the log-anisotropy correction."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._vggt_omega.models.heads.dense_head import DenseHead, _make_prediction_head
from .gaussians import activate_shape, gauss_channels

LOGIT_MAX = 9.0
LOGIT_MIN = -12.0


def _sobel(x):
    """(N, 3, h, w) -> (N, 6, h, w): horizontal and vertical Sobel responses of every channel."""
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=x.device, dtype=x.dtype).view(1, 1, 3, 3) / 8.0
    ky = kx.transpose(-1, -2)
    c = x.shape[1]
    gx = F.conv2d(x, kx.expand(c, 1, 3, 3), padding=1, groups=c)
    gy = F.conv2d(x, ky.expand(c, 1, 3, 3), padding=1, groups=c)
    return torch.cat([gx, gy], dim=1)


class ShapeHead(nn.Module):
    """Predicts the raw shape output r = (r_delta, r_omega) of every Gaussian from F and the Sobel responses
    of the input image (Eq. 3 of the paper). The output layer starts at zero, so delta = omega = 0."""

    def __init__(self, in_ch, shuffle, hidden=128, amp=4.0, rot_amp=1.0):
        super().__init__()
        self.shuffle = int(shuffle)
        self.amp = float(amp)
        self.rot_amp = float(rot_amp)
        self.img = nn.Conv2d(6, hidden, 3, padding=1)
        self.c1 = nn.Conv2d(in_ch, hidden, 3, padding=1)
        self.c2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.blocks = nn.ModuleList()
        self.out = nn.Conv2d(hidden, 6 * self.shuffle ** 2, 1)
        with torch.no_grad():
            self.out.weight.zero_()
            self.out.bias.zero_()

    def forward(self, feat, images_bv):
        """feat (N, C, h, w), images_bv (N, 3, H, W) -> raw shape output (N, H, W, 6)."""
        h, w = feat.shape[-2:]
        im = images_bv
        if im.shape[-2:] != (h, w):
            im = F.interpolate(im.float(), size=(h, w), mode="bilinear", align_corners=False)
        x = self.c1(feat) + self.img(_sobel(im.to(feat.dtype)))
        x = F.gelu(x)
        x = F.gelu(self.c2(x))
        for blk in self.blocks:
            x = x + blk(x)
        d = self.out(x)
        if self.shuffle > 1:
            d = F.pixel_shuffle(d, self.shuffle)
        return d.permute(0, 2, 3, 1)

    def activate(self, raw):
        return activate_shape(raw, self.amp, self.rot_amp)


class GaussianDenseHead(DenseHead):
    """The DPT dense head of VGGT-Omega (depth and confidence at full resolution) with a Gaussian branch.

    The branch upsamples the fused feature from 1/4 to 1/2 of the image resolution (the feature map F of
    the paper), adds a zero-initialized embedding of the input image, applies a residual convolutional
    trunk and decodes the 23 raw channels of the Gaussian of every pixel by pixel shuffling. The shape head
    reads the same F."""

    def __init__(self, dim_in=2048, patch_size=16, sh_degree=1, gauss_upsample=2, app_trunk_depth=6,
                 shape_hidden=128, shape_amp=4.0, shape_rot_amp=1.0):
        super().__init__(dim_in=dim_in, patch_size=patch_size)
        self.sh_degree = sh_degree
        self.gauss_ch = gauss_channels(sh_degree)
        feats = self.proj.in_channels
        self.gauss_upsample = int(gauss_upsample)
        self.gauss_shuffle = max(1, self.final_shuffle_factor // self.gauss_upsample)
        self.gauss_up = (nn.ConvTranspose2d(feats, feats, kernel_size=self.gauss_upsample, stride=self.gauss_upsample)
                         if self.gauss_upsample > 1 else None)
        self.rgb_embed = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(128, feats, 3, padding=1),
        )
        layers = []
        for _ in range(app_trunk_depth):
            layers += [nn.Conv2d(feats, feats, 3, padding=1), nn.GELU()]
        layers += [nn.Conv2d(feats, feats, 3, padding=1)]
        self.app_trunk = nn.Sequential(*layers)
        self.proj_gauss = _make_prediction_head(feats, self.gauss_shuffle ** 2 * self.gauss_ch)
        self.aniso_head = ShapeHead(feats, self.gauss_shuffle, hidden=shape_hidden, amp=shape_amp,
                                    rot_amp=shape_rot_amp)

    def forward(self, aggregated_tokens_list, images, patch_token_start):
        """Returns depth (B, V, H, W), depth confidence (B, V, H, W), raw Gaussian channels (B, V, H, W, 23),
        raw shape output (B, V, H, W, 6) and the feature map F (B*V, 256, H/2, W/2)."""
        B, V, _, H, W = images.shape
        ph, pw = H // self.patch_size, W // self.patch_size
        ms = []
        for fi, li in enumerate(self.intermediate_layer_idx):
            x = aggregated_tokens_list[li]
            x = x[:, :, patch_token_start:]
            if x.dtype != torch.float32:
                x = x.float()
            x = x.reshape(B * V, -1, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], ph, pw)
            x = self.projects[fi](x)
            x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[fi](x)
            ms.append(x)
        fused = self.scratch_forward(ms)
        fused = self._apply_pos_embed(fused, W, H)

        dl = self.proj(fused)
        dl = F.pixel_shuffle(dl, self.final_shuffle_factor).permute(0, 2, 3, 1)
        dl = torch.nan_to_num(dl, nan=0.0, posinf=LOGIT_MAX, neginf=LOGIT_MIN).clamp(LOGIT_MIN, LOGIT_MAX)
        depth = torch.exp(dl)

        cl = self.proj_conf(fused)
        cl = F.pixel_shuffle(cl, self.final_shuffle_factor).permute(0, 2, 3, 1).squeeze(-1)
        cl = torch.nan_to_num(cl, nan=0.0, posinf=LOGIT_MAX, neginf=LOGIT_MIN).clamp(LOGIT_MIN, LOGIT_MAX)
        depth_conf = 1.0 + torch.exp(cl)

        gfeat = fused
        rgb = images.reshape(B * V, 3, H, W).float()
        gfeat = gfeat + self.rgb_embed(rgb)
        if self.gauss_up is not None:
            gfeat = self.gauss_up(gfeat)
        gfeat = gfeat + self.app_trunk(gfeat)

        shape_raw = self.aniso_head(gfeat, images.reshape(B * V, 3, H, W)).view(B, V, H, W, 6)
        gp = self.proj_gauss(gfeat)
        gp = F.pixel_shuffle(gp, self.gauss_shuffle).permute(0, 2, 3, 1)

        depth = depth.view(B, V, *depth.shape[1:])
        depth_conf = depth_conf.view(B, V, *depth_conf.shape[1:])
        gp = gp.view(B, V, H, W, self.gauss_ch)
        return depth[..., 0], depth_conf, gp, shape_raw, gfeat

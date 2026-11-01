# Copyright (c) 2026 The GradSplat Authors.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""In-forward gradient refinement (Sec. 3.3 of the paper).

Each step renders the current Gaussians into the input views, backpropagates the mean squared error
L_in through the rasterizer to a vector r of raw network outputs, and applies a learned update
U_psi(r_k, g_k / rms, delta r_{k-1}, v_k, F, h_{k-1}) -> (delta r_k, h_k). The gradient is an input of the
update, computed on a detached copy of r, so no second derivative of the rasterizer is needed and the
steps run unchanged at inference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gaussians import activate_shape, assemble, flatten_gaussians


class Block(nn.Module):
    """Pre-norm transformer block: multi-head self-attention and an MLP with ratio 4."""

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        assert dim % heads == 0, (dim, heads)
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        hid = int(dim * mlp_ratio)
        self.fc1, self.fc2 = nn.Linear(dim, hid), nn.Linear(hid, dim)

    def forward(self, x):
        N, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(N, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(N, T, D))
        return x + self.fc2(F.gelu(self.fc1(self.n2(x))))


class TokenMixer(nn.Module):
    """Transformer blocks on the encoding of an update, alternating attention within a view and over the
    tokens of all views. The encoding at the resolution of F is cut into patches that form the token grid
    of the backbone (14 x 14 per view at 224 x 224). The last layer starts at zero."""

    def __init__(self, ch, depth, dim=768, heads=12, mid=64, patch=8, grid=14):
        super().__init__()
        self.patch, self.grid, self.mid = patch, grid, mid
        self.red = nn.Conv2d(ch, mid, 1)
        self.inp = nn.Conv2d(mid, dim, patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, grid * grid, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.outp = nn.Linear(dim, mid * patch * patch)
        self.exp = nn.Conv2d(mid, ch, 1)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # At another resolution the token grid differs (16 x 16 at 256 x 256); the position embedding is
        # then interpolated bicubically to the grid of this model, as for the position embedding of a ViT.
        k = prefix + "pos"
        if k in state_dict and state_dict[k].shape != self.pos.shape:
            src = state_dict[k]
            g0, g = int(round(src.shape[1] ** 0.5)), self.grid
            assert g0 * g0 == src.shape[1] and src.shape[2] == self.pos.shape[2], (src.shape, self.pos.shape)
            q = F.interpolate(src.float().reshape(1, g0, g0, -1).permute(0, 3, 1, 2), size=(g, g), mode="bicubic",
                              align_corners=False)
            state_dict[k] = q.permute(0, 2, 3, 1).reshape(1, g * g, -1).to(src.dtype)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x, nviews):
        BV, C, h, w = x.shape
        p, g = self.patch, self.grid
        assert h == w == p * g and BV % nviews == 0, (x.shape, p, g, nviews)
        B, T = BV // nviews, g * g
        t = self.inp(F.gelu(self.red(x))).flatten(2).transpose(1, 2) + self.pos
        D = t.shape[-1]
        for i, blk in enumerate(self.blocks):
            if i % 2 == 0:
                t = blk(t)
            else:
                t = blk(t.reshape(B, nviews * T, D)).reshape(BV, T, D)
        o = self.outp(self.norm(t)).transpose(1, 2).reshape(BV, self.mid * p * p, g, g)
        return self.exp(F.gelu(F.pixel_shuffle(o, p)))


class LearnedUpdate(nn.Module):
    """The learned update U_psi: a convolutional encoding of (r_k, g_k / rms, delta r_{k-1}, v_k, F), the token
    mixer, and a convolutional GRU whose hidden state is mapped to delta r_k by a zero-initialized layer."""

    def __init__(self, feats=256, gauss_ch=23, shuffle=4, hidden=192, ctx_dim=96, mixer_depth=4, mixer_dim=768,
                 mixer_heads=12, mixer_mid=64, mixer_patch=8, mixer_grid=14):
        super().__init__()
        self.shuffle = shuffle
        self.hidden = hidden
        gc = gauss_ch * shuffle ** 2
        self.enc = nn.Sequential(
            nn.Conv2d(4 * gc + feats, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU())
        self.ctx = nn.Conv2d(feats, ctx_dim, 3, padding=1)
        cin = hidden + hidden + ctx_dim
        self.gz = nn.Conv2d(cin, hidden, 3, padding=1)
        self.gr = nn.Conv2d(cin, hidden, 3, padding=1)
        self.gq = nn.Conv2d(cin, hidden, 3, padding=1)
        self.out = nn.Conv2d(hidden, gc, 1)
        self.mixer = TokenMixer(hidden, mixer_depth, dim=mixer_dim, heads=mixer_heads, mid=mixer_mid,
                                patch=mixer_patch, grid=mixer_grid) if mixer_depth > 0 else None

    def forward(self, theta_lo, grad_lo, fused, state, prev_delta, grad_sq, nviews):
        g = grad_lo / grad_lo.flatten(2).pow(2).mean(-1).sqrt().clamp_min(1e-8)[..., None, None]
        pd = torch.zeros_like(theta_lo) if prev_delta is None else prev_delta
        gs = g.detach() ** 2 if grad_sq is None else grad_sq
        m = self.enc(torch.cat([theta_lo, g, pd, gs, fused], dim=1))
        if self.mixer is not None:
            m = m + self.mixer(m, nviews)
        if state is None:
            b, _, h, w = m.shape
            state = m.new_zeros(b, self.hidden, h, w)
        c = torch.cat([m, state, self.ctx(fused)], dim=1)
        z = torch.sigmoid(self.gz(c))
        r = torch.sigmoid(self.gr(c))
        q = torch.tanh(self.gq(torch.cat([m, r * state, self.ctx(fused)], dim=1)))
        state = (1 - z) * state + z * q
        return self.out(state), state


def _normalized_K(K_pix, H, W):
    Kn = K_pix.detach().clone()
    Kn[..., 0, :] = Kn[..., 0, :] / W
    Kn[..., 1, :] = Kn[..., 1, :] / H
    return Kn


def appearance_gradient(gauss_params, world_means, images, cfg, c2w, K_pix, render_fn):
    """Gradient of L_in w.r.t. the raw Gaussian channels, taken with the base scale and rotation."""
    H, W = images.shape[-2], images.shape[-1]
    with torch.enable_grad():
        leaf = gauss_params.detach().clone().requires_grad_(True)
        gd = assemble(leaf, world_means.detach(), images, cfg.sh_degree, scale_max=cfg.scale_max,
                      dmean_scale=cfg.dmean_scale, scale_mult=cfg.scale_mult)
        rec, _ = render_fn(flatten_gaussians(gd), c2w.detach(), _normalized_K(K_pix, H, W), H, W)
        loss = ((rec - images) ** 2).mean()
        (g,) = torch.autograd.grad(loss, leaf)
    return g.detach(), float(loss.detach())


def shape_gradient(shape_raw, gauss_params, world_means, images, cfg, c2w, K_pix, render_fn):
    """Gradient of L_in w.r.t. the raw shape output, through the activation of Eq. 3."""
    H, W = images.shape[-2], images.shape[-1]
    with torch.enable_grad():
        leaf = shape_raw.detach().clone().requires_grad_(True)
        a, w = activate_shape(leaf, cfg.shape_amp, cfg.shape_rot_amp)
        gd = assemble(gauss_params.detach(), world_means.detach(), images, cfg.sh_degree, scale_max=cfg.scale_max,
                      dmean_scale=cfg.dmean_scale, scale_mult=cfg.scale_mult, aniso=a, rot_delta=w)
        rec, _ = render_fn(flatten_gaussians(gd), c2w.detach(), _normalized_K(K_pix, H, W), H, W)
        loss = ((rec - images) ** 2).mean()
        (g,) = torch.autograd.grad(loss, leaf)
    return g.detach(), float(loss.detach())


def appearance_mask(delta_lo, gauss_ch, shuffle, sh_degree=1):
    """Keep the opacity and color channels of an update at the resolution of F, zero the others."""
    r2 = shuffle ** 2
    assert delta_lo.shape[1] == gauss_ch * r2, (delta_lo.shape, gauss_ch, r2)
    ns = (sh_degree + 1) ** 2
    m = torch.zeros(1, gauss_ch * r2, 1, 1, device=delta_lo.device, dtype=delta_lo.dtype)
    m[:, 0:1 * r2] = 1.0
    m[:, 8 * r2:(8 + 3 * ns) * r2] = 1.0
    return delta_lo * m

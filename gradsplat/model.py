# Copyright (c) 2026 The GradSplat Authors.
# Portions adapted from VGGT-Omega (Copyright (c) Meta Platforms, Inc. and affiliates).
#
# This source code is licensed under the FAIR Noncommercial Research License of VGGT-Omega, found in
# gradsplat/_vggt_omega/LICENSE.

"""GradSplat: a feed-forward 3D Gaussian Splatting model that refines its own Gaussians by learned
gradient steps on the input views inside its forward pass.

A VGGT-Omega backbone feeds a camera head, a dense head and a shape head. The dense head predicts a depth and
the raw parameters of one Gaussian per input pixel; the shape head predicts a log-anisotropy correction of
its scale and a rotation increment. The refinement then takes K_a = 3 appearance steps (opacity and color)
and K_s = 5 shape steps, each rendering the Gaussians into the input views and applying a learned update to
the gradient of the photometric loss. No target view is used.
"""
import os
from dataclasses import asdict, dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._vggt_omega.models.aggregator import Aggregator
from ._vggt_omega.models.heads import CameraHead
from .gaussians import activate_shape, assemble, flatten_gaussians, gauss_channels
from .geometry import baseline_scale, normalize_K, pose_enc_to_c2w_K, unproject_zdepth
from .heads import GaussianDenseHead
from .refinement import LearnedUpdate, appearance_gradient, appearance_mask, shape_gradient
from .render import render

try:
    from huggingface_hub import PyTorchModelHubMixin
except ImportError:                                    # huggingface_hub is optional for local weights
    class PyTorchModelHubMixin:                        # noqa: D401
        """Placeholder when huggingface_hub is not installed."""
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__()


# download options that from_pretrained passes on (proxies and resume_download before huggingface_hub 1.0)
_HUB_KWARGS = ("revision", "cache_dir", "force_download", "local_files_only", "token", "proxies", "resume_download")


@dataclass
class GradSplatConfig:
    resolution: int = 224          # square input size; the token mixers are built for its token grid
    patch_size: int = 16
    embed_dim: int = 1024
    sh_degree: int = 1
    scale_max: float = 0.3
    scale_mult: float = 0.001      # tau of Eq. 2
    dmean_scale: float = 0.05      # bound of the learned center offset
    gauss_upsample: int = 2        # F at 1/2 of the image resolution
    app_trunk_depth: int = 6
    shape_hidden: int = 128
    shape_amp: float = 4.0
    shape_rot_amp: float = 1.0
    appearance_steps: int = 3      # K_a
    shape_steps: int = 5           # K_s
    appearance_update_hidden: int = 192
    shape_update_hidden: int = 128
    mixer_depth: int = 4
    mixer_dim: int = 768
    mixer_heads: int = 12
    mixer_mid: int = 64
    kappa: float = 4.04            # scale constant of pose-free inference (Sec. 3.1)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


class GradSplat(nn.Module, PyTorchModelHubMixin,
                library_name="gradsplat", repo_url="https://github.com/Jingong-Chen/GradSplat",
                license="other", tags=["gaussian-splatting", "novel-view-synthesis", "3d-reconstruction",
                                       "feed-forward"]):
    def __init__(self, **config):
        super().__init__()
        cfg = GradSplatConfig.from_dict(config)
        self.cfg = cfg
        if cfg.resolution % cfg.patch_size != 0:
            raise ValueError(f"resolution {cfg.resolution} must be a multiple of the patch size {cfg.patch_size}")
        self.aggregator = Aggregator(patch_size=cfg.patch_size, embed_dim=cfg.embed_dim)
        self.camera_head = CameraHead(dim_in=2 * cfg.embed_dim)
        self.dense_head = GaussianDenseHead(dim_in=2 * cfg.embed_dim, patch_size=cfg.patch_size,
                                            sh_degree=cfg.sh_degree, gauss_upsample=cfg.gauss_upsample,
                                            app_trunk_depth=cfg.app_trunk_depth, shape_hidden=cfg.shape_hidden,
                                            shape_amp=cfg.shape_amp, shape_rot_amp=cfg.shape_rot_amp)
        shuffle = self.dense_head.gauss_shuffle
        f_res = cfg.resolution // shuffle               # side of F
        grid = cfg.resolution // cfg.patch_size         # token grid of the backbone per side
        assert f_res % grid == 0, (f_res, grid)
        mixer = dict(mixer_depth=cfg.mixer_depth, mixer_dim=cfg.mixer_dim, mixer_heads=cfg.mixer_heads,
                     mixer_mid=cfg.mixer_mid, mixer_patch=f_res // grid, mixer_grid=grid)
        self.shape_updater = LearnedUpdate(feats=256, gauss_ch=6, shuffle=1, hidden=cfg.shape_update_hidden, **mixer)
        self.app_updater = LearnedUpdate(feats=256, gauss_ch=gauss_channels(cfg.sh_degree), shuffle=shuffle,
                                         hidden=cfg.appearance_update_hidden, **mixer)

    @property
    def config(self):
        return asdict(self.cfg)

    @classmethod
    def _from_pretrained(cls, *, model_id, variant=None, map_location="cpu", strict=True, **kwargs):
        """Called by from_pretrained. The weights are model.safetensors (trained on RealEstate10K) or, with a variant,
        model.<variant>.safetensors: GradSplat.from_pretrained("Jingong/GradSplat", variant="dl3dv") loads the weights
        fine-tuned on DL3DV."""
        hub = {k: kwargs.pop(k) for k in _HUB_KWARGS if k in kwargs}
        filename = f"model.{variant}.safetensors" if variant else "model.safetensors"
        if os.path.isdir(model_id):
            path = os.path.join(model_id, filename)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{path} does not exist")
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(model_id, filename, **{k: v for k, v in hub.items() if v is not None})
        return cls._load_as_safetensor(cls(**kwargs), path, map_location, strict)

    def _assemble(self, gauss_params, world_means, images, aniso=None, rot_delta=None):
        c = self.cfg
        return assemble(gauss_params, world_means, images, c.sh_degree, scale_max=c.scale_max,
                        dmean_scale=c.dmean_scale, scale_mult=c.scale_mult, aniso=aniso, rot_delta=rot_delta)

    def forward(self, images, c2w=None, K=None, depth_scale=None, appearance_steps=None, shape_steps=None,
                return_iterates=False):
        """Predict and refine the Gaussians of a batch of scenes.

        Args:
            images: (B, V, 3, H, W) input views in [0, 1], H = W = cfg.resolution.
            c2w: (B, V, 4, 4) camera-to-world poses of the input views (OpenCV axes), or None.
            K: (B, V, 3, 3) pixel intrinsics of the input views, or None.
            depth_scale: scale of the predicted depth and camera translations when no poses are given,
                None for cfg.kappa.
            appearance_steps, shape_steps: number of steps (None: K_a = 3 and K_s = 5, the trained setting).
            return_iterates: also return the Gaussians before the steps and after every step.
        With c2w and K the Gaussians are placed with the given cameras, and the predicted depth is scaled by the
        ratio of the given to the predicted distance between the first two cameras. Without them the
        predicted cameras are used, with their translations and the depth multiplied by depth_scale.
        Returns a dict with gaussians (flattened, (B, V*H*W, ...)), the cameras used for the input views
        (c2w, K in pixels), the predicted cameras (c2w_pred, K_pred, unscaled), the depth and its scale.
        """
        cfg = self.cfg
        B, V, _, H, W = images.shape
        if H != cfg.resolution or W != cfg.resolution:
            raise ValueError(f"the model is built for {cfg.resolution} x {cfg.resolution} views, got {H} x {W}")
        if (c2w is None) != (K is None):
            raise ValueError("give both c2w and K, or neither")
        k_app = cfg.appearance_steps if appearance_steps is None else int(appearance_steps)
        k_shape = cfg.shape_steps if shape_steps is None else int(shape_steps)

        use_amp = images.is_cuda
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            tokens_list, pts = self.aggregator(images)
        with torch.autocast(device_type="cuda", enabled=False):
            pose_enc = self.camera_head(tokens_list, patch_token_start=pts).float()
            depth, depth_conf, gauss_params, shape_raw, feat = self.dense_head(tokens_list, images.float(), pts)
        c2w_pred, K_pred = pose_enc_to_c2w_K(pose_enc, (H, W))

        if c2w is not None:
            s = baseline_scale(c2w_pred, c2w).view(B, 1, 1, 1)
        else:
            k = cfg.kappa if depth_scale is None else depth_scale
            s = torch.as_tensor(k, device=images.device, dtype=c2w_pred.dtype).reshape(-1, 1, 1, 1).expand(B, 1, 1, 1)
        depth_scaled = depth * s
        trans_scaled = c2w_pred[..., :3, 3] * s.view(B, 1, 1)
        top = torch.cat([c2w_pred[..., :3, :3], trans_scaled[..., None]], dim=-1)
        c2w_pred_scaled = torch.cat([top, c2w_pred[..., 3:4, :]], dim=-2)
        if c2w is not None:
            ctx_c2w, ctx_K = c2w, K
        else:
            ctx_c2w, ctx_K = c2w_pred_scaled, K_pred

        world_means = unproject_zdepth(depth_scaled, ctx_K, ctx_c2w)
        aniso, rot_d = activate_shape(shape_raw, cfg.shape_amp, cfg.shape_rot_amp)
        iterates, trace = [], []
        if return_iterates:
            iterates.append(flatten_gaussians(self._assemble(gauss_params, world_means, images, aniso, rot_d)))
            trace.append("prediction")

        # appearance steps: opacity and color, gradient with the base shape
        sh_a = self.dense_head.gauss_shuffle
        sta = pda = gsqa = None
        for i in range(k_app):
            ga, _ = appearance_gradient(gauss_params, world_means, images, cfg, ctx_c2w, ctx_K, render)
            to_lo = lambda t: F.pixel_unshuffle(t.reshape(B * V, H, W, -1).permute(0, 3, 1, 2), sh_a)  # noqa: E731
            g_lo, th_lo = to_lo(ga), to_lo(gauss_params)
            gna = g_lo / g_lo.flatten(2).pow(2).mean(-1).sqrt().clamp_min(1e-8)[..., None, None]
            gsqa = gna.detach() ** 2 if gsqa is None else 0.9 * gsqa + 0.1 * gna.detach() ** 2
            da, sta = self.app_updater(th_lo, g_lo, feat, state=sta, prev_delta=pda, grad_sq=gsqa, nviews=V)
            pda = da.detach()
            da = appearance_mask(da, gauss_channels(cfg.sh_degree), sh_a, cfg.sh_degree)
            gauss_params = gauss_params + F.pixel_shuffle(da, sh_a).permute(0, 2, 3, 1).reshape(B, V, H, W, -1)
            if return_iterates:
                iterates.append(flatten_gaussians(self._assemble(gauss_params, world_means, images, aniso, rot_d)))
                trace.append(f"appearance {i + 1}")

        # shape steps: log-anisotropy and rotation, gradient at the refined appearance
        raw = shape_raw
        hh, ww = feat.shape[-2:]
        st = pd = gsq = None
        for i in range(k_shape):
            g, _ = shape_gradient(raw, gauss_params, world_means, images, cfg, ctx_c2w, ctx_K, render)
            to_lo = lambda t: F.interpolate(t.reshape(B * V, H, W, 6).permute(0, 3, 1, 2), size=(hh, ww),  # noqa: E731
                                            mode="bilinear", align_corners=False)
            g_lo, r_lo = to_lo(g), to_lo(raw)
            gn = g_lo / g_lo.flatten(2).pow(2).mean(-1).sqrt().clamp_min(1e-8)[..., None, None]
            gsq = gn.detach() ** 2 if gsq is None else 0.9 * gsq + 0.1 * gn.detach() ** 2
            d, st = self.shape_updater(r_lo, g_lo, feat, state=st, prev_delta=pd, grad_sq=gsq, nviews=V)
            pd = d.detach()
            up = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)
            raw = raw + up.permute(0, 2, 3, 1).reshape(B, V, H, W, 6)
            aniso, rot_d = activate_shape(raw, cfg.shape_amp, cfg.shape_rot_amp)
            if return_iterates:
                iterates.append(flatten_gaussians(self._assemble(gauss_params, world_means, images, aniso, rot_d)))
                trace.append(f"shape {i + 1}")

        gaussians = flatten_gaussians(self._assemble(gauss_params, world_means, images, aniso, rot_d))
        out = dict(gaussians=gaussians, c2w=ctx_c2w, K=ctx_K, c2w_pred=c2w_pred, K_pred=K_pred,
                   c2w_pred_scaled=c2w_pred_scaled, depth=depth_scaled, depth_conf=depth_conf, depth_scale=s.reshape(B))
        if return_iterates:
            out.update(iterates=iterates, iterate_names=trace)
        return out

    @torch.no_grad()
    def predict_cameras(self, images):
        """Camera-to-world poses and pixel intrinsics predicted from images (B, V, 3, H, W) alone, unscaled, in the
        frame of the first view."""
        H, W = images.shape[-2:]
        amp_dtype = torch.bfloat16 if (images.is_cuda and torch.cuda.is_bf16_supported()) else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=images.is_cuda):
            tokens_list, pts = self.aggregator(images)
        with torch.autocast(device_type="cuda", enabled=False):
            pose_enc = self.camera_head(tokens_list, patch_token_start=pts).float()
        return pose_enc_to_c2w_K(pose_enc, (H, W))

    @torch.no_grad()
    def infer(self, images, c2w=None, K=None, **kwargs):
        """One scene without a batch axis: images (V, 3, H, W) in [0, 1], optional c2w (V, 4, 4) and pixel K (V, 3, 3).
        The model predicts under no_grad; the refinement computes its gradients internally."""
        dev = next(self.parameters()).device
        images = images.to(dev).float()[None]
        if c2w is not None:
            c2w, K = c2w.to(dev).float()[None], K.to(dev).float()[None]
        return self.forward(images, c2w, K, **kwargs)

    def render(self, gaussians, c2w, K, H=None, W=None):
        """Render flattened Gaussians (B, N, ...) into cameras c2w (B, T, 4, 4) with pixel intrinsics K (B, T, 3, 3).
        Returns colors (B, T, 3, H, W) and depths (B, T, H, W)."""
        H = H or self.cfg.resolution
        W = W or self.cfg.resolution
        return render(gaussians, c2w, normalize_K(K, H, W), H, W)

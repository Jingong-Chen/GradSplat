# Copyright (c) 2026 The GradSplat Authors.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""CPU tests: conventions, activations, input preparation and export. No weights and no GPU needed."""
import json
import math
import os
import re

import numpy as np
import pytest
import torch
from PIL import Image

from gradsplat import GradSplat
from gradsplat.export import interpolate_path, load_ply, save_ply
from gradsplat.gaussians import activate_shape, apply_rot_delta, assemble, flatten_gaussians, gauss_channels, sh_dc_to_rgb
from gradsplat.geometry import baseline_scale, normalize_K, unproject_zdepth
from gradsplat.io import load_views, square_crop_resize
from gradsplat.refinement import appearance_mask

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.join(os.path.dirname(HERE), "examples", "re10k_dining_room")


def _rand_c2w(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        q = torch.randn(4, generator=g)
        q = q / q.norm()
        w, x, y, z = q.tolist()
        R = torch.tensor([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                          [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                          [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        M = torch.eye(4)
        M[:3, :3] = R
        M[:3, 3] = torch.randn(3, generator=g)
        out.append(M)
    return torch.stack(out)


def test_unproject_reprojects_to_the_pixel():
    H = W = 8
    K = torch.tensor([[10.0, 0, 4.2], [0, 11.0, 3.9], [0, 0, 1]])
    c2w = _rand_c2w(1)[0]
    depth = torch.rand(H, W) + 1.0
    X = unproject_zdepth(depth, K, c2w)                          # (H, W, 3)
    w2c = torch.linalg.inv(c2w)
    pc = (w2c[:3, :3] @ X.reshape(-1, 3).T + w2c[:3, 3:]).T
    uv = (K @ pc.T).T
    uv = uv[:, :2] / uv[:, 2:]
    v, u = torch.meshgrid(torch.arange(H) + 0.5, torch.arange(W) + 0.5, indexing="ij")
    assert torch.allclose(uv, torch.stack([u, v], -1).reshape(-1, 2).float(), atol=1e-4)
    assert torch.allclose(pc[:, 2], depth.reshape(-1), atol=1e-5)


def test_baseline_scale():
    c2w = _rand_c2w(3)[None]
    scaled = c2w.clone()
    scaled[..., :3, 3] *= 2.5
    assert torch.allclose(baseline_scale(c2w, scaled), torch.tensor([2.5]), atol=1e-5)


def test_shape_correction_is_zero_sum_and_bounded():
    raw = torch.randn(1000, 6) * 5
    a, w = activate_shape(raw)
    assert torch.allclose(a.sum(-1), torch.zeros(1000), atol=1e-5)
    assert float((a.max(-1).values - a.min(-1).values).max()) <= 8.0
    assert float(w.norm(dim=-1).max()) < math.sqrt(3) + 1e-6


def test_rotation_increment_identity():
    q = torch.nn.functional.normalize(torch.randn(50, 4), dim=-1)
    assert torch.equal(apply_rot_delta(q, torch.zeros(50, 3)), torch.nn.functional.normalize(q, dim=-1)) or \
        torch.allclose(apply_rot_delta(q, torch.zeros(50, 3)), q, atol=1e-7)


def test_assemble_anchors_color_and_volume():
    H = W = 4
    gp = torch.zeros(1, 2, H, W, gauss_channels(1))
    rgb = torch.rand(1, 2, 3, H, W)
    means = torch.randn(1, 2, H, W, 3)
    g = assemble(gp, means, rgb, 1)
    assert torch.allclose(sh_dc_to_rgb(g["sh"][..., 0, :]), rgb.movedim(-3, -1), atol=1e-6)
    assert torch.equal(g["means"], means)
    raw = torch.randn(1, 2, H, W, 6)
    a, w = activate_shape(raw)
    g2 = assemble(gp, means, rgb, 1, aniso=a, rot_delta=w)
    vol0 = g["scales"].prod(-1)
    assert torch.allclose(g2["scales"].prod(-1), vol0, rtol=1e-5)          # constant volume below the bound
    f = flatten_gaussians(g2)
    assert f["means"].shape == (1, 2 * H * W, 3) and f["sh"].shape == (1, 2 * H * W, 4, 3)


def test_appearance_mask_keeps_opacity_and_color():
    gc, sh = gauss_channels(1), 2
    d = torch.ones(1, gc * sh * sh, 3, 3)
    m = appearance_mask(d, gc, sh)
    kept = [c for c in range(gc) if float(m[0, c * 4:(c + 1) * 4].sum()) > 0]
    assert kept == [0] + list(range(8, 20))       # opacity and the 12 SH coefficients


def test_square_crop_resize_maps_intrinsics():
    im = torch.rand(3, 360, 640)
    K = torch.tensor([[500.0, 0, 320], [0, 500.0, 180], [0, 0, 1]])
    out, K2 = square_crop_resize(im, 224, K)
    assert out.shape == (3, 224, 224)
    assert torch.allclose(K2, torch.tensor([[500 * 224 / 360, 0, 112], [0, 500 * 224 / 360, 112], [0, 0, 1]]), atol=1e-4)


def test_load_example_views():
    v = load_views(os.path.join(EXAMPLE, "images"), cameras=os.path.join(EXAMPLE, "cameras.json"))
    assert v.images.shape == (6, 3, 224, 224) and v.c2w.shape == (6, 4, 4) and v.K.shape == (6, 3, 3)
    assert torch.allclose(v.c2w[0], torch.eye(4), atol=1e-5)               # in the frame of the first view
    assert -1e-4 <= float(v.images.min()) and float(v.images.max()) <= 1.0 + 1e-4


def test_ply_round_trip(tmp_path):
    n = 100
    g = dict(means=torch.randn(n, 3), scales=torch.rand(n, 3) * 0.01 + 1e-4,
             quats=torch.nn.functional.normalize(torch.randn(n, 4), dim=-1), opacities=torch.rand(n) * 0.98 + 0.01,
             sh=torch.randn(n, 4, 3))
    save_ply(str(tmp_path / "g.ply"), g)
    h = load_ply(str(tmp_path / "g.ply"))
    for k in g:
        assert torch.allclose(g[k], h[k], rtol=1e-5, atol=1e-6), k


def test_path_passes_through_the_cameras():
    c2w = _rand_c2w(4, seed=3)
    path = interpolate_path(c2w, 4 * 3, loop=False)
    assert torch.allclose(path[0], c2w[0], atol=1e-5) and torch.allclose(path[-1], c2w[-1], atol=1e-4)


def test_variant_selects_the_weights_file(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"resolution": 224}))
    for variant, name in ((None, "model.safetensors"), ("dl3dv", "model.dl3dv.safetensors")):
        with pytest.raises(FileNotFoundError, match=re.escape(os.path.join(str(tmp_path), name))):
            GradSplat.from_pretrained(str(tmp_path), variant=variant)

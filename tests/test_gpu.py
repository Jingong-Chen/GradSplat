# Copyright (c) 2026 The GradSplat Authors.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""GPU tests with the released weights (skipped without CUDA). GRADSPLAT_WEIGHTS selects the weights
(default: Jingong/GradSplat on Hugging Face)."""
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.join(os.path.dirname(HERE), "examples", "re10k_dining_room")


@pytest.fixture(scope="module")
def model():
    from gradsplat import GradSplat
    return GradSplat.from_pretrained(os.environ.get("GRADSPLAT_WEIGHTS", "Jingong/GradSplat")).cuda().eval()


def _psnr(a, b):
    return float((-10 * torch.log10(((a - b) ** 2).mean(dim=(-1, -2, -3)))).mean())


def test_refinement_improves_the_input_views(model):
    from gradsplat.geometry import normalize_K
    from gradsplat.io import load_views
    from gradsplat.render import render
    v = load_views(os.path.join(EXAMPLE, "images"), cameras=os.path.join(EXAMPLE, "cameras.json"))
    out = model.infer(v.images, v.c2w, v.K, return_iterates=True)
    assert out["gaussians"]["means"].shape == (1, 6 * 224 * 224, 3)
    assert out["iterate_names"][0] == "prediction" and len(out["iterates"]) == 9
    gt = v.images[None].cuda()
    K = normalize_K(out["K"], 224, 224)
    first = _psnr(render(out["iterates"][0], out["c2w"], K, 224, 224)[0], gt)
    last = _psnr(render(out["gaussians"], out["c2w"], K, 224, 224)[0], gt)
    assert last > first + 1.0 and last > 30.0, (first, last)


def test_pose_free_runs(model):
    from gradsplat.io import load_views
    v = load_views(os.path.join(EXAMPLE, "images"))
    out = model.infer(v.images)
    assert torch.isfinite(out["gaussians"]["means"]).all()
    assert torch.allclose(out["depth_scale"], torch.tensor([model.cfg.kappa], device="cuda"))


def test_display_frames_keep_renderings(model):
    from gradsplat.export import to_display_frame, to_orbit_view
    from gradsplat.gaussians import select
    from gradsplat.geometry import normalize_K
    from gradsplat.io import load_views
    from gradsplat.render import render
    v = load_views(os.path.join(EXAMPLE, "images"), cameras=os.path.join(EXAMPLE, "cameras.json"))
    out = model.infer(v.images, v.c2w, v.K)
    K = normalize_K(out["K"], 224, 224)
    a, _ = render(out["gaussians"], out["c2w"], K, 224, 224)
    g0 = select(out["gaussians"])
    for g, T in (to_display_frame(g0), to_orbit_view(g0, float(g0["means"][:, 2].median()), 60.0)):
        b, _ = render({k: t[None] for k, t in g.items()}, (T @ out["c2w"][0])[None], K, 224, 224)
        d = (a - b).abs()
        assert float(d.mean()) < 1e-5 and float(d.max()) < 1e-2, (float(d.mean()), float(d.max()))

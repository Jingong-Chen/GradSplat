#!/usr/bin/env python3
# Copyright (c) 2026 The GradSplat Authors.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Evaluate GradSplat on the protocols of the paper.

    python eval/evaluate.py --protocol re10k_narrow --data data/re10k --out results/re10k_narrow
    python eval/evaluate.py --protocol re10k_narrow --data data/re10k --pose-free --out results/re10k_narrow_posefree
    python eval/evaluate.py --protocol dl3dv --data data/dl3dv --variant dl3dv --out results/dl3dv_ft

Writes per_scene.csv (scene, PSNR, SSIM, LPIPS, input-view PSNR) and summary.json, and with --compare-paper
prints the values of the paper next to the measured ones.
"""
import argparse
import csv
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from chunks import index_scenes, iter_scenes                    # noqa: E402
from metrics import LPIPS, psnr_per_view, ssim                  # noqa: E402

from gradsplat import GradSplat                                 # noqa: E402
from gradsplat.render import render                             # noqa: E402

IDX = os.path.join(REPO, "assets", "indices")
PROTOCOLS = {
    "re10k_narrow": os.path.join(IDX, "re10k_narrow_volsplat.json"),
    "re10k_wide": os.path.join(IDX, "re10k_wide_yonosplat.json"),
    "dl3dv": os.path.join(IDX, "dl3dv_resplat.json"),
    "scannetpp": os.path.join(IDX, "scannetpp_val.json"),
    **{f"re10k_views_{v}": os.path.join(IDX, f"re10k_views_{v}.json") for v in (2, 3, 4, 6, 12, 24, 32, 42)},
}


def norm_K(K_pix, res):
    K = K_pix.clone()
    K[..., 0, :] /= res
    K[..., 1, :] /= res
    return K


def load_model(weights, device, resolution, variant=None):
    kwargs = dict(resolution=resolution) if resolution != 224 else {}
    return GradSplat.from_pretrained(weights, variant=variant, **kwargs).to(device).eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS), help="a protocol of the paper (sets --index)")
    ap.add_argument("--index", help="evaluation index (JSON: scene -> {context, target})")
    ap.add_argument("--data", required=True, help="dataset root with test/*.torch and test/index.json")
    ap.add_argument("--weights", default="Jingong/GradSplat", help="Hugging Face repo id or local directory")
    ap.add_argument("--variant", default=None,
                    help="load model.<variant>.safetensors: dl3dv for the weights fine-tuned on DL3DV "
                         "(default: model.safetensors, trained on RealEstate10K)")
    ap.add_argument("--pose-free", action="store_true",
                    help="no input poses: the Gaussians use the predicted cameras with kappa, and the target "
                         "cameras come from the joint forward of the input and target images (AnySplat protocol)")
    ap.add_argument("--kappa", type=float, default=None, help="scale constant of pose-free inference (default: config)")
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N scenes (for a quick check)")
    ap.add_argument("--save-images", action="store_true", help="also write the renderings of the target views")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare-paper", action="store_true")
    a = ap.parse_args()

    index = a.index or (PROTOCOLS.get(a.protocol) if a.protocol else None)
    if not index:
        ap.error("give --protocol or --index")
    for label, path in (("--data", os.path.join(a.data, "test", "index.json")), ("--index", index)):
        if not os.path.exists(path):
            sys.exit(f"{label}: {path} does not exist")
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")

    model = load_model(a.weights, dev, a.resolution, a.variant)
    kappa = model.cfg.kappa if a.kappa is None else a.kappa
    lpips_fn = LPIPS(dev)
    _, chunk_of, n_index = index_scenes(a.data, index, a.limit)
    weights = a.weights + (f" ({a.variant})" if a.variant else "")
    setting = f"pose-free, kappa {kappa}" if a.pose_free else "input poses"
    print(f"index {os.path.basename(index)}: {n_index} scenes, {len(chunk_of)} in the data"
          f"{' (limited)' if a.limit else ''}; weights {weights}; {setting}", flush=True)

    rows, t0 = [], time.time()
    res = a.resolution
    with torch.no_grad():
        for i, d in enumerate(iter_scenes(a.data, index, res=res, limit=a.limit)):
            imgs = d["ctx_rgb"][None].to(dev)
            gt = d["tgt_rgb"][None].to(dev)
            if a.pose_free:
                out = model(imgs, depth_scale=kappa)
                joint = torch.cat([imgs, gt], dim=1).float()
                c2w_all, K_all = model.predict_cameras(joint)
                nc = imgs.shape[1]
                tgt_c2w = c2w_all[:, nc:].clone()
                s = float(out["depth_scale"].reshape(-1)[0])
                if abs(s - 1.0) > 1e-9:
                    tgt_c2w[..., :3, 3] = tgt_c2w[..., :3, 3] * s
                tgt_K = norm_K(K_all[:, nc:], res)
            else:
                ctx_c2w, ctx_K = d["ctx_c2w"][None].to(dev), d["ctx_K"][None].to(dev)
                out = model(imgs, ctx_c2w, ctx_K)
                tgt_c2w, tgt_K = d["tgt_c2w"][None].to(dev), norm_K(d["tgt_K"][None].to(dev), res)
            color, _ = render(out["gaussians"], tgt_c2w, tgt_K, res, res)
            psnr = float(psnr_per_view(color, gt).flatten().mean())
            ss = ssim(color[0].clamp(0, 1), gt[0])
            lp = lpips_fn(color[0].clamp(0, 1), gt[0])
            inp = float("nan")
            if not a.pose_free:
                back, _ = render(out["gaussians"], ctx_c2w, norm_K(ctx_K, res), res, res)
                inp = float(-10 * torch.log10(((back - imgs.float()) ** 2).mean().clamp_min(1e-9)))
            rows.append((d["scene"], psnr, ss, lp, inp))
            if a.save_images:
                from torchvision.utils import save_image
                sd = os.path.join(a.out, "renders", d["scene"])
                os.makedirs(sd, exist_ok=True)
                for j in range(color.shape[1]):
                    save_image(color[0, j], os.path.join(sd, f"{j:02d}.png"))
            if (i + 1) % 100 == 0 or i == 0:
                el = time.time() - t0
                print(f"  {i + 1}/{len(chunk_of)} scenes  PSNR {sum(r[1] for r in rows) / len(rows):.3f}  "
                      f"{el / (i + 1):.2f} s/scene", flush=True)

    rows.sort(key=lambda r: r[0])
    n = len(rows)

    def mean(j):
        return sum(r[j] for r in rows) / max(1, n)
    summary = dict(protocol=a.protocol, index=os.path.basename(index), weights=a.weights, variant=a.variant,
                   pose_free=a.pose_free, kappa=kappa if a.pose_free else None, scenes=n,
                   psnr=mean(1), ssim=mean(2), lpips=mean(3), input_view_psnr=None if a.pose_free else mean(4),
                   seconds=round(time.time() - t0, 1), device=torch.cuda.get_device_name(0), torch=torch.__version__)
    with open(os.path.join(a.out, "per_scene.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "psnr", "ssim", "lpips", "input_view_psnr"])
        w.writerows(rows)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print(f"\n{a.protocol or os.path.basename(index)}: {n} scenes  PSNR {summary['psnr']:.3f}  "
          f"SSIM {summary['ssim']:.4f}  LPIPS {summary['lpips']:.4f}")

    if a.compare_paper and a.protocol:
        paper = json.load(open(os.path.join(HERE, "paper_numbers.json")))
        key = f"{a.protocol}{'_posefree' if a.pose_free else ''}@{a.variant or 're10k'}"
        ref = paper.get(key)
        if ref is None:
            print(f"(no value of the paper for {key})")
        else:
            print(f"paper ({key}): " + "  ".join(f"{k.upper()} {v}" for k, v in ref.items() if v is not None))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (c) 2026 The GradSplat Authors.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Reconstruct a scene from a few images and render it.

    # with the cameras of the views (the setting of the paper's main results)
    python demo.py --images examples/re10k_dining_room/images --cameras examples/re10k_dining_room/cameras.json \\
        --out outputs/dining_room

    # without cameras: GradSplat predicts them
    python demo.py --images my_scene/ --out outputs/my_scene
    python demo.py --video my_video.mp4 --num-frames 6 --out outputs/my_video

Writes gaussians.ply (3D Gaussian Splatting format), video.mp4 (a path through the input cameras),
steps.jpg (one novel view before the refinement steps, after the appearance steps and after the shape steps)
and cameras.json (the input cameras that were used, at the model resolution).
"""
import argparse
import os
import time

import numpy as np
import torch
from PIL import Image

from gradsplat import GradSplat
from gradsplat.export import (interpolate_path, render_path, save_cameras, save_ply, save_video, scale_K, tile)
from gradsplat.gaussians import select
from gradsplat.io import frames_from_video, list_images, load_views


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", help="folder of images (sorted by name)")
    src.add_argument("--video", help="video file; --num-frames frames are taken at regular intervals")
    ap.add_argument("--cameras", help="cameras.json with c2w and K for every image (optional)")
    ap.add_argument("--num-frames", type=int, default=6)
    ap.add_argument("--weights", default="Jingong/GradSplat", help="Hugging Face repo id or local directory")
    ap.add_argument("--variant", default=None, help="dl3dv for the weights fine-tuned on DL3DV")
    ap.add_argument("--render-size", type=int, default=448, help="side of the rendered video and step images")
    ap.add_argument("--video-frames", type=int, default=150)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = torch.device("cuda")
    os.makedirs(a.out, exist_ok=True)
    model = GradSplat.from_pretrained(a.weights, variant=a.variant).to(dev).eval()
    size = model.cfg.resolution

    if a.video:
        if a.cameras:
            ap.error("--cameras goes with --images")
        tmp = os.path.join(a.out, "frames")
        os.makedirs(tmp, exist_ok=True)
        for i, im in enumerate(frames_from_video(a.video, a.num_frames)):
            im.save(os.path.join(tmp, f"{i:03d}.jpg"), quality=95)
        paths = list_images(tmp)
    else:
        paths = list_images(a.images)
    views = load_views(paths, size=size, cameras=a.cameras)
    V = views.images.shape[0]
    if V < 2:
        ap.error("GradSplat needs at least two views")

    t0 = time.time()
    with torch.no_grad():
        out = model.infer(views.images, views.c2w, views.K, return_iterates=True)
    torch.cuda.synchronize()
    print(f"{V} views {'with' if views.c2w is not None else 'without'} cameras: "
          f"{out['gaussians']['means'].shape[1]:,} Gaussians in {time.time() - t0:.2f} s")

    gs = out["gaussians"]
    save_ply(os.path.join(a.out, "gaussians.ply"), select(gs))
    c2w, K = out["c2w"][0].cpu(), out["K"][0].cpu()
    save_cameras(os.path.join(a.out, "cameras.json"), c2w, K, size, views.names)

    R = a.render_size
    K_r = scale_K(K.mean(0), size, R)
    path = interpolate_path(c2w, a.video_frames)
    save_video(os.path.join(a.out, "video.mp4"), render_path(gs, path, K_r, R))

    # one novel view between the first two inputs: prediction, after K_a appearance steps, after K_s shape steps
    mid = interpolate_path(c2w[:2], 3, loop=False)[1:2]
    names = out["iterate_names"]
    picks = [0, names.index(f"appearance {model.cfg.appearance_steps}"), len(names) - 1]
    row = np.concatenate([render_path(out["iterates"][i], mid, K_r, R) for i in picks])
    Image.fromarray(tile(row, len(picks))).save(os.path.join(a.out, "steps.jpg"), quality=95)
    print(f"wrote {a.out}/gaussians.ply, video.mp4, steps.jpg, cameras.json")


if __name__ == "__main__":
    main()

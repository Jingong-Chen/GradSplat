# Copyright (c) 2026 The GradSplat Authors.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Loading input views: images are center-cropped to a square and resized to the model resolution, the way the
model was trained and evaluated; cameras given for the original images are adjusted accordingly.

cameras.json lists one entry per image, in the order of the sorted file names:
    {"frames": [{"file": "000.jpg", "c2w": 4x4 camera-to-world (OpenCV axes: x right, y down, z forward),
                 "K": 3x3 pixel intrinsics of the original image}, ...]}
"""
import json
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


@dataclass
class Views:
    images: torch.Tensor          # (V, 3, S, S) in [0, 1]
    c2w: torch.Tensor = None      # (V, 4, 4) or None
    K: torch.Tensor = None        # (V, 3, 3) pixel intrinsics at S x S, or None
    names: list = None


def list_images(folder):
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXT))
    if not files:
        raise FileNotFoundError(f"no images in {folder}")
    return [os.path.join(folder, f) for f in files]


def square_crop_resize(im, size, K=None):
    """im (3, H, W) float -> (3, size, size): center crop to the shorter side, then antialiased bilinear resize.
    K (3, 3) pixel intrinsics of the original image are mapped to the output."""
    _, h, w = im.shape
    s = min(h, w)
    r0, c0 = (h - s) // 2, (w - s) // 2
    im = im[:, r0:r0 + s, c0:c0 + s]
    im = F.interpolate(im[None], size=(size, size), mode="bilinear", align_corners=False, antialias=True)[0]
    if K is None:
        return im, None
    K = torch.as_tensor(K, dtype=torch.float32).clone()
    sc = size / s
    K[0, 2] -= c0
    K[1, 2] -= r0
    K[:2] *= sc
    return im, K


def load_image(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255


def load_views(paths, size=224, cameras=None):
    """paths: image files (or a folder). cameras: None, a cameras.json path or its parsed dict."""
    if isinstance(paths, str) and os.path.isdir(paths):
        paths = list_images(paths)
    frames = None
    if cameras is not None:
        meta = json.load(open(cameras)) if isinstance(cameras, str) else cameras
        frames = meta["frames"]
        if len(frames) != len(paths):
            raise ValueError(f"{len(paths)} images but {len(frames)} cameras")
        by_name = {f.get("file"): f for f in frames}
        if all(os.path.basename(p) in by_name for p in paths):
            frames = [by_name[os.path.basename(p)] for p in paths]
    ims, Ks, c2ws = [], [], []
    for i, p in enumerate(paths):
        K = None if frames is None else frames[i]["K"]
        im, K = square_crop_resize(load_image(p), size, K)
        ims.append(im)
        if frames is not None:
            Ks.append(K)
            c2ws.append(torch.as_tensor(frames[i]["c2w"], dtype=torch.float32))
    v = Views(images=torch.stack(ims), names=[os.path.basename(p) for p in paths])
    if frames is not None:
        c2w = torch.stack(c2ws)
        v.c2w = torch.linalg.inv(c2w[0]) @ c2w          # in the frame of the first view
        v.K = torch.stack(Ks)
    return v


def frames_from_video(video, n=8):
    """n frames at regular intervals of a video, as PIL images (needs imageio with its ffmpeg plugin)."""
    import imageio.v3 as iio
    frames = iio.imread(video, plugin="pyav") if _has_pyav() else iio.imread(video)
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [Image.fromarray(frames[i]) for i in idx]


def _has_pyav():
    try:
        import av  # noqa: F401
        return True
    except ImportError:
        return False

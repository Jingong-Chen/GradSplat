# Copyright (c) 2026 The GradSplat Authors.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Scenes in the chunk format of pixelSplat (RealEstate10K, and DL3DV / ScanNet++ after conversion).

A dataset root holds <root>/test/*.torch chunks and <root>/test/index.json (scene key -> chunk file). Each
chunk is a list of scenes with JPEG bytes in "images" and, per frame, "cameras" = [fx, fy, cx, cy] normalized by
the image size, two unused values and the 3 x 4 world-to-camera matrix (OpenCV axes).

Frames are center-cropped to a square and resized to the model resolution; cameras are expressed in the frame
of the first input view. Scenes are read one chunk at a time, in the order of their keys.
"""
import io
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _decode_jpg(b):
    im = Image.open(io.BytesIO(bytes(b.numpy()))).convert("RGB")
    return torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255


def center_square_resize(im, K_norm_fxfycxcy, res):
    _, h, w = im.shape
    fx, fy, cx, cy = (float(K_norm_fxfycxcy[0]) * w, float(K_norm_fxfycxcy[1]) * h,
                      float(K_norm_fxfycxcy[2]) * w, float(K_norm_fxfycxcy[3]) * h)
    s = min(h, w)
    r0, c0 = (h - s) // 2, (w - s) // 2
    im = im[:, r0:r0 + s, c0:c0 + s]
    im = F.interpolate(im[None], size=(res, res), mode="bilinear", align_corners=False, antialias=True)[0]
    sc = res / s
    K = torch.tensor([[fx * sc, 0.0, (cx - c0) * sc], [0.0, fy * sc, (cy - r0) * sc], [0.0, 0.0, 1.0]])
    return im, K


def load_frames(ex, frame_ids, res):
    ims, Ks, c2ws = [], [], []
    for fi in frame_ids:
        cam = ex["cameras"][fi]
        im, K = center_square_resize(_decode_jpg(ex["images"][fi]), cam[:4], res)
        w2c = torch.eye(4)
        w2c[:3] = torch.as_tensor(cam[6:18], dtype=torch.float32).reshape(3, 4)
        ims.append(im)
        Ks.append(K)
        c2ws.append(torch.linalg.inv(w2c))
    return torch.stack(ims), torch.stack(Ks), torch.stack(c2ws)


def index_scenes(root, index_json, limit=None):
    """Keys of the evaluation index that are present in the data, sorted, and the chunk of each."""
    eval_index = json.load(open(index_json))
    chunk_index = json.load(open(os.path.join(root, "test", "index.json")))
    keys = [k for k in sorted(eval_index) if eval_index.get(k) is not None]
    present = [k for k in keys if k in chunk_index]
    if limit is not None:
        present = present[:limit]
    return eval_index, {k: chunk_index[k] for k in present}, len(keys)


def iter_scenes(root, index_json, res=224, limit=None):
    """Yield the scenes of an evaluation index, one chunk at a time, as dicts of ctx_rgb (V, 3, res, res),
    ctx_K (V, 3, 3) in pixels, ctx_c2w (V, 4, 4), the same for the targets, and the scene key.
    Scenes whose index refers to frames beyond the end of the sequence are skipped."""
    eval_index, chunk_of, _ = index_scenes(root, index_json, limit)
    by_chunk = {}
    for k, ch in chunk_of.items():
        by_chunk.setdefault(ch, []).append(k)
    for ch in sorted(by_chunk):
        data = torch.load(os.path.join(root, "test", ch), weights_only=False)
        by_key = {e["key"]: e for e in data}
        for s in by_chunk[ch]:
            ex = by_key.get(s)
            if ex is None:
                continue
            ctx_ids, tgt_ids = eval_index[s]["context"], eval_index[s]["target"]
            if max(ctx_ids + tgt_ids) >= len(ex["images"]):
                continue
            imgs, Ks, c2ws = load_frames(ex, list(ctx_ids) + list(tgt_ids), res)
            ref = torch.linalg.inv(c2ws[0])
            c2ws = torch.einsum("ij,njk->nik", ref, c2ws)
            nc = len(ctx_ids)
            yield dict(ctx_rgb=imgs[:nc], ctx_K=Ks[:nc], ctx_c2w=c2ws[:nc],
                       tgt_rgb=imgs[nc:], tgt_K=Ks[nc:], tgt_c2w=c2ws[nc:], scene=s)
        del data, by_key

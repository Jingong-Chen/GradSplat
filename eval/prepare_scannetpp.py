#!/usr/bin/env python3
# Copyright (c) 2026 The GradSplat Authors.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Build the ScanNet++ evaluation set of the paper: the 50 scenes of the validation split nvs_sem_val, six input
and three target DSLR frames per scene, as one chunk in the format of pixelSplat.

    python eval/prepare_scannetpp.py --scannetpp data/scannetpp_raw --out data/scannetpp

--scannetpp is the root of a ScanNet++ download with <root>/data/<scene>/dslr/ and <root>/splits/nvs_sem_val.txt.
Only three assets of the validation scenes are read (set them in the download configuration of ScanNet++):
dslr_train_test_lists_path, dslr_resized_undistorted_dir and dslr_nerfstudio_transform_undistorted_path.

Frames are ordered along a greedy nearest-neighbor path over the camera positions. Six frames at regular
intervals of the first 60 positions are the inputs and three of the remaining positions are the targets.
The rule is deterministic and reads no score. Images are stored as JPEG (quality 95), cameras as normalized
intrinsics and OpenCV world-to-camera matrices. The scenes must be those of assets/indices/scannetpp_val.json, the
index of the paper (inputs at positions 0 to 5 and targets at 6 to 8 of every scene); missing scenes are reported.
"""
import argparse
import io
import json
import os

import numpy as np
import torch
from PIL import Image

N_CTX, N_TGT, SPAN = 6, 3, 60


def gl_c2w_to_cv_w2c(m):
    c2w = np.array(m, dtype=np.float64)
    c2w[:, 1] *= -1
    c2w[:, 2] *= -1
    return np.linalg.inv(c2w)[:3, :4].reshape(-1)


def cam_pos(m):
    c2w = np.array(m, dtype=np.float64)
    c2w[:, 1] *= -1
    c2w[:, 2] *= -1
    return c2w[:3, 3]


def build(src, scenes):
    recs, skipped = [], []
    for sc in scenes:
        base = os.path.join(src, sc, "dslr")
        tj = os.path.join(base, "nerfstudio", "transforms_undistorted.json")
        if not os.path.exists(tj):
            tj = os.path.join(base, "nerfstudio", "transforms.json")
        imdir = os.path.join(base, "undistorted_images")
        if not os.path.isdir(imdir):
            imdir = os.path.join(base, "resized_undistorted_images")
        if not (os.path.exists(tj) and os.path.isdir(imdir)):
            skipped.append((sc, "no transforms or image folder"))
            continue
        meta = json.load(open(tj))
        have = set(os.listdir(imdir))
        frames = [f for f in meta.get("frames", []) if f["file_path"] in have and not f.get("is_bad", False)]
        if len(frames) < N_CTX + N_TGT:
            skipped.append((sc, f"{len(frames)} usable frames"))
            continue
        W, H = meta.get("w") or meta.get("width"), meta.get("h") or meta.get("height")
        fx, fy, cx, cy = meta.get("fl_x"), meta.get("fl_y"), meta.get("cx"), meta.get("cy")
        if None in (W, H, fx, fy, cx, cy):
            skipped.append((sc, "incomplete intrinsics"))
            continue

        P = np.stack([cam_pos(f["transform_matrix"]) for f in frames])
        cur, path, left = 0, [0], set(range(1, len(P)))
        while left:
            nxt = min(left, key=lambda j: np.linalg.norm(P[cur] - P[j]))
            path.append(nxt)
            left.discard(nxt)
            cur = nxt
        span = min(len(path), SPAN)
        win = path[:span]
        ci = list(dict.fromkeys(win[round(i * (span - 1) / (N_CTX - 1))] for i in range(N_CTX)))
        rest = [f for f in win if f not in set(ci)]
        if len(ci) < N_CTX or len(rest) < N_TGT:
            skipped.append((sc, "too few frames in the window"))
            continue
        ti = list(dict.fromkeys(rest[round(i * (len(rest) - 1) / max(1, N_TGT - 1))] for i in range(N_TGT)))
        if len(ti) < N_TGT:
            skipped.append((sc, "too few targets"))
            continue
        assert not (set(ci) & set(ti))

        sel = [frames[i] for i in ci] + [frames[i] for i in ti]
        imgs, cams = [], []
        for fr in sel:
            buf = io.BytesIO()
            Image.open(os.path.join(imdir, fr["file_path"])).convert("RGB").save(buf, format="JPEG", quality=95)
            imgs.append(torch.frombuffer(bytearray(buf.getvalue()), dtype=torch.uint8))
            cams.append(np.concatenate([[fx / W, fy / H, cx / W, cy / H, 0.0, 0.0],
                                        gl_c2w_to_cv_w2c(fr["transform_matrix"])]))
        recs.append({"key": sc, "images": imgs, "cameras": torch.tensor(np.stack(cams), dtype=torch.float32),
                     "files": [fr["file_path"] for fr in sel]})
    return recs, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scannetpp", required=True, help="ScanNet++ root with data/ and splits/")
    ap.add_argument("--out", required=True, help="output root; writes <out>/test/000000.torch and index.json")
    ap.add_argument("--index", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                   "assets", "indices", "scannetpp_val.json"))
    a = ap.parse_args()
    src = os.path.join(a.scannetpp, "data")
    split = os.path.join(a.scannetpp, "splits", "nvs_sem_val.txt")
    scenes = [l.strip() for l in open(split) if l.strip()] if os.path.exists(split) else sorted(os.listdir(src))
    recs, skipped = build(src, scenes)
    for r in recs:
        r.pop("files")
    expected = json.load(open(a.index))
    missing = sorted(set(expected) - {r["key"] for r in recs})
    os.makedirs(os.path.join(a.out, "test"), exist_ok=True)
    torch.save(recs, os.path.join(a.out, "test", "000000.torch"))
    json.dump({r["key"]: "000000.torch" for r in recs}, open(os.path.join(a.out, "test", "index.json"), "w"))
    print(f"{len(recs)} scenes written to {a.out}/test, {len(skipped)} skipped")
    for s, why in skipped[:10]:
        print(f"  skipped {s}: {why}")
    if missing:
        print(f"WARNING: {len(missing)} scenes of the paper's index are missing: {missing[:5]}")


if __name__ == "__main__":
    main()

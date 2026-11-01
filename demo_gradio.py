# Copyright (c) 2026 The GradSplat Authors.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Interactive GradSplat demo: upload a few images or a video, get 3D Gaussians, a rendered video and the effect of
the refinement steps.

    pip install -e ".[demo]"
    python demo_gradio.py [--share]
"""
import argparse
import glob
import os
import tempfile

import gradio as gr
import numpy as np
import torch
from PIL import Image

from gradsplat import GradSplat
from gradsplat.export import (interpolate_path, render_path, save_cameras, save_ply, save_video, scale_K, tile,
                              to_orbit_view)
from gradsplat.gaussians import select
from gradsplat.io import frames_from_video, load_views

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "examples")
REPO = "Jingong/GradSplat"
WEIGHTS = {"RealEstate10K (paper)": None, "DL3DV fine-tuned": "dl3dv"}     # variant of the weights file
MODELS = {}


def get_model(name):
    if name not in MODELS:
        MODELS.clear()                      # one model on the GPU at a time
        torch.cuda.empty_cache()
        MODELS[name] = GradSplat.from_pretrained(REPO, variant=WEIGHTS[name]).to("cuda").eval()
    return MODELS[name]


def _paths(files):
    return [f if isinstance(f, str) else f.name for f in (files or [])]


def reconstruct(files, video, n_frames, cam_file, use_cameras, weights, render_size):
    work = tempfile.mkdtemp(prefix="gradsplat_")
    paths = sorted(_paths(files), key=os.path.basename)
    if video:
        d = os.path.join(work, "frames")
        os.makedirs(d)
        for i, im in enumerate(frames_from_video(video, int(n_frames))):
            im.save(os.path.join(d, f"{i:03d}.jpg"), quality=95)
        paths = sorted(glob.glob(os.path.join(d, "*.jpg")))
    if len(paths) < 2:
        raise gr.Error("Upload at least two images, or a video.")
    cameras = _paths([cam_file])[0] if (cam_file is not None and use_cameras) else None
    model = get_model(weights)
    size = model.cfg.resolution
    views = load_views(paths, size=size, cameras=cameras)
    out = model.infer(views.images, views.c2w, views.K, return_iterates=True)

    gs = out["gaussians"]
    ply = os.path.join(work, "gaussians.ply")
    save_ply(ply, select(gs))
    # for the 3D viewer only: placed so that its default camera opens on the view of the first input camera
    g0 = select(gs)
    fov = 2 * np.degrees(np.arctan(size / 2 / float(out["K"][0, 0, 1, 1])))
    shown = os.path.join(work, "gaussians_view.ply")
    save_ply(shown, to_orbit_view(g0, float(g0["means"][:, 2].median()), fov)[0])
    c2w, K = out["c2w"][0].cpu(), out["K"][0].cpu()
    cams = os.path.join(work, "cameras.json")
    save_cameras(cams, c2w, K, size, views.names)
    R = int(render_size)
    K_r = scale_K(K.mean(0), size, R)
    vid = os.path.join(work, "video.mp4")
    save_video(vid, render_path(gs, interpolate_path(c2w, 150), K_r, R))

    mid = interpolate_path(c2w[:2], 3, loop=False)[1:2]
    names = out["iterate_names"]
    picks = [0, names.index(f"appearance {model.cfg.appearance_steps}"), len(names) - 1]
    steps = tile(np.concatenate([render_path(out["iterates"][i], mid, K_r, R) for i in picks]), 3)
    inputs = [(np.asarray(Image.open(p).convert("RGB")), os.path.basename(p)) for p in paths]
    info = (f"**{len(paths)} views {'with' if views.c2w is not None else 'without'} cameras** · "
            f"{gs['means'].shape[1]:,} Gaussians")
    return shown, vid, steps, inputs, [ply, vid, cams], info


def load_example(name):
    d = os.path.join(EXAMPLES, name)
    return sorted(glob.glob(os.path.join(d, "images", "*"))), None, os.path.join(d, "cameras.json"), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    theme = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")
    with gr.Blocks(title="GradSplat", theme=theme, css=".gradio-container {max-width: 1280px !important}") as demo:
        gr.Markdown("# GradSplat: gradients inside the forward pass for feed-forward 3D Gaussian Splatting\n"
                    "Upload a few views of a scene (or a short video). GradSplat predicts one 3D Gaussian per pixel and "
                    "refines them with learned gradient steps on the input views, inside its forward pass. "
                    "Cameras are optional: without them, GradSplat uses its own predicted cameras. "
                    "[Project page](https://jingong-chen.github.io/GradSplat/) · "
                    "[Code](https://github.com/Jingong-Chen/GradSplat) · [Paper](https://arxiv.org/abs/2610.38951)")
        with gr.Row():
            with gr.Column(scale=4):
                files = gr.File(label="Images (2 or more)", file_count="multiple", file_types=["image"])
                video = gr.Video(label="... or a video")
                n_frames = gr.Slider(2, 16, value=6, step=1, label="Frames taken from the video")
                cam_file = gr.File(label="cameras.json (optional, see examples/)", file_types=[".json"])
                use_cameras = gr.Checkbox(True, label="Use the cameras of cameras.json")
                weights = gr.Radio(list(WEIGHTS), value="RealEstate10K (paper)", label="Weights")
                render_size = gr.Slider(224, 672, value=448, step=32, label="Rendering size (px)")
                run = gr.Button("Reconstruct", variant="primary")
                info = gr.Markdown()
            with gr.Column(scale=7):
                viewer = gr.Model3D(label="3D Gaussians (drag to rotate)", height=480, clear_color=[1, 1, 1, 1])
                video_out = gr.Video(label="Rendered path through the input cameras", autoplay=True, loop=True)
        steps = gr.Image(label="A novel view: as predicted · after the appearance steps · after the shape steps")
        with gr.Row():
            inputs_gallery = gr.Gallery(label="Input views", columns=6, height="auto", object_fit="contain")
            downloads = gr.File(label="Downloads", file_count="multiple")

        gr.Markdown("### Examples (RealEstate10K test scenes, with their cameras)")
        with gr.Row():
            for name, label in [("re10k_dining_room", "Dining room (narrow protocol)"),
                                ("re10k_rock", "Sculpture (narrow protocol)"),
                                ("re10k_bathroom", "Bathroom to bedroom (wide protocol)")]:
                if os.path.isdir(os.path.join(EXAMPLES, name)):
                    gr.Button(label, size="sm").click(lambda n=name: load_example(n), None,
                                                       [files, video, cam_file, use_cameras])
        run.click(reconstruct, [files, video, n_frames, cam_file, use_cameras, weights, render_size],
                  [viewer, video_out, steps, inputs_gallery, downloads, info])
    demo.queue().launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()

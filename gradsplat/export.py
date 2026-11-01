# Copyright (c) 2026 The GradSplat Authors.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Saving results: Gaussians as a 3DGS PLY file, renderings along a camera path, and the refinement steps."""
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from .geometry import normalize_K
from .render import render


def save_ply(path, gaussians):
    """Write the Gaussians of one scene (dict of (N, ...) tensors) in the PLY layout of 3D Gaussian Splatting,
    readable by common viewers (SuperSplat, the 3DGS viewers, nerfstudio, ...). Coordinates are those of the
    cameras (OpenCV axes: y down). The four rotation values are written in the order the gsplat rasterizer reads
    them (w, x, y, z)."""
    from plyfile import PlyData, PlyElement
    means = gaussians["means"].detach().float().cpu().numpy()
    scales = gaussians["scales"].detach().float().clamp_min(1e-12).log().cpu().numpy()
    quats = gaussians["quats"].detach().float().cpu().numpy()
    op = gaussians["opacities"].detach().float().clamp(1e-6, 1 - 1e-6)
    logit = torch.log(op / (1 - op)).cpu().numpy()
    sh = gaussians["sh"].detach().float().cpu().numpy()                 # (N, K, 3)
    f_dc = sh[:, 0, :]
    f_rest = sh[:, 1:, :].transpose(0, 2, 1).reshape(sh.shape[0], -1)    # channel-major, as 3DGS
    attrs = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)] + \
        [f"f_rest_{i}" for i in range(f_rest.shape[1])] + ["opacity"] + [f"scale_{i}" for i in range(3)] + \
        [f"rot_{i}" for i in range(4)]
    data = np.concatenate([means, np.zeros_like(means), f_dc, f_rest, logit[:, None], scales, quats], axis=1)
    arr = np.empty(data.shape[0], dtype=[(a, "f4") for a in attrs])
    arr[:] = list(map(tuple, data.astype(np.float32)))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")]).write(path)


def _quat_of(R):
    """Rotation matrix (3, 3) -> unit quaternion (w, x, y, z)."""
    m = R.double()
    t = float(m[0, 0] + m[1, 1] + m[2, 2])
    if t > 0:
        r = math.sqrt(1.0 + t) * 2
        q = [0.25 * r, (m[2, 1] - m[1, 2]) / r, (m[0, 2] - m[2, 0]) / r, (m[1, 0] - m[0, 1]) / r]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        r = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / r, 0.25 * r, (m[0, 1] + m[1, 0]) / r, (m[0, 2] + m[2, 0]) / r]
    elif m[1, 1] > m[2, 2]:
        r = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / r, (m[0, 1] + m[1, 0]) / r, 0.25 * r, (m[1, 2] + m[2, 1]) / r]
    else:
        r = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / r, (m[0, 2] + m[2, 0]) / r, (m[1, 2] + m[2, 1]) / r, 0.25 * r]
    return torch.tensor([float(v) for v in q], dtype=R.dtype, device=R.device)


def transform_gaussians(gaussians, R, t, scale=1.0):
    """Apply x -> scale * R @ x + t to the Gaussians of one scene (dict of (N, ...) tensors). Rotations and the
    degree-1 spherical harmonics turn with the scene, so renderings from cameras transformed the same way are
    unchanged. Returns the new dict and the 4 x 4 transform T to apply to camera-to-world matrices (T @ c2w,
    with the scale taken out of the rotation part)."""
    g = {k: v.detach().clone() for k, v in gaussians.items()}
    R = R.to(g["means"])
    g["means"] = scale * g["means"] @ R.T + t.to(g["means"])
    g["scales"] = g["scales"] * scale
    w1, x1, y1, z1 = _quat_of(R).to(g["quats"]).unbind(-1)
    w2, x2, y2, z2 = g["quats"].unbind(-1)
    g["quats"] = torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                              w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], dim=-1)
    if g["sh"].shape[-2] > 1:      # degree 1: color = ... + C1 * (-y sh1 + z sh2 - x sh3) = ... + C1 * a . d
        sh = g["sh"]
        a = torch.stack([-sh[:, 3], -sh[:, 1], sh[:, 2]], dim=1)          # (N, 3 xyz, 3 rgb)
        a = torch.einsum("ij,njc->nic", R, a)
        sh[:, 1], sh[:, 2], sh[:, 3] = -a[:, 1], a[:, 2], -a[:, 0]
    T = torch.eye(4, dtype=g["means"].dtype, device=g["means"].device)
    T[:3, :3] = scale * R
    T[:3, 3] = t.to(g["means"])
    return g, T


def to_display_frame(gaussians):
    """Gaussians of one scene for a y-up viewer: rotated by 180 degrees about x (OpenCV y-down, z-forward axes become
    y-up, z-backward) and centered at the median of the centers. Returns the new dict and the transform T."""
    c = gaussians["means"].median(dim=0).values
    R = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=c.device))
    return transform_gaussians(gaussians, R, -(R.to(c) @ c))


def to_orbit_view(gaussians, look_depth, fov_y_deg, radius=5.0, alpha=0.5, beta=0.5, focal=1132.0, height=440):
    """Gaussians of one scene placed for the default orbit camera of gsplat.js (the 3D viewer of Gradio), so that
    the viewer opens on the first input camera's view: the point at depth look_depth on its optical axis goes to
    the orbit target (the origin), the viewing direction becomes the default one, and the scale makes the field of
    view of the input camera fill the viewer (focal length in pixels, canvas height). Returns the dict and T."""
    p0 = torch.tensor([radius * math.sin(alpha) * math.cos(beta), -radius * math.sin(beta),
                       -radius * math.cos(alpha) * math.cos(beta)])
    b = -p0 / p0.norm()
    th, ps = math.asin(-float(b[1])), math.atan2(float(b[0]), float(b[2]))
    Rx = torch.tensor([[1, 0, 0], [0, math.cos(th), -math.sin(th)], [0, math.sin(th), math.cos(th)]])
    Ry = torch.tensor([[math.cos(ps), 0, math.sin(ps)], [0, 1, 0], [-math.sin(ps), 0, math.cos(ps)]])
    R0 = Ry @ Rx                                            # camera-to-world rotation of the default camera
    assert torch.allclose(R0 @ torch.tensor([0.0, 0.0, 1.0]), b, atol=1e-6)
    s = radius * (height / 2 / focal) / (look_depth * math.tan(math.radians(fov_y_deg) / 2))
    dev = gaussians["means"].device
    R0 = R0.to(dev)
    t = -(s * R0 @ torch.tensor([0.0, 0.0, float(look_depth)], device=dev))
    return transform_gaussians(gaussians, R0, t, s)


def load_ply(path, device="cpu"):
    """Read a PLY written by save_ply back into a dict of (N, ...) tensors."""
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]
    names = v.data.dtype.names
    col = lambda n: torch.from_numpy(np.asarray(v[n], dtype=np.float32))  # noqa: E731
    n_rest = len([n for n in names if n.startswith("f_rest_")])
    k = n_rest // 3 + 1
    rest = torch.stack([col(f"f_rest_{i}") for i in range(n_rest)], 1).reshape(-1, 3, k - 1).transpose(1, 2)
    dc = torch.stack([col(f"f_dc_{i}") for i in range(3)], 1)[:, None]
    return dict(means=torch.stack([col("x"), col("y"), col("z")], 1).to(device),
                scales=torch.stack([col(f"scale_{i}") for i in range(3)], 1).exp().to(device),
                quats=torch.stack([col(f"rot_{i}") for i in range(4)], 1).to(device),
                opacities=torch.sigmoid(col("opacity")).to(device),
                sh=torch.cat([dc, rest], 1).to(device))


def _slerp(q0, q1, t):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - t) * th) * q0 + math.sin(t * th) * q1) / math.sin(th)


def _mat_to_quat(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def interpolate_path(c2w, n_frames=120, loop=True):
    """A smooth camera path through the input cameras (V, 4, 4): Catmull-Rom on the centers, slerp on the
    rotations, back and forth along the views when loop=True. Returns (n_frames, 4, 4)."""
    c2w = c2w.detach().double().cpu().numpy()
    V = c2w.shape[0]
    if V == 1:
        return torch.from_numpy(np.repeat(c2w, n_frames, 0)).float()
    keys = list(range(V)) + (list(range(V - 2, 0, -1)) if loop and V > 2 else ([0] if loop else []))
    P = c2w[keys, :3, 3]
    Q = [_mat_to_quat(c2w[k, :3, :3]) for k in keys]
    segs = len(keys) - 1
    out = []
    for f in range(n_frames):
        u = f / n_frames * segs if loop else f / max(1, n_frames - 1) * segs
        i = min(int(u), segs - 1)
        t = u - i
        p0, p1, p2, p3 = P[max(i - 1, 0)], P[i], P[i + 1], P[min(i + 2, segs)]
        pos = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
                     (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)
        M = np.eye(4)
        M[:3, :3] = _quat_to_mat(_slerp(Q[i], Q[i + 1], t))
        M[:3, 3] = pos
        out.append(M)
    return torch.from_numpy(np.stack(out)).float()


@torch.no_grad()
def render_path(gaussians, c2w_path, K, size, batch=16):
    """Render flattened Gaussians (1, N, ...) along a path (T, 4, 4) with pixel intrinsics K (3, 3) given at
    `size`. Returns uint8 frames (T, size, size, 3)."""
    dev = gaussians["means"].device
    Kn = normalize_K(K.to(dev).float()[None, None], size, size)
    frames = []
    for i in range(0, c2w_path.shape[0], batch):
        cp = c2w_path[i:i + batch].to(dev)[None]
        col, _ = render(gaussians, cp, Kn.expand(1, cp.shape[1], 3, 3), size, size)
        frames.append((col[0].permute(0, 2, 3, 1).clamp(0, 1) * 255 + 0.5).to(torch.uint8).cpu())
    return torch.cat(frames).numpy()


def scale_K(K, from_size, to_size):
    K = K.clone().float()
    K[..., :2, :] *= to_size / from_size
    return K


def save_video(path, frames, fps=30):
    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as w:
        for f in frames:
            w.append_data(f)


def save_cameras(path, c2w, K, size, names=None):
    frames = []
    for i in range(c2w.shape[0]):
        f = {"c2w": c2w[i].tolist(), "K": K[i].tolist()}
        if names:
            f["file"] = names[i]
        frames.append(f)
    with open(path, "w") as fh:
        json.dump({"image_size": [size, size], "frames": frames}, fh, indent=1)


def tile(images, cols):
    """uint8 images (N, H, W, 3) -> one grid image."""
    n, h, w, _ = images.shape
    rows = int(math.ceil(n / cols))
    grid = np.full((rows * h, cols * w, 3), 255, np.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = images[i]
    return grid


def upsample_uint8(img, factor):
    t = torch.from_numpy(img).permute(2, 0, 1)[None].float()
    t = F.interpolate(t, scale_factor=factor, mode="nearest")
    return t[0].permute(1, 2, 0).to(torch.uint8).numpy()

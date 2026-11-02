# Third-party notices

GradSplat is built on the following work. Their licenses apply to the corresponding parts of this repository and of
the released weights.

## VGGT-Ω

GradSplat's backbone, camera head and dense head are those of VGGT-Ω, and the GradSplat weights are trained from the
VGGT-Ω weights. The code in `gradsplat/_vggt_omega/` is a copy of the VGGT-Ω inference package (imports renamed,
otherwise unchanged), `gradsplat/heads.py` extends its dense head and `gradsplat/model.py` follows its forward pass.

Copyright (c) Meta Platforms, Inc. and affiliates. Licensed under the FAIR Noncommercial Research License, a copy of
which is provided in [`gradsplat/_vggt_omega/LICENSE`](gradsplat/_vggt_omega/LICENSE). As that license requires, the
code in `gradsplat/_vggt_omega/`, `gradsplat/heads.py` and `gradsplat/model.py`, the GradSplat weights and any
derivative of them are distributed under the same terms, for noncommercial research use only. The files of
`gradsplat/_vggt_omega/models/layers/` that carry the notice of the DINOv3 License are also subject to that license, a
copy of which is provided in [`gradsplat/_vggt_omega/LICENSE_DINOv3.md`](gradsplat/_vggt_omega/LICENSE_DINOv3.md).

## Evaluation indices

The files in `assets/indices/` list, for every scene, the input and target frames of an evaluation protocol.

| File | Source | License |
|:---|:---|:---|
| `re10k_narrow_volsplat.json` | the RealEstate10K index released with [VolSplat](https://github.com/ziplab/VolSplat) | MIT, Copyright (c) 2025 Zhuang Intelligent Processing Lab |
| `re10k_wide_yonosplat.json` | the RealEstate10K index released with [YoNoSplat](https://github.com/cvg/yonosplat) | MIT, Copyright (c) 2026 Botao Ye |
| `dl3dv_resplat.json` | `dl3dv_start_0_distance_60_ctx_6v_tgt_8v.json` of [ReSplat](https://github.com/cvg/resplat) | MIT, Copyright (c) 2026 Haofei Xu |
| `re10k_views_*.json` | built in this work from the scenes and targets of the narrow index | MIT, as the GradSplat code |
| `scannetpp_val.json` | built in this work (`eval/prepare_scannetpp.py`) | MIT, as the GradSplat code |

The MIT License of these repositories reads:

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Example data

The images in `examples/` are six frames of three RealEstate10K test scenes, as distributed in the test set of
pixelSplat (each folder has a `SOURCE.md`). The camera data of
[RealEstate10K](https://google.github.io/realestate10k/) is licensed by Google LLC under a Creative Commons
Attribution 4.0 International License. The frames come from the YouTube videos that the dataset lists and are included
only as examples for noncommercial research.

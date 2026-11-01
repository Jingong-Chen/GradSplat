# Evaluation

`evaluate.py` reproduces the GradSplat rows of the paper's tables from the released weights. Every protocol is one
command; the evaluation indices of the paper are in [`../assets/indices/`](../assets/indices).

```bash
pip install -e ".[eval]"     # from the repository root: adds LPIPS
```

## Data

All datasets use the chunk format of [pixelSplat](https://github.com/dcharatan/pixelsplat): `<root>/test/*.torch`
and `<root>/test/index.json`.

| Dataset | How to get it | `--data` |
|:---|:---|:---|
| RealEstate10K | the processed 360p test set linked in the [pixelSplat README](https://github.com/dcharatan/pixelsplat#acquiring-datasets) | `data/re10k` |
| DL3DV | the 140 scenes of [DL3DV-Benchmark](https://huggingface.co/datasets/DL3DV/DL3DV-Benchmark), converted at 270 × 480 (`images_8`) as described in the [ReSplat DATASETS.md](https://github.com/cvg/resplat/blob/main/DATASETS.md) (`convert_dl3dv_test.py`, then `generate_dl3dv_index.py`) | `data/dl3dv` |
| ScanNet++ | the validation split `nvs_sem_val` of [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/), converted with `python eval/prepare_scannetpp.py --scannetpp <download> --out data/scannetpp` | `data/scannetpp` |

DL3DV and ScanNet++ have their own terms of use; download them from their sources. For ScanNet++, only three assets
of the validation scenes are needed: `dslr_train_test_lists_path`, `dslr_resized_undistorted_dir` and
`dslr_nerfstudio_transform_undistorted_path`. One DL3DV scene has images of half the size in the benchmark download and
is dropped by the conversion, so 139 of the 140 scenes of the index are evaluated, as in the paper.

## Protocols

| `--protocol` | Index | Scenes | Views |
|:---|:---|:---:|:---|
| `re10k_narrow` | released with [VolSplat](https://github.com/ziplab/VolSplat) | 7,128 | 6 input, 8 target |
| `re10k_wide` | released with [YoNoSplat](https://github.com/cvg/yonosplat) | 1,600 | 6 input, 8 target |
| `re10k_views_{2,3,4,6,12,24,32,42}` | the targets of the narrow protocol with V input views | 6,444 | V input, 8 target |
| `dl3dv` | released with [ReSplat](https://github.com/cvg/resplat) | 140 (139 in the data) | 6 input, 8 target |
| `scannetpp` | built in this work, on `nvs_sem_val` | 50 | 6 input, 3 target |

Views are center-cropped to a square and resized to 224 × 224 pixels. By default GradSplat receives the cameras of
the input views and is rendered at the ground-truth target cameras. With `--pose-free` it receives the images alone:
the Gaussians use the predicted cameras with the constant κ, and the target cameras are estimated from the input
and target images together, the protocol of AnySplat. `--variant dl3dv` evaluates the weights fine-tuned on DL3DV.

```bash
python eval/evaluate.py --protocol re10k_narrow --data data/re10k --out results/re10k_narrow --compare-paper
python eval/evaluate.py --protocol re10k_wide --data data/re10k --out results/re10k_wide --compare-paper
python eval/evaluate.py --protocol re10k_narrow --data data/re10k --pose-free --out results/re10k_narrow_posefree --compare-paper
python eval/evaluate.py --protocol re10k_views_24 --data data/re10k --out results/re10k_views_24 --compare-paper
python eval/evaluate.py --protocol dl3dv --data data/dl3dv --out results/dl3dv --compare-paper
python eval/evaluate.py --protocol dl3dv --data data/dl3dv --variant dl3dv --out results/dl3dv_ft --compare-paper
python eval/evaluate.py --protocol scannetpp --data data/scannetpp --out results/scannetpp --compare-paper
```

Each run writes `per_scene.csv` (PSNR, SSIM, LPIPS and the PSNR of the input views rendered back) and `summary.json`.
`--limit N` evaluates the first N scenes for a quick check, and `--save-images` keeps the renderings.

## Metrics

The metrics are those every method of the paper was measured with. PSNR is computed per target view and averaged
over the views of a scene; SSIM uses an 11 × 11 uniform window; LPIPS uses the VGG network of the `lpips` package.
The value of a protocol is the mean over its scenes.

## Results

The values of the paper, computed on NVIDIA GH200 GPUs. `--compare-paper` prints them next to the values of a run.

| Arguments | Scenes | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|:---|:---:|:---:|:---:|:---:|
| `--protocol re10k_narrow` | 7,128 | 33.683 | 0.9593 | 0.0640 |
| `--protocol re10k_wide` | 1,600 | 26.755 | 0.8993 | 0.1301 |
| `--protocol re10k_narrow --pose-free` | 7,128 | 26.914 | 0.8863 | 0.1053 |
| `--protocol re10k_wide --pose-free` | 1,600 | 20.901 | 0.6945 | 0.2426 |
| `--protocol dl3dv` | 139 | 24.787 | 0.8448 | 0.1919 |
| `--protocol dl3dv --variant dl3dv` | 139 | 26.104 | 0.8802 | 0.1505 |
| `--protocol scannetpp` | 50 | 21.609 | 0.7540 | 0.3184 |

PSNR of `--protocol re10k_views_V` (6,444 scenes):

| V | 2 | 3 | 4 | 6 | 12 | 24 | 32 | 42 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| default | 28.492 | 31.490 | 32.501 | 33.636 | 34.631 | 34.401 | 34.382 | 34.801 |
| `--pose-free` | – | – | – | 26.792 | 28.896 | 30.098 | 30.517 | 31.007 |

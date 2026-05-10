# Data on AutoDL

Large remote sensing datasets should live on the AutoDL machine or AutoDL data
disk, not on the local laptop. Keep this repository local for development, then
sync code to AutoDL and download datasets there.

## Recommended Directory

```text
/autodl-tmp/
  GeoToken/
  datasets/
    us3d_dfc2019/
    whu_omvs/
  outputs/
    geotoken/
```

## Dataset Choices

### US3D / DFC2019

Use this as the first serious benchmark when possible. It provides multi-view
WorldView-3 satellite imagery and LiDAR DSM ground truth.

The public DFC2019 files are hosted through IEEE DataPort:

```text
https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019
```

IEEE DataPort may require browser login or manual download. In that case,
download on AutoDL through the browser/terminal workflow supplied by AutoDL, or
upload the downloaded archive once to the AutoDL data disk.

### WHU-OMVS

Use this for oblique multi-view geometry and building-detail experiments. The
Ada-MVS authors document the expected layout:

```text
WHU_OMVS/
  train/
  test/
  predict/
```

Official dataset entry:

```text
http://gpcv.whu.edu.cn/data
```

## Intermediate Manifest

GeoToken uses a manifest layer so each dataset can be normalized into the same
training interface.

JSONL example:

```json
{"id":"tile_0001","images":["images/tile_0001_v0.png","images/tile_0001_v1.png","images/tile_0001_v2.png"],"camera_params":"cameras/tile_0001.json","gt_dsm":"dsm/tile_0001.npy"}
```

The dataset returns:

```text
images:        [V, 3, H, W]
camera_params: RPC or intrinsic/extrinsic metadata
gt_dsm:        [H, W]
valid_mask:    [H, W]
```

## AutoDL Setup Sketch

```bash
cd /autodl-tmp
git clone <your-repo-url> GeoToken
cd GeoToken
pip install -e .

python scripts/run_compression_sweep.py --dry-run
```

## First Sanity Check

Before spending time on the full dataset, run a cheap learning sanity check:

```bash
python scripts/run_sanity_check.py \
  --steps 200 \
  --token-count 128 \
  --query-points 1024
```

This uses synthetic multi-view geometry and should show RMSE dropping quickly. If
it does not, debug the model before touching the real dataset.

After building a real manifest, overfit one small real batch:

```bash
python scripts/run_sanity_check.py \
  --manifest /autodl-tmp/datasets/us3d_dfc2019/train_manifest.jsonl \
  --data-root /autodl-tmp/datasets/us3d_dfc2019 \
  --steps 200 \
  --batch-size 2 \
  --token-count 128
```

This is the "confidence run": it does not prove the paper, but it should confirm
that the code path, data, rays, and DSM supervision can learn something.

After a dataset manifest is built:

```bash
python scripts/validate_manifest.py \
  --manifest /autodl-tmp/datasets/us3d_dfc2019/train_manifest.jsonl \
  --root /autodl-tmp/datasets/us3d_dfc2019 \
  --min-views 3 \
  --check-open
```

```bash
python scripts/run_compression_sweep.py \
  --config configs/compression_sweep.json \
  --manifest /autodl-tmp/datasets/us3d_dfc2019/train_manifest.jsonl \
  --val-fraction 0.1 \
  --token-count 128 \
  --epochs 1 \
  --limit-steps 20
```

If your data has already been normalized into one folder per sample:

```text
sample_0001/
  images/
    view_00.png
    view_01.png
    view_02.png
  dsm.npy
  camera.json
```

you can build a manifest with:

```bash
python scripts/build_manifest_from_folders.py \
  --root /autodl-tmp/datasets/us3d_dfc2019/normalized \
  --output /autodl-tmp/datasets/us3d_dfc2019/train_manifest.jsonl
```

TensorBoard logs are written under each token-count output directory:

```bash
tensorboard --logdir /autodl-tmp/GeoToken/outputs/compression_sweep
```

If you already have separate manifests:

```bash
python scripts/run_compression_sweep.py \
  --config configs/compression_sweep.json \
  --manifest /autodl-tmp/datasets/us3d_dfc2019/train_manifest.jsonl \
  --val-manifest /autodl-tmp/datasets/us3d_dfc2019/val_manifest.jsonl
```

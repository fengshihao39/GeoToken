# GeoToken

Geometry-preserving visual tokenization for remote sensing 3D reconstruction.

GeoToken studies whether high-resolution multi-view remote sensing imagery can be
compressed into a small set of geometry-aware latent tokens while preserving the
3D structure needed for DSM/depth recovery and downstream reconstruction.

## Core Question

How many visual tokens are needed to preserve remote sensing 3D geometry?

The first milestone is a compression curve:

- Input: multi-view image patches plus camera/RPC-derived ray attributes.
- Bottleneck: fixed-size GeoTokens, e.g. 32, 64, 128, 256, 512.
- Output: DSM/depth, structural edges, and surface normals.
- Metrics: DSM RMSE/MAE, boundary F1, height-discontinuity error,
  reprojection error, memory, and runtime.

## Initial Repo Layout

```text
configs/
  compression_sweep.json       # Standard-library config used by scripts.
  compression_sweep.yaml       # Token-count sweep template.
docs/
  data_autodl.md               # AutoDL dataset placement and manifest format.
  research_plan.md             # Paper story and staged roadmap.
  experiment_protocol.md       # Main experiments and ablations.
scripts/
  build_manifest_from_folders.py # Build a manifest from normalized sample folders.
  run_compression_sweep.py     # Entry point for experiment 1.
  run_sanity_check.py          # Cheap synthetic or one-batch overfit confidence run.
  validate_manifest.py         # Check manifest paths and camera metadata.
src/geotoken/
  backbone.py                  # Raw image to patch feature extraction.
  dataset.py                   # Manifest dataset returning images/camera/DSM.
  rays.py                      # Camera/RPC metadata to 10-D ray convention.
  metrics.py                   # DSM and structural metrics.
  ray_encoding.py              # Ray-aware positional encoding.
  tokenizer.py                 # Cross-view token bottleneck.
  decoder.py                   # Query-based token-to-geometry decoder.
  swin_decoder.py              # Swin-style dense decoder for ablations.
  losses.py                    # Geometry-preserving losses.
```

## Working Hypothesis

Remote sensing geometry is lower entropy than raw imagery. Roof planes,
facades, terrain surfaces, and height discontinuities should be compressible into
a small cross-view token set, provided the tokenizer sees ray geometry and is
trained with metric consistency objectives.

## First Implementation Target

Build and test a minimal `GeoTokenizer -> GeometryDecoder` pipeline on cropped
multi-view patches. The key plot is token count versus DSM accuracy. The plot is
more important than a large system at this stage.

The tokenizer uses two-stage view aggregation: view-local latent tokens are
formed before cross-view GeoTokens are produced. Rays use the base 10-D geometry
convention plus optional normalized height hypotheses (`ray_dim = 10 + N`).

The default query decoder trains with sparse random DSM coordinate queries
(`query_points: 2048`) to avoid dense self-attention memory blowups. Validation
metrics and DSM preview images are logged to TensorBoard.

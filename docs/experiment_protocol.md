# Experiment Protocol

## Experiment 1: Token Count vs Geometry Accuracy

Purpose: identify the geometry information bottleneck.

Sweep token counts:

- 32
- 64
- 128
- 256
- 512
- 1024

Report:

- DSM RMSE and MAE.
- Height discontinuity MAE near building edges.
- Boundary F1.
- Reprojection error.
- Peak GPU memory.
- Training and inference time.

Expected result: accuracy should improve rapidly at small token counts, then
reach a knee point where additional tokens provide diminishing returns.

For the query-based main decoder, training uses sparse random coordinate
queries, e.g. 2048 DSM points per patch. Dense full-map inference should be done
separately, preferably in chunks, to avoid quadratic query self-attention memory.

## Experiment 2: Geometry Mechanism Ablation

Compare:

- Full GeoToken.
- No ray-aware encoding.
- Single-view tokenizer.
- Late-fusion multi-view tokenizer.
- Generic visual tokens from DINO/MAE-style features.
- Random latent bottleneck with the same token count.

The key claim is not just better accuracy, but better preservation of metric
consistency and height discontinuities under high compression.

## Experiment 3: Token-Guided Reconstruction

Integrate GeoToken outputs into 3DGS or NeRF:

- Initialize Gaussian heights from decoded DSM.
- Use structural edges to preserve discontinuities.
- Use normals/planarity to regularize weak-texture areas.

Report:

- Convergence steps to target PSNR/DSM error.
- Final DSM error.
- Weak-texture and shadow-region robustness.
- Runtime and memory.

Reprojection loss is intentionally disabled in the initial config until the
US3D/WHU camera adapters provide trustworthy RPC or intrinsic/extrinsic
projection functions. It should be enabled only after that adapter is verified.

## Geometry-Aware Tokenizer Ablation

The default tokenizer uses separable view aggregation:

```text
per-view features + ray encoding
  -> shared view-local latent queries
  -> inter-view token mixer
  -> final GeoTokens
```

Compare against the older flatten-all-views tokenizer in a future ablation if
needed. Full epipolar/RPC-biased attention should wait until dataset-specific
camera adapters are verified.

Height hypotheses are represented as optional ray dimensions. The current config
uses `ray_dim=16`, i.e. the 10-D base ray plus 6 normalized height offsets across
`+-50m`.

Reprojection loss remains disabled until true camera/RPC projection is available;
otherwise it would optimize a false geometry signal.

## Minimum Baselines

- Classical MVS or stereo matching baseline if available.
- Dense CNN/UNet DSM prediction without token bottleneck.
- Token bottleneck without ray encoding.
- Token bottleneck without cross-view joint compression.
- Query-based dense decoder versus Swin-style hierarchical decoder.

## Decoder Ablation

```text
GeoTokens [B, T, C]
  -> learned 8x8 grid queries cross-attend to GeoTokens
  -> window attention + token-guided upsampling
  -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256
  -> DSM / edge / normal heads
```

The default decoder remains query-based. The Swin-style dense decoder should be
reported as an ablation against the main architecture, with both accuracy and
runtime/memory included.

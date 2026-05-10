# GeoToken Research Plan

## Thesis

GeoToken reframes remote sensing 3D reconstruction as a geometry bottleneck
problem: instead of optimizing dense pixels directly, learn a compact set of
cross-view tokens that preserves metric 3D structure.

## Scientific Questions

1. What is the task-near-lossless compression limit for DSM/depth recovery?
2. Which geometry cues must survive the bottleneck: rays, disparity, boundaries,
   normals, or semantic structure?
3. Can discrete/latent visual tokens be decoded back into continuous metric
   coordinates with stable cross-view consistency?
4. Do GeoTokens improve 3DGS/NeRF initialization and convergence in weak-texture
   regions?

## Model Concept

The first model should be deliberately small and auditable:

1. A 2D image backbone extracts dense per-view patch features.
2. Ray-aware encoding injects camera center, ray direction, pixel coordinate,
   scale, and view identity.
3. Shared view-local latent queries first compress each view independently.
4. Inter-view attention mixes those view tokens into a compact cross-view set.
5. Final GeoTokens read from the cross-view token set.
6. A query-based geometry decoder maps coordinate/ray queries back to continuous
   DSM/depth, structural edges, and normals.

The main model keeps the decoder query-based because the central question is
whether compact GeoTokens can support continuous coordinate recovery. A
Swin-style dense decoder is retained as an ablation to test whether hierarchical
window attention improves efficiency or spatial regularity without changing the
tokenizer. Height hypotheses are exposed as ray dimensions, but they are an
interface for later height-aware projection bias rather than a full geometric
cost volume.

This makes the bottleneck explicit: changing the number of latent queries changes
the geometric information capacity.

## Paper Story

The paper should lead with the question rather than the architecture:

> How many tokens are needed to preserve 3D geometry in remote sensing?

The strongest evidence is a clear compression curve plus ablations proving that
ray-aware, cross-view tokenization preserves geometry better than generic visual
tokens or late-fusion baselines.

## Milestones

### M0: Repo and Protocol

- Define model interfaces.
- Define the compression sweep.
- Define metrics and ablations.

### M1: DSM Bottleneck Prototype

- Train on cropped multi-view patches.
- Predict DSM/depth from 32-1024 tokens.
- Produce the first compression ratio versus accuracy curve.

### M2: Structural Geometry

- Add edge and normal heads.
- Evaluate building boundary F1 and height discontinuity error.
- Compare to generic DINO/MAE features.

### M3: Reconstruction Integration

- Use decoded DSM/structure to initialize 3DGS heights.
- Add token-derived smoothness and planar priors.
- Report convergence speed and weak-texture robustness.

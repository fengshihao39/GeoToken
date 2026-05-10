# Swin Decoder Ablation Notes

The main GeoToken decoder is the query-based `GeometryDecoder` in
`src/geotoken/decoder.py`. It is the default because the central scientific
question is continuous coordinate recovery from compact GeoTokens.

The Swin-style decoder is kept as an ablation in
`src/geotoken/swin_decoder.py`.

## When To Use It

Use `SwinGeometryDecoder` to test whether a dense hierarchical decoder improves:

- runtime and memory at high output resolution,
- spatial regularity of DSM predictions,
- building-edge sharpness,
- multi-scale geometry recovery.

It should not replace the main architecture in the core compression-limit
experiment unless results show a compelling reason.

## Config

The default compression sweep uses:

```json
"decoder": "query"
```

The Swin ablation config uses:

```json
"decoder": "swin_dense"
```

Run the ablation with:

```bash
python scripts/run_compression_sweep.py \
  --config configs/ablation_swin_decoder.json \
  --manifest /autodl-tmp/datasets/us3d_dfc2019/train_manifest.jsonl
```

## Architecture

```text
GeoTokens [B, T, C]
  -> learned 8x8 grid queries cross-attend to GeoTokens
  -> Swin-style window attention
  -> token-guided upsampling stages
  -> DSM / edge / normal heads
```

The implementation is pure PyTorch and does not depend on `timm`.


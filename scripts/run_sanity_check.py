from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_synthetic_batch(
    batch_size: int,
    views: int,
    size: int,
    device,
) -> dict[str, Any]:
    import torch

    y = torch.linspace(-1.0, 1.0, size, device=device)
    x = torch.linspace(-1.0, 1.0, size, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    images = []
    dsms = []
    for item in range(batch_size):
        phase = item / max(batch_size - 1, 1)
        dsm = 0.25 * xx + 0.15 * yy
        dsm = dsm + 0.55 * rectangle(xx, yy, -0.65 + 0.2 * phase, -0.1, -0.55, 0.05)
        dsm = dsm + 0.35 * rectangle(xx, yy, 0.1, 0.65, -0.1 + 0.1 * phase, 0.45)
        dsm = dsm + 0.08 * torch.sin(4.0 * xx + phase) * torch.cos(3.0 * yy)

        view_images = []
        for view in range(views):
            shift = (view - views // 2) * 2
            shifted = torch.roll(dsm, shifts=shift, dims=1)
            rgb = torch.stack(
                [
                    normalize(shifted),
                    normalize(xx + 0.15 * view),
                    normalize(yy - 0.1 * view),
                ],
                dim=0,
            )
            view_images.append(rgb)

        images.append(torch.stack(view_images, dim=0))
        dsms.append(dsm)

    gt_dsm = torch.stack(dsms, dim=0)
    gt_edges = build_edges(gt_dsm)
    gt_normals = build_normals(gt_dsm)
    valid_mask = torch.ones_like(gt_dsm, dtype=torch.bool)
    camera_params = [make_camera_params(views) for _ in range(batch_size)]

    return {
        "images": torch.stack(images, dim=0),
        "gt_dsm": gt_dsm,
        "gt_edges": gt_edges,
        "gt_normals": gt_normals,
        "valid_mask": valid_mask,
        "camera_params": camera_params,
    }


def rectangle(xx, yy, x0: float, x1: float, y0: float, y1: float):
    return ((xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)).float()


def normalize(value):
    return (value - value.min()) / (value.max() - value.min()).clamp_min(1e-6)


def build_edges(dsm):
    import torch

    grad_x = torch.zeros_like(dsm)
    grad_y = torch.zeros_like(dsm)
    grad_x[:, :, 1:] = dsm[:, :, 1:] - dsm[:, :, :-1]
    grad_y[:, 1:, :] = dsm[:, 1:, :] - dsm[:, :-1, :]
    magnitude = torch.sqrt(grad_x.square() + grad_y.square())
    threshold = torch.quantile(magnitude.flatten(1), 0.9, dim=1).reshape(-1, 1, 1)
    return magnitude >= threshold


def build_normals(dsm):
    import torch

    grad_x = torch.zeros_like(dsm)
    grad_y = torch.zeros_like(dsm)
    grad_x[:, :, 1:-1] = 0.5 * (dsm[:, :, 2:] - dsm[:, :, :-2])
    grad_y[:, 1:-1, :] = 0.5 * (dsm[:, 2:, :] - dsm[:, :-2, :])
    normals = torch.stack([-grad_x, -grad_y, torch.ones_like(dsm)], dim=1)
    return torch.nn.functional.normalize(normals, dim=1)


def make_camera_params(views: int) -> dict[str, Any]:
    view_params = []
    for view in range(views):
        tx = (view - views // 2) * 0.1
        view_params.append(
            {
                "intrinsics": [[80.0, 0.0, 32.0], [0.0, 80.0, 32.0], [0.0, 0.0, 1.0]],
                "extrinsics": [
                    [1.0, 0.0, 0.0, tx],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        )
    return {"views": view_params}


def load_manifest_batch(args, device):
    import torch
    from torch.utils.data import DataLoader

    from geotoken import MultiViewGeometryDataset, geotoken_collate

    dataset = MultiViewGeometryDataset(
        manifest_path=args.manifest,
        root=args.data_root,
        views=args.views,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=geotoken_collate,
    )
    batch = next(iter(loader))
    for key in ("images", "gt_dsm", "gt_edges", "gt_normals", "valid_mask"):
        batch[key] = batch[key].to(device)
    return batch


def sample_queries(batch, ray_builder, image_size, query_points: int, device, dtype):
    import torch

    gt_dsm = batch["gt_dsm"]
    gt_edges = batch["gt_edges"]
    gt_normals = batch["gt_normals"]
    valid_mask = batch["valid_mask"]

    dense_rays = ray_builder(
        camera_params=batch["camera_params"],
        views=1,
        feature_size=gt_dsm.shape[-2:],
        image_size=image_size,
        device=device,
        dtype=dtype,
    )[:, 0]

    batch_size, pixels, ray_dim = dense_rays.shape
    flat_dsm = gt_dsm.flatten(1)
    flat_edges = gt_edges.flatten(1)
    flat_valid = valid_mask.flatten(1)
    flat_normals = gt_normals.permute(0, 2, 3, 1).reshape(batch_size, pixels, 3)

    rays = []
    dsms = []
    edges = []
    normals = []
    masks = []
    for item in range(batch_size):
        valid = torch.nonzero(flat_valid[item], as_tuple=False).squeeze(1)
        if valid.numel() == 0:
            valid = torch.arange(pixels, device=device)
        draw = torch.randint(0, valid.numel(), (query_points,), device=device)
        index = valid[draw]
        rays.append(dense_rays[item, index])
        dsms.append(flat_dsm[item, index])
        edges.append(flat_edges[item, index])
        normals.append(flat_normals[item, index])
        masks.append(flat_valid[item, index])

    return {
        "query_rays": torch.stack(rays).reshape(batch_size, query_points, ray_dim),
        "gt_dsm": torch.stack(dsms),
        "gt_edges": torch.stack(edges),
        "gt_normals": torch.stack(normals),
        "valid_mask": torch.stack(masks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a cheap GeoToken sanity check.")
    parser.add_argument("--manifest", type=Path, help="Optional real manifest for one-batch overfit.")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/sanity/sanity_summary.json"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--token-count", type=int, default=128)
    parser.add_argument("--query-points", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    import torch

    from geotoken import CameraToRays, GeoTokenizer, GeometryDecoder, SimpleImageBackbone
    from geotoken.losses import GeometryLoss
    from geotoken.metrics import dsm_rmse

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.manifest:
        batch = load_manifest_batch(args, device)
        mode = "manifest_overfit"
    else:
        batch = make_synthetic_batch(args.batch_size, args.views, args.size, device)
        mode = "synthetic"

    image_size = batch["images"].shape[-2:]
    image_feature_dim = 128
    ray_dim = 16
    backbone = SimpleImageBackbone(out_dim=image_feature_dim).to(device)
    tokenizer = GeoTokenizer(
        image_feature_dim=image_feature_dim,
        ray_dim=ray_dim,
        token_dim=128,
        token_count=args.token_count,
        attention_heads=4,
        encoder_layers=2,
    ).to(device)
    decoder = GeometryDecoder(
        query_ray_dim=ray_dim,
        token_dim=128,
        decoder_dim=128,
        attention_heads=4,
        decoder_layers=2,
    ).to(device)
    ray_builder = CameraToRays(ray_dim=ray_dim, height_hypotheses=6, height_range_m=50.0)
    criterion = GeometryLoss(edge_bce_weight=0.1, normal_cosine_weight=0.05)

    params = list(backbone.parameters()) + list(tokenizer.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    history = []
    initial_rmse = None
    for step in range(args.steps):
        backbone.train()
        tokenizer.train()
        decoder.train()
        optimizer.zero_grad(set_to_none=True)

        features, feature_size = backbone(batch["images"])
        rays = ray_builder(
            camera_params=batch["camera_params"],
            views=batch["images"].shape[1],
            feature_size=feature_size,
            image_size=image_size,
            device=device,
            dtype=features.dtype,
        )
        tokens = tokenizer(features, rays)
        sampled = sample_queries(batch, ray_builder, image_size, args.query_points, device, features.dtype)
        pred = decoder(tokens, sampled["query_rays"])
        losses = criterion(
            pred=pred,
            target_dsm=sampled["gt_dsm"],
            target_edges=sampled["gt_edges"],
            target_normals=sampled["gt_normals"],
            valid_mask=sampled["valid_mask"],
        )
        losses["total"].backward()
        optimizer.step()

        rmse = float(dsm_rmse(pred.dsm.detach(), sampled["gt_dsm"], sampled["valid_mask"]).cpu())
        if initial_rmse is None:
            initial_rmse = rmse
        if step % 10 == 0 or step == args.steps - 1:
            row = {
                "step": step,
                "loss": float(losses["total"].detach().cpu()),
                "rmse": rmse,
            }
            history.append(row)
            print(json.dumps(row))

    final_rmse = history[-1]["rmse"]
    summary = {
        "mode": mode,
        "device": str(device),
        "steps": args.steps,
        "token_count": args.token_count,
        "query_points": args.query_points,
        "initial_rmse": initial_rmse,
        "final_rmse": final_rmse,
        "rmse_ratio": final_rmse / max(initial_rmse, 1e-8),
        "healthy": final_rmse < 0.7 * initial_rmse,
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


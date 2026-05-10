from __future__ import annotations

import argparse
import copy
import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix == ".json":
            return json.load(handle)

        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "YAML config loading requires PyYAML. Use the JSON config or install dependencies."
            ) from exc
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_models(config: dict[str, Any], token_count: int):
    from geotoken import GeoTokenizer, GeometryDecoder, SimpleImageBackbone, SwinGeometryDecoder

    model_cfg = config["model"]
    backbone = SimpleImageBackbone(out_dim=model_cfg["image_feature_dim"])
    tokenizer = GeoTokenizer(
        image_feature_dim=model_cfg["image_feature_dim"],
        ray_dim=model_cfg["ray_dim"],
        token_dim=model_cfg["token_dim"],
        token_count=token_count,
        attention_heads=model_cfg["attention_heads"],
        encoder_layers=model_cfg["encoder_layers"],
    )
    decoder_cls = {
        "query": GeometryDecoder,
        "swin_dense": SwinGeometryDecoder,
    }.get(model_cfg.get("decoder", "query"))
    if decoder_cls is None:
        raise ValueError(f"unsupported decoder: {model_cfg.get('decoder')}")

    decoder = decoder_cls(
        query_ray_dim=model_cfg["ray_dim"],
        token_dim=model_cfg["token_dim"],
        decoder_dim=model_cfg["decoder_dim"],
        attention_heads=model_cfg["attention_heads"],
        decoder_layers=model_cfg["decoder_layers"],
        output_size=model_cfg.get("output_size", 256),
        initial_grid_size=model_cfg.get("initial_grid_size", 8),
        window_size=model_cfg.get("window_size", 8),
    )
    return backbone, tokenizer, decoder


def build_loss(config: dict[str, Any]):
    from geotoken.losses import GeometryLoss

    loss_cfg = config["loss"]
    return GeometryLoss(
        dsm_l1_weight=loss_cfg.get("dsm_l1_weight", 1.0),
        dsm_rmse_weight=loss_cfg.get("dsm_rmse_weight", 0.5),
        edge_bce_weight=loss_cfg.get("edge_bce_weight", 0.2),
        normal_cosine_weight=loss_cfg.get("normal_cosine_weight", 0.1),
        reprojection_weight=loss_cfg.get("reprojection_weight", 0.0),
    )


def autocast_context(device: str, precision: str):
    import torch

    if device != "cuda" or precision == "fp32":
        return nullcontext()

    if precision == "auto":
        if torch.cuda.is_bf16_supported():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    raise ValueError(f"unsupported precision: {precision}")


def should_enable_grad_scaler(device: str, precision: str) -> bool:
    import torch

    if device != "cuda":
        return False
    if precision == "fp16":
        return True
    if precision == "auto":
        return not torch.cuda.is_bf16_supported()
    return False


def sample_query_targets(
    query_rays: "torch.Tensor",
    gt_dsm: "torch.Tensor",
    gt_edges: "torch.Tensor",
    gt_normals: "torch.Tensor",
    valid_mask: "torch.Tensor",
    query_points: int,
) -> dict[str, "torch.Tensor"]:
    import torch

    batch, pixels, ray_dim = query_rays.shape
    flat_dsm = gt_dsm.flatten(1)
    flat_edges = gt_edges.flatten(1)
    flat_valid = valid_mask.flatten(1)
    flat_normals = gt_normals.permute(0, 2, 3, 1).reshape(batch, pixels, 3)

    sampled_rays = []
    sampled_dsm = []
    sampled_edges = []
    sampled_normals = []
    sampled_valid = []
    for item in range(batch):
        valid_indices = torch.nonzero(flat_valid[item], as_tuple=False).squeeze(1)
        if valid_indices.numel() == 0:
            valid_indices = torch.arange(pixels, device=query_rays.device)

        replace = valid_indices.numel() < query_points
        draw = torch.randint(
            low=0,
            high=valid_indices.numel(),
            size=(query_points,),
            device=query_rays.device,
        )
        if not replace:
            draw = torch.randperm(valid_indices.numel(), device=query_rays.device)[:query_points]
        indices = valid_indices[draw]

        sampled_rays.append(query_rays[item, indices])
        sampled_dsm.append(flat_dsm[item, indices])
        sampled_edges.append(flat_edges[item, indices])
        sampled_normals.append(flat_normals[item, indices])
        sampled_valid.append(flat_valid[item, indices])

    return {
        "query_rays": torch.stack(sampled_rays, dim=0).reshape(batch, query_points, ray_dim),
        "gt_dsm": torch.stack(sampled_dsm, dim=0),
        "gt_edges": torch.stack(sampled_edges, dim=0),
        "gt_normals": torch.stack(sampled_normals, dim=0),
        "valid_mask": torch.stack(sampled_valid, dim=0),
    }


def split_manifest(
    manifest_path: Path,
    output_dir: Path,
    val_fraction: float,
    seed: int,
) -> tuple[Path, Path]:
    from geotoken.manifest import load_manifest, write_manifest_jsonl

    samples = load_manifest(manifest_path)
    if len(samples) < 2:
        raise ValueError("need at least two manifest samples to create a train/val split")

    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    val_count = max(1, int(round(len(samples) * val_fraction)))
    val_indices = set(indices[:val_count])
    train_samples = [sample for idx, sample in enumerate(samples) if idx not in val_indices]
    val_samples = [sample for idx, sample in enumerate(samples) if idx in val_indices]
    if not train_samples:
        train_samples, val_samples = val_samples[:-1], val_samples[-1:]

    split_dir = output_dir / "manifest_splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_path = split_dir / f"{manifest_path.stem}_train.jsonl"
    val_path = split_dir / f"{manifest_path.stem}_val.jsonl"
    write_manifest_jsonl(train_path, train_samples)
    write_manifest_jsonl(val_path, val_samples)
    return train_path, val_path


def dry_run(config: dict[str, Any]) -> None:
    summary = []
    for token_count in config["model"]["token_counts"]:
        try:
            backbone, tokenizer, decoder = build_models(config, token_count)
            row = {
                "token_count": token_count,
                "backbone_parameters": sum(p.numel() for p in backbone.parameters()),
                "tokenizer_parameters": sum(p.numel() for p in tokenizer.parameters()),
                "decoder_parameters": sum(p.numel() for p in decoder.parameters()),
                "decoder": config["model"].get("decoder", "query"),
            }
        except ModuleNotFoundError as exc:
            if exc.name != "torch":
                raise
            row = {
                "token_count": token_count,
                "status": "config_valid_torch_not_installed",
                "decoder": config["model"].get("decoder", "query"),
            }
        summary.append(row)
    print(json.dumps(summary, indent=2))


def print_manifest_report(name: str, report) -> None:
    print(
        f"{name} manifest: samples={report.samples}, "
        f"errors={len(report.errors)}, warnings={len(report.warnings)}"
    )
    for issue in report.errors[:10]:
        print(f"ERROR [{issue.sample_id}] {issue.message}")
    for issue in report.warnings[:10]:
        print(f"WARN  [{issue.sample_id}] {issue.message}")
    if len(report.errors) > 10:
        print(f"... {len(report.errors) - 10} more errors")
    if len(report.warnings) > 10:
        print(f"... {len(report.warnings) - 10} more warnings")


def forward_batch(
    batch: dict[str, Any],
    backbone,
    tokenizer,
    decoder,
    criterion,
    ray_builder,
    config: dict[str, Any],
    device,
    precision: str,
    train: bool,
) -> tuple[dict[str, Any], dict[str, float]]:
    import torch

    from geotoken import boundary_f1, discontinuity_mae, dsm_mae, dsm_rmse

    images = batch["images"].to(device, non_blocking=True)
    gt_dsm = batch["gt_dsm"].to(device, non_blocking=True)
    gt_edges = batch["gt_edges"].to(device, non_blocking=True)
    gt_normals = batch["gt_normals"].to(device, non_blocking=True)
    valid_mask = batch["valid_mask"].to(device, non_blocking=True)

    with autocast_context(device.type, precision):
        features, feature_size = backbone(images)
        rays = ray_builder(
            camera_params=batch["camera_params"],
            views=images.shape[1],
            feature_size=feature_size,
            image_size=images.shape[-2:],
            device=device,
            dtype=features.dtype,
        )
        tokens = tokenizer(features, rays)
        if config["model"].get("decoder", "query") == "query":
            dense_query_rays = ray_builder(
                camera_params=batch["camera_params"],
                views=1,
                feature_size=gt_dsm.shape[-2:],
                image_size=images.shape[-2:],
                device=device,
                dtype=features.dtype,
            )[:, 0]
            query_points = config["train"].get("query_points", 2048)
            sampled = sample_query_targets(
                query_rays=dense_query_rays,
                gt_dsm=gt_dsm,
                gt_edges=gt_edges,
                gt_normals=gt_normals,
                valid_mask=valid_mask,
                query_points=query_points,
            )
            pred = decoder(tokens, query_rays=sampled["query_rays"])
            loss_target_dsm = sampled["gt_dsm"]
            loss_target_edges = sampled["gt_edges"]
            loss_target_normals = sampled["gt_normals"]
            loss_valid_mask = sampled["valid_mask"]
        else:
            pred = decoder(tokens, output_size=gt_dsm.shape[-2:])
            loss_target_dsm = gt_dsm
            loss_target_edges = gt_edges
            loss_target_normals = gt_normals
            loss_valid_mask = valid_mask
        losses = criterion(
            pred=pred,
            target_dsm=loss_target_dsm,
            target_edges=loss_target_edges,
            target_normals=loss_target_normals,
            valid_mask=loss_valid_mask,
        )

    metrics = {
        "loss": float(losses["total"].detach().cpu()),
        "dsm_rmse": float(dsm_rmse(pred.dsm, loss_target_dsm, loss_valid_mask).detach().cpu()),
        "dsm_mae": float(dsm_mae(pred.dsm, loss_target_dsm, loss_valid_mask).detach().cpu()),
        "boundary_f1": float(
            boundary_f1(pred.edge_logits, loss_target_edges, loss_valid_mask).detach().cpu()
        ),
        "discontinuity_mae": float(
            discontinuity_mae(
                pred.dsm,
                loss_target_dsm,
                loss_target_edges,
                loss_valid_mask,
            ).detach().cpu()
        ),
    }

    output = {
        "losses": losses,
        "pred": pred,
        "target_dsm": loss_target_dsm,
        "target_edges": loss_target_edges,
        "valid_mask": loss_valid_mask,
        "images": images,
    }
    return output, metrics


def average_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def log_metrics(writer, prefix: str, metrics: dict[str, float], step: int) -> None:
    if writer is None:
        return
    for key, value in metrics.items():
        writer.add_scalar(f"{prefix}/{key}", value, step)


def log_dsm_images(writer, prefix: str, output: dict[str, Any], step: int) -> None:
    if writer is None:
        return
    import torch

    pred = output["pred"].dsm.detach()
    target = output["target_dsm"].detach()
    if pred.ndim != 3 or target.ndim != 3:
        return

    pred_img = normalize_for_image(pred[:1])
    target_img = normalize_for_image(target[:1])
    writer.add_image(f"{prefix}/pred_dsm", pred_img, step)
    writer.add_image(f"{prefix}/target_dsm", target_img, step)


def normalize_for_image(value: "torch.Tensor") -> "torch.Tensor":
    import torch

    image = value.float()
    finite = torch.isfinite(image)
    if not finite.any():
        return torch.zeros_like(image[0:1]).cpu()
    valid = image[finite]
    min_value = valid.min()
    max_value = valid.max()
    image = (image - min_value) / (max_value - min_value).clamp_min(1e-6)
    return image[0].unsqueeze(0).cpu()


def train_one_token_count(args: argparse.Namespace, config: dict[str, Any], token_count: int) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm

    from geotoken import (
        CameraToRays,
        MultiViewGeometryDataset,
        geotoken_collate,
    )
    from geotoken.manifest import validate_manifest

    set_seed(config["experiment"].get("seed", 7))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg = config["train"]
    data_cfg = config["data"]
    eval_cfg = config.get("eval", {})

    val_manifest = args.val_manifest
    train_manifest = args.manifest
    val_fraction = args.val_fraction if args.val_fraction is not None else eval_cfg.get("val_fraction", 0.1)
    if val_manifest is None and val_fraction > 0:
        train_manifest, val_manifest = split_manifest(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            val_fraction=val_fraction,
            seed=config["experiment"].get("seed", 7),
        )

    if args.validate_manifest:
        train_report = validate_manifest(
            manifest_path=train_manifest,
            root=args.data_root,
            min_views=data_cfg.get("views", 2),
            check_open=args.check_data_open,
        )
        print_manifest_report("train", train_report)
        if not train_report.ok:
            raise ValueError(f"train manifest validation failed with {len(train_report.errors)} errors")
        if val_manifest is not None:
            val_report = validate_manifest(
                manifest_path=val_manifest,
                root=args.data_root,
                min_views=data_cfg.get("views", 2),
                check_open=args.check_data_open,
            )
            print_manifest_report("val", val_report)
            if not val_report.ok:
                raise ValueError(f"val manifest validation failed with {len(val_report.errors)} errors")

    train_dataset = MultiViewGeometryDataset(
        manifest_path=train_manifest,
        root=args.data_root,
        views=data_cfg.get("views"),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=geotoken_collate,
        pin_memory=device.type == "cuda",
    )
    val_loader = None
    if val_manifest is not None:
        val_dataset = MultiViewGeometryDataset(
            manifest_path=val_manifest,
            root=args.data_root,
            views=data_cfg.get("views"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size or train_cfg.get("batch_size", 4),
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=geotoken_collate,
            pin_memory=device.type == "cuda",
        )

    backbone, tokenizer, decoder = build_models(config, token_count)
    backbone.to(device)
    tokenizer.to(device)
    decoder.to(device)

    criterion = build_loss(config)
    ray_builder = CameraToRays(
        ray_dim=config["model"]["ray_dim"],
        height_hypotheses=config["model"].get("height_hypotheses"),
        height_range_m=config["model"].get("height_range_m", 50.0),
    )
    params = list(backbone.parameters()) + list(tokenizer.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=train_cfg.get("lr", 2e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    precision = train_cfg.get("precision", "auto")
    scaler = torch.cuda.amp.GradScaler(
        enabled=should_enable_grad_scaler(device.type, precision)
    )
    epochs = args.epochs or train_cfg.get("epochs", 80)
    val_interval = args.val_interval or eval_cfg.get("val_interval", 1)
    val_limit_steps = args.val_limit_steps
    if val_limit_steps is None:
        val_limit_steps = eval_cfg.get("val_limit_steps")
    output_dir = args.output_dir / f"tokens_{token_count}"
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir / "tb")) if args.tensorboard else None
    last_metrics: dict[str, float] = {}
    global_step = 0

    try:
        for epoch in range(epochs):
            backbone.train()
            tokenizer.train()
            decoder.train()
            train_rows = []
            progress = tqdm(train_loader, desc=f"tokens={token_count} epoch={epoch + 1}/{epochs}")

            for step, batch in enumerate(progress):
                optimizer.zero_grad(set_to_none=True)
                output, step_metrics = forward_batch(
                    batch=batch,
                    backbone=backbone,
                    tokenizer=tokenizer,
                    decoder=decoder,
                    criterion=criterion,
                    ray_builder=ray_builder,
                    config=config,
                    device=device,
                    precision=precision,
                    train=True,
                )

                scaler.scale(output["losses"]["total"]).backward()
                scaler.step(optimizer)
                scaler.update()

                train_rows.append(step_metrics)
                last_metrics = step_metrics
                log_metrics(writer, "train_step", step_metrics, global_step)
                if global_step % args.image_log_interval == 0:
                    log_dsm_images(writer, "train", output, global_step)
                global_step += 1
                progress.set_postfix(step_metrics)

                if args.limit_steps is not None and step + 1 >= args.limit_steps:
                    break

            train_metrics = average_metrics(train_rows)
            log_metrics(writer, "train_epoch", train_metrics, epoch + 1)
            last_metrics = {f"train_{key}": value for key, value in train_metrics.items()}

            val_metrics = {}
            if val_loader is not None and (epoch + 1) % val_interval == 0:
                val_metrics = evaluate(
                    loader=val_loader,
                    backbone=backbone,
                    tokenizer=tokenizer,
                    decoder=decoder,
                    criterion=criterion,
                    ray_builder=ray_builder,
                    config=config,
                    device=device,
                    precision=precision,
                    limit_steps=val_limit_steps,
                    writer=writer,
                    epoch=epoch + 1,
                )
                log_metrics(writer, "val", val_metrics, epoch + 1)
                last_metrics.update({f"val_{key}": value for key, value in val_metrics.items()})

            checkpoint = {
                "token_count": token_count,
                "epoch": epoch + 1,
                "backbone": backbone.state_dict(),
                "tokenizer": tokenizer.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metrics": last_metrics,
                "config": config,
            }
            torch.save(checkpoint, output_dir / "last.pt")
    finally:
        if writer is not None:
            writer.close()

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(last_metrics, indent=2), encoding="utf-8")
    return {"token_count": token_count, **last_metrics}


def evaluate(
    loader,
    backbone,
    tokenizer,
    decoder,
    criterion,
    ray_builder,
    config: dict[str, Any],
    device,
    precision: str,
    limit_steps: int | None,
    writer,
    epoch: int,
) -> dict[str, float]:
    import torch
    from tqdm import tqdm

    backbone.eval()
    tokenizer.eval()
    decoder.eval()

    rows = []
    last_output = None
    with torch.no_grad():
        progress = tqdm(loader, desc=f"val epoch={epoch}")
        for step, batch in enumerate(progress):
            output, metrics = forward_batch(
                batch=batch,
                backbone=backbone,
                tokenizer=tokenizer,
                decoder=decoder,
                criterion=criterion,
                ray_builder=ray_builder,
                config=config,
                device=device,
                precision=precision,
                train=False,
            )
            rows.append(metrics)
            last_output = output
            progress.set_postfix(metrics)
            if limit_steps is not None and step + 1 >= limit_steps:
                break

    if last_output is not None:
        log_dsm_images(writer, "val", last_output, epoch)
    return average_metrics(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GeoToken compression sweep.")
    parser.add_argument("--config", type=Path, default=Path("configs/compression_sweep.json"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/compression_sweep"))
    parser.add_argument("--token-count", type=int, help="Run one token count instead of the full sweep.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--val-limit-steps", type=int)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-log-interval", type=int, default=100)
    parser.add_argument("--validate-manifest", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-data-open", action="store_true")
    parser.add_argument("--limit-steps", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.dry_run:
        dry_run(config)
        return

    if args.manifest is None:
        raise ValueError("--manifest is required for training")

    token_counts = [args.token_count] if args.token_count else config["model"]["token_counts"]
    results = [train_one_token_count(args, config, token_count) for token_count in token_counts]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import torch


def dsm_mae(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    error = pred[valid_mask] - target[valid_mask]
    return error.abs().mean()


def dsm_rmse(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    error = pred[valid_mask] - target[valid_mask]
    return torch.sqrt(error.square().mean().clamp_min(1e-8))


def boundary_f1(
    pred_edge_logits: torch.Tensor,
    target_edges: torch.Tensor,
    valid_mask: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    pred_edges = pred_edge_logits.sigmoid() >= threshold
    target_edges = target_edges.bool()
    pred_edges = pred_edges & valid_mask
    target_edges = target_edges & valid_mask

    true_positive = (pred_edges & target_edges).sum().float()
    precision = true_positive / pred_edges.sum().clamp_min(1).float()
    recall = true_positive / target_edges.sum().clamp_min(1).float()
    return 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)


def discontinuity_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_edges: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    mask = target_edges.bool() & valid_mask
    if mask.sum() == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    return (pred[mask] - target[mask]).abs().mean()


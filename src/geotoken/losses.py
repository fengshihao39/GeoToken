from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .decoder import GeometryPrediction


class GeometryLoss(nn.Module):
    """Weighted losses for DSM, structural edges, and normals."""

    def __init__(
        self,
        dsm_l1_weight: float = 1.0,
        dsm_rmse_weight: float = 0.5,
        edge_bce_weight: float = 0.2,
        normal_cosine_weight: float = 0.1,
        reprojection_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.dsm_l1_weight = dsm_l1_weight
        self.dsm_rmse_weight = dsm_rmse_weight
        self.edge_bce_weight = edge_bce_weight
        self.normal_cosine_weight = normal_cosine_weight
        self.reprojection_weight = reprojection_weight

    def forward(
        self,
        pred: GeometryPrediction,
        target_dsm: torch.Tensor,
        target_edges: torch.Tensor | None = None,
        target_normals: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        reprojection_residual: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if valid_mask is None:
            valid_mask = torch.ones_like(target_dsm, dtype=torch.bool)

        dsm_error = pred.dsm[valid_mask] - target_dsm[valid_mask]
        losses = {
            "dsm_l1": dsm_error.abs().mean(),
            "dsm_rmse": torch.sqrt((dsm_error.square()).mean().clamp_min(1e-8)),
        }

        total = self.dsm_l1_weight * losses["dsm_l1"]
        total = total + self.dsm_rmse_weight * losses["dsm_rmse"]

        if target_edges is not None:
            edge_loss = F.binary_cross_entropy_with_logits(
                pred.edge_logits[valid_mask],
                target_edges[valid_mask].float(),
            )
            losses["edge_bce"] = edge_loss
            total = total + self.edge_bce_weight * edge_loss

        if target_normals is not None:
            pred_normals = pred.normals
            if pred_normals.shape[1] == 3:
                pred_normals = pred_normals.permute(0, 2, 3, 1)
            if target_normals.shape[1] == 3:
                target_normals = target_normals.permute(0, 2, 3, 1)
            normal_loss = 1.0 - F.cosine_similarity(
                pred_normals[valid_mask],
                target_normals[valid_mask],
                dim=-1,
            ).mean()
            losses["normal_cosine"] = normal_loss
            total = total + self.normal_cosine_weight * normal_loss

        if reprojection_residual is not None:
            reprojection_loss = reprojection_residual[valid_mask].abs().mean()
            losses["reprojection"] = reprojection_loss
            total = total + self.reprojection_weight * reprojection_loss

        losses["total"] = total
        return losses

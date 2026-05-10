from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .ray_encoding import FourierRayEncoding


@dataclass
class GeometryPrediction:
    dsm: torch.Tensor
    edge_logits: torch.Tensor
    normals: torch.Tensor


class GeometryDecoder(nn.Module):
    """Query-based GeoToken decoder used as the main architecture.

    The decoder maps coordinate/ray queries to continuous DSM, edge, and normal
    predictions. This keeps the primary model faithful to the geometry-token
    bottleneck question: how much continuous geometry can be recovered by
    querying a compact cross-view token set?
    """

    def __init__(
        self,
        query_ray_dim: int,
        token_dim: int = 256,
        decoder_dim: int = 256,
        attention_heads: int = 8,
        decoder_layers: int = 3,
        output_size: int | tuple[int, int] = 256,
        initial_grid_size: int = 8,
        window_size: int = 8,
    ) -> None:
        super().__init__()
        del output_size
        del initial_grid_size
        del window_size

        self.query_encoding = FourierRayEncoding(query_ray_dim, decoder_dim)
        self.token_proj = nn.Linear(token_dim, decoder_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=decoder_dim,
            num_heads=attention_heads,
            batch_first=True,
        )
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=decoder_dim,
                    nhead=attention_heads,
                    dim_feedforward=decoder_dim * 4,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(decoder_layers)
            ]
        )
        self.norm = nn.LayerNorm(decoder_dim)
        self.dsm_head = nn.Linear(decoder_dim, 1)
        self.edge_head = nn.Linear(decoder_dim, 1)
        self.normal_head = nn.Linear(decoder_dim, 3)

    def forward(
        self,
        tokens: torch.Tensor,
        query_rays: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> GeometryPrediction:
        """Predict geometry for each query.

        Args:
            tokens: ``(batch, token_count, token_dim)``.
            query_rays: ``(batch, queries, query_ray_dim)``.
            output_size: Optional ``(height, width)`` used to reshape dense
                query outputs back into maps.
        """

        queries = self.query_encoding(query_rays)
        memory = self.token_proj(tokens)
        decoded, _ = self.cross_attention(query=queries, key=memory, value=memory)

        for layer in self.layers:
            decoded = layer(decoded)

        decoded = self.norm(decoded)
        dsm = self.dsm_head(decoded).squeeze(-1)
        edge_logits = self.edge_head(decoded).squeeze(-1)
        normals = nn.functional.normalize(self.normal_head(decoded), dim=-1)

        if output_size is not None:
            batch, query_count = dsm.shape
            expected_queries = output_size[0] * output_size[1]
            if query_count != expected_queries:
                raise ValueError(
                    f"cannot reshape {query_count} queries to output_size={output_size}"
                )
            height, width = output_size
            dsm = dsm.reshape(batch, height, width)
            edge_logits = edge_logits.reshape(batch, height, width)
            normals = normals.reshape(batch, height, width, 3).permute(0, 3, 1, 2).contiguous()

        return GeometryPrediction(dsm=dsm, edge_logits=edge_logits, normals=normals)


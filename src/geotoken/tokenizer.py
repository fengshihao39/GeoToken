from __future__ import annotations

import torch
from torch import nn

from .ray_encoding import FourierRayEncoding


class GeoTokenizer(nn.Module):
    """Cross-view latent bottleneck for geometry-preserving compression.

    The tokenizer uses two-stage aggregation instead of flattening all view
    patches directly:

    1. Shared latent queries compress each view into view-local geometry tokens.
    2. Final GeoTokens cross-attend to the concatenated view tokens and mix
       information across views.

    This is a conservative geometry-aware step toward epipolar/RPC-biased
    attention: ray features are already injected before each view is compressed,
    while the inter-view mixer operates on a much smaller token set.
    """

    def __init__(
        self,
        image_feature_dim: int,
        ray_dim: int,
        token_dim: int = 256,
        token_count: int = 128,
        attention_heads: int = 8,
        encoder_layers: int = 4,
    ) -> None:
        super().__init__()
        self.token_count = token_count
        self.token_dim = token_dim

        self.image_proj = nn.Linear(image_feature_dim, token_dim)
        self.ray_encoding = FourierRayEncoding(ray_dim=ray_dim, out_dim=token_dim)
        self.view_latents = nn.Parameter(torch.randn(token_count, token_dim) * 0.02)
        self.geo_latents = nn.Parameter(torch.randn(token_count, token_dim) * 0.02)

        self.view_cross_attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=attention_heads,
            batch_first=True,
        )
        self.view_cross_norm = nn.LayerNorm(token_dim)

        inter_view_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=attention_heads,
            dim_feedforward=token_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.inter_view_mixer = nn.TransformerEncoder(
            inter_view_layer,
            num_layers=max(1, encoder_layers // 2),
        )

        self.geo_cross_attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=attention_heads,
            batch_first=True,
        )
        self.geo_cross_norm = nn.LayerNorm(token_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=attention_heads,
            dim_feedforward=token_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_mixer = nn.TransformerEncoder(layer, num_layers=encoder_layers)

    def forward(self, image_features: torch.Tensor, rays: torch.Tensor) -> torch.Tensor:
        """Compress multi-view features into GeoTokens.

        Args:
            image_features: ``(batch, views, patches, image_feature_dim)``.
            rays: ``(batch, views, patches, ray_dim)``.

        Returns:
            GeoTokens shaped ``(batch, token_count, token_dim)``.
        """

        if image_features.ndim != 4:
            raise ValueError("image_features must be shaped (batch, views, patches, channels)")
        if rays.shape[:3] != image_features.shape[:3]:
            raise ValueError("rays must share batch/view/patch dimensions with image_features")

        batch, views, patches, _ = image_features.shape
        features = self.image_proj(image_features) + self.ray_encoding(rays)

        tokens_per_view = max(1, self.token_count // views)
        view_queries = self.view_latents[:tokens_per_view].unsqueeze(0)
        view_queries = view_queries.expand(batch * views, -1, -1)
        view_features = features.reshape(batch * views, patches, self.token_dim)

        view_attended, _ = self.view_cross_attention(
            query=view_queries,
            key=view_features,
            value=view_features,
        )
        view_tokens = self.view_cross_norm(view_queries + view_attended)
        view_tokens = view_tokens.reshape(batch, views * tokens_per_view, self.token_dim)
        view_tokens = self.inter_view_mixer(view_tokens)

        geo_queries = self.geo_latents.unsqueeze(0).expand(batch, -1, -1)
        attended, _ = self.geo_cross_attention(query=geo_queries, key=view_tokens, value=view_tokens)
        tokens = self.geo_cross_norm(geo_queries + attended)
        return self.token_mixer(tokens)

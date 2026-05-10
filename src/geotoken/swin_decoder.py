from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .decoder import GeometryPrediction


class WindowAttentionBlock(nn.Module):
    """Swin-style local window attention block for decoder ablations."""

    def __init__(
        self,
        dim: int,
        attention_heads: int = 8,
        window_size: int = 8,
        shift_size: int = 0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, attention_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        shortcut = x

        x = x.permute(0, 2, 3, 1).contiguous()
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        pad_height = (self.window_size - height % self.window_size) % self.window_size
        pad_width = (self.window_size - width % self.window_size) % self.window_size
        if pad_height or pad_width:
            x = F.pad(x, (0, 0, 0, pad_width, 0, pad_height))

        padded_height, padded_width = x.shape[1:3]
        windows = self._partition_windows(x)
        attended = self.norm1(windows)
        attended, _ = self.attention(attended, attended, attended)
        windows = windows + attended
        windows = windows + self.mlp(self.norm2(windows))
        x = self._merge_windows(windows, batch, padded_height, padded_width)

        if pad_height or pad_width:
            x = x[:, :height, :width, :]
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.permute(0, 3, 1, 2).contiguous()
        return shortcut + x

    def _partition_windows(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = x.shape
        x = x.view(
            batch,
            height // self.window_size,
            self.window_size,
            width // self.window_size,
            self.window_size,
            channels,
        )
        return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, self.window_size * self.window_size, channels)

    def _merge_windows(
        self,
        windows: torch.Tensor,
        batch: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        channels = windows.shape[-1]
        x = windows.view(
            batch,
            height // self.window_size,
            width // self.window_size,
            self.window_size,
            self.window_size,
            channels,
        )
        return x.permute(0, 1, 3, 2, 4, 5).reshape(batch, height, width, channels)


class TokenGuidedUpsampleStage(nn.Module):
    """Upsample feature maps and refresh them with global GeoToken guidance."""

    def __init__(
        self,
        dim: int,
        attention_heads: int,
        window_size: int,
        use_shift: bool,
    ) -> None:
        super().__init__()
        self.local_a = WindowAttentionBlock(dim, attention_heads, window_size, shift_size=0)
        self.local_b = WindowAttentionBlock(
            dim,
            attention_heads,
            window_size,
            shift_size=window_size // 2 if use_shift else 0,
        )
        self.token_norm = nn.LayerNorm(dim)
        self.token_attention = nn.MultiheadAttention(dim, attention_heads, batch_first=True)
        self.token_fuse = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.local_a(x)
        x = self.local_b(x)

        batch, channels, height, width = x.shape
        queries = x.flatten(2).transpose(1, 2)
        refreshed, _ = self.token_attention(
            query=self.token_norm(queries),
            key=tokens,
            value=tokens,
        )
        queries = self.token_fuse(queries + refreshed)
        return queries.transpose(1, 2).reshape(batch, channels, height, width)


class SwinGeometryDecoder(nn.Module):
    """Swin-style dense decoder kept for decoder ablation experiments."""

    def __init__(
        self,
        query_ray_dim: int | None = None,
        token_dim: int = 256,
        decoder_dim: int = 256,
        attention_heads: int = 8,
        decoder_layers: int = 3,
        output_size: int | tuple[int, int] = 256,
        initial_grid_size: int = 8,
        window_size: int = 8,
    ) -> None:
        super().__init__()
        del query_ray_dim
        del decoder_layers

        self.output_size = (output_size, output_size) if isinstance(output_size, int) else output_size
        self.initial_grid_size = initial_grid_size
        self.grid_queries = nn.Parameter(
            torch.randn(initial_grid_size * initial_grid_size, decoder_dim) * 0.02
        )
        self.token_proj = nn.Linear(token_dim, decoder_dim)
        self.grid_attention = nn.MultiheadAttention(decoder_dim, attention_heads, batch_first=True)
        self.grid_norm = nn.LayerNorm(decoder_dim)

        stages = []
        current_size = initial_grid_size
        while current_size < max(self.output_size):
            stages.append(
                TokenGuidedUpsampleStage(
                    dim=decoder_dim,
                    attention_heads=attention_heads,
                    window_size=window_size,
                    use_shift=True,
                )
            )
            current_size *= 2
        self.stages = nn.ModuleList(stages)

        self.dsm_head = nn.Conv2d(decoder_dim, 1, kernel_size=1)
        self.edge_head = nn.Conv2d(decoder_dim, 1, kernel_size=1)
        self.normal_head = nn.Conv2d(decoder_dim, 3, kernel_size=1)

    def forward(
        self,
        tokens: torch.Tensor,
        query_rays: torch.Tensor | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> GeometryPrediction:
        del query_rays
        batch = tokens.shape[0]
        output_size = output_size or self.output_size
        memory = self.token_proj(tokens)

        grid_queries = self.grid_queries.unsqueeze(0).expand(batch, -1, -1)
        grid, _ = self.grid_attention(query=grid_queries, key=memory, value=memory)
        grid = self.grid_norm(grid_queries + grid)
        x = grid.transpose(1, 2).reshape(
            batch,
            -1,
            self.initial_grid_size,
            self.initial_grid_size,
        )

        for stage in self.stages:
            x = stage(x, memory)

        if x.shape[-2:] != output_size:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)

        normals = F.normalize(self.normal_head(x), dim=1)
        return GeometryPrediction(
            dsm=self.dsm_head(x).squeeze(1),
            edge_logits=self.edge_head(x).squeeze(1),
            normals=normals,
        )


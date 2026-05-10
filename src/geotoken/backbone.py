from __future__ import annotations

import torch
from torch import nn


class SimpleImageBackbone(nn.Module):
    """Small CNN backbone that converts raw views into patch features.

    This keeps the first training path self-contained. On AutoDL we can later
    swap this for torchvision ResNet/Swin backbones without changing the
    tokenizer interface.
    """

    def __init__(self, out_dim: int = 256, base_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            self._block(3, base_dim, stride=2),
            self._block(base_dim, base_dim * 2, stride=2),
            self._block(base_dim * 2, base_dim * 4, stride=2),
            self._block(base_dim * 4, out_dim, stride=2),
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Extract per-view patch features.

        Args:
            images: Tensor shaped ``(batch, views, 3, height, width)``.

        Returns:
            A pair ``(features, feature_size)`` where features are shaped
            ``(batch, views, patches, out_dim)``.
        """

        if images.ndim != 5:
            raise ValueError("images must be shaped (batch, views, 3, height, width)")

        batch, views, channels, height, width = images.shape
        view_batch = images.reshape(batch * views, channels, height, width)
        feature_map = self.net(view_batch)
        _, feature_dim, feature_height, feature_width = feature_map.shape
        features = feature_map.flatten(2).transpose(1, 2)
        features = features.reshape(batch, views, feature_height * feature_width, feature_dim)
        return features, (feature_height, feature_width)

    def _block(self, in_dim: int, out_dim: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )


from __future__ import annotations

import math

import torch
from torch import nn


class FourierRayEncoding(nn.Module):
    """Encode physical ray attributes with Fourier features and a linear mixer.

    Expected ray attributes may include normalized pixel coordinates, camera
    center, ray direction, scale, and view id. The module is agnostic to the
    exact convention as long as the last dimension is fixed.
    """

    def __init__(self, ray_dim: int, out_dim: int, num_frequencies: int = 8) -> None:
        super().__init__()
        self.ray_dim = ray_dim
        self.out_dim = out_dim
        self.num_frequencies = num_frequencies

        encoded_dim = ray_dim * (1 + 2 * num_frequencies)
        self.proj = nn.Sequential(
            nn.Linear(encoded_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

        frequencies = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies * math.pi, persistent=False)

    def forward(self, rays: torch.Tensor) -> torch.Tensor:
        """Return ray encodings.

        Args:
            rays: Tensor shaped ``(..., ray_dim)``.

        Returns:
            Tensor shaped ``(..., out_dim)``.
        """

        if rays.shape[-1] != self.ray_dim:
            raise ValueError(f"expected ray_dim={self.ray_dim}, got {rays.shape[-1]}")

        angles = rays.unsqueeze(-1) * self.frequencies
        fourier = torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2)
        return self.proj(torch.cat([rays, fourier], dim=-1))


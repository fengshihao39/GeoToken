from __future__ import annotations

from typing import Any
import warnings

import torch


class CameraToRays:
    """Convert camera metadata into the ray convention used by GeoToken.

    Base ray convention:

    0. normalized x in ``[-1, 1]``
    1. normalized y in ``[-1, 1]``
    2-4. camera center xyz, normalized or raw if supplied
    5-7. ray direction xyz
    8. relative image scale
    9. normalized view id

    Optional dimensions 10 onward are normalized height hypotheses. They are a
    lightweight interface for height-aware matching and should later be coupled
    to RPC/intrinsic projection bias rather than treated as a full cost volume.
    """

    base_ray_dim = 10

    def __init__(
        self,
        ray_dim: int = 10,
        height_hypotheses: int | list[float] | tuple[float, ...] | None = None,
        height_range_m: float = 50.0,
    ) -> None:
        if ray_dim < self.base_ray_dim:
            raise ValueError("ray_dim must be at least 10")

        inferred_hypotheses = ray_dim - self.base_ray_dim
        if height_hypotheses is None:
            height_count = inferred_hypotheses
            heights = None
        elif isinstance(height_hypotheses, int):
            height_count = height_hypotheses
            heights = None
        else:
            heights = torch.as_tensor(list(height_hypotheses), dtype=torch.float32)
            height_count = int(heights.numel())

        if ray_dim != self.base_ray_dim + height_count:
            raise ValueError(
                f"ray_dim={ray_dim} is inconsistent with {height_count} height hypotheses"
            )

        self.ray_dim = ray_dim
        self.height_count = height_count
        self.height_range_m = height_range_m
        if height_count == 0:
            self.height_hypotheses_m = torch.empty(0, dtype=torch.float32)
        elif heights is not None:
            self.height_hypotheses_m = heights
        else:
            self.height_hypotheses_m = torch.linspace(-height_range_m, height_range_m, height_count)
        self._warned_missing_camera = False

    def __call__(
        self,
        camera_params: list[Any],
        views: int,
        feature_size: tuple[int, int],
        image_size: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        feature_height, feature_width = feature_size
        image_height, image_width = image_size
        batch = len(camera_params)

        base_xy = self._normalized_grid(feature_height, feature_width, device, dtype)
        rays = []
        for sample_params in camera_params:
            view_params = self._extract_view_params(sample_params, views)
            sample_rays = []
            for view_index in range(views):
                params = view_params[view_index] if view_index < len(view_params) else {}
                sample_rays.append(
                    self._build_view_rays(
                        base_xy=base_xy,
                        params=params,
                        view_index=view_index,
                        views=views,
                        image_height=image_height,
                        image_width=image_width,
                    )
                )
            rays.append(torch.stack(sample_rays, dim=0))
        return torch.stack(rays, dim=0).reshape(
            batch,
            views,
            feature_height * feature_width,
            self.ray_dim,
        )

    def _build_view_rays(
        self,
        base_xy: torch.Tensor,
        params: dict[str, Any],
        view_index: int,
        views: int,
        image_height: int,
        image_width: int,
    ) -> torch.Tensor:
        device = base_xy.device
        dtype = base_xy.dtype
        feature_height, feature_width, _ = base_xy.shape

        center = torch.zeros(feature_height, feature_width, 3, device=device, dtype=dtype)
        direction = torch.cat([base_xy, torch.ones_like(base_xy[..., :1])], dim=-1)

        intrinsics = self._tensor_or_none(params.get("intrinsics"), device, dtype)
        extrinsics = self._tensor_or_none(params.get("extrinsics"), device, dtype)
        if intrinsics is None and extrinsics is None and not self._warned_missing_camera:
            warnings.warn(
                "No camera intrinsics/extrinsics found; rays are defaulting to normalized image grids. "
                "Check the dataset manifest/camera adapter before trusting geometry metrics.",
                stacklevel=2,
            )
            self._warned_missing_camera = True
        if intrinsics is not None and intrinsics.shape[-2:] == (3, 3):
            pixels = self._pixel_grid(feature_height, feature_width, image_height, image_width, device, dtype)
            intrinsics_inv = torch.linalg.inv(intrinsics)
            direction = pixels @ intrinsics_inv.transpose(0, 1)

        if extrinsics is not None and extrinsics.shape[-2:] == (4, 4):
            rotation = extrinsics[:3, :3]
            translation = extrinsics[:3, 3]
            center_value = -rotation.transpose(0, 1) @ translation
            center = center + center_value.reshape(1, 1, 3)
            direction = direction @ rotation

        direction = torch.nn.functional.normalize(direction, dim=-1)
        scale = torch.full(
            (feature_height, feature_width, 1),
            fill_value=max(image_height, image_width) / max(feature_height, feature_width),
            device=device,
            dtype=dtype,
        )
        view_id = torch.full(
            (feature_height, feature_width, 1),
            fill_value=view_index / max(views - 1, 1),
            device=device,
            dtype=dtype,
        )
        base_ray = torch.cat([base_xy, center, direction, scale, view_id], dim=-1)
        if self.height_count == 0:
            return base_ray

        normalized_heights = (self.height_hypotheses_m / max(self.height_range_m, 1e-6)).to(
            device=device,
            dtype=dtype,
        )
        height_grid = normalized_heights.reshape(1, 1, self.height_count).expand(
            feature_height,
            feature_width,
            -1,
        )
        return torch.cat([base_ray, height_grid], dim=-1)

    def _extract_view_params(self, camera_params: Any, views: int) -> list[dict[str, Any]]:
        if isinstance(camera_params, dict):
            if isinstance(camera_params.get("views"), list):
                return camera_params["views"]
            return [camera_params for _ in range(views)]
        if isinstance(camera_params, list):
            return [item if isinstance(item, dict) else {} for item in camera_params]
        return [{} for _ in range(views)]

    def _normalized_grid(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([xx, yy], dim=-1)

    def _pixel_grid(
        self,
        feature_height: int,
        feature_width: int,
        image_height: int,
        image_width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y = torch.linspace(0.0, image_height - 1.0, feature_height, device=device, dtype=dtype)
        x = torch.linspace(0.0, image_width - 1.0, feature_width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        ones = torch.ones_like(xx)
        return torch.stack([xx, yy, ones], dim=-1)

    def _tensor_or_none(
        self,
        value: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        try:
            return torch.as_tensor(value, device=device, dtype=dtype)
        except (TypeError, ValueError):
            return None

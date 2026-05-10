from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class GeoTokenBatchKeys:
    images: str = "images"
    camera_params: str = "camera_params"
    gt_dsm: str = "gt_dsm"
    gt_edges: str = "gt_edges"
    gt_normals: str = "gt_normals"
    valid_mask: str = "valid_mask"
    sample_id: str = "sample_id"


class MultiViewGeometryDataset(Dataset):
    """Manifest-based dataset for remote sensing multi-view geometry.

    The dataset intentionally uses a small intermediate manifest format so US3D,
    WHU-OMVS, or future datasets can be adapted without changing model code.

    Each manifest row should contain:

    ```json
    {
      "id": "tile_0001",
      "images": ["images/tile_0001_v0.png", "images/tile_0001_v1.png"],
      "camera_params": "cameras/tile_0001.json",
      "gt_dsm": "dsm/tile_0001.npy"
    }
    ```

    Returns:
        - images: float tensor shaped ``(V, 3, H, W)``.
        - camera_params: dict/list loaded from JSON, or raw/numeric TXT content.
        - gt_dsm: float tensor shaped ``(H, W)``.
        - valid_mask: bool tensor shaped ``(H, W)``.
    """

    keys = GeoTokenBatchKeys()

    def __init__(
        self,
        manifest_path: str | Path,
        root: str | Path | None = None,
        views: int | None = None,
        normalize_images: bool = True,
        edge_percentile: float = 90.0,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = Path(root) if root is not None else self.manifest_path.parent
        self.views = views
        self.normalize_images = normalize_images
        self.edge_percentile = edge_percentile
        self.samples = self._load_manifest(self.manifest_path)

        if not self.samples:
            raise ValueError(f"manifest has no samples: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image_paths = sample["images"]
        if self.views is not None:
            image_paths = image_paths[: self.views]

        images = torch.stack([self._load_image(path) for path in image_paths], dim=0)
        gt_dsm = self._load_raster(sample["gt_dsm"]).float()
        valid_mask = torch.isfinite(gt_dsm)

        if "valid_mask" in sample:
            valid_mask = self._load_raster(sample["valid_mask"]).bool() & valid_mask

        gt_dsm = torch.nan_to_num(gt_dsm, nan=0.0, posinf=0.0, neginf=0.0)
        gt_edges = self._build_edges(gt_dsm, valid_mask)
        gt_normals = self._build_normals(gt_dsm)

        return {
            self.keys.sample_id: sample.get("id", str(index)),
            self.keys.images: images,
            self.keys.camera_params: self._load_camera_params(sample["camera_params"]),
            self.keys.gt_dsm: gt_dsm,
            self.keys.gt_edges: gt_edges,
            self.keys.gt_normals: gt_normals,
            self.keys.valid_mask: valid_mask,
        }

    def _load_manifest(self, path: Path) -> list[dict[str, Any]]:
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]
        if path.suffix == ".json":
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                data = data.get("samples", [])
            if not isinstance(data, list):
                raise ValueError("JSON manifest must be a list or {'samples': [...]}")
            return data
        raise ValueError(f"unsupported manifest format: {path.suffix}")

    def _resolve(self, path: str | Path) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path
        return self.root / path

    def _load_image(self, path: str | Path) -> torch.Tensor:
        image_path = self._resolve(path)
        image = Image.open(image_path).convert("RGB")
        array = np.asarray(image, dtype=np.float32)
        if self.normalize_images:
            array = array / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def _load_raster(self, path: str | Path) -> torch.Tensor:
        raster_path = self._resolve(path)
        suffix = raster_path.suffix.lower()

        if suffix == ".npy":
            array = np.load(raster_path)
        elif suffix == ".npz":
            data = np.load(raster_path)
            key = "arr_0" if "arr_0" in data else next(iter(data.keys()))
            array = data[key]
        elif suffix in {".tif", ".tiff"}:
            array = self._load_tiff(raster_path)
        else:
            image = Image.open(raster_path)
            array = np.asarray(image, dtype=np.float32)

        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 3:
            array = array[..., 0]
        return torch.from_numpy(array)

    def _load_tiff(self, path: Path) -> np.ndarray:
        try:
            import rasterio

            with rasterio.open(path) as dataset:
                return dataset.read(1)
        except ImportError:
            image = Image.open(path)
            return np.asarray(image, dtype=np.float32)

    def _load_camera_params(self, camera_params: str | Path | dict[str, Any] | list[Any]) -> Any:
        if isinstance(camera_params, (dict, list)):
            return camera_params

        path = self._resolve(camera_params)
        suffix = path.suffix.lower()

        if suffix == ".json":
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)

        if suffix == ".txt":
            text = path.read_text(encoding="utf-8")
            numeric_rows = []
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    numeric_rows.append([float(value) for value in stripped.split()])
                except ValueError:
                    continue
            return {"raw_text": text, "numeric_rows": numeric_rows}

        raise ValueError(f"unsupported camera parameter format: {path}")

    def _build_edges(self, dsm: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        grad_y = torch.zeros_like(dsm)
        grad_x = torch.zeros_like(dsm)
        grad_y[1:, :] = dsm[1:, :] - dsm[:-1, :]
        grad_x[:, 1:] = dsm[:, 1:] - dsm[:, :-1]
        magnitude = torch.sqrt(grad_x.square() + grad_y.square())
        valid_values = magnitude[valid_mask]
        if valid_values.numel() == 0:
            return torch.zeros_like(dsm, dtype=torch.bool)
        threshold = torch.quantile(valid_values, self.edge_percentile / 100.0)
        if threshold <= 0:
            return (magnitude > 0) & valid_mask
        return (magnitude >= threshold) & valid_mask

    def _build_normals(self, dsm: torch.Tensor) -> torch.Tensor:
        grad_y = torch.zeros_like(dsm)
        grad_x = torch.zeros_like(dsm)
        grad_y[1:-1, :] = 0.5 * (dsm[2:, :] - dsm[:-2, :])
        grad_x[:, 1:-1] = 0.5 * (dsm[:, 2:] - dsm[:, :-2])
        normals = torch.stack([-grad_x, -grad_y, torch.ones_like(dsm)], dim=0)
        return torch.nn.functional.normalize(normals, dim=0)


def geotoken_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate variable-size samples while keeping camera metadata as a list."""

    keys = MultiViewGeometryDataset.keys
    max_views = max(item[keys.images].shape[0] for item in batch)
    max_height = max(item[keys.images].shape[-2] for item in batch)
    max_width = max(item[keys.images].shape[-1] for item in batch)
    max_dsm_height = max(item[keys.gt_dsm].shape[-2] for item in batch)
    max_dsm_width = max(item[keys.gt_dsm].shape[-1] for item in batch)

    images = []
    dsms = []
    edges = []
    normals = []
    masks = []
    for item in batch:
        images.append(_pad_image_views(item[keys.images], max_views, max_height, max_width))
        dsms.append(_pad_2d(item[keys.gt_dsm], max_dsm_height, max_dsm_width, fill_value=0.0))
        edges.append(_pad_2d(item[keys.gt_edges].float(), max_dsm_height, max_dsm_width, fill_value=0.0).bool())
        normals.append(_pad_normals(item[keys.gt_normals], max_dsm_height, max_dsm_width))
        masks.append(_pad_2d(item[keys.valid_mask].float(), max_dsm_height, max_dsm_width, fill_value=0.0).bool())

    return {
        keys.sample_id: [item[keys.sample_id] for item in batch],
        keys.images: torch.stack(images, dim=0),
        keys.camera_params: [item[keys.camera_params] for item in batch],
        keys.gt_dsm: torch.stack(dsms, dim=0),
        keys.gt_edges: torch.stack(edges, dim=0),
        keys.gt_normals: torch.stack(normals, dim=0),
        keys.valid_mask: torch.stack(masks, dim=0),
    }


def _pad_image_views(images: torch.Tensor, views: int, height: int, width: int) -> torch.Tensor:
    output = torch.zeros(views, images.shape[1], height, width, dtype=images.dtype)
    output[: images.shape[0], :, : images.shape[-2], : images.shape[-1]] = images
    return output


def _pad_2d(tensor: torch.Tensor, height: int, width: int, fill_value: float) -> torch.Tensor:
    output = torch.full((height, width), fill_value=fill_value, dtype=tensor.dtype)
    output[: tensor.shape[-2], : tensor.shape[-1]] = tensor
    return output


def _pad_normals(normals: torch.Tensor, height: int, width: int) -> torch.Tensor:
    output = torch.zeros(normals.shape[0], height, width, dtype=normals.dtype)
    output[2, :, :] = 1.0
    output[:, : normals.shape[-2], : normals.shape[-1]] = normals
    return output

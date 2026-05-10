from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DSM_SUFFIXES = {".npy", ".npz", ".tif", ".tiff", ".png"}


@dataclass
class ManifestIssue:
    level: str
    sample_id: str
    message: str


@dataclass
class ManifestReport:
    samples: int = 0
    errors: list[ManifestIssue] = field(default_factory=list)
    warnings: list[ManifestIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, level: str, sample_id: str, message: str) -> None:
        issue = ManifestIssue(level=level, sample_id=sample_id, message=message)
        if level == "error":
            self.errors.append(issue)
        else:
            self.warnings.append(issue)


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
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


def write_manifest_jsonl(path: str | Path, samples: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def resolve_path(root: str | Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(root) / path


def validate_manifest(
    manifest_path: str | Path,
    root: str | Path | None = None,
    min_views: int = 2,
    check_open: bool = False,
) -> ManifestReport:
    manifest_path = Path(manifest_path)
    root = Path(root) if root is not None else manifest_path.parent
    samples = load_manifest(manifest_path)
    report = ManifestReport(samples=len(samples))

    for index, sample in enumerate(samples):
        sample_id = str(sample.get("id", index))
        for key in ("images", "camera_params", "gt_dsm"):
            if key not in sample:
                report.add("error", sample_id, f"missing required field: {key}")

        images = sample.get("images", [])
        if not isinstance(images, list) or len(images) < min_views:
            report.add("error", sample_id, f"expected at least {min_views} image views")
        else:
            for image_path in images:
                check_existing_file(report, root, sample_id, image_path, IMAGE_SUFFIXES, "image")

        if "gt_dsm" in sample:
            dsm_path = check_existing_file(
                report,
                root,
                sample_id,
                sample["gt_dsm"],
                DSM_SUFFIXES,
                "DSM",
            )
            if dsm_path is not None and check_open:
                check_raster_open(report, sample_id, dsm_path)

        if "valid_mask" in sample:
            check_existing_file(report, root, sample_id, sample["valid_mask"], DSM_SUFFIXES, "valid_mask")

        if "camera_params" in sample:
            validate_camera_entry(report, root, sample_id, sample["camera_params"])

    return report


def check_existing_file(
    report: ManifestReport,
    root: Path,
    sample_id: str,
    value: Any,
    allowed_suffixes: set[str],
    label: str,
) -> Path | None:
    if not isinstance(value, str):
        report.add("error", sample_id, f"{label} path must be a string")
        return None
    path = resolve_path(root, value)
    if not path.exists():
        report.add("error", sample_id, f"{label} file does not exist: {path}")
        return None
    if path.suffix.lower() not in allowed_suffixes:
        report.add("warning", sample_id, f"unusual {label} suffix: {path.suffix}")
    return path


def validate_camera_entry(
    report: ManifestReport,
    root: Path,
    sample_id: str,
    camera_params: Any,
) -> None:
    if isinstance(camera_params, str):
        path = resolve_path(root, camera_params)
        if not path.exists():
            report.add("error", sample_id, f"camera file does not exist: {path}")
            return
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as handle:
                camera_params = json.load(handle)
        else:
            report.add("warning", sample_id, "camera file is not JSON; geometry adapter may be needed")
            return

    if not has_camera_geometry(camera_params):
        report.add(
            "warning",
            sample_id,
            "camera params lack intrinsics/extrinsics/RPC keys; rays will fall back to normalized grids",
        )


def has_camera_geometry(camera_params: Any) -> bool:
    if isinstance(camera_params, list):
        return any(has_camera_geometry(item) for item in camera_params)
    if not isinstance(camera_params, dict):
        return False
    if isinstance(camera_params.get("views"), list):
        return any(has_camera_geometry(item) for item in camera_params["views"])

    direct_keys = {
        "intrinsics",
        "extrinsics",
        "rpc",
        "rpc_coefficients",
        "rpcCoefficients",
        "rational_polynomial_coefficients",
    }
    return any(key in camera_params for key in direct_keys)


def check_raster_open(report: ManifestReport, sample_id: str, path: Path) -> None:
    try:
        if path.suffix.lower() in {".npy", ".npz"}:
            import numpy as np

            data = np.load(path)
            if hasattr(data, "files") and not data.files:
                report.add("error", sample_id, f"empty raster archive: {path}")
        else:
            from PIL import Image

            with Image.open(path) as image:
                image.verify()
    except Exception as exc:  # noqa: BLE001
        report.add("error", sample_id, f"failed to open raster {path}: {exc}")


from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "geotoken"))

from manifest import IMAGE_SUFFIXES, write_manifest_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a GeoToken manifest from sample folders. Each sample folder "
            "should contain an image directory, one DSM file, and optionally a camera JSON."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images-dir", default="images")
    parser.add_argument("--dsm-name", default="dsm.npy")
    parser.add_argument("--camera-name", default="camera.json")
    parser.add_argument("--relative-to-root", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    samples = []
    for sample_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
        image_dir = sample_dir / args.images_dir
        dsm_path = sample_dir / args.dsm_name
        camera_path = sample_dir / args.camera_name
        if not image_dir.exists() or not dsm_path.exists():
            continue

        image_paths = sorted(
            path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if len(image_paths) < 2:
            continue

        sample = {
            "id": sample_dir.name,
            "images": [format_path(path, args.root, args.relative_to_root) for path in image_paths],
            "gt_dsm": format_path(dsm_path, args.root, args.relative_to_root),
            "camera_params": (
                format_path(camera_path, args.root, args.relative_to_root)
                if camera_path.exists()
                else {"views": [{} for _ in image_paths]}
            ),
        }
        samples.append(sample)

    write_manifest_jsonl(args.output, samples)
    print(json.dumps({"samples": len(samples), "output": str(args.output)}, indent=2))


def format_path(path: Path, root: Path, relative: bool) -> str:
    if not relative:
        return str(path)
    return str(path.relative_to(root))


if __name__ == "__main__":
    main()

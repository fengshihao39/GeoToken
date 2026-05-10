from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "geotoken"))

from manifest import validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a GeoToken dataset manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--check-open", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    report = validate_manifest(
        manifest_path=args.manifest,
        root=args.root,
        min_views=args.min_views,
        check_open=args.check_open,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "ok": report.ok,
                    "samples": report.samples,
                    "errors": [issue.__dict__ for issue in report.errors],
                    "warnings": [issue.__dict__ for issue in report.warnings],
                },
                indent=2,
            )
        )
    else:
        print(f"samples: {report.samples}")
        print(f"errors: {len(report.errors)}")
        for issue in report.errors:
            print(f"ERROR [{issue.sample_id}] {issue.message}")
        print(f"warnings: {len(report.warnings)}")
        for issue in report.warnings:
            print(f"WARN  [{issue.sample_id}] {issue.message}")

    if not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

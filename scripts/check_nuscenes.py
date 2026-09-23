#!/usr/bin/env python3
"""Read-only audit of nuScenes archives and already extracted data."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_nuscenes import (  # noqa: E402
    configured_cameras,
    configured_path,
    discover_archives,
    image_error,
    inspect_archive,
    join_camera_records,
    load_config,
    no_archive_message,
    print_archive,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/nuscenes.yaml"))
    parser.add_argument("--validate-images", action="store_true", help="Open every candidate image with Pillow")
    args = parser.parse_args()
    config = load_config(configured_path(args.config))
    archive_dir = configured_path(config["dataset"]["archive_dir"])
    pattern = str(config["dataset"].get("archive_pattern", "*.taz"))
    extract_root = configured_path(config["dataset"]["extract_dir"])
    archives = discover_archives(archive_dir, pattern)
    if not archives:
        raise SystemExit(no_archive_message(archive_dir, pattern))
    print("Discovered archives:")
    for archive in archives:
        print(f"- {archive.resolve()}")
    infos = []
    for archive in archives:
        info = inspect_archive(archive)
        infos.append(info)
        print_archive(info)
    report = {
        "archives_found": len(archives),
        "metadata_archives": sum(info.classification == "METADATA" for info in infos),
        "keyframe_archives": sum(info.classification == "KEYFRAME_BLOBS" for info in infos),
        "extract_dir": str(extract_root.resolve()),
        "extract_dir_exists": extract_root.is_dir(),
    }
    if extract_root.is_dir():
        try:
            selection = config["selection"]
            cameras = configured_cameras(selection.get("cameras", selection.get("camera", "CAM_FRONT")))
            records = join_camera_records(extract_root, cameras, bool(selection["keyframes_only"]))
            report["camera_channels"] = list(cameras)
            report["camera_keyframes"] = len(records)
            report["per_camera"] = dict(Counter(row["camera_channel"] for row in records))
            report["missing_images"] = sum(not Path(row["image_path"]).is_file() for row in records)
            if args.validate_images:
                errors = [image_error(Path(row["image_path"])) for row in records]
                report["valid_images"] = sum(error is None for error in errors)
                report["corrupted_images"] = sum(bool(error and error != "missing") for error in errors)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            report["extracted_metadata_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

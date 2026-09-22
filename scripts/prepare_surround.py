"""Prepare all local nuScenes camera keyframes without a 10K cap."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.surround import prepare_dataset
from src.utils.config import load_config, resolve_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/surround.yaml")
    parser.add_argument("--skip-image-verification", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = config["dataset"]
    previous = dataset.get("preserve_test_csv")
    report = prepare_dataset(
        resolve_path(config, dataset["root"]), resolve_path(config, dataset["processed_dir"]),
        version=dataset["version"], seed=config["seed"],
        validation_fraction=dataset["validation_fraction"], test_fraction=dataset["test_fraction"],
        previous_test=resolve_path(config, previous) if previous else None,
        verify_images=not args.skip_image_verification,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

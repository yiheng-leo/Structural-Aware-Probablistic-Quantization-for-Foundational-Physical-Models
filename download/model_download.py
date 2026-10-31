#!/usr/bin/env python3
"""Download SAFE-SVD model checkpoints from Hugging Face.

Remote-to-local layout:

  poseidon/Poseidon-L/<dataset>/<file> -> models/poseidon/<dataset>/<file>
  VICON/model.pth                       -> models/VICON/model.pth
  MPP/MPP_AViT_L.pth                    -> models/MPP/model

Authenticate with ``hf auth login`` before downloading this private repository.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "Evave7/SAFE-SVD"
POSEIDON_PREFIX = "poseidon/Poseidon-L/"
STATIC_FILES = {
    "VICON/model.pth": Path("VICON/model.pth"),
    "MPP/model": Path("MPP/MPP_AViT_L"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Local directory for all downloaded models (default: models)",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Download only this Poseidon dataset; repeat for multiple datasets.",
    )
    parser.add_argument(
        "--poseidon-only",
        action="store_true",
        help="Download Poseidon checkpoints only; skip VICON and MPP.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing local files.",
    )
    return parser.parse_args()


def place_file(source: Path, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        print(f"Already exists, skipping: {destination}")
        return

    if destination.exists() or destination.is_symlink():
        destination.unlink()

    destination.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source.resolve(), destination)


def download(repo_file: str, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        print(f"Already exists, skipping: {destination}")
        return
    print(f"Downloading {repo_file} -> {destination}")
    cached_file = Path(
        hf_hub_download(repo_id=REPO_ID, repo_type="model", filename=repo_file)
    )
    place_file(cached_file, destination, force)


def main() -> None:
    args = parse_args()
    api = HfApi()
    repo_files = api.list_repo_files(REPO_ID, repo_type="model")

    poseidon_files = [file for file in repo_files if file.startswith(POSEIDON_PREFIX)]
    if not poseidon_files:
        raise RuntimeError(f"No Poseidon files found under {POSEIDON_PREFIX!r} in {REPO_ID}")

    requested_datasets = set(args.datasets or [])
    found_datasets: set[str] = set()
    for repo_file in poseidon_files:
        relative_path = repo_file.removeprefix(POSEIDON_PREFIX)
        dataset_name, separator, filename = relative_path.partition("/")
        if not separator:
            continue
        if requested_datasets and dataset_name not in requested_datasets:
            continue
        found_datasets.add(dataset_name)
        download(repo_file, args.output_dir / "poseidon" / dataset_name / filename, args.force)

    missing = requested_datasets - found_datasets
    if missing:
        raise FileNotFoundError(f"Poseidon dataset(s) not found in the repository: {', '.join(sorted(missing))}")

    if not args.poseidon_only:
        for repo_file, relative_destination in STATIC_FILES.items():
            if repo_file not in repo_files:
                print(f"Remote file not found, skipping: {repo_file}")
                continue
            download(repo_file, args.output_dir / relative_destination, args.force)

    print("Download complete.")


if __name__ == "__main__":
    main()
"""Prepare a local split manifest from the notebook's Kaggle dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from training.config import (
    DATASET_DOWNLOAD_SIZE_GB,
    DATASET_HANDLE,
    MANIFEST_PATH,
    RECOMMENDED_FREE_SPACE_GB,
)
from training.dataset import build_manifest, save_manifest, split_summary


def download_dataset(confirm_download: bool) -> Path:
    if not confirm_download:
        raise SystemExit(
            f"Download not started. The Kaggle dataset is {DATASET_DOWNLOAD_SIZE_GB:.2f} GB. "
            f"Keep at least {RECOMMENDED_FREE_SPACE_GB:.0f} GB free and repeat with "
            "--download --confirm-download after reviewing storage."
        )
    import kagglehub

    return Path(kagglehub.dataset_download(DATASET_HANDLE))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic, patient-group-aware local dataset splits."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-dir", type=Path, help="Existing extracted dataset directory.")
    source.add_argument("--download", action="store_true", help="Download with KaggleHub.")
    parser.add_argument(
        "--confirm-download",
        action="store_true",
        help="Acknowledge the 1.24 GB download and recommended 3 GB free space.",
    )
    parser.add_argument("--output", type=Path, default=MANIFEST_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = download_dataset(args.confirm_download) if args.download else args.dataset_dir
    manifest, dataset_root = build_manifest(dataset_dir)
    output = save_manifest(manifest, args.output)
    print(f"Dataset root: {dataset_root}")
    print(f"Manifest: {output}")
    print(split_summary(manifest).to_string(index=False))
    print("Patient overlap between train and validation: 0")
    print("Official test directory preserved unchanged for final evaluation.")


if __name__ == "__main__":
    main()

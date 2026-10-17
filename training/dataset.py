"""Dataset discovery, patient-safe splitting, manifests, and tf.data loading."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from training.config import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    RANDOM_SEED,
    SUPPORTED_EXTENSIONS,
    VALIDATION_FRACTION,
)


REQUIRED_MANIFEST_COLUMNS = {
    "relative_path",
    "label",
    "class_name",
    "split",
    "source_split",
    "patient_id",
}


class DatasetError(ValueError):
    """Raised when the local dataset cannot satisfy the training contract."""


def resolve_dataset_root(path: str | Path) -> Path:
    """Locate the train/test/val directory across known Kaggle archive layouts."""
    supplied = Path(path).expanduser().resolve()
    candidates = (
        supplied,
        supplied / "chest_xray",
        supplied / "chest_xray" / "chest_xray",
    )
    for candidate in candidates:
        if all((candidate / split).is_dir() for split in ("train", "test", "val")):
            return candidate
    raise DatasetError(
        "Could not find train, test, and val folders. Point --dataset-dir to the downloaded "
        "Kaggle dataset root or its chest_xray folder."
    )


def extract_patient_id(path: Path) -> str:
    """Extract the public dataset's patient identifier from common filename forms."""
    stem = path.stem.lower()
    person_match = re.match(r"(person\d+)", stem)
    if person_match:
        return person_match.group(1)

    normal_match = re.search(r"((?:normal\d+-)?im-\d+)", stem)
    if normal_match:
        return normal_match.group(1)

    # Conservative fallback: strip known diagnosis/image suffixes only.
    fallback = re.sub(r"_(?:bacteria|virus)_\d+$", "", stem)
    fallback = re.sub(r"-\d{4}$", "", fallback)
    return fallback


def _collect_split(dataset_root: Path, source_split: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for class_name in CLASS_NAMES:
        class_dir = dataset_root / source_split / class_name
        if not class_dir.is_dir():
            raise DatasetError(f"Missing dataset directory: {class_dir}")
        paths = sorted(
            path for path in class_dir.iterdir() if path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not paths:
            raise DatasetError(f"No supported images found in {class_dir}")
        for path in paths:
            rows.append(
                {
                    "relative_path": path.relative_to(dataset_root).as_posix(),
                    "label": CLASS_TO_INDEX[class_name],
                    "class_name": class_name,
                    "source_split": source_split,
                    "patient_id": extract_patient_id(path),
                }
            )
    return pd.DataFrame(rows)


def _patient_group_split(source_train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each class by patient group to prevent train/validation overlap."""
    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []

    for class_name in CLASS_NAMES:
        class_rows = source_train[source_train["class_name"] == class_name].copy()
        patient_ids = sorted(class_rows["patient_id"].unique().tolist())
        if len(patient_ids) < 2:
            raise DatasetError(f"Not enough patient groups to split class {class_name}.")
        train_patients, validation_patients = train_test_split(
            patient_ids,
            test_size=VALIDATION_FRACTION,
            random_state=RANDOM_SEED,
            shuffle=True,
        )
        train_parts.append(class_rows[class_rows["patient_id"].isin(train_patients)])
        validation_parts.append(class_rows[class_rows["patient_id"].isin(validation_patients)])

    train_rows = pd.concat(train_parts, ignore_index=True).sort_values("relative_path")
    validation_rows = pd.concat(validation_parts, ignore_index=True).sort_values("relative_path")
    return train_rows.reset_index(drop=True), validation_rows.reset_index(drop=True)


def build_manifest(dataset_dir: str | Path) -> tuple[pd.DataFrame, Path]:
    """Create deterministic train/validation/test assignments without moving images."""
    dataset_root = resolve_dataset_root(dataset_dir)
    source_train = _collect_split(dataset_root, "train")
    train_rows, validation_rows = _patient_group_split(source_train)
    test_rows = _collect_split(dataset_root, "test").sort_values("relative_path").reset_index(drop=True)
    source_validation = (
        _collect_split(dataset_root, "val").sort_values("relative_path").reset_index(drop=True)
    )

    train_rows["split"] = "train"
    validation_rows["split"] = "validation"
    test_rows["split"] = "test"
    # The notebook did not train on the dataset's 16-image source val folder.
    source_validation["split"] = "source_validation_unused"

    manifest = pd.concat(
        [train_rows, validation_rows, test_rows, source_validation], ignore_index=True
    )
    manifest = manifest[
        ["relative_path", "label", "class_name", "split", "source_split", "patient_id"]
    ]
    validate_manifest(manifest, dataset_root, verify_files=True)
    return manifest, dataset_root


def validate_manifest(
    manifest: pd.DataFrame, dataset_root: str | Path, *, verify_files: bool = False
) -> None:
    """Validate class mapping, split coverage, files, and patient isolation."""
    missing = REQUIRED_MANIFEST_COLUMNS.difference(manifest.columns)
    if missing:
        raise DatasetError(f"Manifest is missing columns: {', '.join(sorted(missing))}")
    if manifest.empty:
        raise DatasetError("Manifest contains no images.")
    if not manifest["label"].isin([0, 1]).all():
        raise DatasetError("Manifest labels must be 0 (NORMAL) or 1 (PNEUMONIA).")
    expected_mapping = manifest["class_name"].map(CLASS_TO_INDEX)
    if expected_mapping.isna().any() or not (expected_mapping == manifest["label"]).all():
        raise DatasetError("Manifest class names do not match the required class mapping.")

    required_splits = {"train", "validation", "test"}
    if not required_splits.issubset(set(manifest["split"])):
        raise DatasetError("Manifest must contain train, validation, and test splits.")

    train_patients = set(manifest.loc[manifest["split"] == "train", "patient_id"])
    validation_patients = set(
        manifest.loc[manifest["split"] == "validation", "patient_id"]
    )
    overlap = train_patients.intersection(validation_patients)
    if overlap:
        preview = ", ".join(sorted(overlap)[:5])
        raise DatasetError(f"Patient leakage detected between train and validation: {preview}")

    test_rows = manifest[manifest["split"] == "test"]
    if not (test_rows["source_split"] == "test").all():
        raise DatasetError("The evaluation split must contain only untouched source test images.")

    if verify_files:
        root = Path(dataset_root)
        missing_files = [
            relative_path
            for relative_path in manifest["relative_path"]
            if not (root / Path(relative_path)).is_file()
        ]
        if missing_files:
            raise DatasetError(f"Manifest references missing file: {missing_files[0]}")


def save_manifest(manifest: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    return output


def load_manifest(path: str | Path, dataset_root: str | Path) -> pd.DataFrame:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise DatasetError(
            f"Manifest not found: {manifest_path}. Run python -m training.prepare_dataset first."
        )
    manifest = pd.read_csv(manifest_path)
    validate_manifest(manifest, dataset_root, verify_files=True)
    return manifest


def make_tf_dataset(
    manifest: pd.DataFrame,
    dataset_root: str | Path,
    split: str,
    batch_size: int,
    *,
    shuffle: bool,
):
    """Build the exact grayscale, resize, normalization pipeline from the notebook."""
    import tensorflow as tf

    split_rows = manifest[manifest["split"] == split].sort_values("relative_path")
    if split_rows.empty:
        raise DatasetError(f"Manifest has no rows for split: {split}")
    root = Path(dataset_root).resolve()
    paths = [str(root / Path(relative_path)) for relative_path in split_rows["relative_path"]]
    labels = split_rows["label"].astype("float32").tolist()

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(paths), seed=RANDOM_SEED, reshuffle_each_iteration=True
        )

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=1, expand_animations=False)
        image.set_shape([None, None, 1])
        image = tf.image.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        image = tf.cast(image, tf.float32) / 255.0
        return image, tf.cast(label, tf.float32)

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(tf.data.AUTOTUNE)


def split_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return image and patient counts for a human-verifiable split report."""
    return (
        manifest.groupby(["split", "class_name"], sort=False)
        .agg(images=("relative_path", "count"), patients=("patient_id", "nunique"))
        .reset_index()
    )

"""Verify dashboard preprocessing and Grad-CAM against an actual trained model."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from training.config import MANIFEST_PATH, MODEL_PATH
from training.dataset import load_manifest, resolve_dataset_root
from training.runtime import configure_reproducibility
from utils.gradcam import make_gradcam_heatmap, overlay_heatmap
from utils.model_loader import predict_pneumonia_score
from utils.preprocessing import preprocess_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a trained model with one untouched test image.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--overlay-output", type=Path, default=Path("models/gradcam_verification.png")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tf = configure_reproducibility()
    dataset_root = resolve_dataset_root(args.dataset_dir)
    manifest = load_manifest(args.manifest, dataset_root)
    test_row = manifest[manifest["split"] == "test"].sort_values("relative_path").iloc[0]
    image_path = dataset_root / Path(test_row["relative_path"])
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise SystemExit(f"Trained model not found: {model_path}")

    model = tf.keras.models.load_model(model_path, compile=False)
    with Image.open(image_path) as source:
        image = source.copy()
    batch = preprocess_image(image, model, encoded_bytes=image_path.read_bytes())
    score = predict_pneumonia_score(model, batch)
    explain_pneumonia = score >= 0.5
    heatmap, layer_name = make_gradcam_heatmap(model, batch, explain_pneumonia)
    overlay = overlay_heatmap(image, heatmap)
    output = args.overlay_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output)

    predicted = "PNEUMONIA" if explain_pneumonia else "NORMAL"
    print(f"Test image: {image_path}")
    print(f"Dataset label: {test_row['class_name']}")
    print(f"Thresholded model class: {predicted}")
    print(f"Raw pneumonia sigmoid output: {score:.6f}")
    print(f"Grad-CAM layer: {layer_name}")
    print(f"Overlay saved to: {output}")
    print("Actual-model dashboard integration verification passed.")


if __name__ == "__main__":
    main()

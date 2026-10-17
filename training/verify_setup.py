"""Fast local environment and architecture verification; does not train."""

from __future__ import annotations

import json

import numpy as np

from training.config import INPUT_SHAPE
from training.model import build_compiled_cnn
from training.runtime import configure_reproducibility, environment_report
from utils.gradcam import make_gradcam_heatmap


EXPECTED_PARAMETER_COUNT = 11_168_513


def main() -> None:
    tf = configure_reproducibility()
    model = build_compiled_cnn()
    if model.input_shape != (None, *INPUT_SHAPE):
        raise RuntimeError(f"Unexpected input shape: {model.input_shape}")
    if model.output_shape != (None, 1):
        raise RuntimeError(f"Unexpected output shape: {model.output_shape}")
    if model.count_params() != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PARAMETER_COUNT:,} parameters, found {model.count_params():,}."
        )

    sample = np.zeros((1, *INPUT_SHAPE), dtype=np.float32)
    output = np.asarray(model(sample, training=False))
    if output.shape != (1, 1) or not np.isfinite(output).all():
        raise RuntimeError("Synthetic forward pass failed.")
    heatmap, layer_name = make_gradcam_heatmap(model, sample, explain_pneumonia=True)
    if heatmap.ndim != 2 or not np.isfinite(heatmap).all():
        raise RuntimeError("Synthetic Grad-CAM integration check failed.")

    print(json.dumps(environment_report(tf), indent=2))
    print(f"Architecture: {model.input_shape} -> {model.output_shape}")
    print(f"Parameters: {model.count_params():,}")
    print(f"Grad-CAM layer: {layer_name}; heatmap shape: {heatmap.shape}")
    print("Setup verification passed. No training was performed.")


if __name__ == "__main__":
    main()

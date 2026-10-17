"""Cached TensorFlow model discovery, validation, and inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_CANDIDATES = (
    ROOT_DIR / "models" / "pneumonia_cnn.keras",
    ROOT_DIR / "pneumonia_cnn.keras",
)


class ModelCompatibilityError(ValueError):
    """Raised when a model does not match the binary classifier contract."""


def find_model_path() -> Path | None:
    """Find the preferred local Keras artifact without scanning outside the project."""
    return next((path for path in MODEL_CANDIDATES if path.is_file()), None)


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """Load and validate a Keras model once per Streamlit process."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is not installed in this environment. Install the project requirements first."
        ) from exc

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as exc:
        raise RuntimeError(f"Keras could not load the model: {exc}") from exc

    output_shape = model.output_shape
    if isinstance(output_shape, list) or len(output_shape) != 2 or output_shape[-1] != 1:
        raise ModelCompatibilityError(
            f"Expected one sigmoid output for binary classification, but the model uses {output_shape}."
        )
    return model


def predict_pneumonia_score(model, image_batch: np.ndarray) -> float:
    """Return the model's raw pneumonia sigmoid output after sanity checks."""
    try:
        prediction = np.asarray(model(image_batch, training=False)).reshape(-1)
    except Exception as exc:
        raise ModelCompatibilityError(f"Inference failed for this image/model combination: {exc}") from exc
    if prediction.size != 1 or not np.isfinite(prediction[0]):
        raise ModelCompatibilityError("The model did not return one finite binary output.")
    score = float(prediction[0])
    if score < 0.0 or score > 1.0:
        raise ModelCompatibilityError(
            "The model output is outside [0, 1]; a final sigmoid activation may be missing."
        )
    return score

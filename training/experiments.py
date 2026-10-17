"""Model builders and class-weight policy for controlled validation experiments."""

from __future__ import annotations

import pandas as pd

from training.config import INPUT_SHAPE
from training.model import build_cnn, compile_model


EXPERIMENT_NAMES = ("baseline", "class_weighted", "mobilenet_v2")


def calculate_balanced_class_weights(manifest: pd.DataFrame) -> dict[int, float]:
    """Calculate inverse-frequency weights from training rows only."""
    training_rows = manifest[manifest["split"] == "train"]
    counts = training_rows["label"].value_counts().to_dict()
    if set(counts) != {0, 1} or any(count <= 0 for count in counts.values()):
        raise ValueError("Training rows must contain both binary classes.")
    total = float(sum(counts.values()))
    return {label: total / (2.0 * float(counts[label])) for label in (0, 1)}


def build_experiment_model(experiment: str, *, transfer_weights: str | None = "imagenet"):
    """Build a baseline-family CNN or frozen lightweight MobileNetV2 candidate."""
    if experiment not in EXPERIMENT_NAMES:
        raise ValueError(f"Unknown experiment: {experiment}")
    if experiment in {"baseline", "class_weighted"}:
        return compile_model(build_cnn())

    import tensorflow as tf

    inputs = tf.keras.layers.Input(shape=INPUT_SHAPE, name="grayscale_xray")
    rgb = tf.keras.layers.Concatenate(name="grayscale_to_rgb")([inputs, inputs, inputs])
    scaled = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0, name="mobilenet_scaling")(rgb)
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(INPUT_SHAPE[0], INPUT_SHAPE[1], 3),
        alpha=0.35,
        include_top=False,
        weights=transfer_weights,
    )
    backbone.trainable = False
    features = backbone(scaled, training=False)
    pooled = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(features)
    dropped = tf.keras.layers.Dropout(0.25, name="transfer_dropout")(pooled)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="pneumonia_output")(dropped)
    model = tf.keras.Model(inputs, output, name="pneumovision_mobilenet_v2_035")
    return compile_model(model)


def class_weights_for_experiment(
    experiment: str, manifest: pd.DataFrame
) -> dict[int, float] | None:
    if experiment == "class_weighted":
        return calculate_balanced_class_weights(manifest)
    return None

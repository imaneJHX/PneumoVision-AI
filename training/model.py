"""Notebook-faithful CNN architecture and compilation."""

from __future__ import annotations

from training.config import INPUT_SHAPE, LEARNING_RATE


def build_cnn():
    """Build the exact sequential CNN defined in notebook cell 14."""
    from tensorflow.keras import layers, models

    return models.Sequential(
        [
            layers.Input(shape=INPUT_SHAPE),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="pneumovision_cnn",
    )


def compile_model(model):
    """Apply the notebook optimizer/loss plus expanded monitoring metrics."""
    from tensorflow.keras.metrics import AUC, BinaryAccuracy, Precision, Recall
    from tensorflow.keras.optimizers import Adam

    from training.metrics import BinaryF1

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            BinaryAccuracy(name="accuracy"),
            Precision(name="precision"),
            Recall(name="recall"),
            BinaryF1(name="f1"),
            # More thresholds reduce approximation error for saturated sigmoid scores.
            # Final reports still use exact sklearn ROC-AUC from saved predictions.
            AUC(name="auc", num_thresholds=1000),
        ],
    )
    return model


def build_compiled_cnn():
    return compile_model(build_cnn())

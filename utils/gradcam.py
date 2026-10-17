"""Grad-CAM generation for compatible TensorFlow/Keras CNNs."""

from __future__ import annotations

import numpy as np
from PIL import Image


class GradCAMError(RuntimeError):
    """Raised when a valid Grad-CAM map cannot be generated."""


def find_last_conv_layer(model):
    """Find the last Conv2D layer used by the model."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise GradCAMError("TensorFlow is required to generate Grad-CAM.") from exc

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise GradCAMError("No Conv2D layer was found in the loaded model.")


def make_gradcam_heatmap(model, image_batch: np.ndarray, explain_pneumonia: bool) -> tuple[np.ndarray, str]:
    """Generate a normalized map for the selected binary class output."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise GradCAMError("TensorFlow is required to generate Grad-CAM.") from exc

    conv_layer = find_last_conv_layer(model)
    try:
        tensor = tf.convert_to_tensor(image_batch, dtype=tf.float32)
        with tf.GradientTape() as tape:
            # Replay the sequential network so the captured feature tensor and
            # prediction are guaranteed to belong to the same eager graph.
            # This avoids disconnected symbolic tensors after a Keras 3 model
            # has been serialized and reloaded.
            activations = tensor
            conv_outputs = None
            for layer in model.layers:
                activations = layer(activations, training=False)
                if layer is conv_layer:
                    conv_outputs = activations
                    tape.watch(conv_outputs)

            if conv_outputs is None:
                raise GradCAMError("The selected convolution layer was not reached during inference.")
            predictions = activations
            pneumonia_score = predictions[:, 0]
            target_score = pneumonia_score if explain_pneumonia else 1.0 - pneumonia_score

        gradients = tape.gradient(target_score, conv_outputs)
        if gradients is None:
            raise GradCAMError("The selected output is not differentiable from the convolution features.")
        weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(conv_outputs[0] * weights, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        maximum = tf.reduce_max(heatmap)
        heatmap = tf.where(maximum > 0, heatmap / maximum, tf.zeros_like(heatmap))
        result = heatmap.numpy()
    except GradCAMError:
        raise
    except Exception as exc:
        raise GradCAMError(f"Grad-CAM could not be generated for this model: {exc}") from exc

    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise GradCAMError("Grad-CAM produced an invalid heatmap.")
    return result, conv_layer.name


def _turbo_like(values: np.ndarray) -> np.ndarray:
    """Map normalized values to a high-contrast blue-to-red palette."""
    stops = np.array(
        [
            [0.00, 20, 25, 92],
            [0.20, 20, 110, 210],
            [0.40, 27, 210, 205],
            [0.60, 238, 222, 72],
            [0.80, 242, 106, 35],
            [1.00, 153, 22, 45],
        ],
        dtype=np.float32,
    )
    flat = values.reshape(-1)
    rgb = np.column_stack(
        [np.interp(flat, stops[:, 0], stops[:, channel]) for channel in range(1, 4)]
    )
    return rgb.reshape((*values.shape, 3)).astype(np.uint8)


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.42) -> Image.Image:
    """Blend a Grad-CAM heatmap over a display-sized grayscale X-ray."""
    base = image.convert("L").convert("RGB")
    heatmap_image = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), mode="L")
    heatmap_array = np.asarray(
        heatmap_image.resize(base.size, Image.Resampling.BILINEAR), dtype=np.float32
    ) / 255.0
    colored = Image.fromarray(_turbo_like(heatmap_array), mode="RGB")
    return Image.blend(base, colored, alpha=alpha)

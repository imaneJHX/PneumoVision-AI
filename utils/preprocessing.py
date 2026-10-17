"""Image validation and notebook-compatible preprocessing."""

from __future__ import annotations

from io import BytesIO
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
SUPPORTED_FORMATS = {"JPEG", "PNG"}


class ImageValidationError(ValueError):
    """Raised when an uploaded image cannot be safely processed."""


def load_uploaded_image(uploaded_file: BinaryIO) -> Image.Image:
    """Validate an in-memory upload and return an orientation-corrected image."""
    raw = uploaded_file.getvalue()
    if not raw:
        raise ImageValidationError("The uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ImageValidationError("The image exceeds the 15 MB upload limit.")

    try:
        with Image.open(BytesIO(raw)) as candidate:
            image_format = candidate.format
            width, height = candidate.size
            candidate.verify()
        if image_format not in SUPPORTED_FORMATS:
            raise ImageValidationError("Only JPG, JPEG, and PNG images are supported.")
        if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
            raise ImageValidationError("The image dimensions are invalid or too large to process safely.")
        with Image.open(BytesIO(raw)) as candidate:
            return ImageOps.exif_transpose(candidate).copy()
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageValidationError("The file is not a readable JPG or PNG image.") from exc


def infer_model_input_spec(model) -> tuple[int, int, int]:
    """Return (height, width, channels) from a compatible Keras model."""
    shape = model.input_shape
    if isinstance(shape, list):
        if len(shape) != 1:
            raise ValueError("Models with multiple inputs are not supported.")
        shape = shape[0]
    if len(shape) != 4:
        raise ValueError(f"Expected a 4D image input, but the model uses {shape}.")

    _, height, width, channels = shape
    if height is None or width is None:
        raise ValueError("The model must define fixed input height and width.")
    if channels not in (1, 3):
        raise ValueError(f"Expected 1 or 3 input channels, but the model uses {channels}.")
    return int(height), int(width), int(channels)


def preprocess_image(
    image: Image.Image, model, *, encoded_bytes: bytes | None = None
) -> np.ndarray:
    """Decode, resize, and normalize using the notebook's TensorFlow operations."""
    height, width, channels = infer_model_input_spec(model)
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for notebook-matched preprocessing.") from exc

    if encoded_bytes is not None:
        tensor = tf.io.decode_image(
            encoded_bytes, channels=channels, expand_animations=False
        )
        tensor.set_shape([None, None, channels])
    else:
        mode = "L" if channels == 1 else "RGB"
        array = np.asarray(image.convert(mode))
        if channels == 1:
            array = array[..., np.newaxis]
        tensor = tf.convert_to_tensor(array)
    tensor = tf.image.resize(tensor, (height, width))
    tensor = tf.cast(tensor, tf.float32) / 255.0
    return np.expand_dims(tensor.numpy(), axis=0)

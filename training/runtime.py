"""Runtime reproducibility and environment reporting helpers."""

from __future__ import annotations

import os
import platform
import random
import sys

import numpy as np

from training.config import RANDOM_SEED


def configure_reproducibility(seed: int = RANDOM_SEED):
    """Configure deterministic seeds before datasets and the model are created."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        pass
    return tf


def environment_report(tf) -> dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "devices": [device.name for device in tf.config.list_physical_devices()],
        "gpu_devices": [device.name for device in tf.config.list_physical_devices("GPU")],
    }

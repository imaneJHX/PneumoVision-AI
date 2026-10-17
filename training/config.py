"""Single source of truth for the notebook-reproduced ML configuration."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "manifests" / "dataset_manifest.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "pneumonia_cnn.keras"
CHECKPOINT_PATH = MODEL_DIR / "checkpoints" / "best.keras"
RUN_DIR = PROJECT_ROOT / "training" / "runs"

DATASET_HANDLE = "paultimothymooney/chest-xray-pneumonia"
DATASET_DOWNLOAD_SIZE_GB = 1.24
RECOMMENDED_FREE_SPACE_GB = 3.0

CLASS_NAMES = ("NORMAL", "PNEUMONIA")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 1
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

BATCH_SIZE = 32
VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42
EPOCHS = 10
LEARNING_RATE = 0.0001
CLASSIFICATION_THRESHOLD = 0.50
EARLY_STOPPING_PATIENCE = 3

SUPPORTED_EXTENSIONS = {".jpeg", ".jpg", ".png"}

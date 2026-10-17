"""Local training and evaluation pipeline for PneumoVision AI."""

from __future__ import annotations

import os
from pathlib import Path


# Keep generated library caches inside the ignored project cache on Windows.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

"""Validated loading for model evaluation labels and scores."""

from __future__ import annotations

from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_EVALUATION_PATH = ROOT_DIR / "models" / "evaluation_data.csv"
REQUIRED_COLUMNS = ("y_true", "y_score")


class EvaluationDataError(ValueError):
    """Raised when evaluation input cannot support valid binary metrics."""


def load_evaluation_data(source: str | Path | IO[bytes]) -> pd.DataFrame:
    """Load and strictly validate binary ground-truth labels and model scores."""
    try:
        frame = pd.read_csv(source)
    except Exception as exc:
        raise EvaluationDataError(f"The evaluation CSV could not be read: {exc}") from exc

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise EvaluationDataError("The CSV must contain columns named y_true and y_score.")

    data = frame.loc[:, REQUIRED_COLUMNS].copy()
    data["y_true"] = pd.to_numeric(data["y_true"], errors="coerce")
    data["y_score"] = pd.to_numeric(data["y_score"], errors="coerce")
    if data.isna().any().any() or not np.isfinite(data.to_numpy()).all():
        raise EvaluationDataError("Labels and scores must be finite numeric values with no missing rows.")
    if data.empty:
        raise EvaluationDataError("The evaluation CSV contains no rows.")
    if not data["y_true"].isin([0, 1]).all():
        raise EvaluationDataError("y_true must use 0 for NORMAL and 1 for PNEUMONIA.")
    if not data["y_score"].between(0.0, 1.0).all():
        raise EvaluationDataError("y_score must contain sigmoid outputs between 0 and 1.")
    if data["y_true"].nunique() < 2:
        raise EvaluationDataError("Evaluation data must include both NORMAL and PNEUMONIA labels.")

    data["y_true"] = data["y_true"].astype(int)
    return data


def summarize_at_threshold(data: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    """Calculate binary metrics from validated labels and raw model scores."""
    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise EvaluationDataError("The classification threshold must be inside (0, 1).")
    if not set(REQUIRED_COLUMNS).issubset(data.columns):
        raise EvaluationDataError("Evaluation data must contain y_true and y_score.")

    y_true = data["y_true"].to_numpy(dtype=int)
    y_score = data["y_score"].to_numpy(dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "cases": int(len(y_true)),
    }

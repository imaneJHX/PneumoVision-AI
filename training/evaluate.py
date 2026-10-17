"""Evaluate a trained model on the untouched source test directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from training.config import (
    BATCH_SIZE,
    CLASSIFICATION_THRESHOLD,
    CLASS_NAMES,
    MANIFEST_PATH,
    MODEL_DIR,
    MODEL_PATH,
)
from training.dataset import load_manifest, make_tf_dataset, resolve_dataset_root
from training.model import compile_model
from training.runtime import configure_reproducibility, environment_report
from utils.evaluation import summarize_at_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PneumoVision CNN on the test split.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--threshold",
        type=float,
        default=CLASSIFICATION_THRESHOLD,
        help="Preselected classification threshold strictly between 0 and 1.",
    )
    return parser.parse_args()


def _save_confusion_matrix(cm: np.ndarray, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=axis, cmap="Blues", values_format="d", colorbar=False
    )
    axis.set_title("PneumoVision CNN — Test Confusion Matrix")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _save_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 5.2))
    axis.plot(fpr, tpr, label=f"CNN (AUC = {roc_auc:.3f})", linewidth=2)
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    axis.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
    axis.set_title("Receiver Operating Characteristic")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _save_precision_recall_curve(
    recall: np.ndarray, precision: np.ndarray, pr_auc: float, output: Path
) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 5.2))
    axis.plot(recall, precision, label=f"CNN (AUC = {pr_auc:.3f})", linewidth=2)
    axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    axis.set_title("Precision–Recall Curve")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise SystemExit("--threshold must be strictly between 0 and 1.")
    tf = configure_reproducibility()
    dataset_root = resolve_dataset_root(args.dataset_dir)
    manifest = load_manifest(args.manifest, dataset_root)
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise SystemExit(f"Trained model not found: {model_path}")

    test_rows = manifest[manifest["split"] == "test"].sort_values("relative_path")
    test_dataset = make_tf_dataset(
        manifest, dataset_root, "test", args.batch_size, shuffle=False
    )
    model = tf.keras.models.load_model(model_path, compile=False)
    compile_model(model)

    keras_results = {
        key: float(value)
        for key, value in model.evaluate(test_dataset, return_dict=True, verbose=1).items()
    }
    y_true = test_rows["label"].to_numpy(dtype=int)
    y_score = model.predict(test_dataset, verbose=1).reshape(-1)
    if len(y_true) != len(y_score):
        raise RuntimeError("Prediction count does not match the untouched test manifest.")
    if not np.isfinite(y_score).all() or np.any((y_score < 0) | (y_score > 1)):
        raise RuntimeError("Model returned invalid sigmoid scores.")

    y_pred = (y_score >= args.threshold).astype(int)
    score_frame = pd.DataFrame({"y_true": y_true, "y_score": y_score})
    threshold_metrics = summarize_at_threshold(score_frame, args.threshold)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report_text = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, zero_division=0
    )
    report_dict = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, zero_division=0, output_dict=True
    )
    roc_auc = float(roc_auc_score(y_true, y_score))
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    pr_auc = float(auc(recall, precision))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    score_frame.to_csv(output_dir / "evaluation_data.csv", index=False)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    pd.DataFrame(
        {"false_positive_rate": fpr, "true_positive_rate": tpr, "threshold": roc_thresholds}
    ).to_csv(output_dir / "roc_curve.csv", index=False)
    pd.DataFrame({"recall": recall, "precision": precision}).to_csv(
        output_dir / "precision_recall_curve.csv", index=False
    )
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    (output_dir / "classification_report.json").write_text(
        json.dumps(report_dict, indent=2), encoding="utf-8"
    )
    summary = {
        "threshold": args.threshold,
        "test_images": int(len(y_true)),
        "roc_auc": roc_auc,
        "precision_recall_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
        "threshold_metrics": threshold_metrics,
        "keras_metric_threshold": CLASSIFICATION_THRESHOLD,
        "keras_evaluation": keras_results,
        "environment": environment_report(tf),
    }
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    _save_confusion_matrix(cm, output_dir / "confusion_matrix.png")
    _save_roc_curve(fpr, tpr, roc_auc, output_dir / "roc_curve.png")
    _save_precision_recall_curve(
        recall, precision, pr_auc, output_dir / "precision_recall_curve.png"
    )

    print(report_text)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Precision–Recall AUC: {pr_auc:.4f}")
    print(f"Evaluation artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()

"""Run one approval-gated model experiment using train/validation data only."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from training.config import (
    BATCH_SIZE,
    CLASSIFICATION_THRESHOLD,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    MANIFEST_PATH,
    RUN_DIR,
)
from training.dataset import load_manifest, make_tf_dataset, resolve_dataset_root, split_summary
from training.experiments import (
    EXPERIMENT_NAMES,
    build_experiment_model,
    class_weights_for_experiment,
)
from training.runtime import configure_reproducibility, environment_report
from utils.evaluation import summarize_at_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one isolated PneumoVision candidate on train/validation only."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--experiment", choices=EXPERIMENT_NAMES, required=True)
    parser.add_argument("--run-name", help="Output folder name; defaults to the experiment name.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--transfer-weights",
        choices=("imagenet", "none"),
        default="imagenet",
        help="ImageNet creates a transfer-learning candidate; none is intended only for offline plumbing tests.",
    )
    parser.add_argument(
        "--confirm-training",
        action="store_true",
        help="Required acknowledgement before any candidate training starts.",
    )
    return parser.parse_args()


def select_validation_threshold(data: pd.DataFrame) -> tuple[float, dict[str, float | int]]:
    """Select a threshold by a predefined balanced-accuracy objective on validation only."""
    candidates = np.linspace(0.01, 0.99, 99)
    y_true = data["y_true"].to_numpy(dtype=int)
    y_score = data["y_score"].to_numpy(dtype=float)
    ranked = []
    for threshold in candidates:
        predictions = (y_score >= threshold).astype(int)
        ranked.append((float(balanced_accuracy_score(y_true, predictions)), float(threshold)))
    _, selected = max(ranked, key=lambda item: (item[0], -abs(item[1] - 0.5)))
    return selected, summarize_at_threshold(data, selected)


def main() -> None:
    args = parse_args()
    if not args.confirm_training:
        raise SystemExit(
            "Experiment not started. Review the configuration, then repeat with --confirm-training."
        )
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs and --batch-size must be positive integers.")

    run_name = args.run_name or args.experiment
    if re.fullmatch(r"[A-Za-z0-9_-]+", run_name) is None:
        raise SystemExit("--run-name may contain only letters, numbers, hyphens, and underscores.")
    output_dir = (RUN_DIR / "experiments" / run_name).resolve()
    if (output_dir / "summary.json").exists() or (output_dir / "candidate.keras").exists():
        raise SystemExit(f"Refusing to overwrite an existing experiment run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tf = configure_reproducibility()
    dataset_root = resolve_dataset_root(args.dataset_dir)
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path, dataset_root)
    print(split_summary(manifest[manifest["split"].isin(["train", "validation"])]).to_string(index=False))
    print("Official test rows are excluded from candidate training, callbacks, and threshold selection.")

    train_dataset = make_tf_dataset(
        manifest, dataset_root, "train", args.batch_size, shuffle=True
    )
    validation_dataset = make_tf_dataset(
        manifest, dataset_root, "validation", args.batch_size, shuffle=False
    )
    transfer_weights = None if args.transfer_weights == "none" else args.transfer_weights
    model = build_experiment_model(args.experiment, transfer_weights=transfer_weights)
    class_weights = class_weights_for_experiment(args.experiment, manifest)

    checkpoint_path = output_dir / "best.keras"
    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.CSVLogger(output_dir / "history.csv"),
    ]
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        shuffle=False,
    )
    candidate_path = output_dir / "candidate.keras"
    model.save(candidate_path)

    validation_rows = (
        manifest[manifest["split"] == "validation"].sort_values("relative_path").reset_index(drop=True)
    )
    validation_scores = model.predict(validation_dataset, verbose=1).reshape(-1)
    predictions = pd.DataFrame(
        {
            "relative_path": validation_rows["relative_path"],
            "y_true": validation_rows["label"].astype(int),
            "y_score": validation_scores,
        }
    )
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    metric_data = predictions[["y_true", "y_score"]]
    default_metrics = summarize_at_threshold(metric_data, CLASSIFICATION_THRESHOLD)
    selected_threshold, selected_metrics = select_validation_threshold(metric_data)

    history_json = {
        key: [float(value) for value in values] for key, values in history.history.items()
    }
    (output_dir / "history.json").write_text(
        json.dumps(history_json, indent=2), encoding="utf-8"
    )
    summary = {
        "experiment": args.experiment,
        "run_name": run_name,
        "model_path": str(candidate_path),
        "production_model_replaced": False,
        "parameters": int(model.count_params()),
        "epochs_completed": len(history.epoch),
        "class_weights": class_weights,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "validation_metrics_at_0_5": default_metrics,
        "validation_threshold_selection": {
            "objective": "maximum balanced accuracy on patient-group-aware validation data",
            "threshold": selected_threshold,
            "metrics": selected_metrics,
        },
        "test_split_used": False,
        "environment": environment_report(tf),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Candidate saved outside the production model path: {candidate_path}")


if __name__ == "__main__":
    main()

"""Train the notebook CNN locally from a prepared split manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    MANIFEST_PATH,
    MODEL_PATH,
    RUN_DIR,
)
from training.dataset import load_manifest, make_tf_dataset, resolve_dataset_root, split_summary
from training.model import build_compiled_cnn
from training.runtime import configure_reproducibility, environment_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PneumoVision CNN locally.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--model-output", type=Path, default=MODEL_PATH)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--confirm-training",
        action="store_true",
        help="Required safety flag acknowledging that full CPU training can take a long time.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm_training:
        raise SystemExit(
            "Training was not started. Review the dataset summary and runtime expectations, then "
            "repeat with --confirm-training."
        )
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs and --batch-size must be positive integers.")

    tf = configure_reproducibility()
    dataset_root = resolve_dataset_root(args.dataset_dir)
    manifest = load_manifest(args.manifest, dataset_root)
    print(split_summary(manifest).to_string(index=False))

    train_dataset = make_tf_dataset(
        manifest, dataset_root, "train", args.batch_size, shuffle=True
    )
    validation_dataset = make_tf_dataset(
        manifest, dataset_root, "validation", args.batch_size, shuffle=False
    )

    model = build_compiled_cnn()
    model.summary()

    model_output = args.model_output.resolve()
    checkpoint = args.checkpoint.resolve()
    model_output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
        ),
        tf.keras.callbacks.CSVLogger(RUN_DIR / "training_history.csv"),
    ]

    report = environment_report(tf)
    (RUN_DIR / "environment.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        shuffle=False,
    )
    model.save(model_output)
    (RUN_DIR / "history.json").write_text(
        json.dumps(history.history, indent=2), encoding="utf-8"
    )
    print(f"Final restored-best model saved to: {model_output}")
    print(f"Best checkpoint saved to: {checkpoint}")


if __name__ == "__main__":
    main()

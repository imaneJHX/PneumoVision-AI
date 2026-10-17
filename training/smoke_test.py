"""One-epoch, small-subset training smoke test kept separate from full training."""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import math
import os
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from training.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    EPOCHS,
    MANIFEST_PATH,
    RANDOM_SEED,
    RUN_DIR,
)
from training.dataset import load_manifest, make_tf_dataset, resolve_dataset_root
from training.model import build_compiled_cnn
from training.runtime import configure_reproducibility, environment_report


SMOKE_EPOCHS = 1
SMOKE_TRAIN_IMAGES_PER_CLASS = 128
SMOKE_VALIDATION_IMAGES_PER_CLASS = 64
SMOKE_OUTPUT_DIR = RUN_DIR / "smoke"


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _windows_memory_sample() -> tuple[int, int, int] | None:
    """Return process RSS, total physical memory, and available memory in bytes."""
    if os.name != "nt":
        return None
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MemoryStatus)]
    kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
    process = kernel32.GetCurrentProcess()
    success = psapi.GetProcessMemoryInfo(
        process, ctypes.byref(counters), counters.cb
    )
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    memory_success = kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    if not success or not memory_success:
        return None
    return counters.WorkingSetSize, status.ullTotalPhys, status.ullAvailPhys


class ResourceSampler:
    def __init__(self, interval_seconds: float = 0.25) -> None:
        self.interval_seconds = interval_seconds
        self.peak_rss_bytes = 0
        self.total_memory_bytes = 0
        self.minimum_available_bytes: int | None = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        sample = _windows_memory_sample()
        if sample is None:
            return
        rss, total, available = sample
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
        self.total_memory_bytes = total
        if self.minimum_available_bytes is None:
            self.minimum_available_bytes = available
        else:
            self.minimum_available_bytes = min(self.minimum_available_bytes, available)

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2)
        self._sample()


def _select_patient_group_subset(
    manifest: pd.DataFrame, split: str, target_per_class: int
) -> pd.DataFrame:
    """Select complete patient groups deterministically until each class target is met."""
    selected_parts: list[pd.DataFrame] = []
    split_offset = 0 if split == "train" else 10_000
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_rows = manifest[
            (manifest["split"] == split) & (manifest["class_name"] == class_name)
        ].copy()
        patient_ids = sorted(class_rows["patient_id"].unique().tolist())
        rng = np.random.default_rng(RANDOM_SEED + split_offset + class_index)
        rng.shuffle(patient_ids)
        selected_patients: list[str] = []
        selected_count = 0
        for patient_id in patient_ids:
            selected_patients.append(patient_id)
            selected_count += int((class_rows["patient_id"] == patient_id).sum())
            if selected_count >= target_per_class:
                break
        selected_parts.append(class_rows[class_rows["patient_id"].isin(selected_patients)])
    return pd.concat(selected_parts, ignore_index=True).sort_values("relative_path")


def _timing_callback(tf):
    class TimingCallback(tf.keras.callbacks.Callback):
        def __init__(self) -> None:
            super().__init__()
            self.epoch_durations: list[float] = []
            self.train_batch_durations: list[float] = []
            self.validation_batch_durations: list[float] = []
            self._epoch_start = 0.0
            self._batch_start = 0.0

        def on_epoch_begin(self, epoch, logs=None):
            self._epoch_start = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            self.epoch_durations.append(time.perf_counter() - self._epoch_start)

        def on_train_batch_begin(self, batch, logs=None):
            self._batch_start = time.perf_counter()

        def on_train_batch_end(self, batch, logs=None):
            self.train_batch_durations.append(time.perf_counter() - self._batch_start)

        def on_test_batch_begin(self, batch, logs=None):
            self._batch_start = time.perf_counter()

        def on_test_batch_end(self, batch, logs=None):
            self.validation_batch_durations.append(time.perf_counter() - self._batch_start)

    return TimingCallback()


def _median_after_warmup(values: list[float]) -> float:
    candidates = values[1:] if len(values) > 1 else values
    return float(statistics.median(candidates))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a separate one-epoch, balanced-subset training smoke test."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=SMOKE_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--train-per-class", type=int, default=SMOKE_TRAIN_IMAGES_PER_CLASS
    )
    parser.add_argument(
        "--validation-per-class",
        type=int,
        default=SMOKE_VALIDATION_IMAGES_PER_CLASS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.batch_size, args.train_per_class, args.validation_per_class) < 1:
        raise SystemExit("Smoke-test sizes and batch size must be positive integers.")

    tf = configure_reproducibility()
    dataset_root = resolve_dataset_root(args.dataset_dir)
    manifest = load_manifest(args.manifest, dataset_root)
    train_subset = _select_patient_group_subset(
        manifest, "train", args.train_per_class
    )
    validation_subset = _select_patient_group_subset(
        manifest, "validation", args.validation_per_class
    )
    smoke_manifest = pd.concat([train_subset, validation_subset], ignore_index=True)

    train_dataset = make_tf_dataset(
        smoke_manifest, dataset_root, "train", args.batch_size, shuffle=True
    )
    validation_dataset = make_tf_dataset(
        smoke_manifest, dataset_root, "validation", args.batch_size, shuffle=False
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.keras"
    final_model_path = output_dir / "final.keras"
    history_csv_path = output_dir / "history.csv"
    summary_path = output_dir / "summary.json"

    sampler = ResourceSampler()
    sampler.start()
    model = build_compiled_cnn()
    timer = _timing_callback(tf)
    callbacks = [
        timer,
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
        ),
        tf.keras.callbacks.CSVLogger(history_csv_path),
    ]

    fit_wall_start = time.perf_counter()
    fit_cpu_start = time.process_time()
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=SMOKE_EPOCHS,
        callbacks=callbacks,
        shuffle=False,
        verbose=2,
    )
    fit_cpu_seconds = time.process_time() - fit_cpu_start
    fit_wall_seconds = time.perf_counter() - fit_wall_start
    model.save(final_model_path)
    sampler.stop()

    # Confirm that both serialized models load and execute after the write completes.
    validation_images, _ = next(iter(validation_dataset))
    checkpoint_model = tf.keras.models.load_model(checkpoint_path, compile=False)
    final_model = tf.keras.models.load_model(final_model_path, compile=False)
    checkpoint_prediction = checkpoint_model(validation_images[:1], training=False).numpy()
    final_prediction = final_model(validation_images[:1], training=False).numpy()
    if not np.isfinite(checkpoint_prediction).all() or not np.isfinite(final_prediction).all():
        raise RuntimeError("A saved smoke-test model did not reload with finite output.")
    if not history_csv_path.is_file() or len(history_csv_path.read_text(encoding="utf-8").splitlines()) < 2:
        raise RuntimeError("CSV training history was not written correctly.")

    full_train_images = int((manifest["split"] == "train").sum())
    full_validation_images = int((manifest["split"] == "validation").sum())
    full_train_steps = math.ceil(full_train_images / args.batch_size)
    full_validation_steps = math.ceil(full_validation_images / args.batch_size)
    median_train_step = _median_after_warmup(timer.train_batch_durations)
    median_validation_step = _median_after_warmup(timer.validation_batch_durations)
    measured_batch_time = sum(timer.train_batch_durations) + sum(
        timer.validation_batch_durations
    )
    fixed_epoch_overhead = max(0.0, timer.epoch_durations[0] - measured_batch_time)
    tracing_overhead = max(0.0, timer.train_batch_durations[0] - median_train_step)
    estimated_steady_epoch = (
        fixed_epoch_overhead
        + median_train_step * full_train_steps
        + median_validation_step * full_validation_steps
    )
    estimated_first_epoch = estimated_steady_epoch + tracing_overhead
    estimated_ten_epochs = estimated_first_epoch + estimated_steady_epoch * (EPOCHS - 1)
    logical_cpus = os.cpu_count() or 1
    aggregate_cpu_percent = 100.0 * fit_cpu_seconds / fit_wall_seconds
    normalized_cpu_percent = aggregate_cpu_percent / logical_cpus

    summary = {
        "purpose": "pipeline smoke test only; not a trained production model",
        "environment": environment_report(tf),
        "smoke_configuration": {
            "epochs": SMOKE_EPOCHS,
            "batch_size": args.batch_size,
            "train_images": int(len(train_subset)),
            "validation_images": int(len(validation_subset)),
            "train_images_by_class": train_subset["class_name"].value_counts().to_dict(),
            "validation_images_by_class": validation_subset["class_name"].value_counts().to_dict(),
        },
        "full_configuration_unchanged": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "train_images": full_train_images,
            "validation_images": full_validation_images,
            "train_steps_per_epoch": full_train_steps,
            "validation_steps_per_epoch": full_validation_steps,
        },
        "timing_seconds": {
            "fit_wall": fit_wall_seconds,
            "epoch": timer.epoch_durations[0],
            "train_batch_durations": timer.train_batch_durations,
            "validation_batch_durations": timer.validation_batch_durations,
            "median_train_batch_after_warmup": median_train_step,
            "median_validation_batch_after_warmup": median_validation_step,
        },
        "resources": {
            "logical_cpu_count": logical_cpus,
            "process_cpu_seconds": fit_cpu_seconds,
            "aggregate_process_cpu_percent": aggregate_cpu_percent,
            "normalized_total_cpu_percent": normalized_cpu_percent,
            "peak_process_rss_mb": sampler.peak_rss_bytes / (1024**2),
            "system_total_memory_gb": sampler.total_memory_bytes / (1024**3),
            "minimum_system_available_memory_gb": (
                sampler.minimum_available_bytes / (1024**3)
                if sampler.minimum_available_bytes is not None
                else None
            ),
        },
        "rough_full_training_estimate_seconds": {
            "first_epoch": estimated_first_epoch,
            "steady_epoch": estimated_steady_epoch,
            "maximum_10_epochs": estimated_ten_epochs,
            "low_10_epochs": estimated_ten_epochs * 0.65,
            "high_10_epochs": estimated_ten_epochs * 1.50,
            "note": "Extrapolated from smoke-test batch timings; CPU contention and caching can change it substantially.",
        },
        "history": {
            key: [float(value) for value in values] for key, values in history.history.items()
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "final_model": str(final_model_path),
            "final_model_bytes": final_model_path.stat().st_size,
            "history_csv": str(history_csv_path),
            "history_csv_bytes": history_csv_path.stat().st_size,
            "checkpoint_reload_verified": True,
            "final_model_reload_verified": True,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Smoke-test output: {output_dir}")
    print(f"Epoch duration: {timer.epoch_durations[0]:.2f} seconds")
    print(f"Peak process memory: {summary['resources']['peak_process_rss_mb']:.1f} MB")
    print(f"Estimated steady full epoch: {estimated_steady_epoch / 60:.1f} minutes")
    print(f"Estimated maximum 10 epochs: {estimated_ten_epochs / 60:.1f} minutes")
    print("Checkpoint, final smoke model, and CSV history reload/write checks passed.")
    print("The production model path was not written and full training was not started.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import unittest

import pandas as pd

from training.experiments import (
    build_experiment_model,
    calculate_balanced_class_weights,
)
from training.run_experiment import select_validation_threshold
from utils.evaluation import summarize_at_threshold


class ExperimentConfigurationTests(unittest.TestCase):
    def test_experiment_names_are_safe_default_run_names(self) -> None:
        import re

        from training.experiments import EXPERIMENT_NAMES

        self.assertTrue(all(re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in EXPERIMENT_NAMES))

    def test_class_weights_use_training_rows_only(self) -> None:
        manifest = pd.DataFrame(
            {
                "split": ["train", "train", "train", "train", "validation"],
                "label": [0, 1, 1, 1, 0],
            }
        )
        weights = calculate_balanced_class_weights(manifest)
        self.assertAlmostEqual(weights[0], 2.0)
        self.assertAlmostEqual(weights[1], 2.0 / 3.0)

    def test_lightweight_transfer_model_preserves_external_input_contract(self) -> None:
        model = build_experiment_model("mobilenet_v2", transfer_weights=None)
        self.assertEqual(model.input_shape, (None, 224, 224, 1))
        self.assertEqual(model.output_shape, (None, 1))
        self.assertLess(model.count_params(), 1_000_000)

    def test_validation_threshold_selection_does_not_require_test_rows(self) -> None:
        data = pd.DataFrame(
            {"y_true": [0, 0, 1, 1], "y_score": [0.1, 0.4, 0.6, 0.9]}
        )
        threshold, metrics = select_validation_threshold(data)
        self.assertGreater(threshold, 0.4)
        self.assertLessEqual(threshold, 0.6)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)

    def test_threshold_metrics_include_error_counts(self) -> None:
        data = pd.DataFrame(
            {"y_true": [0, 0, 1, 1], "y_score": [0.2, 0.8, 0.4, 0.9]}
        )
        metrics = summarize_at_threshold(data, 0.5)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["cases"], 4)


if __name__ == "__main__":
    unittest.main()

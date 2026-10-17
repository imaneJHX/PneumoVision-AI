from __future__ import annotations

import unittest

from PIL import Image

from utils.diagnostics import assess_image_quality, build_screening_summary
from utils.reporting import build_educational_report


class ScreeningSummaryTests(unittest.TestCase):
    def test_score_at_threshold_is_positive(self) -> None:
        summary = build_screening_summary(0.5, 0.5)
        self.assertTrue(summary.is_positive)
        self.assertEqual(summary.label, "PNEUMONIA")
        self.assertIn("close to its cutoff", summary.boundary_note)

    def test_negative_summary_warns_about_false_negatives(self) -> None:
        summary = build_screening_summary(0.2, 0.5)
        self.assertFalse(summary.is_positive)
        self.assertEqual(summary.label, "NORMAL")
        self.assertIn("false-negative", summary.explanation)

    def test_invalid_score_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_screening_summary(1.1, 0.5)


class ImageQualityTests(unittest.TestCase):
    def test_small_flat_image_reports_limitations(self) -> None:
        image = Image.new("L", (100, 120), color=127)
        quality = assess_image_quality(image)
        self.assertEqual((quality.width, quality.height), (100, 120))
        self.assertTrue(any("upscaled" in note for note in quality.notes))
        self.assertTrue(any("low global" in note for note in quality.notes))

    def test_report_escapes_filename_and_does_not_embed_image(self) -> None:
        image = Image.new("L", (224, 224), color=127)
        quality = assess_image_quality(image)
        summary = build_screening_summary(0.25, 0.5)
        report = build_educational_report(
            filename="patient<script>.png",
            model_name="pneumonia_cnn.keras",
            score=0.25,
            threshold=0.5,
            summary=summary,
            quality=quality,
            gradcam_layer="conv2d_2",
        )
        self.assertIn("patient&lt;script&gt;.png", report)
        self.assertNotIn("patient<script>.png", report)
        self.assertNotIn("data:image", report)


if __name__ == "__main__":
    unittest.main()

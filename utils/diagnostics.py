"""Research-only summaries for a single X-ray model prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ScreeningSummary:
    """Plain-language interpretation of a binary model decision."""

    label: str
    headline: str
    explanation: str
    boundary_note: str
    is_positive: bool


@dataclass(frozen=True)
class ImageQualitySummary:
    """Simple pixel-level checks; these are not radiographic quality assurance."""

    width: int
    height: int
    mean_intensity: float
    contrast: float
    dynamic_range: float
    notes: tuple[str, ...]


def build_screening_summary(score: float, threshold: float) -> ScreeningSummary:
    """Describe the thresholded CNN output without treating it as a diagnosis."""
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("The pneumonia score must be finite and inside [0, 1].")
    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("The classification threshold must be inside (0, 1).")

    is_positive = score >= threshold
    distance = abs(score - threshold)
    if distance <= 0.10:
        boundary_note = (
            f"The output is {distance:.3f} from the decision threshold, so the model decision "
            "is close to its cutoff."
        )
    else:
        boundary_note = (
            f"The output is {distance:.3f} from the decision threshold. This distance is not a "
            "calibrated confidence measure."
        )

    if is_positive:
        return ScreeningSummary(
            label="PNEUMONIA",
            headline="Pneumonia-associated pattern flagged",
            explanation=(
                "The CNN output met or exceeded its PNEUMONIA decision threshold. "
                "A qualified clinician must review the original radiograph and clinical context."
            ),
            boundary_note=boundary_note,
            is_positive=True,
        )

    return ScreeningSummary(
        label="NORMAL",
        headline="No pneumonia-associated pattern flagged",
        explanation=(
            "The CNN output stayed below its PNEUMONIA decision threshold. This result cannot "
            "rule out pneumonia, another abnormality, or a false-negative model error."
        ),
        boundary_note=boundary_note,
        is_positive=False,
    )


def assess_image_quality(image: Image.Image, target_size: tuple[int, int] = (224, 224)) -> ImageQualitySummary:
    """Return transparent pixel statistics and conservative technical warnings."""
    grayscale = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    if grayscale.size == 0:
        raise ValueError("The image contains no pixels.")

    mean_intensity = float(np.mean(grayscale))
    contrast = float(np.std(grayscale))
    low, high = np.percentile(grayscale, [1, 99])
    dynamic_range = float(high - low)
    notes: list[str] = []

    target_width, target_height = target_size
    if image.width < target_width or image.height < target_height:
        notes.append(
            "The source is smaller than the model input in at least one dimension and will be upscaled."
        )
    if contrast < 0.08:
        notes.append("The image has low global pixel contrast, which may reduce model reliability.")
    if dynamic_range < 0.35:
        notes.append("The central 98% of pixel values occupy a narrow intensity range.")
    if not notes:
        notes.append("No obvious limitation was found by the basic resolution and pixel-distribution checks.")

    return ImageQualitySummary(
        width=image.width,
        height=image.height,
        mean_intensity=mean_intensity,
        contrast=contrast,
        dynamic_range=dynamic_range,
        notes=tuple(notes),
    )

# Saved-model audit — 2026-10-17

This audit reproduces the current `models/pneumonia_cnn.keras` behavior without changing the model, dataset, manifest, or production threshold.

## Artifact contract

- SHA-256: `271A48C2FB7112FE9EE5FD50C08C518BB6287D5460E1927CB45B3111D35E6CDF`
- Input: `224 × 224 × 1`, grayscale, TensorFlow resize, float32 scaling to `[0,1]`
- Output: one sigmoid unit
- Mapping: `NORMAL = 0`, `PNEUMONIA = 1`
- Production threshold: `0.50`
- Parameters: `11,168,513`

## Reproduced official-test results

The evaluation was rerun from the saved model against all 624 untouched official test images. It reproduced the existing prediction CSV and metrics:

| Metric | Result |
|---|---:|
| Accuracy | 0.7420 |
| Pneumonia precision | 0.7093 |
| Pneumonia recall | 0.9949 |
| Pneumonia F1 | 0.8282 |
| Exact ROC-AUC | 0.9271 |
| Precision-recall AUC | 0.9366 |
| True negatives | 75 |
| False positives | 159 |
| False negatives | 2 |
| True positives | 388 |

## False-positive investigation

The training split contains 3,087 PNEUMONIA images and 1,071 NORMAL images. Unweighted binary cross-entropy therefore sees approximately 2.88 pneumonia images for each normal image. This is a plausible contributor to a pneumonia-favoring decision boundary, but does not by itself prove causation.

The larger signal is validation-to-test shift:

| Split at threshold 0.50 | Accuracy | Precision | Recall | False positives | False negatives | Exact ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Patient-group validation (1,058) | 0.9688 | 0.9760 | 0.9822 | 19 | 14 | 0.9925 |
| Official test (624) | 0.7420 | 0.7093 | 0.9949 | 159 | 2 | 0.9271 |

The median pneumonia score for validation NORMAL images is `0.044`, while the median for official-test NORMAL images is `0.857`. This substantial score-distribution shift explains why a threshold that behaves well on validation generates many test false positives. Simple resized-image mean intensity and global contrast have only weak correlation with NORMAL test scores (`0.118` and `-0.134`), so basic brightness or contrast alone does not explain the failure.

Scores are highly saturated: the median official-test PNEUMONIA score is approximately `0.99997`. This limits the usefulness of treating score magnitude as confidence and contributes to approximation error in Keras's default 200-threshold AUC. The exact scikit-learn ROC-AUC is the reporting value; experiment monitoring now uses 1,000 thresholds and final summaries use exact saved-score calculations.

No mismatch was found in label mapping, dashboard/training preprocessing, model input/output shape, or the fixed production threshold. Candidate experiments should therefore focus on class weighting, a smaller pretrained feature extractor, validation-only threshold selection, and improved generalization—not on changing the test threshold after seeing test errors.

## Reproduction command

```powershell
python -m training.evaluate `
  --dataset-dir "C:\PneumoVision-Datasets\chest_xray" `
  --output-dir ".cache\model_audit_reproduction"
```

The production artifact was not replaced.

# Model-improvement experiment results — 2026-10-17

Three candidates were trained locally with TensorFlow 2.21 on CPU. Every run used the same patient-group-aware manifest (`SHA-256 0746a830e675b95679dc2a138d83adfcf9383501d90c060885c93eff21ec4860`). The official test split was not used for training, early stopping, model selection, or threshold selection.

## Validation comparison

All fixed-threshold rows below use `0.50` and all 1,058 validation cases.

| Candidate | Epochs | Accuracy | Precision | Recall | F1 | ROC-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline CNN | 8 | 0.9660 | 0.9759 | 0.9784 | 0.9772 | 0.9928 | 19 | 17 |
| Class-weighted CNN | 10 | 0.9660 | 0.9821 | 0.9721 | 0.9770 | 0.9930 | 14 | 22 |
| MobileNetV2 0.35 | 10 | 0.9357 | 0.9478 | 0.9670 | 0.9573 | 0.9758 | 42 | 26 |

The class-weighted CNN was selected before test evaluation. It had the highest validation ROC-AUC and reduced validation false positives by 26% versus the baseline at the common threshold, with a small recall trade-off. MobileNetV2 used only 411,489 parameters and trained much faster on CPU, but its validation discrimination and false-positive count were worse.

The predefined maximum-balanced-accuracy rule selected a threshold of `0.63` for the class-weighted CNN. At that threshold its validation confusion matrix was TN 263, FP 7, FN 26, TP 762; accuracy was 0.9688 and F1 was 0.9788. This threshold was locked before the official test data was evaluated.

## One-time official-test result for the selected candidate

The class-weighted candidate was evaluated once on all 624 official test images at its locked `0.63` threshold:

| Metric | Result |
|---|---:|
| Accuracy | 0.7901 |
| Pneumonia precision | 0.7524 |
| Pneumonia recall | 0.9897 |
| Pneumonia F1 | 0.8549 |
| ROC-AUC | 0.9329 |
| Precision-recall AUC | 0.9459 |
| True negatives | 107 |
| False positives | 127 |
| False negatives | 4 |
| True positives | 386 |

Compared with the current production model at threshold `0.50`, this candidate reduced false positives from 159 to 127 and raised accuracy from 0.7420 to 0.7901. It also increased false negatives from 2 to 4. The large validation-to-test performance gap remains, so class weighting improved the result but did not solve dataset shift or establish clinical generalization.

## Decision

The candidate has **not** replaced `models/pneumonia_cnn.keras`. Its model, histories, validation predictions, and test artifacts remain isolated under `training/runs/experiments/class_weighted/`. Promotion requires an explicit decision after reviewing the error trade-off and research limitations.

Raw scores are model outputs, not calibrated clinical probabilities. These results are for research and education and do not establish diagnostic safety.

# Local machine-learning workflow

This folder migrates the executable machine-learning workflow from
`AI_Medical_Image_Analysis_Pneumonia_Detection.ipynb` into reproducible Python modules.
The notebook remains unchanged as the original experimental record.

## Notebook contract

- Dataset: Kaggle `paultimothymooney/chest-xray-pneumonia`, version 2.
- Input: `224 × 224 × 1`, decoded in grayscale.
- Preprocessing: TensorFlow resize followed by float conversion and division by `255.0`.
- Mapping: `NORMAL = 0`, `PNEUMONIA = 1`.
- Architecture: Conv2D 32 → pool → Conv2D 64 → pool → Conv2D 128 → pool →
  flatten → Dense 128 → Dropout 0.5 → Dense 1 sigmoid.
- Optimizer: Adam, learning rate `0.0001`.
- Loss: binary cross-entropy.
- Notebook metrics: binary accuracy, precision, recall, and AUC. Local experiment runs also log F1.
- Batch size: 32.
- Maximum epochs: 10.
- Early stopping: validation loss, patience 3, restore best weights.
- Classification threshold: 0.50.

Notebook cell audit:

| Concern | Notebook source | Reproduced locally |
|---|---|---|
| Dataset | Cell 0, Kaggle handle `paultimothymooney/chest-xray-pneumonia` | `training.prepare_dataset` |
| Source counts | Cells 7, 8, and 22: 5,216 source-train images, 4,172/1,044 image-level train/validation rows, 624 test images | Manifest summary, with patient grouping applied before the 80%/20% split |
| Preprocessing | Cell 9: decode with one channel, resize to 224 × 224, cast float32, divide by 255 | `training.dataset.make_tf_dataset` and `utils.preprocessing.preprocess_image` |
| Architecture | Cell 14: three convolution/pooling stages, 128-unit dense head, dropout 0.5, sigmoid output | `training.model.build_cnn` |
| Compilation | Cell 16: Adam 0.0001, binary cross-entropy, accuracy/precision/recall/AUC | `training.model.compile_model` |
| Training | Cell 19: batch 32, maximum 10 epochs, early stopping patience 3 with restored best weights | `training.train`, plus compatible best-model checkpointing |
| Test evaluation | Cells 22–24 and 37–40: untouched test folder, threshold 0.5, report, confusion matrix, ROC-AUC, error analysis | `training.evaluate` |
| Saved model | Cell 41: `pneumonia_cnn.keras` | `models/pneumonia_cnn.keras` |

The notebook used `train_test_split(..., test_size=0.20, random_state=42,
stratify=labels)` at image level on the source `train` folder. Public filenames expose
patient-like identifiers, so the local pipeline makes the necessary safety correction of
splitting patient groups independently within each class. It still uses 80%/20%, seed 42,
never moves the source images, leaves the official test folder untouched, and records the
small official `val` folder as `source_validation_unused` because the notebook did not train
or evaluate on it.

## Commands

Run commands from the repository root with `.venv` activated.

Prepare an already extracted dataset:

```powershell
python -m training.prepare_dataset --dataset-dir "C:\path\to\chest_xray"
```

The source is a 1.24 GB Kaggle dataset. Its download is intentionally gated:

```powershell
python -m training.prepare_dataset --download
```

The command above only reports the requirement. After confirming at least 3 GB of free
space for download/cache overhead, explicitly allow it with:

```powershell
python -m training.prepare_dataset --download --confirm-download
```

Verify TensorFlow, architecture, a forward pass, and Grad-CAM without training:

```powershell
python -m training.verify_setup
```

Run the separate one-epoch balanced-subset smoke test. Its artifacts are written only
under `training/runs/smoke/` and never replace the production model:

```powershell
python -m training.smoke_test --dataset-dir "C:\path\to\chest_xray"
```

Start the long training run only after reviewing the generated split summary:

```powershell
python -m training.train --dataset-dir "C:\path\to\chest_xray" --confirm-training
```

Evaluate the resulting model on the untouched official test set:

```powershell
python -m training.evaluate --dataset-dir "C:\path\to\chest_xray"
```

Verify dashboard preprocessing and Grad-CAM with the actual trained model:

```powershell
python -m training.verify_model --dataset-dir "C:\path\to\chest_xray"
```

## Controlled model-improvement experiments

The experiment runner uses only the manifest's patient-group-aware `train` and
`validation` rows. It never trains, selects a model, or selects a threshold from the
official test split. Every command requires `--confirm-training`, writes to an isolated
folder under `training/runs/experiments/`, and cannot replace
`models/pneumonia_cnn.keras`.

The three prepared comparisons are:

1. `baseline`: the existing custom CNN without class weights.
2. `class_weighted`: the same CNN with balanced weights calculated from training rows only.
3. `mobilenet_v2`: a frozen MobileNetV2 (`alpha=0.35`) feature extractor with the external
   `224 × 224 × 1` input contract preserved. Grayscale-to-RGB conversion and MobileNet
   scaling happen inside the candidate model.

Review configuration without training:

```powershell
python -m training.run_experiment --help
```

Run candidates only after explicitly approving the CPU time:

```powershell
python -m training.run_experiment --dataset-dir "C:\PneumoVision-Datasets\chest_xray" --experiment baseline --run-name baseline_repro --confirm-training
python -m training.run_experiment --dataset-dir "C:\PneumoVision-Datasets\chest_xray" --experiment class_weighted --run-name class_weighted_repro --confirm-training
python -m training.run_experiment --dataset-dir "C:\PneumoVision-Datasets\chest_xray" --experiment mobilenet_v2 --run-name mobilenet_v2_repro --confirm-training
```

The transfer-learning run uses public ImageNet initialization and may download the
MobileNetV2 weights on first use. Use `--transfer-weights none` only to test offline
plumbing; that option is not a transfer-learning result.

Each run records loss, accuracy, precision, recall, F1, and approximate ROC-AUC by epoch,
plus exact validation metrics from saved scores. Early stopping, best-model checkpointing,
and reduce-on-plateau learning-rate scheduling are enabled. A validation-only threshold is
also selected using the predefined maximum-balanced-accuracy objective. That threshold is
not applied to the production dashboard automatically.

After real runs exist, the Streamlit Performance page discovers their `history.csv` and
`summary.json` files and displays training curves and a validation comparison. It never
creates placeholder candidate metrics.

The current saved-model investigation is documented in
[`docs/model_audit.md`](../docs/model_audit.md).

Generated model and evaluation artifacts are local and excluded from Git.

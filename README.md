# PneumoVision-AI
### Deep Learning-Based Pneumonia Detection from Chest X-Ray Images

## Project Overview

PneumoVision-AI is a deep learning project focused on automated pneumonia classification using chest X-ray images.

The project implements a **Convolutional Neural Network (CNN)** to classify medical images into two categories: NORMAL and PNEUMONIA.

It combines medical image preprocessing, deep learning, model performance evaluation, and Explainable Artificial Intelligence (XAI) techniques to investigate model predictions and limitations.

## Project Objectives

- Develop a CNN architecture for binary chest X-ray classification.
- Perform image preprocessing and normalization.
- Train and evaluate the deep learning model.
- Analyze classification performance using multiple evaluation metrics.
- Apply Grad-CAM to visualize image regions influencing predictions.
- Investigate model limitations through error analysis.

## Dataset

The project uses a publicly available chest X-ray dataset containing images organized into two classes:

- **NORMAL:** Chest X-ray images labeled as normal.
- **PNEUMONIA:** Chest X-ray images labeled as pneumonia.

Images are preprocessed and resized before being provided to the CNN model.

## Methodology

### 1. Data Preprocessing

- Dataset loading and exploration.
- Image resizing and normalization.
- Preparation of training, validation, and testing datasets.

### 2. CNN Architecture

A Convolutional Neural Network is developed using TensorFlow and Keras.

The network learns image features through convolutional operations and uses these features to perform binary classification.

### 3. Model Training

The CNN is trained on labeled chest X-ray images to distinguish between normal and pneumonia cases.

### 4. Model Evaluation

The trained model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Receiver Operating Characteristic – Area Under the Curve (ROC-AUC)

### 5. Explainable AI

Gradient-weighted Class Activation Mapping (Grad-CAM) is applied to visualize regions that influence the model's predictions.

These visualizations support model interpretation but do not represent clinically validated disease localization.

## Current Saved-Model Results

The current `models/pneumonia_cnn.keras` artifact was reproduced on the untouched
624-image official test split at the fixed `0.50` threshold.

| Evaluation Metric | Result |
|---|---|
| Accuracy | 0.742 |
| ROC-AUC | 0.927 |
| Pneumonia Precision | 0.709 |
| Pneumonia Recall | 0.995 |
| Pneumonia F1-score | 0.828 |
| Normal Precision | 0.974 |
| Normal Recall | 0.321 |
| False positives | 159 |
| False negatives | 2 |

### Results Analysis

The CNN demonstrates high sensitivity to pneumonia cases, achieving recall above 99%.

However, the relatively low recall for normal images indicates a substantial number of false-positive pneumonia predictions.

These findings highlight the importance of evaluating medical image classification models using multiple performance metrics rather than relying exclusively on accuracy.

The reproducible false-positive investigation, artifact hash, validation comparison, and
score-distribution findings are recorded in [`docs/model_audit.md`](docs/model_audit.md).

## Technologies and Tools

**Programming Language**
- Python

**Deep Learning Framework**
- TensorFlow
- Keras

**Data Processing**
- NumPy

**Machine Learning Evaluation**
- Scikit-learn

**Data Visualization**
- Matplotlib

**Explainability**
- Grad-CAM

**Development Environment**
- Visual Studio Code on Windows
- Local Python virtual environment
- Jupyter notebook retained as the original experimental record

**Version Control**
- Git
- GitHub

## Repository Structure

PneumoVision-AI/
├── AI_Medical_Image_Analysis_Pneumonia_Detection.ipynb
├── README.md
└── requirements.txt

## Installation and Execution

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

The original experiment was developed using Google Colab and TensorFlow/Keras. That reference remains in the unchanged notebook; all active development, training, evaluation, and dashboard execution now run locally in VS Code.

---

## Interactive Streamlit Application

The repository also includes a polished Streamlit research dashboard that turns the notebook workflow into a modular local application. The original notebook remains unchanged and continues to serve as the experimental record.

### Application Features

- Premium dark medical-imaging interface with responsive navigation and reusable styling.
- Session-persistent light/dark appearance switch available from every page.
- Responsive medical-research homepage hero using a project-local image asset and accessible alternative text.
- Session-only JPG, JPEG, and PNG chest X-ray upload.
- Notebook-compatible grayscale conversion, 224 × 224 resizing, and `[0, 1]` pixel scaling.
- Cached loading of a local `.keras` model with input/output compatibility checks.
- Binary `NORMAL` / `PNEUMONIA` classification only when a real compatible model is present.
- Raw sigmoid score visualization that is explicitly separated from clinical probability or severity.
- Interactive raw-score gauge and a session-generated educational HTML report that does not embed the X-ray.
- Grad-CAM generation from the final convolutional layer and a side-by-side image overlay.
- Interactive confusion matrix, ROC curve, precision–recall curve, threshold analysis, and error counts.
- Real training-history charts and validation-only baseline/candidate comparison when experiment artifacts exist.
- Strict evaluation-data validation and an informative empty state when real outputs are unavailable.
- Research and medical-safety disclaimers throughout the interface.

### Application Technologies

- Python and Streamlit
- TensorFlow / Keras
- NumPy and Pandas
- Pillow
- Plotly
- Scikit-learn
- Custom CSS

### Expanded Project Structure

```text
PneumoVision-AI/
├── app.py
├── pages/
│   ├── 1_Xray_Analysis.py
│   ├── 2_AI_Performance.py
│   └── 3_About.py
├── training/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── metrics.py
│   ├── experiments.py
│   ├── run_experiment.py
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── verify_setup.py
│   └── verify_model.py
├── utils/
│   ├── __init__.py
│   ├── evaluation.py
│   ├── gradcam.py
│   ├── model_loader.py
│   ├── diagnostics.py
│   ├── reporting.py
│   ├── preprocessing.py
│   └── ui.py
├── models/
│   └── README.md
├── assets/
│   └── style.css
├── .streamlit/
│   └── config.toml
├── .vscode/
│   ├── settings.json
│   └── tasks.json
├── tests/
│   ├── test_training_pipeline.py
│   ├── test_diagnostics.py
│   └── test_experiments.py
├── docs/
│   └── model_audit.md
├── .gitignore
├── requirements.txt
├── README.md
└── AI_Medical_Image_Analysis_Pneumonia_Detection.ipynb
```

## Local Installation

The configured Windows environment uses Python 3.12, which is supported by TensorFlow 2.21. From a VS Code terminal opened at the project root:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks environment activation, run the interpreter and Streamlit executable directly from `.venv\Scripts` or use a VS Code terminal configured for the environment.

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

Streamlit prints the local URL in the terminal, normally `http://localhost:8501`. The dashboard launches successfully without a model; inference stays disabled and shows local-training instructions.

## Local Dataset Preparation and Training

The original dataset source is Kaggle dataset `paultimothymooney/chest-xray-pneumonia`, version 2. It contains 5,863 images and requires approximately 1.24 GB for the dataset. Keep at least 3 GB free for download and cache overhead.

The project does not download the dataset automatically. If it is already extracted locally, create the deterministic split manifest with:

```powershell
python -m training.prepare_dataset --dataset-dir "C:\path\to\chest_xray"
```

To use the original Kaggle source through KaggleHub, first review the storage requirement. The first command below will not download anything; the second includes explicit confirmation:

```powershell
python -m training.prepare_dataset --download
python -m training.prepare_dataset --download --confirm-download
```

The notebook split training images at image level. Because the public filenames contain patient-like identifiers, the local pipeline uses an 80%/20% patient-group split within each class with the notebook seed of 42. The official test folder remains untouched. The small source `val` folder is recorded but unused, matching the notebook workflow.

Verify TensorFlow, the 11,168,513-parameter architecture, a synthetic forward pass, and Grad-CAM without training:

```powershell
python -m training.verify_setup
```

Run a separate one-epoch, balanced-subset smoke test without touching the production model path:

```powershell
python -m training.smoke_test --dataset-dir "C:\path\to\chest_xray"
```

Start full local training only after reviewing the generated manifest summary:

```powershell
python -m training.train --dataset-dir "C:\path\to\chest_xray" --confirm-training
```

This reproduces the notebook configuration: 224 × 224 grayscale input, `[0,1]` scaling, batch size 32, Adam at `0.0001`, binary cross-entropy, a maximum of 10 epochs, and early stopping on validation loss with patience 3. A best-model checkpoint is also retained. The restored-best final model is saved to:

```text
models/pneumonia_cnn.keras
```

Current TensorFlow releases train on CPU under native Windows. For NVIDIA GPU acceleration with modern TensorFlow, use WSL2; the native VS Code workflow remains fully functional on CPU.

## Provide Real Evaluation Data

The performance laboratory does not reconstruct charts from summary metrics. Generate all evaluation artifacts from the untouched official test folder with:

```powershell
python -m training.evaluate --dataset-dir "C:\path\to\chest_xray"
```

This creates the classification report, confusion matrix, ROC curve, precision–recall curve, evaluation summary, and the dashboard CSV at:

```text
models/evaluation_data.csv
```

Required class mapping:

- `0` = `NORMAL`
- `1` = `PNEUMONIA`

Threshold selection should be performed on validation data using a predefined objective. The final test dataset should remain reserved for unbiased evaluation.

Finally, verify preprocessing and Grad-CAM with the actual trained model and one untouched test image:

```powershell
python -m training.verify_model --dataset-dir "C:\path\to\chest_xray"
```

Full implementation notes are available in `training/README.md`. Dataset files, `.venv`, model artifacts, and evaluation outputs are excluded from Git.

## Model-Improvement Experiments

Three candidates are configured and were run locally without replacing the production model:

- Existing baseline CNN
- The same CNN with training-only balanced class weights
- Frozen MobileNetV2 (`alpha=0.35`) transfer learning for a lighter CPU-suitable candidate

All candidates use the existing patient-group-aware train/validation manifest, early
stopping, reduce-on-plateau learning-rate scheduling, isolated checkpoints, and expanded
metric tracking. They never replace the production model. The completed validation and
selected-candidate test comparison is documented in
[`docs/experiment_results.md`](docs/experiment_results.md). Review the interface with:

```powershell
python -m training.run_experiment --help
```

To reproduce a run, use one candidate at a time with a new `--run-name` so existing
artifacts are not overwritten:

```powershell
python -m training.run_experiment --dataset-dir "C:\PneumoVision-Datasets\chest_xray" --experiment baseline --run-name baseline_repro --confirm-training
python -m training.run_experiment --dataset-dir "C:\PneumoVision-Datasets\chest_xray" --experiment class_weighted --run-name class_weighted_repro --confirm-training
python -m training.run_experiment --dataset-dir "C:\PneumoVision-Datasets\chest_xray" --experiment mobilenet_v2 --run-name mobilenet_v2_repro --confirm-training
```

The first MobileNetV2 transfer run may download its public ImageNet weights. Candidate
metrics appear in the dashboard only after real summaries are generated.

## Safety, Privacy, and Limitations

- This project is an educational and research prototype, not a medical device or diagnostic tool.
- A raw sigmoid model score is not a calibrated clinical probability and does not indicate disease severity.
- Grad-CAM shows regions that influenced a model output. It is not clinically validated pneumonia localization or lesion segmentation.
- Dataset performance may not generalize across patient populations, institutions, equipment, acquisition settings, or disease prevalence.
- Binary classification does not represent the full range of normal variants, artifacts, comorbidities, or alternative diagnoses.
- Images should be de-identified before upload. The application processes uploads in the active Streamlit session and does not implement persistent image storage.
- A qualified healthcare professional must interpret all chest X-rays and clinical findings.

## Future Improvements

- External and subgroup validation across institutions and acquisition devices.
- Prospective threshold selection and probability calibration using validation data.
- Dataset-shift and image-quality monitoring.
- DICOM ingestion with careful metadata handling and orientation verification.
- Model cards, experiment versioning, and reproducible artifact tracking.
- Clinician-centered usability studies and accessibility review.
- Authenticated deployment with an approved privacy and security architecture if the project scope expands beyond local research.

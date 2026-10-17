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

## Experimental Results

The CNN was evaluated on a test dataset containing 624 chest X-ray images.

| Evaluation Metric | Result |
|---|---|
| Accuracy | 77% |
| ROC-AUC | 0.925 |
| Pneumonia Precision | 0.73 |
| Pneumonia Recall | 0.99 |
| Pneumonia F1-score | 0.84 |
| Normal Precision | 0.98 |
| Normal Recall | 0.40 |
| Normal F1-score | 0.57 |

### Results Analysis

The CNN demonstrates high sensitivity to pneumonia cases, achieving a recall of 99%.

However, the relatively low recall for normal images indicates a substantial number of false-positive pneumonia predictions.

These findings highlight the importance of evaluating medical image classification models using multiple performance metrics rather than relying exclusively on accuracy.

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
- Google Colab
- Jupyter Notebook

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

The project was developed using Google Colab and TensorFlow/Keras. A compatible Python environment is required to execute the notebook.

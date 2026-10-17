"""PneumoVision AI dashboard landing page."""

import base64
from pathlib import Path

import streamlit as st

from training.config import CLASSIFICATION_THRESHOLD
from utils.evaluation import (
    DEFAULT_EVALUATION_PATH,
    EvaluationDataError,
    load_evaluation_data,
    summarize_at_threshold,
)
from utils.ui import configure_page, disclaimer, metric_card, section_header


configure_page("Dashboard")

hero_image_path = Path(__file__).resolve().parent / "assets" / "medical-home-hero.png"
hero_image_data = base64.b64encode(hero_image_path.read_bytes()).decode("ascii")
hero_image_uri = f"data:image/png;base64,{hero_image_data}"

st.markdown(
    f"""
    <div class="hero-shell hero-shell-image">
        <div class="hero-backdrop" role="img"
             aria-label="Doctor using a tablet with abstract digital healthcare graphics"
             style="background-image:url('{hero_image_uri}')"></div>
        <div class="hero-grid">
            <div>
                <div class="eyebrow">Explainable medical imaging research</div>
                <div class="hero-title">Pneumo<br><span>Vision AI</span></div>
                <p class="hero-copy">A focused research console for chest X-ray classification,
                model evaluation, and visual explanation—built around a custom TensorFlow CNN.</p>
            </div>
            <div class="hero-module">
                <div class="hero-module-label">MODEL PIPELINE</div>
                <span class="hero-module-value">CNN / Binary classification</span>
                <span class="hero-module-detail">224 × 224 grayscale input</span>
                <div class="signal-line"></div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

evaluation_summary = None
if DEFAULT_EVALUATION_PATH.is_file():
    try:
        evaluation_summary = summarize_at_threshold(
            load_evaluation_data(DEFAULT_EVALUATION_PATH), CLASSIFICATION_THRESHOLD
        )
    except EvaluationDataError:
        evaluation_summary = None

if evaluation_summary is not None:
    section_header(
        "Verified benchmark",
        "Current saved-model test performance",
        f"Calculated from {evaluation_summary['cases']:,} untouched test images at the fixed "
        f"{CLASSIFICATION_THRESHOLD:.2f} threshold.",
    )
    headline_metrics = [
        ("Accuracy", f"{evaluation_summary['accuracy']:.1%}", "Official test split", "cyan"),
        ("ROC–AUC", f"{evaluation_summary['roc_auc']:.3f}", "Threshold-independent ranking", "blue"),
        ("Pneumonia recall", f"{evaluation_summary['recall']:.1%}", f"{evaluation_summary['fn']} false negatives", "cyan"),
        ("Pneumonia precision", f"{evaluation_summary['precision']:.1%}", f"{evaluation_summary['fp']} false positives", "violet"),
    ]
else:
    section_header(
        "Notebook benchmark",
        "Historical experimental results",
        "Reference values only; generate models/evaluation_data.csv to display current saved-model results.",
    )
    headline_metrics = [
        ("Accuracy", "77%", "Historical notebook metric", "cyan"),
        ("ROC–AUC", "0.925", "Historical notebook metric", "blue"),
        ("Pneumonia recall", "99%", "Historical notebook metric", "cyan"),
        ("Pneumonia precision", "73%", "Historical notebook metric", "violet"),
    ]

metric_cols = st.columns(4)
for column, (label, value, detail, accent) in zip(metric_cols, headline_metrics):
    with column:
        metric_card(label, value, detail, accent)

section_header(
    "Workspace",
    "From image input to evidence review",
    "The application separates single-image inference from dataset-level evaluation so results stay traceable and correctly scoped.",
)

features = st.columns(3)
feature_content = [
    ("01 / INFERENCE", "X-ray analysis", "Upload one JPG or PNG image, apply notebook-matched preprocessing, and run the local model when available."),
    ("02 / EXPLAINABILITY", "Grad-CAM review", "Inspect the regions that most influenced the selected model output without treating activation as lesion localization."),
    ("03 / VALIDATION", "Performance laboratory", "Load labeled evaluation scores, test thresholds, and examine ROC, precision–recall, and error counts."),
]
for col, (number, title, body) in zip(features, feature_content):
    with col:
        st.markdown(
            f'<div class="feature-card"><div class="feature-number">{number}</div><h3>{title}</h3><p>{body}</p></div>',
            unsafe_allow_html=True,
        )

nav_left, nav_right, _ = st.columns([1, 1, 1.5])
with nav_left:
    st.page_link("pages/1_Xray_Analysis.py", label="Open X-ray analysis", width="stretch")
with nav_right:
    st.page_link("pages/2_AI_Performance.py", label="Open performance lab", width="stretch")

section_header("Architecture", "Purpose-built CNN", "The live app follows the architecture documented in the project notebook.")
arch_cols = st.columns([1.25, 1])
with arch_cols[0]:
    st.markdown(
        """
        <div class="content-card">
            <div class="feature-number">FEATURE EXTRACTION</div>
            <h3>Three convolution stages</h3>
            <p>Conv2D blocks with 32, 64, and 128 filters learn progressively richer spatial features.
            Each block is followed by max pooling. The final feature map supplies the Grad-CAM signal.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with arch_cols[1]:
    st.markdown(
        """
        <div class="content-card">
            <div class="feature-number">CLASSIFICATION HEAD</div>
            <h3>Binary sigmoid output</h3>
            <p>Flattened features feed a 128-unit dense layer and dropout before a single sigmoid output.
            The notebook class order is NORMAL = 0 and PNEUMONIA = 1.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

disclaimer()

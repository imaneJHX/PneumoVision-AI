"""Interactive evaluation laboratory backed only by supplied real data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from utils.evaluation import DEFAULT_EVALUATION_PATH, EvaluationDataError, load_evaluation_data
from utils.ui import PLOTLY_CONFIG, configure_page, disclaimer, page_header, plotly_layout


configure_page("AI Performance")
page_header(
    "Validation workspace",
    "AI performance laboratory",
    "Interrogate classification behavior across thresholds using ground-truth labels and real CNN outputs.",
)

st.info(
    "Thresholds must be selected using validation data and a predefined objective. Do not optimize a threshold on the final test set."
)

with st.expander("Metric guide", expanded=False):
    guide_cols = st.columns(4)
    guides = [
        ("Accuracy", "The fraction of all cases classified correctly; sensitive to class imbalance."),
        ("Precision", "Among pneumonia classifications, the fraction labeled pneumonia in the dataset."),
        ("Recall", "Among dataset pneumonia cases, the fraction classified as pneumonia."),
        ("F1 score", "The harmonic mean of pneumonia precision and recall at the selected threshold."),
    ]
    for col, (name, explanation) in zip(guide_cols, guides):
        with col:
            st.markdown(f'<div class="feature-card"><h3>{name}</h3><p>{explanation}</p></div>', unsafe_allow_html=True)

with st.expander("Evaluation data source", expanded=False):
    st.caption(
        "The saved test evaluation loads automatically. Upload a CSV only to inspect another "
        "labeled evaluation set in this browser session."
    )
    uploaded_data = st.file_uploader(
        "Optional evaluation CSV",
        type=["csv"],
        help="CSV columns: y_true (0=NORMAL, 1=PNEUMONIA) and y_score (raw pneumonia sigmoid output).",
    )

data = None
data_source = None
try:
    if uploaded_data is not None:
        data = load_evaluation_data(uploaded_data)
        data_source = "Session upload"
    elif DEFAULT_EVALUATION_PATH.is_file():
        data = load_evaluation_data(DEFAULT_EVALUATION_PATH)
        data_source = "models/evaluation_data.csv"
except EvaluationDataError as exc:
    st.error(str(exc))

if data is None:
    st.markdown(
        """
        <div class="content-card" style="padding:2rem;margin-top:1rem">
            <div class="feature-number">NO EVALUATION DATA</div>
            <h3>Charts are intentionally empty</h3>
            <p>No confusion counts, ROC points, or threshold metrics will be synthesized. Export the true test or validation
            labels and corresponding model scores to a CSV file, then upload it here or save it as
            <code>models/evaluation_data.csv</code>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code("y_true,y_score", language="csv")
    st.caption("Required mapping: 0 = NORMAL, 1 = PNEUMONIA. Each row must pair one real label with its model output.")
    st.markdown("#### Historical notebook context")
    st.write(
        "The project documents historical results of 77% accuracy, 0.925 ROC–AUC, 99% pneumonia recall, "
        "and 73% pneumonia precision. These summaries are not enough to reconstruct valid interactive curves."
    )
    disclaimer()
    st.stop()

st.success(f"Evaluation data loaded from {data_source}: {len(data):,} labeled cases")
threshold = st.slider(
    "Classification threshold",
    min_value=0.01,
    max_value=0.99,
    value=0.50,
    step=0.01,
    help="Scores at or above this value are classified as PNEUMONIA.",
)

y_true = data["y_true"].to_numpy()
y_score = data["y_score"].to_numpy()
y_pred = (y_score >= threshold).astype(int)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

metric_values = [
    ("Accuracy", accuracy_score(y_true, y_pred)),
    ("Pneumonia precision", precision_score(y_true, y_pred, zero_division=0)),
    ("Pneumonia recall", recall_score(y_true, y_pred, zero_division=0)),
    ("F1 score", f1_score(y_true, y_pred, zero_division=0)),
]
metric_cols = st.columns(4)
for col, (label, value) in zip(metric_cols, metric_values):
    with col:
        st.metric(label, f"{value:.3f}")

matrix_col, errors_col = st.columns([1.15, 0.85], gap="large")
with matrix_col:
    matrix_figure = go.Figure(
        go.Heatmap(
            z=cm,
            x=["Predicted NORMAL", "Predicted PNEUMONIA"],
            y=["Actual NORMAL", "Actual PNEUMONIA"],
            colorscale=[[0, "#08192a"], [0.5, "#176aa5"], [1, "#2de2e6"]],
            showscale=False,
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 22, "color": "white"},
            hovertemplate="%{y}<br>%{x}<br>Cases: %{z}<extra></extra>",
        )
    )
    matrix_figure.update_layout(**plotly_layout(f"Confusion matrix at threshold {threshold:.2f}", 410))
    matrix_figure.update_yaxes(autorange="reversed")
    st.plotly_chart(matrix_figure, width="stretch", config=PLOTLY_CONFIG)

with errors_col:
    st.markdown("#### Error analysis")
    st.metric("False positives", f"{fp:,}", help="NORMAL labels classified as PNEUMONIA")
    st.metric("False negatives", f"{fn:,}", help="PNEUMONIA labels classified as NORMAL")
    st.markdown(
        f"At **{threshold:.2f}**, the model produces **{fp:,} false-positive** and **{fn:,} false-negative** classifications. "
        "Changing the threshold redistributes these errors; the appropriate trade-off depends on a prospectively defined use case."
    )

fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_score)

roc_col, pr_col = st.columns(2, gap="large")
with roc_col:
    roc_figure = go.Figure()
    roc_figure.add_trace(
        go.Scatter(x=fpr, y=tpr, mode="lines", name=f"CNN · AUC {roc_auc:.3f}", line={"color": "#2de2e6", "width": 3})
    )
    roc_figure.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance reference", line={"color": "#587185", "dash": "dash"})
    )
    roc_figure.update_layout(**plotly_layout("Receiver operating characteristic", 410))
    roc_figure.update_xaxes(title="False-positive rate", range=[0, 1], gridcolor="rgba(100,150,180,.12)")
    roc_figure.update_yaxes(title="True-positive rate", range=[0, 1], gridcolor="rgba(100,150,180,.12)")
    st.plotly_chart(roc_figure, width="stretch", config=PLOTLY_CONFIG)

with pr_col:
    pr_figure = go.Figure(
        go.Scatter(
            x=pr_recall,
            y=pr_precision,
            mode="lines",
            name="CNN",
            line={"color": "#8b7dff", "width": 3},
            hovertemplate="Recall %{x:.3f}<br>Precision %{y:.3f}<extra></extra>",
        )
    )
    pr_figure.update_layout(**plotly_layout("Precision–recall trade-off", 410))
    pr_figure.update_xaxes(title="Recall", range=[0, 1], gridcolor="rgba(100,150,180,.12)")
    pr_figure.update_yaxes(title="Precision", range=[0, 1], gridcolor="rgba(100,150,180,.12)")
    st.plotly_chart(pr_figure, width="stretch", config=PLOTLY_CONFIG)

thresholds = np.linspace(0.01, 0.99, 99)
precision_values = []
recall_values = []
f1_values = []
for candidate in thresholds:
    candidate_pred = (y_score >= candidate).astype(int)
    precision_values.append(precision_score(y_true, candidate_pred, zero_division=0))
    recall_values.append(recall_score(y_true, candidate_pred, zero_division=0))
    f1_values.append(f1_score(y_true, candidate_pred, zero_division=0))

tradeoff_figure = go.Figure()
tradeoff_figure.add_trace(go.Scatter(x=thresholds, y=precision_values, name="Precision", line={"color": "#3288ff", "width": 2.5}))
tradeoff_figure.add_trace(go.Scatter(x=thresholds, y=recall_values, name="Recall", line={"color": "#2de2e6", "width": 2.5}))
tradeoff_figure.add_trace(go.Scatter(x=thresholds, y=f1_values, name="F1 score", line={"color": "#8b7dff", "width": 2.5}))
tradeoff_figure.add_vline(x=threshold, line_dash="dot", line_color="#f5c164", annotation_text="Selected")
tradeoff_figure.update_layout(**plotly_layout("Threshold sensitivity", 430))
tradeoff_figure.update_xaxes(title="Classification threshold", gridcolor="rgba(100,150,180,.12)")
tradeoff_figure.update_yaxes(title="Metric value", range=[0, 1], gridcolor="rgba(100,150,180,.12)")
st.plotly_chart(tradeoff_figure, width="stretch", config=PLOTLY_CONFIG)

st.caption("All displayed counts, curves, and metrics are calculated from the loaded evaluation rows in the current session.")

st.markdown("---")
st.markdown("### Training history")
st.caption(
    "Training curves use saved CSV logs only. They are not reconstructed from final test metrics."
)
project_root = Path(__file__).resolve().parents[1]
history_options: dict[str, Path] = {}
baseline_history = project_root / "training" / "runs" / "training_history.csv"
if baseline_history.is_file():
    history_options["Current production baseline"] = baseline_history
experiment_root = project_root / "training" / "runs" / "experiments"
if experiment_root.is_dir():
    for history_path in sorted(experiment_root.glob("*/history.csv")):
        history_options[f"Experiment · {history_path.parent.name}"] = history_path

if not history_options:
    st.info("No real training-history CSV is available yet.")
else:
    selected_history = st.selectbox("Training run", list(history_options))
    history_data = pd.read_csv(history_options[selected_history])
    epoch_values = history_data.get("epoch", pd.Series(range(len(history_data)))) + 1
    history_figure = go.Figure()
    curve_colors = {
        "loss": "#ff9f68",
        "val_loss": "#ffc56c",
        "accuracy": "#3288ff",
        "val_accuracy": "#2de2e6",
    }
    for metric_name, color in curve_colors.items():
        if metric_name in history_data.columns:
            history_figure.add_trace(
                go.Scatter(
                    x=epoch_values,
                    y=history_data[metric_name],
                    mode="lines+markers",
                    name=metric_name.replace("_", " ").title(),
                    line={"color": color, "width": 2.5},
                )
            )
    history_figure.update_layout(**plotly_layout("Loss and accuracy by epoch", 420))
    history_figure.update_xaxes(title="Epoch", dtick=1, gridcolor="rgba(100,150,180,.12)")
    history_figure.update_yaxes(title="Metric value", gridcolor="rgba(100,150,180,.12)")
    st.plotly_chart(history_figure, width="stretch", config=PLOTLY_CONFIG)

    secondary_metrics = [
        name
        for name in ("precision", "val_precision", "recall", "val_recall", "f1", "val_f1", "auc", "val_auc")
        if name in history_data.columns
    ]
    if secondary_metrics:
        detail_figure = go.Figure()
        palette = ["#3288ff", "#70b8ff", "#2de2e6", "#8af5e4", "#8b7dff", "#b6adff", "#ff9f68", "#ffc56c"]
        for metric_name, color in zip(secondary_metrics, palette):
            detail_figure.add_trace(
                go.Scatter(
                    x=epoch_values,
                    y=history_data[metric_name],
                    mode="lines",
                    name=metric_name.replace("_", " ").title(),
                    line={"color": color, "width": 2.2},
                )
            )
        detail_figure.update_layout(**plotly_layout("Precision, recall, F1, and ROC-AUC history", 420))
        detail_figure.update_xaxes(title="Epoch", dtick=1, gridcolor="rgba(100,150,180,.12)")
        detail_figure.update_yaxes(title="Metric value", range=[0, 1], gridcolor="rgba(100,150,180,.12)")
        st.plotly_chart(detail_figure, width="stretch", config=PLOTLY_CONFIG)
    if "f1" not in history_data.columns and "val_f1" not in history_data.columns:
        st.caption("The legacy baseline run did not log epoch-level F1; prepared experiment runs will record it.")

st.markdown("### Baseline versus candidate models")
comparison_rows: list[dict[str, object]] = []
if experiment_root.is_dir():
    for summary_path in sorted(experiment_root.glob("*/summary.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics = payload["validation_metrics_at_0_5"]
            comparison_rows.append(
                {
                    "Model": payload["experiment"],
                    "Accuracy": metrics["accuracy"],
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1": metrics["f1"],
                    "ROC-AUC": metrics["roc_auc"],
                    "False positives": metrics["fp"],
                    "False negatives": metrics["fn"],
                }
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue

if comparison_rows:
    comparison_data = pd.DataFrame(comparison_rows)
    st.dataframe(
        comparison_data.style.format(
            {"Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}", "ROC-AUC": "{:.3f}"}
        ),
        width="stretch",
        hide_index=True,
    )
    comparison_figure = go.Figure()
    for metric_name, color in (("Precision", "#3288ff"), ("Recall", "#2de2e6"), ("F1", "#8b7dff"), ("ROC-AUC", "#ff9f68")):
        comparison_figure.add_trace(
            go.Bar(x=comparison_data["Model"], y=comparison_data[metric_name], name=metric_name, marker_color=color)
        )
    comparison_figure.update_layout(**plotly_layout("Validation-set model comparison", 430), barmode="group")
    comparison_figure.update_yaxes(range=[0, 1], title="Metric value", gridcolor="rgba(100,150,180,.12)")
    st.plotly_chart(comparison_figure, width="stretch", config=PLOTLY_CONFIG)
    st.caption("Candidate models are compared on the patient-group-aware validation split, never the official test set.")
else:
    st.info(
        "No candidate results are shown because the prepared experiments have not been trained. "
        "This prevents placeholder or fabricated comparisons."
    )
disclaimer()

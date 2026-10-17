"""Single-image CNN inference and Grad-CAM review."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from training.config import CLASSIFICATION_THRESHOLD
from utils.diagnostics import assess_image_quality, build_screening_summary
from utils.gradcam import GradCAMError, make_gradcam_heatmap, overlay_heatmap
from utils.model_loader import (
    ModelCompatibilityError,
    find_model_path,
    load_model,
    predict_pneumonia_score,
)
from utils.preprocessing import ImageValidationError, load_uploaded_image, preprocess_image
from utils.reporting import build_educational_report
from utils.ui import PLOTLY_CONFIG, configure_page, disclaimer, page_header, plotly_layout


configure_page("X-ray Analysis")
page_header(
    "Inference workspace",
    "Chest X-ray analysis",
    "Run the notebook-trained binary classifier on one local image and inspect the model's visual attention with Grad-CAM.",
)

model_path = find_model_path()
if model_path is None:
    st.warning(
        "No trained model is available. The interface is ready, but inference remains disabled until "
        "`models/pneumonia_cnn.keras` is added. No placeholder prediction will be generated."
    )
    with st.expander("Train the model locally in VS Code"):
        st.code(
            """# Prepare the local dataset manifest first, then run:
python -m training.train --dataset-dir "C:\\path\\to\\chest_xray" --confirm-training

# Evaluate the untouched test split:
python -m training.evaluate --dataset-dir "C:\\path\\to\\chest_xray""" ,
            language="powershell",
        )
        st.markdown("Local training writes `models/pneumonia_cnn.keras`; refresh this page after it finishes.")
else:
    st.success(f"Local model detected: `{model_path.relative_to(model_path.parents[1])}`")

st.markdown("### Image input")
st.caption("Supported formats: JPG, JPEG, PNG · Maximum size: 15 MB · The image stays in this Streamlit session.")
uploaded_file = st.file_uploader(
    "Upload a chest X-ray",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown(
        """
        <div class="content-card" style="text-align:center;padding:3rem 1.5rem;margin-top:1rem">
            <div class="feature-number">AWAITING INPUT</div>
            <h3>Choose a de-identified chest X-ray to begin</h3>
            <p>The application does not upload or persist the image outside the active app session.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    disclaimer()
    st.stop()

try:
    image = load_uploaded_image(uploaded_file)
except ImageValidationError as exc:
    st.error(str(exc))
    disclaimer()
    st.stop()

preview_col, context_col = st.columns([1.15, 0.85], gap="large")
with preview_col:
    st.image(image.convert("L"), caption="Uploaded X-ray · grayscale preview", width="stretch")
with context_col:
    st.markdown("#### Input review")
    st.write(f"Dimensions: **{image.width} × {image.height} px**")
    st.write(f"Source format: **{image.format or uploaded_file.type.split('/')[-1].upper()}**")
    st.write("Preprocessing: **grayscale · resize · [0,1] scaling**")
    st.info(
        "Confirm that the image is a chest radiograph and contains no identifying patient information before analysis."
    )

if model_path is None:
    disclaimer()
    st.stop()

try:
    model = load_model(str(model_path))
    image_batch = preprocess_image(image, model, encoded_bytes=uploaded_file.getvalue())
except (RuntimeError, ValueError, ModelCompatibilityError) as exc:
    st.error(f"The local model is not compatible with this application. {exc}")
    disclaimer()
    st.stop()

if not st.button("Run CNN analysis", type="primary", width="stretch"):
    st.caption("Analysis runs only after you select the button. No result has been generated yet.")
    disclaimer()
    st.stop()

try:
    with st.spinner("Running the local CNN and generating its visual explanation..."):
        pneumonia_score = predict_pneumonia_score(model, image_batch)
        predicted_pneumonia = pneumonia_score >= CLASSIFICATION_THRESHOLD
        predicted_label = "PNEUMONIA" if predicted_pneumonia else "NORMAL"
        screening_summary = build_screening_summary(
            pneumonia_score, CLASSIFICATION_THRESHOLD
        )
        quality_summary = assess_image_quality(image)
        heatmap, conv_layer_name = make_gradcam_heatmap(
            model, image_batch, explain_pneumonia=predicted_pneumonia
        )
        overlay = overlay_heatmap(image, heatmap)
except (ModelCompatibilityError, GradCAMError, ValueError) as exc:
    st.error(f"Analysis could not be completed. {exc}")
    disclaimer()
    st.stop()

st.markdown("---")
st.markdown("### AI screening summary")
summary_class = "positive" if screening_summary.is_positive else "negative"
st.markdown(
    f"""
    <div class="screening-summary {summary_class}">
        <div class="screening-kicker">MODEL FINDING · RESEARCH USE ONLY</div>
        <h3>{screening_summary.headline}</h3>
        <p>{screening_summary.explanation}</p>
        <div class="screening-boundary">{screening_summary.boundary_note}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Model output")
result_col, chart_col = st.columns([0.8, 1.2], gap="large")
with result_col:
    st.metric("Thresholded class", predicted_label)
    st.metric("Pneumonia model score", f"{pneumonia_score:.4f}")
    st.caption(
        f"Classification threshold: {CLASSIFICATION_THRESHOLD:.2f} · "
        "Notebook class mapping: NORMAL = 0, PNEUMONIA = 1"
    )
    st.warning(
        "This raw sigmoid score is not a calibrated clinical probability and does not describe disease severity."
    )

with chart_col:
    score_figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pneumonia_score,
            number={"valueformat": ".3f", "font": {"color": "#edf7ff", "size": 38}},
            title={"text": "Raw sigmoid output · not a clinical probability", "font": {"size": 15}},
            gauge={
                "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": "#6f8ca2"},
                "bar": {"color": "#2de2e6", "thickness": 0.28},
                "bgcolor": "rgba(4,15,27,.75)",
                "bordercolor": "rgba(95,164,205,.24)",
                "steps": [
                    {"range": [0, CLASSIFICATION_THRESHOLD], "color": "rgba(50,136,255,.14)"},
                    {"range": [CLASSIFICATION_THRESHOLD, 1], "color": "rgba(255,130,85,.11)"},
                ],
                "threshold": {
                    "line": {"color": "#ffc56c", "width": 4},
                    "thickness": 0.8,
                    "value": CLASSIFICATION_THRESHOLD,
                },
            },
        )
    )
    score_figure.update_layout(**plotly_layout(height=300))
    st.plotly_chart(score_figure, width="stretch", config=PLOTLY_CONFIG)

st.markdown("### Technical image review")
quality_col1, quality_col2, quality_col3 = st.columns(3)
quality_col1.metric("Source resolution", f"{quality_summary.width} × {quality_summary.height}")
quality_col2.metric("Mean intensity", f"{quality_summary.mean_intensity:.3f}")
quality_col3.metric("Global contrast", f"{quality_summary.contrast:.3f}")
for note in quality_summary.notes:
    st.write(f"- {note}")
st.caption(
    "These automated pixel checks cannot verify positioning, anatomy coverage, exposure quality, "
    "artifacts, or whether the image is a valid diagnostic chest radiograph."
)

st.markdown("### Grad-CAM inspection")
original_col, gradcam_col = st.columns(2, gap="large")
with original_col:
    st.image(image.convert("L"), caption="Original grayscale image", width="stretch")
with gradcam_col:
    st.image(overlay, caption=f"Grad-CAM overlay · layer: {conv_layer_name}", width="stretch")

st.info(
    "Grad-CAM highlights image regions that influenced the selected model output. It is a model explanation, "
    "not clinically validated pneumonia localization, segmentation, or evidence of a lesion."
)

report_html = build_educational_report(
    filename=uploaded_file.name,
    model_name=str(model_path.name),
    score=pneumonia_score,
    threshold=CLASSIFICATION_THRESHOLD,
    summary=screening_summary,
    quality=quality_summary,
    gradcam_layer=conv_layer_name,
)
download_col, trace_col = st.columns([1, 1.5], gap="large")
with download_col:
    st.download_button(
        "Download educational analysis report",
        data=report_html.encode("utf-8"),
        file_name="pneumovision_educational_analysis.html",
        mime="text/html",
        width="stretch",
    )
with trace_col:
    with st.expander("Technical trace"):
        st.code(
            f"Model: {model_path}\nInput: {model.input_shape}\n"
            f"Mapping: NORMAL=0, PNEUMONIA=1\nGrad-CAM layer: {conv_layer_name}"
        )
disclaimer()

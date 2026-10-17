"""Project methodology, provenance, and responsible-use notes."""

import streamlit as st

from utils.ui import configure_page, disclaimer, page_header, section_header


configure_page("About")
page_header(
    "Project dossier",
    "About PneumoVision AI",
    "A transparent view of the research question, technical pipeline, and boundaries of this educational prototype.",
)

section_header("Mission", "Explain the model, not just the label")
st.markdown(
    """
    PneumoVision AI explores binary chest X-ray classification with a custom convolutional neural network.
    The project combines reproducible preprocessing, threshold-aware evaluation, and Grad-CAM to make model
    behavior easier to examine. It is designed as a machine-learning research portfolio project—not a clinical product.
    """
)

pipeline_cols = st.columns(4)
pipeline = [
    ("01", "Prepare", "Convert to grayscale, resize to 224 × 224, and scale pixels to [0,1]."),
    ("02", "Classify", "Apply the local TensorFlow CNN with NORMAL = 0 and PNEUMONIA = 1."),
    ("03", "Explain", "Use gradients from the selected binary output to weight the final convolution features."),
    ("04", "Evaluate", "Measure behavior from paired ground-truth labels and real model scores."),
]
for col, (number, title, body) in zip(pipeline_cols, pipeline):
    with col:
        st.markdown(
            f'<div class="feature-card"><div class="feature-number">{number}</div><h3>{title}</h3><p>{body}</p></div>',
            unsafe_allow_html=True,
        )

section_header("Technical profile", "Model and data contract")
left, right = st.columns(2, gap="large")
with left:
    st.markdown(
        """
        #### CNN architecture

        - Input: 224 × 224 × 1
        - Conv2D: 32, 64, and 128 filters
        - Max pooling after every convolution block
        - Dense layer with 128 units and dropout
        - Single sigmoid output
        """
    )
with right:
    st.markdown(
        """
        #### Research stack

        - TensorFlow and Keras
        - Streamlit
        - NumPy and Pandas
        - Scikit-learn
        - Plotly
        - Pillow
        """
    )

section_header("Responsible AI", "Known limitations")
st.markdown(
    """
    - Performance from one public dataset may not generalize to new hospitals, devices, patient populations, or acquisition protocols.
    - A raw sigmoid output is not a calibrated clinical probability and is not a severity measure.
    - Grad-CAM indicates model influence at coarse spatial resolution; it does not identify or confirm pathology.
    - Threshold choice changes false-positive and false-negative behavior and must be established on validation data.
    - Uploaded images should be de-identified and are retained only in the active application session by this implementation.
    """
)

section_header("Provenance", "Notebook-first development")
st.write(
    "The original notebook remains the experimental record for dataset preparation, training, evaluation, and early Grad-CAM work. "
    "This application reuses its preprocessing and class contract while separating presentation and reusable inference utilities."
)

disclaimer()

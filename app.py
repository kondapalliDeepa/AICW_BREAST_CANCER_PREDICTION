# app.py

import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle
import warnings

warnings.filterwarnings("ignore")

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Breast Cancer Prediction",
    layout="centered"
)

# -------------------------------
# Add background image
# -------------------------------
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://www.omegahospitals.com/blog/storage/2024/01/blog_breast_cancer_11-e1705062587725.png");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

[data-testid="stToolbar"] {
    background: rgba(0,0,0,0);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -------------------------------
# Title
# -------------------------------
st.title("Breast Cancer Prediction")
st.write("Enter the tumor feature values and predict the class.")

# -------------------------------
# Load model & scaler (with safe checks and optional upload)
# -------------------------------
def load_artifacts():
    import tempfile
    from pathlib import Path

    model_path = Path("BC_model.h5")
    scaler_path = Path("BC_scaler.pkl")

    # If both files exist locally, load them
    if model_path.exists() and scaler_path.exists():
        model = tf.keras.models.load_model(str(model_path))
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler

    st.warning("Model or scaler not found in the app folder. You can upload them below.")

    model_file = st.file_uploader("Upload BC_model.h5 (Keras HDF5)", type=["h5", "hdf5"], key="model_upload")
    scaler_file = st.file_uploader("Upload BC_scaler.pkl (pickle)", type=["pkl"], key="scaler_upload")

    if model_file is None or scaler_file is None:
        st.info("Please upload both `BC_model.h5` and `BC_scaler.pkl` to continue.")
        st.stop()

    # Save uploaded model to a temporary file and load
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as mtmp:
        mtmp.write(model_file.read())
        mtmp.flush()
        model = tf.keras.models.load_model(mtmp.name)

    # Load scaler from uploaded bytes
    scaler = pickle.load(scaler_file)

    return model, scaler


model, scaler = load_artifacts()

# -------------------------------
# Input fields
# -------------------------------
st.subheader("Input Features")

input_data = {
    'radius_mean': st.number_input('radius_mean', value=14.0),
    'texture_mean': st.number_input('texture_mean', value=20.0),
    'perimeter_mean': st.number_input('perimeter_mean', value=90.0),
    'area_mean': st.number_input('area_mean', value=600.0),
    'smoothness_mean': st.number_input('smoothness_mean', value=0.1),
    'compactness_mean': st.number_input('compactness_mean', value=0.15),
    'concavity_mean': st.number_input('concavity_mean', value=0.2),
    'concave points_mean': st.number_input('concave points_mean', value=0.1),
    'symmetry_mean': st.number_input('symmetry_mean', value=0.2),
    'fractal_dimension_mean': st.number_input('fractal_dimension_mean', value=0.06),

    'radius_se': st.number_input('radius_se', value=0.2),
    'texture_se': st.number_input('texture_se', value=1.0),
    'perimeter_se': st.number_input('perimeter_se', value=1.5),
    'area_se': st.number_input('area_se', value=20.0),
    'smoothness_se': st.number_input('smoothness_se', value=0.005),
    'compactness_se': st.number_input('compactness_se', value=0.02),
    'concavity_se': st.number_input('concavity_se', value=0.03),
    'concave points_se': st.number_input('concave points_se', value=0.01),
    'symmetry_se': st.number_input('symmetry_se', value=0.03),
    'fractal_dimension_se': st.number_input('fractal_dimension_se', value=0.004),

    'radius_worst': st.number_input('radius_worst', value=16.0),
    'texture_worst': st.number_input('texture_worst', value=25.0),
    'perimeter_worst': st.number_input('perimeter_worst', value=105.0),
    'area_worst': st.number_input('area_worst', value=800.0),
    'smoothness_worst': st.number_input('smoothness_worst', value=0.12),
    'compactness_worst': st.number_input('compactness_worst', value=0.2),
    'concavity_worst': st.number_input('concavity_worst', value=0.3),
    'concave points_worst': st.number_input('concave points_worst', value=0.15),
    'symmetry_worst': st.number_input('symmetry_worst', value=0.25),
    'fractal_dimension_worst': st.number_input('fractal_dimension_worst', value=0.08),
}

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])

    # Align input columns to scaler expected features if available
    if hasattr(scaler, "feature_names_in_"):
        expected = list(scaler.feature_names_in_)
        cols = list(input_df.columns)

        def _norm(s):
            return ''.join(str(s).lower().replace('_', ' ').replace('-', ' ').split())

        rename = {}
        missing = []
        for exp in expected:
            if exp in cols:
                continue
            alt = exp.replace(' ', '_')
            if alt in cols:
                rename[alt] = exp
                continue
            key = _norm(exp)
            found = next((c for c in cols if _norm(c) == key), None)
            if found:
                rename[found] = exp
            else:
                missing.append(exp)

        if missing:
            st.error(f"Missing features expected by scaler: {missing}")
            st.stop()

        input_df = input_df.rename(columns=rename)[expected]
    else:
        # scaler has no stored feature names; expect 30 features in correct order
        if input_df.shape[1] != 30:
            st.error("Input must contain 30 features in the training order when scaler has no feature names.")
            st.stop()

    # Ensure numeric dtype
    input_df = input_df.astype(float)

    # Transform and predict
    try:
        input_scaled = scaler.transform(input_df)
    except Exception as e:
        st.error(f"Error during scaling: {e}")
        st.stop()

    try:
        prediction = model.predict(input_scaled)[0][0]
    except Exception as e:
        st.error(f"Error during model prediction: {e}")
        st.stop()

    predicted_class = "Malignant" if prediction > 0.5 else "Benign"

    st.subheader("Result")
    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Probability:** {prediction:.4f}")

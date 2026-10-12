
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="College Cutoff Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 College Cutoff Predictor")
st.write("College admission cutoff ka prediction karein.")

MODEL_PATH = Path(__file__).parent / "Model (3).pkl"


@st.cache_resource
def load_models():
    return joblib.load(MODEL_PATH)


try:
    models = load_models()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()


if not isinstance(models, dict):
    st.error("Model file dictionary format mein nahi hai.")
    st.stop()

if "score_model" not in models or "rank_model" not in models:
    st.error("score_model ya rank_model key nahi mili.")
    st.stop()

score_model = models["score_model"]
rank_model = models["rank_model"]

features = getattr(score_model, "feature_names_in_", None)

if features is None:
    st.error("Model ke feature names nahi mile.")
    st.stop()


st.subheader("Admission Details")

with st.form("cutoff_form"):
    input_values = {}

    for feature in features:
        feature_name = str(feature)

        if "year" in feature_name.lower():
            input_values[feature_name] = st.number_input(
                feature_name.replace("_", " ").title(),
                min_value=2000,
                max_value=2100,
                value=2026,
                step=1
            )
        else:
            input_values[feature_name] = st.text_input(
                feature_name.replace("_", " ").title()
            )

    prediction_type = st.selectbox(
        "Prediction Type",
        ["score", "rank"]
    )

    submitted = st.form_submit_button("Predict Cutoff")


if submitted:
    missing = [
        name
        for name, value in input_values.items()
        if isinstance(value, str) and not value.strip()
    ]

    if missing:
        st.warning("Please fill in: " + ", ".join(missing))
    else:
        input_df = pd.DataFrame(
            [input_values],
            columns=list(features)
        )

        if prediction_type == "score":
            selected_model = score_model
        else:
            selected_model = rank_model

        try:
            prediction = selected_model.predict(input_df)[0]
            st.subheader("Predicted Cutoff")
            st.success(f"{prediction:,.2f}")

        except Exception as e:
            st.error(f"Prediction error: {e}")
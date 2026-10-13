
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="College Cutoff Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Model (3).pkl"


# ==================================================
# 2. CSS DESIGN
# ==================================================
st.markdown("""
<style>
    #MainMenu, footer, header {
        visibility: hidden;
    }

    .stApp {
        background: #ffffff;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .hero {
        background: linear-gradient(110deg, #222e68, #405be8);
        padding: 25px 22px;
        border-radius: 0 0 16px 16px;
        margin-bottom: 28px;
    }

    .hero h1 {
        color: white;
        font-size: 36px;
        font-weight: 750;
        margin: 0 0 18px 0;
    }

    .hero p {
        color: #f2f4ff;
        font-size: 14px;
        margin: 0;
    }

    .section-title {
        font-size: 24px;
        font-weight: 650;
        color: #253454;
        margin-bottom: 8px;
    }

    .section-subtitle {
        color: #9295a0;
        font-size: 12px;
        margin-bottom: 16px;
    }

    div[data-testid="stForm"] {
        border: 1px solid #d6def4;
        border-radius: 14px;
        padding: 20px;
        background: white;
    }

    div[data-testid="stTextInput"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label {
        color: #253454;
        font-size: 13px;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {
        background-color: #eff1f5;
        border: none;
        border-radius: 8px;
        min-height: 42px;
    }

    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background-color: #eff1f5;
        border: none;
        border-radius: 8px;
        min-height: 40px;
    }

    div[data-testid="stFormSubmitButton"] button {
        background: #fa5050;
        color: white;
        border: none;
        border-radius: 8px;
        height: 44px;
        font-size: 14px;
        font-weight: 600;
        width: 100%;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background: #e94343;
        color: white;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #d6def4;
        padding: 18px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. LOAD MODEL
# ==================================================
@st.cache_resource
def load_models(model_path):
    return joblib.load(model_path)


def normalize_column(column):
    return (
        str(column)
        .strip()
        .lower()
        .replace("_", "")
        .replace(" ", "")
        .replace("-", "")
    )


# ==================================================
# 4. PREPARE MODEL INPUT
# ==================================================
def prepare_input(model, values):

    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            "Model ke expected feature names nahi mile."
        )

    feature_names = list(model.feature_names_in_)

    normalized_values = {
        normalize_column(key): value
        for key, value in values.items()
    }

    row = {}

    for feature in feature_names:
        key = normalize_column(feature)

        if key not in normalized_values:
            raise ValueError(
                f"Required model feature form mein nahi mila: {feature}"
            )

        value = normalized_values[key]

        # IMPORTANT:
        # Trained model ke numeric features year aur round hain.
        # Dropdown label jaise 'Round 1' kabhi model ko nahi bhejna.
        if key in ("year", "round"):
            try:
                value = int(value)
            except (ValueError, TypeError):
                raise ValueError(
                    f"{feature} numeric hona chahiye. Received: {value!r}"
                )

        row[feature] = value

    return pd.DataFrame([row], columns=feature_names)


# ==================================================
# 5. HEADER
# ==================================================
st.markdown("""
<div class="hero">
    <h1>🎓 College Cutoff Predictor</h1>
    <p>Choose admission details and predict your expected cutoff.</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">📋 Admission Details</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Fill in the required details below.'
    '</div>',
    unsafe_allow_html=True
)


# ==================================================
# 6. ADMISSION FORM
# ==================================================
with st.form("admission_form"):

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        university_id = st.text_input(
            "University Id",
            placeholder="Enter university id"
        )

    with col2:
        year = st.number_input(
            "Year",
            min_value=2020,
            max_value=2050,
            value=2026,
            step=1
        )

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        programme_id = st.text_input(
            "Programme Id",
            placeholder="Enter programme id"
        )

    with col2:
        college = st.text_input(
            "College",
            placeholder="Enter college"
        )

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        variant = st.selectbox(
            "Variant",
            ["General", "Ladies", "Other"]
        )

    with col2:
        # User sees "Round 1", but variable stores integer 1.
        round_number = st.selectbox(
            "Round",
            options=[1, 2, 3, 4, 5],
            format_func=lambda number: f"Round {number}"
        )

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        category = st.selectbox(
            "Category",
            ["OPEN", "OBC", "SC", "ST", "EWS", "SEBC", "Other"]
        )

    with col2:
        seat_type = st.selectbox(
            "Seat Type",
            ["Home State", "Other State", "All India"]
        )

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        basis = st.selectbox(
            "Basis",
            ["score", "rank"]
        )

    submitted = st.form_submit_button(
        "🔮 Predict Cutoff",
        use_container_width=True
    )


# ==================================================
# 7. PREDICTION
# ==================================================
if submitted:

    if not university_id.strip():
        st.error("Please enter the University Id.")

    elif not programme_id.strip():
        st.error("Please enter the Programme Id.")

    elif not college.strip():
        st.error("Please enter the College name.")

    elif not MODEL_PATH.exists():
        st.error(
            f"Model file nahi mili: {MODEL_PATH}\n\n"
            "Model (3).pkl ko app.py ke same folder mein rakhein."
        )

    else:
        try:
            with st.spinner("Predicting cutoff..."):

                saved_models = load_models(str(MODEL_PATH))

                if not isinstance(saved_models, dict):
                    raise ValueError(
                        "Saved model dictionary format mein nahi hai."
                    )

                if basis == "score":
                    model = saved_models.get("score_model")
                else:
                    model = saved_models.get("rank_model")

                if model is None:
                    raise ValueError(
                        f"{basis}_model nahi mila. "
                        f"Available keys: {list(saved_models.keys())}"
                    )

                # Always send an integer for round.
                form_values = {
                    "university_id": university_id.strip(),
                    "year": int(year),
                    "programme_id": programme_id.strip(),
                    "college": college.strip(),
                    "variant": variant,
                    "round": int(round_number),
                    "category": category,
                    "seat_type": seat_type,
                    "basis": basis
                }

                input_df = prepare_input(model, form_values)

                # Debug safeguard: round must be numeric.
                if not pd.api.types.is_numeric_dtype(input_df["round"]):
                    raise ValueError(
                        "Round numeric nahi hai. "
                        "Expected value 1, 2, 3, 4 ya 5."
                    )

                prediction = model.predict(input_df)
                predicted_value = float(prediction[0])

            st.success("Prediction completed successfully!")

            st.markdown("### 🎯 Predicted Cutoff")

            st.metric(
                label=f"Predicted {basis.title()}",
                value=f"{predicted_value:,.2f}"
            )

            with st.expander("View Admission Details"):
                st.dataframe(
                    pd.DataFrame([form_values]),
                    use_container_width=True,
                    hide_index=True
                )

            st.caption(f"Model used: {MODEL_PATH.name}")

        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.warning(
                "Agar error mein 'Round 1' ab bhi dikh raha hai, "
                "toh check karein ki app.py save hua hai aur sahi "
                "project folder se run ho raha hai."
            )


# ==================================================
# 8. FOOTER
# ==================================================
st.markdown("---")
st.caption("College Cutoff Predictor | Machine Learning Application")
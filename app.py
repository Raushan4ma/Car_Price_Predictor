import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

MODEL_PATH = "linear_regression_model.pkl"
ENCODER_PATH = "label_encoders.pkl"  # optional (see README notes)

# Order MUST match the columns used while training (X_train columns)
FEATURE_ORDER = [
    "Brand",
    "Body",
    "Mileage",
    "EngineV",
    "Engine Type",
    "Registration",
    "Year",
]

# LabelEncoder sorts categories alphabetically, so these lists are in the same
# order the notebook's LabelEncoder used (index in list == encoded number).
DEFAULT_CATEGORIES = {
    "Brand": ["Audi", "BMW", "Mercedes-Benz", "Mitsubishi", "Renault", "Toyota", "Volkswagen"],
    "Body": ["crossover", "hatch", "other", "sedan", "vagon", "van"],
    "Engine Type": ["Diesel", "Gas", "Other", "Petrol"],
    "Registration": ["no", "yes"],
}


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_encoders():
    """Use saved LabelEncoders if available, otherwise fall back to defaults."""
    if os.path.exists(ENCODER_PATH):
        return joblib.load(ENCODER_PATH)
    return None


def get_options(col, encoders):
    if encoders is not None and col in encoders:
        return list(encoders[col].classes_)
    return DEFAULT_CATEGORIES[col]


def encode(col, value, encoders):
    if encoders is not None and col in encoders:
        return int(encoders[col].transform([value])[0])
    return DEFAULT_CATEGORIES[col].index(value)


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🚗 Used Car Price Predictor")
st.caption("Linear Regression model trained on the Used Car Sales dataset (predicts price in USD).")

if not os.path.exists(MODEL_PATH):
    st.error(
        f"`{MODEL_PATH}` nahi mili. Notebook se model save karke is file (app.py) ke "
        "same folder me rakho."
    )
    st.stop()

model = load_model()
encoders = load_encoders()

st.subheader("Car details")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", get_options("Brand", encoders))
    body = st.selectbox("Body type", get_options("Body", encoders))
    engine_type = st.selectbox("Engine type", get_options("Engine Type", encoders))
    registration = st.selectbox("Registered?", get_options("Registration", encoders))

with col2:
    year = st.slider("Year of manufacture", min_value=1969, max_value=2016, value=2008)
    engine_v = st.slider("Engine volume (litres)", min_value=0.6, max_value=6.5, value=2.0, step=0.1)
    mileage = st.number_input(
        "Mileage",
        min_value=0,
        max_value=1000,
        value=150,
        step=10,
        help="Same unit as in the training dataset (usually thousands of km).",
    )

if st.button("Predict Price", type="primary", use_container_width=True):
    input_df = pd.DataFrame(
        [
            {
                "Brand": encode("Brand", brand, encoders),
                "Body": encode("Body", body, encoders),
                "Mileage": mileage,
                "EngineV": engine_v,
                "Engine Type": encode("Engine Type", engine_type, encoders),
                "Registration": encode("Registration", registration, encoders),
                "Year": year,
            }
        ]
    )[FEATURE_ORDER]

    # Model predicts log(price) -> convert back to actual price
    log_price = float(model.predict(input_df)[0])
    price = float(np.exp(log_price))

    st.success("Prediction ready!")
    st.metric("Estimated Price", f"${price:,.0f}")
    with st.expander("Details"):
        st.write(f"Predicted log price: `{log_price:.4f}`")
        st.dataframe(input_df, use_container_width=True)

st.markdown("---")
st.caption(
    "Note: Ye estimate sirf training data ke basis par hai. Dataset ke range "
    "(Year 1969–2016, EngineV ≤ 6.5) se bahar ke inputs par result reliable nahi hoga."
)

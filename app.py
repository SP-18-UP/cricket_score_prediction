
import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("cricket_score_prediction.pkl")

st.set_page_config(
    page_title="Cricket Score Predictor",
    page_icon="🏏",
    layout="centered"
)

st.title("🏏 Cricket Score Predictor")
st.write("Predict the final cricket score using match details.")

st.divider()

# User inputs
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox(
        "Batting Team",
        ["India", "Australia", "England", "Pakistan",
         "South Africa", "New Zealand", "Other"]
    )

    overs = st.number_input(
        "Overs Completed",
        min_value=5.0,
        max_value=19.0,
        value=10.0,
        step=1.0
    )

    runs = st.number_input(
        "Current Runs",
        min_value=0,
        value=100
    )

with col2:
    wickets = st.number_input(
        "Wickets Lost",
        min_value=0,
        max_value=9,
        value=3
    )

    runs_last_5 = st.number_input(
        "Runs in Last 5 Overs",
        min_value=0,
        value=40
    )

    wickets_last_5 = st.number_input(
        "Wickets in Last 5 Overs",
        min_value=0,
        max_value=10,
        value=1
    )

if st.button("Predict Final Score", type="primary"):
    try:
        # Update this feature order to match your trained model
        features = np.array([[
            runs,
            wickets,
            overs,
            runs_last_5,
            wickets_last_5
        ]])

        prediction = model.predict(features)[0]

        st.success("Prediction completed!")
        st.metric(
            label="Predicted Final Score",
            value=f"{max(0, round(float(prediction)))} runs"
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.info(
            "Check the feature names, number, and order "
            "of inputs used during model training."
)
  

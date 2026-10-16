import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("cricket_score_prediction.pkl")

st.set_page_config(
    page_title="Cricket Score Prediction",
    page_icon="🏏",
    layout="wide"
)

st.title("🏏 Cricket Score Prediction")
st.write("Predict the score using match and player statistics.")

st.divider()

st.subheader("Enter Match Details")

# Input form
with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        runs_batter = st.number_input(
            "Runs scored by batter",
            min_value=0, value=0, step=1
        )

        team_balls = st.number_input(
            "Team balls faced",
            min_value=0, value=0, step=1
        )

        team_wicket = st.number_input(
            "Team wickets lost",
            min_value=0, max_value=10, value=0, step=1
        )

        batter_runs = st.number_input(
            "Batter's total runs",
            min_value=0, value=0, step=1
        )

        batter_balls = st.number_input(
            "Batter's balls faced",
            min_value=0, value=0, step=1
        )

    with col2:
        bowler_wicket = st.number_input(
            "Bowler wickets",
            min_value=0, max_value=10, value=0, step=1
        )

        over = st.number_input(
            "Current over",
            min_value=0, value=0, step=1
        )

        ball = st.number_input(
            "Ball within the over",
            min_value=1, max_value=6, value=1, step=1
        )

        ball_no = st.number_input(
            "Total ball number",
            min_value=1, value=1, step=1
        )

        runs_extras = st.number_input(
            "Extra runs",
            min_value=0, value=0, step=1
        )

    submitted = st.form_submit_button(
        "🏏 Predict Final Score",
        use_container_width=True
    )

# Prediction
if submitted:

    input_data = pd.DataFrame([{
        "runs_batter": runs_batter,
        "team_balls": team_balls,
        "team_wicket": team_wicket,
        "batter_runs": batter_runs,
        "batter_balls": batter_balls,
        "bowler_wicket": bowler_wicket,
        "over": over,
        "ball": ball,
        "ball_no": ball_no,
        "runs_extras": runs_extras
    }])

    # Preserve the exact feature order used during training
    if hasattr(model, "feature_names_in_"):
        input_data = input_data.loc[
            :, model.feature_names_in_
        ]

    prediction = float(model.predict(input_data)[0])

    st.divider()
    st.subheader("Prediction Result")

    st.metric(
        label="Predicted Final Score",
        value=f"{max(0, prediction):.0f} runs"
    )

    st.caption(
        "This is a machine-learning estimate, not a guaranteed score."
)
    

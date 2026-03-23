import streamlit as st
import numpy as np
import joblib
import pandas as pd

model = joblib.load("productivity_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("Remote Work Productivity Predictor")

age = st.slider("Age", 20, 60, 30)
years_experience = st.slider("Years of Experience", 0, 30, 5)
daily_meetings = st.slider("Daily Meetings", 0, 10, 2)
hours_worked = st.slider("Hours Worked per Day", 4, 12, 8)
internet_speed = st.slider("Internet Speed ( in Mbps)", 10, 200, 50)
home_office_score = st.slider("Home Office Score (Work Environment rating out of 10)", 1, 10, 5)
distractions = st.slider("Number of Distractions per Day", 0, 15, 3)

if st.button("Predict Productivity"):

    input_data = pd.DataFrame([[age, years_experience, daily_meetings,
                                hours_worked, internet_speed,
                                home_office_score, distractions]],
                              columns=columns)

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    st.success(f"Predicted Productivity Score: {round(prediction[0], 2)}")

st.markdown("""
**Productivity Score Interpretation:**
- Below 20: Low Productivity Score
- 20-25: Average Productivity Score
- 25-30: Good Productivity Score
- Above 30: Excellent Productivity Score
""")

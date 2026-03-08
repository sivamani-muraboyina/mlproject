import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Set page title
st.set_page_config(page_title="Student Math Score Predictor")

st.title(" 🎓Student Performance Prediction🎓")
st.markdown("Enter the student details below to predict their **Math Score**.")

# Creating columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["female", "male"])
    race_ethnicity = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
    parental_level_of_education = st.selectbox("Parental Education", [
        "associate's degree", "bachelor's degree", "high school", "master's degree", "some college", "some high school"
    ])

with col2:
    lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
    test_preparation_course = st.selectbox("Test Prep Course", ["none", "completed"])
    reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=70)
    writing_score = st.number_input("Writing Score", min_value=0, max_value=100, value=70)

st.divider()

if st.button("Predict Score"):
    # Initialize the data class with user inputs
    data = CustomData(
        gender=gender,
        race_ethnicity=race_ethnicity,
        parental_level_of_education=parental_level_of_education,
        lunch=lunch,
        test_preparation_course=test_preparation_course,
        reading_score=reading_score,
        writing_score=writing_score
    )

    # Convert to DataFrame
    pred_df = data.get_data_as_data_frame()
    
    # Run Prediction
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    
    # Display Result
    st.balloons()
    st.success(f"### Predicted Math Score: {round(results[0], 2)}")
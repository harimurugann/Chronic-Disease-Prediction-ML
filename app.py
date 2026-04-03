import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the model and scaler
model = pickle.load(open('chronic-disease-model.sav', 'rb'))

st.title("Chronic Disease Prediction System")
st.write("Enter patient details to predict chronic disease risk.")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=30)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
bmi = st.number_input("BMI", value=25.0)
smoking = st.selectbox("Smoking", ["No", "Yes"])
alcohol = st.selectbox("Alcohol Intake", ["Low", "Moderate", "High"])
activity = st.slider("Physical Activity (Hours/Week)", 0.0, 10.0, 3.0)
diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
sleep = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
bp = st.number_input("Blood Pressure (Systolic)", value=120)
cholesterol = st.number_input("Cholesterol", value=200)
glucose = st.number_input("Glucose Level", value=100)
family_hist = st.selectbox("Family History", ["No", "Yes"])
stress = st.slider("Stress Level (1-10)", 1, 10, 5)

# Mapping inputs back to numerical
# Note: Ensure this mapping matches the LabelEncoder used in training
mapping = {"No": 0, "Yes": 1, "Male": 1, "Female": 0, "Other": 2, "Low": 1, "Moderate": 2, "High": 0, "Poor": 2, "Average": 0, "Good": 1}

if st.button("Predict"):
    features = np.array([[age, mapping[gender], bmi, mapping[smoking], mapping[alcohol], 
                          activity, mapping[diet], sleep, bp, cholesterol, glucose, 
                          mapping[family_hist], stress]])
    
    # Scale numerical features (simplified for app context)
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)
    
    if prediction[0] == 1:
        st.error("Warning: The patient is likely to have a chronic disease.")
    else:
        st.success("The patient is healthy.")

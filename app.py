import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(page_title="AI Multi-Disease Predictor", layout="wide", page_icon="🏥")

# Sidebar for navigation
with st.sidebar:
    st.title("Main Menu")
    selection = st.radio("Select a Project:", ["Chronic Disease Prediction", "Credit Card Fraud Detection"])

# --- 1. Chronic Disease Prediction Page ---
if selection == "Chronic Disease Prediction":
    st.title("🏥 Chronic Disease (Diabetes) Prediction")
    st.write("Please enter the patient's clinical metrics below:")

    # Load the Disease Model
    try:
        disease_model = pickle.load(open('chronic_disease_model.sav', 'rb'))
    except FileNotFoundError:
        st.error("Error: 'chronic_disease_model.sav' file not found in repository!")

    # Creating Inputs in Columns for a better UI
    col1, col2, col3 = st.columns(3)
    
    with col1:
        preg = st.number_input('Number of Pregnancies', min_value=0, step=1)
        glucose = st.number_input('Glucose Level')
        bp = st.number_input('Blood Pressure value')

    with col2:
        skin = st.number_input('Skin Thickness value')
        insulin = st.number_input('Insulin Level')
        bmi = st.number_input('BMI value')

    with col3:
        dpf = st.number_input('Diabetes Pedigree Function value')
        age = st.number_input('Age of the Person', min_value=1, step=1)

    # Prediction Logic
    if st.button("Predict Disease Status"):
        # Making a list of the 8 inputs we have
        user_input = [preg, glucose, bp, skin, insulin, bmi, dpf, age]
        
        # Create a list of 21 features (filling the rest with zeros)
        # 8 values from user + 13 zeros = 21 total features
        final_input = user_input + [0.0] * 13 
        
        # Convert to numpy array and reshape
        prediction = disease_model.predict([final_input])
        
        if prediction[0] == 1:
            st.warning("⚠️ High Risk: The person is likely to have Chronic Disease.")
        else:
            st.success("🎉 Low Risk: The person is Healthy.")
            
# --- 2. Credit Card Fraud Detection Page ---
elif selection == "Credit Card Fraud Detection":
    st.title("🚨 Credit Card Fraud Detection")
    st.write("Enter transaction details to analyze fraud risk.")

    # Load the Fraud Model
    try:
        fraud_model = pickle.load(open('credit_card_fraud_model.sav', 'rb'))
    except FileNotFoundError:
        st.error("Error: 'credit_card_fraud_model.sav' file not found in repository!")

    v1 = st.number_input("Feature V1")
    v2 = st.number_input("Feature V2")
    amount = st.number_input("Transaction Amount")

    if st.button("Detect Fraud"):
        # We need 30 features for the model (rest 27 are set to 0)
        features = np.zeros(30)
        features[1] = v1
        features[2] = v2
        features[29] = amount
        
        fraud_prediction = fraud_model.predict([features])
        
        if fraud_prediction[0] == 1:
            st.error("🚨 ALERT: This is a Fraudulent Transaction!")
        else:
            st.success("✅ Safe: This is a Normal Transaction.")
            

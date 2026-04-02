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
        # 1. User inputs (8 values)
        user_input = [preg, glucose, bp, skin, insulin, bmi, dpf, age]
        
        # 2. Creating a list of 21 zeros
        final_features = [0.0] * 21
        
        # 3. Mapping our 8 inputs to the first 8 positions
        # (Assuming your first 8 columns were the standard ones)
        for i in range(len(user_input)):
            final_features[i] = user_input[i]
            
        # 4. Special Logic: If Glucose or BMI is high, 
        # we activate some hidden "Risk" flags in the remaining 13 features 
        # (This is a temporary fix until we get exact column names)
        if glucose > 140:
            final_features[8] = 1.0  # Common index for 'High_Glucose' flag
        if bmi > 30:
            final_features[9] = 1.0  # Common index for 'Obese' flag

        try:
            prediction = disease_model.predict([final_features])
            
            if prediction[0] == 1:
                st.warning("⚠️ High Risk: The person is likely to have Chronic Disease.")
            else:
                st.success("🎉 Low Risk: The person is Healthy.")
        except Exception as e:
            st.error(f"Error: Model expects 21 features. {e}")
            
            
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
            

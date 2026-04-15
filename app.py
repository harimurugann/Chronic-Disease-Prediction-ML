import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
from fpdf import FPDF

# ==========================================
# 1. FIXED PDF FUNCTION (Corruption Proof)
# ==========================================
def create_pdf(name, age, gender, result, prob, recs, medical_data):
    pdf = FPDF()
    pdf.add_page()
    
    # Universal English Headers to avoid encoding issues
    pdf.set_fill_color(44, 62, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 20, txt="CLINICAL ASSESSMENT REPORT", ln=True, align='C', fill=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"Patient: {name} | Age: {age} | Gender: {gender}", ln=True, border='B')
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=" CLINICAL MEASUREMENTS", ln=True, fill=True)
    pdf.set_font("Arial", size=10)
    for key, value in medical_data.items():
        pdf.cell(95, 8, f" {key}", border=1)
        pdf.cell(95, 8, f" {value}", border=1, ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"Final Diagnosis: {result} (Risk: {prob:.1f}%)", ln=True)
    
    # Return as raw bytes (Safest method for Streamlit)
    return pdf.output(dest='S')

# ==========================================
# 2. MAIN APP SETUP
# ==========================================
st.set_page_config(page_title="Professional AI Health", layout="wide")

try:
    pipeline = joblib.load('full_pipeline_compressed.sav')
except:
    st.error("Model file not found!")

st.title("🏥 Professional Health Prediction & Analytics")

col1, col2 = st.columns(2)
with col1:
    st.subheader("👤 Patient Profile")
    # FIX: Default name changed to 'Enter Name'
    patient_name = st.text_input("Patient Full Name", value="", placeholder="Enter Name")
    age = st.number_input("Age", 1, 120, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    bmi = st.number_input("BMI (kg/m2)", 10.0, 50.0, 25.0)
    smoking = st.selectbox("Smoking Status", ["No", "Yes"])
    activity = st.slider("Activity (Hrs/Week)", 0.0, 10.0, 3.0)

with col2:
    st.subheader("🩸 Clinical Metrics")
    bp = st.number_input("Systolic BP (mmHg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 200)
    glucose = st.number_input("Glucose Level (mg/dL)", 50, 300, 100)
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)
    diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
    family_hist = st.selectbox("Family History", ["No", "Yes"])

# Prepare Clinical Summary
clinical_summary = {
    "Systolic Blood Pressure": f"{bp} mmHg",
    "Blood Glucose Level": f"{glucose} mg/dL",
    "Total Cholesterol": f"{cholesterol} mg/dL",
    "Body Mass Index": f"{bmi:.1f}",
    "Physical Activity": f"{activity} Hrs/Wk"
}

st.markdown("---")

# ==========================================
# 3. ANALYSIS & DOWNLOAD
# ==========================================
if st.button("🚀 Generate Full Health Analysis"):
    # Features for model
    features = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel']
    input_df = pd.DataFrame([[age, gender, bmi, smoking, "Low", activity, diet, 7.0, bp, cholesterol, glucose, family_hist, stress]], columns=features)
    
    prob = pipeline.predict_proba(input_df)[0][1] * 100
    res_text = "Chronic Disease Likely" if prob > 50 else "No Chronic Disease"

    if prob > 50:
        st.error(f"### Status: {res_text} ({prob:.1f}%)")
    else:
        st.success(f"### Status: {res_text} ({prob:.1f}%)")

    # Metrics Display
    m1, m2, m3 = st.columns(3)
    m1.metric("BP", f"{bp} mmHg", delta=f"{bp-120}", delta_color="inverse")
    m2.metric("Glucose", f"{glucose} mg/dL", delta=f"{glucose-100}", delta_color="inverse")
    m3.metric("BMI", f"{bmi:.1f}", delta=f"{bmi-22.0:.1f}", delta_color="inverse")

    # PDF Download Section
    st.markdown("---")
    try:
        # Pass data to the fixed PDF function
        pdf_data = create_pdf(patient_name, age, gender, res_text, prob, [], clinical_summary)
        
        st.download_button(
            label="📥 Download Clinical Report (PDF)",
            data=bytes(pdf_data), # Ensure data is sent as bytes
            file_name=f"Health_Report_{patient_name if patient_name else 'Patient'}.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Error generating PDF: {e}")

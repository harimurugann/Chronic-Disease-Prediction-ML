import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
from fpdf import FPDF

# ==========================================
# 1. PROFESSIONAL PDF FUNCTION
# ==========================================
def create_pdf(name, age, gender, result, prob, score, medical_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(44, 62, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 20, txt="CLINICAL ASSESSMENT REPORT", ln=True, align='C', fill=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"Patient: {name if name else 'N/A'} | Age: {age} | Gender: {gender}", ln=True, border='B')
    
    pdf.ln(5)
    pdf.cell(0, 10, txt=f"Lifestyle Health Score: {score}/100", ln=True)
    pdf.cell(0, 10, txt=f"Final Diagnosis: {result} (Risk Score: {prob:.1f}%)", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=" CLINICAL MEASUREMENTS", ln=True, fill=True)
    pdf.set_font("Arial", size=10)
    for key, value in medical_data.items():
        pdf.cell(95, 8, f" {key}", border=1)
        pdf.cell(95, 8, f" {value}", border=1, ln=True)
    
    pdf_output = pdf.output(dest='S')
    if isinstance(pdf_output, str):
        return pdf_output.encode('latin-1')
    return pdf_output

# ==========================================
# 2. MAIN APP SETUP
# ==========================================
st.set_page_config(page_title="Pro Health AI", layout="wide")
pipeline = joblib.load('full_pipeline_compressed.sav')

st.title("🏥 Professional Health Dashboard & Predictive Analytics")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.subheader("👤 Patient Profile")
    patient_name = st.text_input("Patient Full Name", value="", placeholder="Enter Name")
    age = st.number_input("Age", 1, 120, 35)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    bmi = st.number_input("BMI (kg/m2)", 10.0, 50.0, 24.5)
    smoking = st.selectbox("Smoking Status", ["No", "Yes"])
    activity = st.slider("Activity (Hrs/Week)", 0.0, 10.0, 4.0)

with col2:
    st.subheader("🩸 Clinical Vitals")
    bp = st.number_input("Systolic BP (mmHg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 190)
    glucose = st.number_input("Glucose Level (mg/dL)", 50, 300, 95)
    stress = st.slider("Stress Level (1-10)", 1, 10, 4)
    diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
    family_hist = st.selectbox("Family History", ["No", "Yes"])

clinical_summary = {"BP": f"{bp} mmHg", "Glucose": f"{glucose} mg/dL", "Chol": f"{cholesterol} mg/dL", "BMI": f"{bmi:.1f}"}

# ==========================================
# 3. CORE ANALYSIS
# ==========================================
if st.button("🚀 Run Comprehensive AI Health Analysis"):
    features = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel']
    input_df = pd.DataFrame([[age, gender, bmi, smoking, "Low", activity, diet, 7.0, bp, cholesterol, glucose, family_hist, stress]], columns=features)
    
    prob = pipeline.predict_proba(input_df)[0][1] * 100
    res_text = "Chronic Disease Risk Detected" if prob > 50 else "Healthy / Low Risk"

    st.markdown("---")
    if prob > 50: st.error(f"### Result: {res_text} ({prob:.1f}%)")
    else: st.success(f"### Result: {res_text} ({prob:.1f}%)")

    # --- FEATURE 8: BMI CLASSIFICATION ---
    st.subheader("⚖️ BMI Health Classification")
    if bmi < 18.5: b_cat, b_col = "Underweight", "blue"
    elif 18.5 <= bmi < 24.9: b_cat, b_col = "Normal Weight", "green"
    elif 25 <= bmi < 29.9: b_cat, b_col = "Overweight", "orange"
    else: b_cat, b_col = "Obese", "red"
    
    st.markdown(f"Status: :{b_col}[**{b_cat}**]")
    

[Image of BMI categories chart]


    # --- FEATURE 10: MULTI-DISEASE RISK CHART ---
    st.markdown("---")
    st.subheader("📊 Multi-Disease Risk Categorization")
    
    # Logic based on clinical correlations
    heart_risk = (prob * 0.8) + (20 if bp > 140 else 0)
    diabetic_risk = (prob * 0.7) + (25 if glucose > 120 else 0)
    resp_risk = (prob * 0.5) + (30 if smoking == "Yes" else 0)
    
    risk_data = pd.DataFrame({
        'Disease Type': ['Cardiovascular', 'Diabetes', 'Respiratory'],
        'Risk %': [min(100, heart_risk), min(100, diabetic_risk), min(100, resp_risk)]
    })
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x='Risk %', y='Disease Type', data=risk_data, palette='magma', ax=ax)
    plt.xlim(0, 100)
    st.pyplot(fig)

    # --- FEATURE 6: LIFESTYLE SCORE ---
    st.markdown("---")
    st.subheader("🎯 Lifestyle Health Score")
    l_score = 100
    if smoking == "Yes": l_score -= 30
    if diet == "Poor": l_score -= 20
    if activity < 2: l_score -= 20
    l_score = max(0, min(100, l_score))
    
    sc1, sc2 = st.columns([3, 1])
    with sc1: st.progress(l_score / 100)
    with sc2: st.markdown(f"**{l_score}/100**")

    # SIMULATOR & PDF
    st.markdown("---")
    t_bp = st.slider("Target BP for Simulation", 80, 200, int(bp))
    sim_df = input_df.copy(); sim_df.at[0, 'BloodPressure'] = t_bp
    s_prob = pipeline.predict_proba(sim_df)[0][1] * 100
    st.metric("Simulated Risk Score", f"{s_prob:.1f}%", delta=f"{s_prob-prob:.1f}%", delta_color="inverse")

    pdf_bytes = create_pdf(patient_name, age, gender, res_text, prob, l_score, clinical_summary)
    st.download_button("📥 Download Analytical Report", pdf_bytes, f"Report_{patient_name}.pdf", "application/pdf")

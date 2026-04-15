import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
from fpdf import FPDF

# ==========================================
# 1. PDF Generation Function
# ==========================================
def create_pdf(name, age, gender, result, prob, recs, medical_data):
    pdf = FPDF()
    pdf.add_page()
    
    # Title Header
    pdf.set_fill_color(44, 62, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 20, txt="CHRONIC DISEASE ASSESSMENT REPORT", ln=True, align='C', fill=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    
    # Patient Info
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 10, txt=" PATIENT INFORMATION", ln=True, fill=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(95, 10, txt=f"Name: {name}", border='B')
    pdf.cell(95, 10, txt=f"Age: {age}", border='B', ln=True)
    pdf.cell(95, 10, txt=f"Gender: {gender}", border='B')
    pdf.cell(95, 10, txt=f"Date: {pd.Timestamp.now().strftime('%d-%m-%Y')}", border='B', ln=True)
    pdf.ln(5)

    # Clinical Table
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=" CLINICAL MEASUREMENTS", ln=True, fill=True)
    pdf.set_font("Arial", size=10)
    for key, value in medical_data.items():
        pdf.cell(95, 8, f" {key}", border=1)
        pdf.cell(95, 8, f" {value}", border=1, ln=True)
    pdf.ln(5)

    # Recommendations
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=" DOCTOR'S RECOMMENDATIONS", ln=True, fill=True)
    pdf.set_font("Arial", size=10)
    pdf.ln(2)
    for r in recs:
        pdf.multi_cell(0, 8, txt=f" {r}")
    
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 2. App Setup & Model Loading
# ==========================================
st.set_page_config(page_title="Advanced Health AI", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('full_pipeline_compressed.sav')

pipeline = load_model()

st.title("🏥 Next-Gen Chronic Disease Diagnostic System")
st.markdown("This dashboard combines **AI Explainability (SHAP)** with **Clinical Benchmarking**.")

# User Inputs
col1, col2 = st.columns(2)
with col1:
    st.subheader("General Info")
    patient_name = st.text_input("Full Name", "Patient Name")
    age = st.number_input("Age", 1, 120, 45)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    bmi = st.number_input("BMI (kg/m2)", 10.0, 50.0, 28.0)
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    activity = st.slider("Activity (Hrs/Week)", 0.0, 10.0, 2.0)

with col2:
    st.subheader("Medical Metrics")
    bp = st.number_input("Systolic BP (mmHg)", 80, 200, 145)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 250)
    glucose = st.number_input("Glucose (mg/dL)", 50, 300, 150)
    stress = st.slider("Stress Level (1-10)", 1, 10, 7)
    diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
    family_hist = st.selectbox("Family History", ["No", "Yes"])

# Data Preparation
clinical_summary = {
    "Body Mass Index": f"{bmi:.1f} kg/m2",
    "Blood Pressure": f"{bp} mmHg",
    "Cholesterol": f"{cholesterol} mg/dL",
    "Glucose": f"{glucose} mg/dL",
    "Physical Activity": f"{activity} Hrs/Wk",
    "Stress Level": f"{stress}/10"
}

features = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel']
input_df = pd.DataFrame([[age, gender, bmi, smoking, "Low", activity, diet, 7.0, bp, cholesterol, glucose, family_hist, stress]], columns=features)

# ==========================================
# 3. Main Analysis Execution
# ==========================================
if st.button("Run Advanced Diagnosis & Generate Report"):
    # Prediction
    prob = pipeline.predict_proba(input_df)[0][1] * 100
    res_text = "High Risk: Chronic Disease Likely" if prob > 50 else "Low Risk: Healthy"
    
    st.markdown("---")
    if prob > 50:
        st.error(f"### Result: {res_text} (Risk Score: {prob:.1f}%)")
    else:
        st.success(f"### Result: {res_text} (Risk Score: {prob:.1f}%)")

    # Dual Analytics Row
    st.subheader("🔬 Deep-Dive Diagnostic Analytics")
    ana_col1, ana_col2 = st.columns(2)

    # --- FEATURE A: SHAP DECISION LOGIC ---
    with ana_col1:
        st.markdown("**1. AI Decision Logic (SHAP Impact)**")
        try:
            model = pipeline.named_steps['classifier']
            preprocessor = pipeline.named_steps['preprocessor']
            X_trans = preprocessor.transform(input_df)
            f_names = preprocessor.get_feature_names_out()
            
            explainer = shap.TreeExplainer(model)
            shap_v = explainer.shap_values(X_trans)
            
            # Safe impact extraction
            if isinstance(shap_v, list):
                impact = shap_v[1].flatten() if len(shap_v) > 1 else shap_v[0].flatten()
            elif len(shap_v.shape) == 3:
                impact = shap_v[0, :, 1]
            else:
                impact = shap_v[0]

            shap_df = pd.DataFrame({'Factor': f_names, 'Impact': impact}).sort_values(by='Impact', ascending=False).head(5)
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            sns.barplot(x='Impact', y='Factor', data=shap_df, palette='Reds_r', ax=ax1)
            plt.title("Factors Driving Your Risk Up")
            st.pyplot(fig1)
        except:
            st.info("💡 SHAP Analysis: Identifying risk triggers...")

    # --- FEATURE B: CLINICAL BENCHMARKING ---
    with ana_col2:
        st.markdown("**2. Patient vs. Clinical Benchmarks**")
        bench_data = {
            'Metric': ['BP', 'Cholesterol', 'Glucose', 'BMI'],
            'Your Value': [bp, cholesterol, glucose, bmi],
            'Healthy Target': [120, 200, 100, 22]
        }
        bench_df = pd.DataFrame(bench_data).melt(id_vars='Metric', var_name='Type', value_name='Value')
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.barplot(x='Metric', y='Value', hue='Type', data=bench_df, palette='coolwarm', ax=ax2)
        plt.title("Measurement Comparison")
        st.pyplot(fig2)

    # Recommendations Logic
    recs = []
    if bp > 135: recs.append("-> BP exceeds 120mmHg benchmark. Clinical sodium restriction advised.")
    if glucose > 140: recs.append("-> Glucose is above 100mg/dL target. Immediate sugar monitoring needed.")
    if bmi > 25: recs.append("-> BMI is outside healthy range (22). Weight management plan suggested.")
    if smoking == "Yes": recs.append("-> Smoking detected as high-impact risk. Seek tobacco cessation support.")
    if stress > 7: recs.append("-> Elevated stress score. Recommend mindfulness or wellness counseling.")
    if not recs: recs.append("-> All metrics within healthy clinical benchmarks. Maintain routine checkups.")

    # PDF Report Download
    pdf_bytes = create_pdf(patient_name, age, gender, res_text, prob, recs, clinical_summary)
    st.download_button(label="📥 Download Detailed Analytical Report (PDF)",
                       data=pdf_bytes,
                       file_name=f"Advanced_Report_{patient_name}.pdf",
                       mime="application/pdf")

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
from fpdf import FPDF

# ==========================================
# 1. PDF Function (Stable with Clinical Data)
# ==========================================
def create_pdf(name, age, gender, result, prob, recs, medical_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(200, 15, txt="CLINICAL DIAGNOSTIC REPORT", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt=f"Patient Name: {name} | Age: {age} | Gender: {gender}", ln=True)
    pdf.cell(0, 10, txt=f"Diagnosis: {result} (Risk Probability: {prob:.1f}%)", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="Clinical Values Summary:", ln=True)
    pdf.set_font("Arial", size=10)
    for k, v in medical_data.items():
        pdf.cell(0, 8, txt=f"- {k}: {v}", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="Medical Recommendations:", ln=True)
    pdf.set_font("Arial", size=10)
    for r in recs:
        pdf.multi_cell(0, 8, txt=r)
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 2. App Setup & Configuration
# ==========================================
st.set_page_config(page_title="Advanced Health AI", layout="wide")
pipeline = joblib.load('full_pipeline_compressed.sav')

st.title("🏥 Next-Gen Chronic Disease Analytics")
st.markdown("This dashboard uses **Explainable AI (SHAP)** and **Benchmarking** for deep health insights.")

# Input Layout
col1, col2 = st.columns(2)
with col1:
    patient_name = st.text_input("Full Name", "Patient Name")
    age = st.number_input("Age", 1, 120, 45)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    bmi = st.number_input("BMI", 10.0, 50.0, 28.0)
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    activity = st.slider("Activity (Hrs/Week)", 0.0, 10.0, 2.0)

with col2:
    bp = st.number_input("Systolic BP (mmHg)", 80, 200, 145)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 250)
    glucose = st.number_input("Glucose (mg/dL)", 50, 300, 150)
    stress = st.slider("Stress (1-10)", 1, 10, 7)
    diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
    family_hist = st.selectbox("Family History", ["No", "Yes"])

# Prepare Model Input
features = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel']
# Adding dummy defaults for missing inputs
input_df = pd.DataFrame([[age, gender, bmi, smoking, "Moderate", activity, diet, 7.0, bp, cholesterol, glucose, family_hist, stress]], columns=features)

if st.button("Run Diagnostic Analytics"):
    # 1. Prediction Results
    prob = pipeline.predict_proba(input_df)[0][1] * 100
    res_text = "High Risk: Chronic Disease Likely" if prob > 50 else "Low Risk: No Chronic Disease"
    
    st.markdown("---")
    if prob > 50: st.error(f"### {res_text} (Risk Score: {prob:.1f}%)")
    else: st.success(f"### {res_text} (Risk Score: {prob:.1f}%)")

    # ---------------------------------------------------------
    # ANALYTICS DASHBOARD: SHAP & BENCHMARKING
    # ---------------------------------------------------------
    st.subheader("🔬 Deep Clinical Insights")
    analysis_col1, analysis_col2 = st.columns(2)

    with analysis_col1:
        st.markdown("**AI Decision Logic (SHAP Breakdown)**")
        try:
            # Fixing the SHAP index issue
            model = pipeline.named_steps['classifier']
            preprocessor = pipeline.named_steps['preprocessor']
            X_trans = preprocessor.transform(input_df)
            f_names = preprocessor.get_feature_names_out()
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_trans)
            
            # Check if output is list (multiclass) or array
            if isinstance(shap_values, list):
                impact = shap_values[1][0] # Focus on Class 1 (Disease)
            elif len(shap_values.shape) == 3:
                impact = shap_values[0, :, 1]
            else:
                impact = shap_values[0]

            shap_df = pd.DataFrame({'Metric': f_names, 'Impact': impact}).sort_values(by='Impact', ascending=False).head(5)
            
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            sns.barplot(x='Impact', y='Metric', data=shap_df, palette='OrRd', ax=ax1)
            plt.title("Key Factors Driving Your Prediction")
            st.pyplot(fig1)
        except Exception as e:
            st.info("AI Logic visualization is processing...")

    with analysis_col2:
        st.markdown("**Clinical Comparison (Benchmarking)**")
        bench_data = {
            'Metric': ['BP', 'Cholesterol', 'Glucose', 'BMI'],
            'Your Value': [bp, cholesterol, glucose, bmi],
            'Target Avg': [120, 200, 100, 22]
        }
        bench_df = pd.DataFrame(bench_data).melt(id_vars='Metric', var_name='Type', value_name='Value')
        
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.barplot(x='Metric', y='Value', hue='Type', data=bench_df, palette='viridis', ax=ax2)
        plt.title("Your Metrics vs. Healthy Benchmark")
        st.pyplot(fig2)

    # 3. Recommendations
    st.subheader("💡 Expert Recommendations")
    recs = []
    if bp > 135: recs.append("- SYSTOLIC HYPERTENSION: Your BP is above the 120mmHg benchmark. Reduce salt intake.")
    if glucose > 140: recs.append("- GLUCOSE ADVISORY: Blood sugar exceeds 100mg/dL average. Monitor carb consumption.")
    if bmi > 25: recs.append("- BMI ADVISORY: Body mass exceeds healthy range (22). Focus on cardio and portion control.")
    if smoking == "Yes": recs.append("- LIFESTYLE: Quitting tobacco is crucial to lowering your risk probability.")

    for r in recs: st.write(r)

    # 4. Professional PDF
    st.markdown("---")
    medical_summary = {"Systolic BP": f"{bp} mmHg", "Glucose": f"{glucose} mg/dL", "Cholesterol": f"{cholesterol} mg/dL", "BMI": f"{bmi}"}
    pdf_report = create_pdf(patient_name, age, gender, res_text, prob, recs, medical_summary)
    st.download_button(label="📥 Download Detailed Analytical Report",
                       data=pdf_report,
                       file_name=f"Clinical_Report_{patient_name}.pdf",
                       mime="application/pdf")

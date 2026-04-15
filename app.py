import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
from fpdf import FPDF

# ==========================================
# 1. PDF Function (Stable Version)
# ==========================================
def create_pdf(name, age, gender, result, prob, recs, medical_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="CHRONIC DISEASE DIAGNOSTIC REPORT", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt=f"Patient: {name} | Age: {age} | Gender: {gender}", ln=True)
    pdf.cell(0, 10, txt=f"Status: {result} (Risk: {prob:.1f}%)", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="Clinical Values:", ln=True)
    pdf.set_font("Arial", size=10)
    for k, v in medical_data.items():
        pdf.cell(0, 8, txt=f"- {k}: {v}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="Advice:", ln=True)
    pdf.set_font("Arial", size=10)
    for r in recs:
        pdf.multi_cell(0, 8, txt=r)
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 2. Main App Setup
# ==========================================
st.set_page_config(page_title="Advanced Health AI", layout="wide")
pipeline = joblib.load('full_pipeline_compressed.sav')

st.title("🏥 Next-Gen Chronic Disease Analytics")
st.markdown("This dashboard uses **Explainable AI (SHAP)** and **Clinical Benchmarking** to analyze health risks.")

# Input Layout
col1, col2 = st.columns(2)
with col1:
    name = st.text_input("Name", "Patient")
    age = st.number_input("Age", 1, 120, 45)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    bmi = st.number_input("BMI", 10.0, 50.0, 28.0)
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    activity = st.slider("Activity (Hrs/Week)", 0.0, 10.0, 2.0)

with col2:
    bp = st.number_input("Systolic BP", 80, 200, 145)
    cholesterol = st.number_input("Cholesterol", 100, 400, 250)
    glucose = st.number_input("Glucose", 50, 300, 150)
    stress = st.slider("Stress (1-10)", 1, 10, 7)
    diet = st.selectbox("Diet", ["Poor", "Average", "Good"])
    family_hist = st.selectbox("Family History", ["No", "Yes"])

# Prepare Data
features = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel']
input_df = pd.DataFrame([[age, gender, bmi, smoking, "Low", activity, diet, 7.0, bp, cholesterol, glucose, family_hist, stress]], columns=features)

if st.button("Run Advanced Diagnosis"):
    # 1. Prediction
    prob = pipeline.predict_proba(input_df)[0][1] * 100
    res_text = "High Risk: Chronic Disease Likely" if prob > 50 else "Low Risk: Healthy"
    
    st.markdown("---")
    st.header(f"Result: {res_text} ({prob:.1f}%)")

    # ---------------------------------------------------------
    # MIXED FEATURES: Choice A (SHAP) & Choice B (Benchmarking)
    # ---------------------------------------------------------
    st.subheader("🔬 Diagnostic Deep-Dive")
    analysis_col1, analysis_col2 = st.columns(2)

    with analysis_col1:
        st.markdown("**Choice A: AI Decision Logic (SHAP)**")
        try:
            model = pipeline.named_steps['classifier']
            preprocessor = pipeline.named_steps['preprocessor']
            X_trans = preprocessor.transform(input_df)
            f_names = preprocessor.get_feature_names_out()
            
            explainer = shap.TreeExplainer(model)
            shap_v = explainer.shap_values(X_trans)
            
            # Extract impact for 'Disease' class
            impact = shap_v[1][0] if isinstance(shap_v, list) else shap_v[0][:, 1]
            
            shap_df = pd.DataFrame({'Factor': f_names, 'Impact': impact}).sort_values(by='Impact', ascending=False).head(5)
            fig1, ax1 = plt.subplots()
            sns.barplot(x='Impact', y='Factor', data=shap_df, palette='OrRd', ax=ax1)
            plt.title("Top Factors Pushing Risk Up")
            st.pyplot(fig1)
        except:
            st.info("Loading AI Logic...")

    with analysis_col2:
        st.markdown("**Choice B: Comparative Benchmarking**")
        # Comparing Patient vs Clinical Average
        bench_data = {
            'Metric': ['BP', 'Cholesterol', 'Glucose', 'BMI'],
            'Your Value': [bp, cholesterol, glucose, bmi],
            'Healthy Avg': [120, 200, 100, 22]
        }
        bench_df = pd.DataFrame(bench_data).melt(id_vars='Metric', var_name='Type', value_name='Value')
        
        fig2, ax2 = plt.subplots()
        sns.barplot(x='Metric', y='Value', hue='Type', data=bench_df, palette='muted', ax=ax2)
        plt.title("Your Metrics vs. Healthy Targets")
        st.pyplot(fig2)

    # Recommendations
    st.subheader("💡 Action Plan")
    recs = []
    if bp > 130: recs.append("-> BP is higher than the 120mmHg benchmark. Reduce salt.")
    if glucose > 140: recs.append("-> Glucose is above 100mg/dL average. Check sugar intake.")
    if bmi > 25: recs.append("-> BMI exceeds healthy benchmark (22). Increase cardio.")
    
    for r in recs: st.write(r)

    # PDF
    clinical_data = {"BP": f"{bp} mmHg", "Glucose": f"{glucose} mg/dL", "BMI": f"{bmi}"}
    pdf_b = create_pdf(name, age, gender, res_text, prob, recs, clinical_data)
    st.download_button("📥 Download Analytical Report", pdf_b, f"{name}_Analysis.pdf")

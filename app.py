import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
from fpdf import FPDF

# ==========================================
# 1. FIXED PDF Function
# ==========================================
def create_pdf(name, age, gender, result, prob, recs, medical_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(44, 62, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 20, txt="CHRONIC DISEASE DIAGNOSTIC REPORT", ln=True, align='C', fill=True)
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
    pdf.cell(0, 10, txt=f"Diagnosis: {result} (Risk Score: {prob:.1f}%)", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=" RECOMMENDATIONS", ln=True, fill=True)
    pdf.set_font("Arial", size=10)
    for r in recs:
        pdf.multi_cell(0, 8, txt=r)
    
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 2. Main App Setup
# ==========================================
st.set_page_config(page_title="Professional AI Health", layout="wide")

# Loading the model safely
try:
    pipeline = joblib.load('full_pipeline_compressed.sav')
except Exception as e:
    st.error(f"Error loading model: {e}")

st.title("🏥 Professional Health Prediction & Analytics")

col1, col2 = st.columns(2)
with col1:
    patient_name = st.text_input("Patient Full Name", "Enter Name")
    age = st.number_input("Age", 1, 120, 45)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    bmi = st.number_input("BMI (kg/m2)", 10.0, 50.0, 25.0)
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    activity = st.slider("Activity (Hrs/Week)", 0.0, 10.0, 2.20)

with col2:
    bp = st.number_input("Systolic BP (mmHg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 200)
    glucose = st.number_input("Glucose Level (mg/dL)", 50, 300, 100)
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)
    diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
    family_hist = st.selectbox("Family History", ["No", "Yes"])

# Data Preparation
features = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel']
input_df = pd.DataFrame([[age, gender, bmi, smoking, "Low", activity, diet, 7.0, bp, cholesterol, glucose, family_hist, stress]], columns=features)

clinical_summary = {
    "Blood Pressure": f"{bp} mmHg",
    "Glucose Level": f"{glucose} mg/dL",
    "Cholesterol": f"{cholesterol} mg/dL",
    "BMI": f"{bmi:.1f}",
    "Activity": f"{activity} Hrs/Wk",
    "Stress": f"{stress}/10"
}

st.markdown("---")

# ==========================================
# 3. TRIGGER: Everything must be INSIDE this if-block
# ==========================================
if st.button("Generate Diagnostic Report"):
    # A. Prediction Analysis
    prob = pipeline.predict_proba(input_df)[0][1] * 100
    res_text = "Chronic Disease Detected" if prob > 50 else "No Chronic Disease"

    if prob > 50:
        st.error(f"### Status: {res_text} ({prob:.1f}%)")
    else:
        st.success(f"### Status: {res_text} ({prob:.1f}%)")

    # B. Analytics Section (SHAP & Benchmarking)
    st.subheader("🔬 Diagnostic Deep-Dive Insights")
    an_col1, an_col2 = st.columns(2)

    with an_col1:
        st.markdown("**AI Decision Logic (SHAP)**")
        try:
            model = pipeline.named_steps['classifier']
            preprocessor = pipeline.named_steps['preprocessor']
            X_trans = preprocessor.transform(input_df)
            f_names = preprocessor.get_feature_names_out()
            
            explainer = shap.TreeExplainer(model)
            shap_v = explainer.shap_values(X_trans)
            
            # Robust mapping for visualization
            if isinstance(shap_v, list): sv = shap_v[1][0]
            elif len(shap_v.shape) == 3: sv = shap_v[0, :, 1]
            else: sv = shap_v[0]
            
            shap_df = pd.DataFrame({'Factor': f_names, 'Impact': np.array(sv).flatten()})
            shap_df = shap_df[shap_df['Impact'] > 0].sort_values(by='Impact', ascending=False).head(5)
            
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            sns.barplot(x='Impact', y='Factor', data=shap_df, palette='Reds_r', ax=ax1)
            plt.title("Primary Risk Drivers", fontweight='bold')
            st.pyplot(fig1)
            plt.close(fig1)
        except Exception as e:
            st.info("Visualizing AI risk factors...")

    with an_col2:
        st.markdown("**Visual Benchmarking (Points)**")
        bench_data = {
            'Metric': ['BP', 'Cholesterol', 'Glucose', 'BMI'],
            'Your Value': [bp, cholesterol, glucose, bmi],
            'Healthy Avg': [120, 200, 100, 22]
        }
        bench_df = pd.DataFrame(bench_data).melt(id_vars='Metric', var_name='Type', value_name='Value')
        
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.barplot(x='Metric', y='Value', hue='Type', data=bench_df, palette='muted', ax=ax2)
        plt.title("Patient Stats vs Target Benchmarks", fontweight='bold')
        st.pyplot(fig2)
        plt.close(fig2)

    # C. Recommendations
    st.subheader("💡 Health Recommendations")
    recs = []
    if bp > 130: recs.append("-> BP is high. Reduce salt intake.")
    if glucose > 140: recs.append("-> Glucose levels are high. Limit sugar.")
    if smoking == "Yes": recs.append("-> Smoking detected as high-risk. Cessation strongly advised.")
    if diet == "Poor": recs.append("-> Diet quality is poor. Focus on fiber and proteins.")
    
    if not recs:
        st.write("Maintain your current healthy lifestyle.")
    else:
        for r in recs: st.write(r)

    # D. PDF Report
    pdf_output = create_pdf(patient_name, age, gender, res_text, prob, recs, clinical_summary)
    st.download_button(label="📥 Download Detailed PDF Report",
                       data=pdf_output,
                       file_name=f"Report_{patient_name}.pdf",
                       mime="application/pdf")

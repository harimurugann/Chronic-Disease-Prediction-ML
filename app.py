import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
from fpdf import FPDF

# ==========================================
# 1. Professional PDF Function
# ==========================================
def create_pdf(name, age, gender, result, prob, recs, medical_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(44, 62, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 20, txt="MEDICAL ASSESSMENT REPORT", ln=True, align='C', fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"Patient: {name} | Age: {age} | Gender: {gender}", ln=True, border='B')
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, txt=f"Diagnosis: {result} (Risk Score: {prob:.1f}%)", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="Clinical Measurements:", ln=True, fill=True)
    pdf.set_font("Arial", size=10)
    for k, v in medical_data.items():
        pdf.cell(95, 8, txt=f" {k}", border=1)
        pdf.cell(95, 8, txt=f" {v}", border=1, ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="Doctor's Recommendations:", ln=True, fill=True)
    pdf.set_font("Arial", size=10)
    for r in recs:
        pdf.multi_cell(0, 8, txt=r)
        
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 2. Main App Configuration
# ==========================================
st.set_page_config(page_title="Advanced Health AI", layout="wide")

# Load model with error handling
try:
    pipeline = joblib.load('full_pipeline_compressed.sav')
except:
    st.error("⚠️ Model file 'full_pipeline_compressed.sav' not found!")

st.title("🏥 Next-Gen Chronic Disease Analytics")
st.markdown("---")

# Input Layout
col1, col2 = st.columns(2)
with col1:
    st.subheader("👤 Patient Profile")
    patient_name = st.text_input("Full Name", "Patient Name")
    age = st.number_input("Age", 1, 120, 45)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    bmi = st.number_input("BMI (kg/m2)", 10.0, 50.0, 28.0)
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    activity = st.slider("Activity (Hrs/Week)", 0.0, 10.0, 2.0)

with col2:
    st.subheader("🩸 Clinical Metrics")
    bp = st.number_input("Systolic BP (mmHg)", 80, 200, 145)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 250)
    glucose = st.number_input("Glucose (mg/dL)", 50, 300, 150)
    stress = st.slider("Stress Level (1-10)", 1, 10, 7)
    diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
    family_hist = st.selectbox("Family History", ["No", "Yes"])

# Prepare Clinical Summary for PDF
clinical_summary = {
    "Systolic Blood Pressure": f"{bp} mmHg",
    "Blood Glucose Level": f"{glucose} mg/dL",
    "Cholesterol Level": f"{cholesterol} mg/dL",
    "BMI": f"{bmi:.1f}",
    "Physical Activity": f"{activity} Hrs/Week",
    "Stress Level": f"{stress}/10"
}

# Model Features (Must match training order)
features = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel']
input_df = pd.DataFrame([[age, gender, bmi, smoking, "Low", activity, diet, 7.0, bp, cholesterol, glucose, family_hist, stress]], columns=features)

# ==========================================
# 3. Execution & Analytics
# ==========================================
if st.button("🚀 Run Comprehensive Diagnosis"):
    # A. Prediction
    prob = pipeline.predict_proba(input_df)[0][1] * 100
    res_text = "Chronic Disease Detected" if prob > 50 else "No Chronic Disease"
    
    st.markdown("---")
    if prob > 50:
        st.error(f"### Diagnosis: {res_text} (Risk Score: {prob:.1f}%)")
    else:
        st.success(f"### Diagnosis: {res_text} (Risk Score: {prob:.1f}%)")

    # B. ANALYTICS SECTION (Choice A & B Mixed)
    st.subheader("🔬 Deep-Dive Diagnostic Insights")
    an_col1, an_col2 = st.columns(2)

    with an_col1:
        st.markdown("**AI Decision Logic (SHAP)**")
        try:
            model = pipeline.named_steps['classifier']
            preprocessor = pipeline.named_steps['preprocessor']
            X_trans = preprocessor.transform(input_df)
            f_names = preprocessor.get_feature_names_out()
            
            # Optimized Explainer
            explainer = shap.TreeExplainer(model)
            shap_v = explainer.shap_values(X_trans)
            
            # Robust extraction of SHAP values
            if isinstance(shap_v, list): sv = shap_v[1][0]
            elif len(shap_v.shape) == 3: sv = shap_v[0, :, 1]
            else: sv = shap_v[0]
            
            shap_df = pd.DataFrame({'Factor': f_names, 'Impact': np.array(sv).flatten()})
            shap_df = shap_df[shap_df['Impact'] > 0].sort_values(by='Impact', ascending=False).head(5)
            
            if not shap_df.empty:
                fig1, ax1 = plt.subplots(figsize=(8, 5))
                sns.barplot(x='Impact', y='Factor', data=shap_df, palette='Reds_r', ax=ax1)
                plt.title("Primary Risk Drivers", fontweight='bold')
                st.pyplot(fig1)
                plt.close(fig1)
            else:
                st.info("AI Logic: No high-risk individual factors detected.")
        except:
            st.warning("SHAP analysis currently processing...")

    with an_col2:
        st.markdown("**Comparative Benchmarking**")
        bench_data = {
            'Metric': ['BP', 'Cholesterol', 'Glucose', 'BMI'],
            'Your Value': [bp, cholesterol, glucose, bmi],
            'Healthy Avg': [120, 200, 100, 22]
        }
        bench_df = pd.DataFrame(bench_data).melt(id_vars='Metric', var_name='Type', value_name='Value')
        
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.barplot(x='Metric', y='Value', hue='Type', data=bench_df, palette='muted', ax=ax2)
        plt.title("Your Stats vs Clinical Targets", fontweight='bold')
        st.pyplot(fig2)
        plt.close(fig2)

    # C. Recommendations
    st.subheader("💡 Personalized Clinical Advice")
    recs = []
    if bp > 130: recs.append("-> BP is above 120mmHg benchmark. Clinical Advice: Reduce sodium and check BP daily.")
    if glucose > 140: recs.append("-> High Glucose detected. Clinical Advice: Limit refined carbs and sugars.")
    if bmi > 25: recs.append("-> BMI exceeds target (22). Clinical Advice: Increase physical activity to 150 mins/week.")
    if smoking == "Yes": recs.append("-> Smoking is a high-impact risk factor. Advice: Consult for tobacco cessation support.")
    
    if not recs:
        st.write("Maintain your current healthy lifestyle and regular annual check-ups.")
    else:
        for r in recs: st.write(r)

    # D. PDF Download
    st.markdown("---")
    pdf_bytes = create_pdf(patient_name, age, gender, res_text, prob, recs, clinical_summary)
    st.download_button(label="📥 Download Detailed Analytical Report",
                       data=pdf_bytes,
                       file_name=f"Report_{patient_name}.pdf",
                       mime="application/pdf")

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
def create_pdf(name, age, gender, result, prob, recs, medical_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(44, 62, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 20, txt="ADVANCED HEALTH ASSESSMENT REPORT", ln=True, align='C', fill=True)
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
    pdf.cell(0, 10, txt=" PERSONALIZED DOCTOR'S ADVICE", ln=True, fill=True)
    pdf.set_font("Arial", size=10)
    for r in recs:
        pdf.multi_cell(0, 8, txt=r)
    
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 2. MAIN APP CONFIG & DATA ENTRY
# ==========================================
st.set_page_config(page_title="Health AI Pro", layout="wide")
pipeline = joblib.load('full_pipeline_compressed.sav')

st.title("🏥 Professional Health Prediction & Improvement Simulator")
st.markdown("Enter your clinical data to analyze risks and simulate lifestyle improvements.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("👤 Basic Profile")
    patient_name = st.text_input("Patient Full Name", "Palanisamy")
    age = st.number_input("Age", 1, 120, 45)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    bmi = st.number_input("BMI (kg/m2)", 10.0, 50.0, 25.0)
    smoking = st.selectbox("Smoking Status", ["No", "Yes"])
    activity = st.slider("Activity (Hrs/Week)", 0.0, 10.0, 2.20)

with col2:
    st.subheader("🩸 Clinical Metrics")
    bp = st.number_input("Systolic BP (mmHg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 200)
    glucose = st.number_input("Glucose Level (mg/dL)", 50, 300, 100)
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)
    diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
    family_hist = st.selectbox("Family History", ["No", "Yes"])

# Data Setup
features = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel']
input_df = pd.DataFrame([[age, gender, bmi, smoking, "Low", activity, diet, 7.0, bp, cholesterol, glucose, family_hist, stress]], columns=features)

clinical_summary = {
    "Blood Pressure": f"{bp} mmHg",
    "Glucose Level": f"{glucose} mg/dL",
    "Total Cholesterol": f"{cholesterol} mg/dL",
    "BMI": f"{bmi:.1f}",
    "Weekly Activity": f"{activity} Hrs",
    "Stress Level": f"{stress}/10"
}

st.markdown("---")

# ==========================================
# 3. ANALYSIS & INTERACTIVE FEATURES
# ==========================================
if st.button("Generate Diagnostic Report & Analysis"):
    # A. PREDICTION
    prob = pipeline.predict_proba(input_df)[0][1] * 100
    res_text = "Chronic Disease Detected" if prob > 50 else "No Chronic Disease"

    st.header(f"Status: {res_text} ({prob:.1f}%)")
    
    # B. WHAT-IF SIMULATION (Advanced Feature 3)
    st.subheader("🛠️ Full Clinical Improvement Simulator")
    st.markdown("*What if you improve your BP, Glucose, and Cholesterol? Adjust below to see your risk change:*")
    
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        target_bp = st.slider("Target Systolic BP", 80, 200, int(bp), key="sbp")
        target_glucose = st.slider("Target Glucose Level", 50, 300, int(glucose), key="glu")
    with sim_col2:
        target_chol = st.slider("Target Cholesterol", 100, 400, int(cholesterol), key="cho")
        target_bmi = st.slider("Target BMI", 10.0, 50.0, float(bmi), key="bmi_sim")

    # Simulation Logic
    sim_df = input_df.copy()
    sim_df['BloodPressure'] = target_bp
    sim_df['Glucose'] = target_glucose
    sim_df['Cholesterol'] = target_chol
    sim_df['BMI'] = target_bmi
    
    sim_prob = pipeline.predict_proba(sim_df)[0][1] * 100
    diff = sim_prob - prob
    
    st.metric(label="Simulated Risk Score", value=f"{sim_prob:.1f}%", delta=f"{diff:.1f}%", delta_color="inverse")
    
    if sim_prob < 50 and prob > 50:
        st.balloons()
        st.success("✨ Incredible! These target levels would bring your health status to 'Low Risk'.")

    # C. DIAGNOSTIC DEEP-DIVE (SHAP & Benchmarking)
    st.markdown("---")
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
        except: st.info("Loading AI Analysis...")

    with an_col2:
        st.markdown("**Visual Benchmarking (Patient vs Healthy)**")
        bench_data = {'Metric': ['BP', 'Cholesterol', 'Glucose', 'BMI'], 'Your Value': [bp, cholesterol, glucose, bmi], 'Healthy Target': [120, 200, 100, 22]}
        bench_df = pd.DataFrame(bench_data).melt(id_vars='Metric', var_name='Type', value_name='Value')
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.barplot(x='Metric', y='Value', hue='Type', data=bench_df, palette='muted', ax=ax2)
        st.pyplot(fig2)
        plt.close(fig2)

    # D. RECOMMENDATIONS & PDF
    st.subheader("💡 Actionable Recommendations")
    recs = []
    if bp > 130: recs.append("-> BP Management: Salt reduction and daily monitoring advised.")
    if glucose > 140: recs.append("-> Blood Sugar: Limit refined sugars and carbs.")
    if smoking == "Yes": recs.append("-> Habits: Smoking is your #1 risk. Cessation support recommended.")
    
    for r in recs: st.write(r)

    pdf_bytes = create_pdf(patient_name, age, gender, res_text, prob, recs, clinical_summary)
    st.download_button(label="📥 Download Analytical Report (PDF)", data=pdf_bytes, file_name=f"Report_{patient_name}.pdf", mime="application/pdf")

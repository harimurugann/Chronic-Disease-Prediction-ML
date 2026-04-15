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
    pdf.cell(0, 20, txt="CHRONIC DISEASE ASSESSMENT REPORT", ln=True, align='C', fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    
    # Patient Info
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"Patient: {name} | Age: {age} | Gender: {gender}", ln=True, border='B')
    pdf.ln(5)

    # Clinical Measurements Table
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=" CLINICAL MEASUREMENTS", ln=True, fill=True)
    pdf.set_font("Arial", size=10)
    for key, value in medical_data.items():
        pdf.cell(95, 8, f" {key}", border=1)
        pdf.cell(95, 8, f" {value}", border=1, ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"Final Diagnosis: {result} (Risk Score: {prob:.1f}%)", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=" DOCTOR'S CLINICAL ADVICE", ln=True, fill=True)
    pdf.set_font("Arial", size=10)
    for r in recs:
        pdf.multi_cell(0, 8, txt=r)
    
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 2. MAIN APP SETUP
# ==========================================
st.set_page_config(page_title="Professional AI Health", layout="wide")

# Loading model with check
try:
    pipeline = joblib.load('full_pipeline_compressed.sav')
except Exception as e:
    st.error(f"Error: Model file not found! {e}")

st.title("🏥 Professional Health Prediction & Improvement Simulator")
st.markdown("---")

# Layout for Inputs
col1, col2 = st.columns(2)
with col1:
    st.subheader("👤 Patient Profile")
    patient_name = st.text_input("Patient Full Name", "Palanisamy")
    age = st.number_input("Age", 1, 120, 45)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    bmi = st.number_input("BMI (kg/m2)", 10.0, 50.0, 25.0)
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    activity = st.slider("Activity (Hrs/Week)", 0.0, 10.0, 2.20)

with col2:
    st.subheader("🩸 Medical Data")
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
    "Total Cholesterol": f"{cholesterol} mg/dL",
    "BMI": f"{bmi:.1f}",
    "Activity Level": f"{activity} Hrs/Wk",
    "Stress Assessment": f"{stress}/10"
}

# ==========================================
# 3. BUTTON TRIGGER & LOGIC
# ==========================================
if st.button("🚀 Generate Full Health Analysis"):
    # A. PREDICTION
    prob = pipeline.predict_proba(input_df)[0][1] * 100
    res_text = "Chronic Disease Detected" if prob > 50 else "No Chronic Disease"

    if prob > 50:
        st.error(f"### Current Status: {res_text} ({prob:.1f}%)")
    else:
        st.success(f"### Current Status: {res_text} ({prob:.1f}%)")

    # B. WHAT-IF SIMULATOR WITH METRIC CARDS
    st.markdown("---")
    st.subheader("🏁 Full Clinical Improvement Simulator")
    st.info("Adjust the targets below to see how lifestyle changes affect your risk:")

    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        target_bp = st.slider("Target Systolic BP", 80, 200, int(bp), key="sbp")
        target_glucose = st.slider("Target Glucose Level", 50, 300, int(glucose), key="glu")
    with sim_col2:
        target_chol = st.slider("Target Cholesterol", 100, 400, int(cholesterol), key="cho")
        target_bmi = st.slider("Target BMI", 10.0, 50.0, float(bmi), key="bmi_sim")

    # Simulation Calculation
    sim_df = input_df.copy()
    sim_df['BloodPressure'] = target_bp
    sim_df['Glucose'] = target_glucose
    sim_df['Cholesterol'] = target_chol
    sim_df['BMI'] = target_bmi
    sim_prob = pipeline.predict_proba(sim_df)[0][1] * 100
    diff = sim_prob - prob

    # METRIC CARDS DISPLAY
    st.markdown("#### Simulated Risk Analysis")
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric(label="Original Risk Score", value=f"{prob:.1f}%")
    with m_col2:
        st.metric(label="Simulated Risk Score", value=f"{sim_prob:.1f}%", delta=f"{diff:.1f}%", delta_color="inverse")
    with m_col3:
        improvement = abs(diff) if diff < 0 else 0
        st.metric(label="Total Improvement", value=f"{improvement:.1f}%")

    if sim_prob < 50 and prob > 50:
        st.balloons()
        st.success("🌟 Positive Change Detected! These targets will significantly improve your health status.")

    # C. ANALYTICS DEEP-DIVE
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
        except: st.info("Visualizing risk impact factors...")

    with an_col2:
        st.markdown("**Visual Benchmarking**")
        bench_data = {'Metric': ['BP', 'Cholesterol', 'Glucose', 'BMI'], 'Your Value': [bp, cholesterol, glucose, bmi], 'Healthy Target': [120, 200, 100, 22]}
        bench_df = pd.DataFrame(bench_data).melt(id_vars='Metric', var_name='Type', value_name='Value')
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.barplot(x='Metric', y='Value', hue='Type', data=bench_df, palette='muted', ax=ax2)
        plt.title("Current vs Healthy Targets", fontweight='bold')
        st.pyplot(fig2)
        plt.close(fig2)

    # D. RECOMMENDATIONS & PDF
    st.subheader("💡 Health Recommendations")
    recs = []
    if bp > 130: recs.append("-> BP Management: Hypertension risk detected. Reduce salt and check BP daily.")
    if glucose > 140: recs.append("-> Blood Sugar: Potential diabetic risk. Limit refined sugars and carbs.")
    if bmi > 25: recs.append("-> Weight Management: BMI is above normal. Focus on cardio and calorie balance.")
    if smoking == "Yes": recs.append("-> Habits: Smoking is a critical risk factor. Immediate cessation support recommended.")
    
    if not recs:
        st.write("Your metrics look healthy. Maintain this lifestyle!")
    else:
        for r in recs: st.write(r)

    pdf_bytes = create_pdf(patient_name, age, gender, res_text, prob, recs, clinical_summary)
    st.download_button(label="📥 Download Comprehensive Analytical Report", data=pdf_bytes, file_name=f"Assessment_Report_{patient_name}.pdf", mime="application/pdf")

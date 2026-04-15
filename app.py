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
    pdf.cell(0, 20, txt="CHRONIC DISEASE DIAGNOSTIC REPORT", ln=True, align='C', fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"Patient: {name} | Age: {age} | Gender: {gender}", ln=True, border='B')
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, txt=f"Diagnostic Status: {result} (Risk Score: {prob:.1f}%)", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="Clinical Measurements:", ln=True, fill=True)
    pdf.set_font("Arial", size=10)
    for k, v in medical_data.items():
        pdf.cell(95, 8, txt=f" {k}", border=1)
        pdf.cell(95, 8, txt=f" {v}", border=1, ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="Detailed Doctor Recommendations:", ln=True, fill=True)
    pdf.set_font("Arial", size=10)
    for r in recs:
        pdf.multi_cell(0, 8, txt=r)
        
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 2. APP CONFIGURATION & MODEL LOADING
# ==========================================
st.set_page_config(page_title="Advanced Health AI Dashboard", layout="wide")

try:
    pipeline = joblib.load('full_pipeline_compressed.sav')
except:
    st.error("⚠️ Model file not found! Ensure 'full_pipeline_compressed.sav' is in the folder.")

st.title("🏥 Next-Gen Chronic Disease Analytics & Simulator")
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
    activity = st.slider("Physical Activity (Hrs/Week)", 0.0, 15.0, 2.0)

with col2:
    st.subheader("🩸 Clinical Metrics")
    bp = st.number_input("Systolic Blood Pressure (mmHg)", 80, 200, 145)
    cholesterol = st.number_input("Cholesterol Level (mg/dL)", 100, 400, 250)
    glucose = st.number_input("Glucose Level (mg/dL)", 50, 300, 150)
    stress = st.slider("Current Stress Level (1-10)", 1, 10, 7)
    diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
    family_hist = st.selectbox("Family History", ["No", "Yes"])

# Clinical data for PDF
clinical_summary = {
    "Blood Pressure": f"{bp} mmHg",
    "Glucose Level": f"{glucose} mg/dL",
    "Cholesterol": f"{cholesterol} mg/dL",
    "BMI": f"{bmi:.1f}",
    "Physical Activity": f"{activity} Hrs/Wk",
    "Stress Level": f"{stress}/10"
}

# Features for Model
features = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel']
input_df = pd.DataFrame([[age, gender, bmi, smoking, "Low", activity, diet, 7.0, bp, cholesterol, glucose, family_hist, stress]], columns=features)

# ==========================================
# 3. ANALYSIS & INTERACTIVE FEATURES
# ==========================================
if st.button("🚀 Run Comprehensive Analysis"):
    # A. Initial Prediction
    prob = pipeline.predict_proba(input_df)[0][1] * 100
    res_text = "Chronic Disease Detected" if prob > 50 else "No Chronic Disease"
    
    st.markdown("---")
    if prob > 50:
        st.error(f"### Diagnosis: {res_text} (Current Risk: {prob:.1f}%)")
    else:
        st.success(f"### Diagnosis: {res_text} (Current Risk: {prob:.1f}%)")

    # ------------------------------------------
    # NEW FEATURE: WHAT-IF SIMULATOR
    # ------------------------------------------
    st.subheader("🛠️ Lifestyle Improvement Simulator")
    st.info("Adjust targets below to see how lifestyle changes could reduce your risk.")
    
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        s_bmi = st.slider("Target BMI", 18.0, 35.0, float(bmi), key="s_bmi")
        s_act = st.slider("Target Activity (Hrs/Wk)", 0.0, 20.0, float(activity), key="s_act")
    with sim_col2:
        s_smoke = st.selectbox("Quit Smoking?", ["No", "Yes"], index=0 if smoking=="No" else 1, key="s_smoke")
        s_stress = st.slider("Target Stress Level", 1, 10, int(stress), key="s_stress")

    # Simulation Prediction
    sim_df = input_df.copy()
    sim_df['BMI'], sim_df['PhysicalActivity'], sim_df['Smoking'], sim_df['StressLevel'] = s_bmi, s_act, s_smoke, s_stress
    sim_prob = pipeline.predict_proba(sim_df)[0][1] * 100
    
    diff = prob - sim_prob
    st.metric(label="Simulated Risk Score", value=f"{sim_prob:.1f}%", delta=f"-{diff:.1f}%" if diff > 0 else f"+{abs(diff):.1f}%", delta_color="inverse")
    if diff > 5:
        st.balloons()
        st.success(f"Great! These changes can lower your risk by {diff:.1f}%")

    # ------------------------------------------
    # PHASE 1 ANALYTICS: SHAP & BENCHMARKING
    # ------------------------------------------
    st.subheader("🔬 Deep-Dive Diagnostic Insights")
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
            
            sv = shap_v[1][0] if isinstance(shap_v, list) else shap_v[0, :, 1]
            shap_df = pd.DataFrame({'Factor': f_names, 'Impact': np.array(sv).flatten()})
            shap_df = shap_df[shap_df['Impact'] > 0].sort_values(by='Impact', ascending=False).head(5)
            
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            sns.barplot(x='Impact', y='Factor', data=shap_df, palette='Reds_r', ax=ax1)
            plt.title("Key Risk Drivers")
            st.pyplot(fig1)
            plt.close(fig1)
        except:
            st.info("AI Logic analysis in progress...")

    with an_col2:
        st.markdown("**Comparative Benchmarking**")
        bench_df = pd.DataFrame({
            'Metric': ['BP', 'Glucose', 'BMI'],
            'Your Value': [bp, glucose, bmi],
            'Healthy Avg': [120, 100, 22]
        }).melt(id_vars='Metric', var_name='Type', value_name='Value')
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.barplot(x='Metric', y='Value', hue='Type', data=bench_df, palette='muted', ax=ax2)
        plt.title("Your Stats vs Targets")
        st.pyplot(fig2)
        plt.close(fig2)

    # D. Clinical Advice & PDF
    st.subheader("💡 Personalized Clinical Advice")
    recs = []
    if bp > 130: recs.append("-> HIGH BLOOD PRESSURE: Reduce sodium intake and engage in daily 30-min brisk walking.")
    if glucose > 140: recs.append("-> GLUCOSE ADVISORY: Limit refined carbohydrates and schedule a HbA1c test.")
    if bmi > 25: recs.append("-> BMI MANAGEMENT: Focus on a calorie-deficit diet and strength training.")
    if s_smoke == "Yes" and diff > 0: recs.append(f"-> QUITTING SMOKING: Can potentially reduce your risk by a significant margin.")
    
    for r in recs: st.write(r)

    st.markdown("---")
    pdf_b = create_pdf(patient_name, age, gender, res_text, prob, recs, clinical_summary)
    st.download_button(label="📥 Download Detailed Analytical Report", data=pdf_b, file_name=f"Report_{patient_name}.pdf")

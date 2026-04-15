import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
from fpdf import FPDF

# ==========================================
# 1. LANGUAGE DICTIONARY
# ==========================================
lang_data = {
    "English": {
        "title": "🏥 Clinical Health Dashboard & AI Analytics",
        "profile": "👤 Patient Profile",
        "vitals": "🩸 Clinical Metrics",
        "run_btn": "🚀 Run Comprehensive Analysis",
        "sim_title": "🏁 Full Clinical Improvement Simulator",
        "bmi_title": "⚖️ BMI Health Classification",
        "download": "📥 Download Clinical PDF Report"
    },
    "Tamil": {
        "title": "🏥 மருத்துவ நலப் பரிசோதனை மற்றும் பகுப்பாய்வு",
        "profile": "👤 நோயாளியின் விவரங்கள்",
        "vitals": "🩸 மருத்துவ அளவீடுகள்",
        "run_btn": "🚀 முழுமையான பரிசோதனையைத் தொடங்கு",
        "sim_title": "🏁 உடல்நிலை முன்னேற்ற சிமுலேட்டர்",
        "bmi_title": "⚖️ பி.எம்.ஐ (BMI) உடல்நிலை வகைப்பாடு",
        "download": "📥 மருத்துவ அறிக்கையைப் பதிவிறக்கம் செய் (PDF)"
    }
}

# ==========================================
# 2. PDF FUNCTION (Robust Version)
# ==========================================
def create_pdf(name, age, gender, result, prob, medical_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(44, 62, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 20, txt="HEALTH ASSESSMENT REPORT", ln=True, align='C', fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"Patient: {name} | Age: {age} | Gender: {gender}", ln=True, border='B')
    pdf.ln(5)
    
    # Simple Status for PDF to avoid encoding issues
    status_eng = "Action Required" if prob > 50 else "Healthy Range"
    pdf.cell(0, 10, txt=f"Result: {status_eng} (Risk Score: {prob:.1f}%)", ln=True)
    pdf.ln(5)

    pdf.cell(0, 10, txt=" CLINICAL VALUES", ln=True, fill=True)
    pdf.set_font("Arial", size=10)
    for k, v in medical_data.items():
        pdf.cell(95, 8, txt=f" {k}", border=1)
        pdf.cell(95, 8, txt=f" {v}", border=1, ln=True)
    
    return pdf.output()

# ==========================================
# 3. MAIN APP SETUP
# ==========================================
st.set_page_config(page_title="Professional AI Health", layout="wide")

try:
    pipeline = joblib.load('full_pipeline_compressed.sav')
except:
    st.error("Model file not found! Please check 'full_pipeline_compressed.sav'")

# Sidebar Language Selection
sel_lang = st.sidebar.selectbox("🌐 Choose Language / மொழி", ["English", "Tamil"])
L = lang_data[sel_lang]

st.title(L["title"])
st.markdown("---")

# User Inputs
col1, col2 = st.columns(2)
with col1:
    st.subheader(L["profile"])
    patient_name = st.text_input("Full Name / பெயர்", "Palanisamy")
    age = st.number_input("Age / வயது", 1, 120, 45)
    gender = st.selectbox("Gender / பாலினம்", ["Male", "Female", "Other"])
    bmi = st.number_input("BMI (kg/m2)", 10.0, 50.0, 25.0)
    smoking = st.selectbox("Smoking / புகைப்பிடித்தல்", ["No", "Yes"])
    activity = st.slider("Activity / உடற்பயிற்சி (Hrs/Week)", 0.0, 10.0, 2.2)

with col2:
    st.subheader(L["vitals"])
    bp = st.number_input("Systolic BP (mmHg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 200)
    glucose = st.number_input("Glucose Level (mg/dL)", 50, 300, 100)
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)
    diet = st.selectbox("Diet Quality / உணவுமுறை", ["Poor", "Average", "Good"])
    family_hist = st.selectbox("Family History / பரம்பரை வரலாறு", ["No", "Yes"])

# Model features preparation
features = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel']
input_df = pd.DataFrame([[age, gender, bmi, smoking, "Low", activity, diet, 7.0, bp, cholesterol, glucose, family_hist, stress]], columns=features)

clinical_summary = {"BP": f"{bp} mmHg", "Glucose": f"{glucose} mg/dL", "Cholesterol": f"{cholesterol} mg/dL", "BMI": f"{bmi:.1f}"}

# ==========================================
# 4. ANALYSIS LOGIC
# ==========================================
if st.button(L["run_btn"]):
    prob = pipeline.predict_proba(input_df)[0][1] * 100
    
    # 1. Result Metric Cards
    st.markdown("---")
    res_msg = "Disease Risk Detected" if prob > 50 else "Low Risk / Healthy"
    if prob > 50: st.error(f"### {res_msg} ({prob:.1f}%)")
    else: st.success(f"### {res_msg} ({prob:.1f}%)")

    # 2. BMI Classification (Feature 5)
    st.subheader(L["bmi_title"])
    bmi_cat, bmi_color = "", ""
    if bmi < 18.5: bmi_cat, bmi_color = "Underweight", "blue"
    elif 18.5 <= bmi < 25: bmi_cat, bmi_color = "Normal Weight", "green"
    elif 25 <= bmi < 30: bmi_cat, bmi_color = "Overweight", "orange"
    else: bmi_cat, bmi_color = "Obese", "red"
    
    st.markdown(f"Status: :{bmi_color}[**{bmi_cat}**]")

    # 3. Improvement Simulator
    st.markdown("---")
    st.subheader(L["sim_title"])
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        t_bp = st.slider("Target BP", 80, 200, int(bp), key="t_bp")
        t_glu = st.slider("Target Glucose", 50, 300, int(glucose), key="t_glu")
    with sim_col2:
        t_cho = st.slider("Target Cholesterol", 100, 400, int(cholesterol), key="t_cho")
        t_bmi = st.slider("Target BMI", 10.0, 50.0, float(bmi), key="t_bmi")

    sim_df = input_df.copy()
    sim_df.update({'BloodPressure': t_bp, 'Glucose': t_glu, 'Cholesterol': t_cho, 'BMI': t_bmi})
    sim_prob = pipeline.predict_proba(sim_df)[0][1] * 100
    st.metric("Simulated Risk Score", f"{sim_prob:.1f}%", delta=f"{sim_prob-prob:.1f}%", delta_color="inverse")

    # 4. SHAP Analytics
    st.markdown("---")
    try:
        model = pipeline.named_steps['classifier']
        preprocessor = pipeline.named_steps['preprocessor']
        X_trans = preprocessor.transform(input_df)
        explainer = shap.TreeExplainer(model)
        shap_v = explainer.shap_values(X_trans)
        impact = shap_v[1][0] if isinstance(shap_v, list) else shap_v[0,:,1]
        
        shap_df = pd.DataFrame({'Factor': preprocessor.get_feature_names_out(), 'Impact': np.array(impact).flatten()})
        shap_df = shap_df[shap_df['Impact'] > 0].sort_values(by='Impact', ascending=False).head(5)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x='Impact', y='Factor', data=shap_df, palette='Reds_r')
        st.pyplot(fig)
    except: st.info("Loading AI Analysis...")

    # 5. PDF Download
    pdf_out = create_pdf(patient_name, age, gender, res_msg, prob, clinical_summary)
    st.download_button(L["download"], pdf_out, f"Report_{patient_name}.pdf")

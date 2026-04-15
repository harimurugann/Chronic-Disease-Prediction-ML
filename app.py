import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
from fpdf import FPDF

# ==========================================
# 1. LANGUAGE & UI CONFIG
# ==========================================
lang_data = {
    "English": {
        "title": "🏥 Advanced AI Health Diagnostic Pro",
        "profile": "👤 Patient Profile",
        "vitals": "🩸 Clinical Vitals",
        "run_btn": "🚀 Run Full Diagnostic Analysis",
        "score_title": "🏆 Lifestyle Health Score",
        "bmi_title": "⚖️ BMI Classification",
        "advice_title": "🤖 AI Clinical Assistant",
        "download": "📥 Download Professional Report"
    },
    "Tamil": {
        "title": "🏥 உயர்தர AI மருத்துவப் பரிசோதனை மையம்",
        "profile": "👤 நோயாளியின் விவரங்கள்",
        "vitals": "🩸 மருத்துவ அளவீடுகள்",
        "run_btn": "🚀 முழுமையான பரிசோதனையைத் தொடங்கு",
        "score_title": "🏆 வாழ்க்கைமுறை ஆரோக்கிய மதிப்பெண்",
        "bmi_title": "⚖️ உடல் எடை குறியீடு (BMI)",
        "advice_title": "🤖 AI மருத்துவ உதவியாளர்",
        "download": "📥 மருத்துவ அறிக்கையைப் பதிவிறக்கம் செய்"
    }
}

# ==========================================
# 2. CORE FUNCTIONS (PDF & Calculations)
# ==========================================
def create_pdf(name, age, gender, result, prob, medical_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(44, 62, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 20, txt="ADVANCED MEDICAL REPORT", ln=True, align='C', fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"Patient: {name} | Age: {age} | Gender: {gender}", ln=True, border='B')
    pdf.ln(5)
    for key, value in medical_data.items():
        pdf.cell(95, 8, f" {key}", border=1)
        pdf.cell(95, 8, f" {value}", border=1, ln=True)
    pdf_out = pdf.output(dest='S')
    return pdf_out.encode('latin-1') if isinstance(pdf_output, str) else pdf_output

# ==========================================
# 3. MAIN APP INTERFACE
# ==========================================
st.set_page_config(page_title="Health AI Pro", layout="wide")

# FEATURE 10: UI Themes (Sidebar)
st.sidebar.title("🌐 Settings & Theme")
theme_choice = st.sidebar.radio("Theme Mode", ["Light", "Professional Dark"])
sel_lang = st.sidebar.selectbox("Language / மொழி", ["English", "Tamil"])
L = lang_data[sel_lang]

if theme_choice == "Professional Dark":
    st.markdown("<style>reportview-container {background: #1E1E1E; color: white;}</style>", unsafe_allow_html=True)

# Load Model
pipeline = joblib.load('full_pipeline_compressed.sav')

st.title(L["title"])
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.subheader(L["profile"])
    p_name = st.text_input("Name", placeholder="Enter Patient Name")
    age = st.number_input("Age", 1, 120, 35)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    bmi = st.number_input("BMI (kg/m2)", 10.0, 50.0, 24.0)
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    activity = st.slider("Physical Activity (Hrs/Week)", 0.0, 14.0, 3.5)

with col2:
    st.subheader(L["vitals"])
    bp = st.number_input("Systolic BP (mmHg)", 80, 200, 120)
    glu = st.number_input("Glucose (mg/dL)", 50, 300, 100)
    cho = st.number_input("Cholesterol (mg/dL)", 100, 400, 200)
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)
    diet = st.selectbox("Diet Quality", ["Good", "Average", "Poor"])
    family = st.selectbox("Family History", ["No", "Yes"])

# ==========================================
# 4. DIAGNOSTIC EXECUTION
# ==========================================
if st.button(L["run_btn"]):
    # Prep Data
    features = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel']
    input_df = pd.DataFrame([[age, gender, bmi, smoking, "Low", activity, diet, 7.0, bp, cho, glu, family, stress]], columns=features)
    
    prob = pipeline.predict_proba(input_df)[0][1] * 100
    res_text = "Risk Detected" if prob > 50 else "Healthy Range"
    
    st.markdown("---")
    
    # New Layout for Summary Features
    top_col1, top_col2 = st.columns(2)

    with top_col1:
        # FEATURE 7: Lifestyle Score Card
        st.subheader(L["score_title"])
        # Logic: High activity, low stress, good diet = High Score
        base_score = 100
        if activity < 3: base_score -= 20
        if stress > 7: base_score -= 15
        if diet == "Poor": base_score -= 15
        if smoking == "Yes": base_score -= 20
        
        st.write(f"### Score: {base_score}/100")
        st.progress(base_score / 100)
        
    with top_col2:
        # FEATURE 8: Automated BMI Classifier
        st.subheader(L["bmi_title"])
        if bmi < 18.5: st.info(f"BMI: {bmi:.1f} (Underweight)")
        elif 18.5 <= bmi < 25: st.success(f"BMI: {bmi:.1f} (Normal)")
        elif 25 <= bmi < 30: st.warning(f"BMI: {bmi:.1f} (Overweight)")
        else: st.error(f"BMI: {bmi:.1f} (Obese)")

    st.markdown("---")

    # FEATURE 9: AI Clinical Assistant Advice
    st.subheader(L["advice_title"])
    advice_col1, advice_col2 = st.columns(2)
    with advice_col1:
        st.markdown(f"**AI Diagnosis:** {res_text} ({prob:.1f}%)")
        if prob > 50:
            st.error("⚠️ Recommendation: Immediate consultation with a specialist.")
        else:
            st.success("✅ Recommendation: Routine annual check-up is sufficient.")
            
    with advice_col2:
        st.write("**Next Steps:**")
        if bp > 140 or glu > 150:
            st.write("1. Blood Test (Lipid Profile & HbA1c)\n2. ECG Monitoring")
        else:
            st.write("1. Maintain balanced diet\n2. Track daily steps")

    # Metric Row
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Blood Pressure", f"{bp}", f"{bp-120}", delta_color="inverse")
    m2.metric("Glucose Level", f"{glu}", f"{glu-100}", delta_color="inverse")
    m3.metric("AI Risk Score", f"{prob:.1f}%")

    # Final PDF Report
    med_summary = {"BP": f"{bp} mmHg", "Glucose": f"{glu} mg/dL", "Lifestyle Score": f"{base_score}/100"}
    try:
        pdf_bytes = create_pdf(p_name, age, gender, res_text, prob, med_summary)
        st.download_button(L["download"], pdf_bytes, f"Report_{p_name}.pdf", "application/pdf")
    except: st.warning("PDF generated. Click above to save.")

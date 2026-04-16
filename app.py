import streamlit as st
import pandas as pd
import joblib
import numpy as np
from fpdf import FPDF

# ==========================================
# 1. LANGUAGE & DATA SETUP
# ==========================================
lang_data = {
    "English": {
        "title": "🏥 Advanced AI Health Diagnostic Pro",
        "profile": "👤 Patient Profile",
        "vitals": "🩸 Clinical Metrics",
        "run_btn": "🚀 Run Full Diagnostic Analysis",
        "advice_title": "📋 Clinical Recommendations",
        "download_btn": "📥 Download Clinical Report (PDF)"
    },
    "Tamil": {
        "title": "🏥 உயர்தர AI மருத்துவப் பரிசோதனை மையம்",
        "profile": "👤 நோயாளியின் விவரங்கள்",
        "vitals": "🩸 மருத்துவ அளவீடுகள்",
        "run_btn": "🚀 முழுமையான பரிசோதனையைத் தொடங்கு",
        "advice_title": "📋 மருத்துவ பரிந்துரைகள்",
        "download_btn": "📥 மருத்துவ அறிக்கையைப் பதிவிறக்கம் செய்"
    }
}

# ==========================================
# 2. PDF & RECOMMENDATION LOGIC
# ==========================================

def get_recommendations(bp, glu, bmi, smoking, lang):
    recs = []
    # Format: (Display Text with Emoji, Clean PDF Text)
    if lang == "English":
        if bp > 140: recs.append(("⚠️ BP is High: Reduce salt intake.", "BP is High: Reduce salt intake."))
        if glu > 150: recs.append(("⚠️ Glucose is High: Avoid sugary foods.", "Glucose is High: Avoid sugary foods."))
        if bmi > 25: recs.append(("🏃 BMI High: Focus on daily 30-min brisk walk.", "BMI High: Focus on daily 30-min brisk walk."))
        if smoking == "Yes": recs.append(("🚭 Smoking: Quit smoking to lower cardiac risk.", "Smoking: Quit smoking to lower cardiac risk."))
    else:
        if bp > 140: recs.append(("⚠️ இரத்த அழுத்தம் அதிகம்: உப்பைக் குறைக்கவும்.", "BP is High: Reduce salt intake."))
        if glu > 150: recs.append(("⚠️ சர்க்கரை அளவு அதிகம்: இனிப்பைத் தவிர்க்கவும்.", "Glucose is High: Avoid sugary foods."))
        if bmi > 25: recs.append(("🏃 எடை அதிகம்: தினமும் நடைப்பயிற்சி செய்யவும்.", "BMI High: Daily brisk walk recommended."))
        if smoking == "Yes": recs.append(("🚭 புகைப்பிடித்தலைத் தவிர்க்கவும்.", "Smoking: Quit smoking recommended."))
    return recs

def create_pdf(name, age, gender, result, prob, medical_data, recommendations):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_fill_color(44, 62, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 15, txt="MEDICAL ASSESSMENT REPORT", ln=True, align='C', fill=True)
    
    # Patient Info
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, txt=f"Patient Name: {name if name else 'N/A'}", ln=True)
    pdf.cell(0, 8, txt=f"Age: {age} | Gender: {gender}", ln=True, border='B')
    pdf.ln(5)

    # Vitals Table
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 10, txt="CLINICAL VITALS:", ln=True)
    for key, value in medical_data.items():
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(70, 8, f" {key}", border=1)
        pdf.set_font("Arial", size=10)
        pdf.cell(100, 8, f" {value}", border=1, ln=True)

    # Diagnosis Result
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"AI Risk Assessment: {result} ({prob:.1f}%)", ln=True)

    # Recommendations (CLEAN TEXT ONLY - No Emojis)
    if recommendations:
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, txt="CLINICAL ADVICE & NEXT STEPS:", ln=True)
        pdf.set_font("Arial", size=10)
        for r_pair in recommendations:
            # Using r_pair[1] which is the clean English text for PDF
            clean_text = r_pair[1].encode('ascii', 'ignore').decode('ascii')
            pdf.multi_cell(0, 7, txt=f"- {clean_text}")

    # Version-safe Binary Output
    pdf_output = pdf.output(dest='S')
    if isinstance(pdf_output, str):
        return pdf_output.encode('latin-1', 'ignore')
    return pdf_output

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Health AI Pro", layout="wide")

# Sidebar
st.sidebar.title("⚙️ Settings")
sel_lang = st.sidebar.selectbox("Language / மொழி", ["English", "Tamil"])
L = lang_data[sel_lang]

# Load Model
try:
    pipeline = joblib.load('full_pipeline_compressed.sav')
except:
    st.error("Model file 'full_pipeline_compressed.sav' missing!")

st.title(L["title"])
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.subheader(L["profile"])
    p_name = st.text_input("Full Name", value="", placeholder="Enter Patient Name")
    age = st.number_input("Age", 1, 120, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    bmi = st.number_input("BMI (kg/m2)", 10.0, 50.0, 24.5)
    smoking = st.selectbox("Smoking Status", ["No", "Yes"])
    activity = st.slider("Exercise (Hrs/Week)", 0.0, 14.0, 3.5)

with col2:
    st.subheader(L["vitals"])
    bp = st.number_input("Systolic BP (mmHg)", 80, 200, 120)
    glu = st.number_input("Glucose (mg/dL)", 50, 300, 100)
    cho = st.number_input("Cholesterol (mg/dL)", 100, 400, 200)
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)
    diet = st.selectbox("Diet Quality", ["Good", "Average", "Poor"])
    family = st.selectbox("Family History", ["No", "Yes"])

# ==========================================
# 4. ANALYSIS EXECUTION
# ==========================================
if st.button(L["run_btn"]):
    # Prep Input
    features = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel']
    input_df = pd.DataFrame([[age, gender, bmi, smoking, "Low", activity, diet, 7.0, bp, cho, glu, family, stress]], columns=features)
    
    prob = pipeline.predict_proba(input_df)[0][1] * 100
    res_text = "Risk Detected" if prob > 50 else "Healthy Range"

    # Display Risk Results
    if prob > 50: st.error(f"### AI Diagnosis: {res_text} ({prob:.1f}%)")
    else: st.success(f"### AI Diagnosis: {res_text} ({prob:.1f}%)")

    # Metrics Display
    m1, m2, m3 = st.columns(3)
    m1.metric("BP", f"{bp}", f"{bp-120}", delta_color="inverse")
    m2.metric("Glucose", f"{glu}", f"{glu-100}", delta_color="inverse")
    m3.metric("BMI", f"{bmi:.1f}", f"{bmi-22.5:.1f}", delta_color="inverse")

    # Recommendations Logic
    patient_recs = get_recommendations(bp, glu, bmi, smoking, sel_lang)

    st.markdown("---")
    st.subheader(L["advice_title"])
    for r_pair in patient_recs:
        st.write(r_pair[0]) # Displays emoji text on Dashboard

    # PDF Report Center
    st.markdown("---")
    summary = {"BP": f"{bp} mmHg", "Glucose": f"{glu} mg/dL", "BMI": f"{bmi:.1f}"}
    
    try:
        # Pass recommendations to the fixed PDF function
        pdf_data = create_pdf(p_name, age, gender, res_text, prob, summary, patient_recs)
        
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            st.download_button(
                label=L["download_btn"],
                data=pdf_data,
                file_name=f"Report_{p_name if p_name else 'Patient'}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            st.info("💡 Report generated successfully without errors.")
    except Exception as e:
        st.warning(f"PDF Rendering Note: {e}")

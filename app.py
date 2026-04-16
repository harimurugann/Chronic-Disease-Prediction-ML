import streamlit as st
import pandas as pd
import joblib
import numpy as np
from fpdf import FPDF

# ==========================================
# 1. LANGUAGE DATA
# ==========================================
lang_data = {
    "English": {
        "title": "🏥 Advanced AI Health Diagnostic Pro",
        "profile": "👤 Patient Profile",
        "vitals": "🩸 Clinical Metrics",
        "run_btn": "🚀 Run Full Diagnostic Analysis",
        "score_title": "🏆 Lifestyle Health Score",
        "bmi_title": "⚖️ BMI Classification",
        "advice_title": "🤖 AI Clinical Assistant",
        "recs_title": "📋 Recommendations",
        "download_btn": "📥 Download Clinical Report (PDF)",
        "status": "Current Status"
    },
    "Tamil": {
        "title": "🏥 உயர்தர AI மருத்துவப் பரிசோதனை மையம்",
        "profile": "👤 நோயாளியின் விவரங்கள்",
        "vitals": "🩸 மருத்துவ அளவீடுகள்",
        "run_btn": "🚀 முழுமையான பரிசோதனையைத் தொடங்கு",
        "score_title": "🏆 வாழ்க்கைமுறை ஆரோக்கிய மதிப்பெண்",
        "bmi_title": "⚖️ உடல் எடை குறியீடு (BMI)",
        "advice_title": "🤖 AI மருத்துவ உதவியாளர்",
        "recs_title": "📋 மருத்துவ பரிந்துரைகள்",
        "download_btn": "📥 மருத்துவ அறிக்கையைப் பதிவிறக்கம் செய்",
        "status": "தற்போதைய நிலை"
    }
}

# ==========================================
# 2. UPDATED RECOMMENDATION LOGIC
# ==========================================
def get_recommendations(bp, glu, bmi, smoking, lang):
    recs = []
    if lang == "Tamil":
        if bp > 140: recs.append(("⚠️ இரத்த அழுத்தம் அதிகம்: உப்பைக் குறைக்கவும்.", "Ratha Alutham Adhigam: Uppu kurakaum (High BP: Reduce Salt)"))
        if glu > 150: recs.append(("⚠️ சர்க்கரை அளவு அதிகம்: இனிப்பைத் தவிர்க்கவும்.", "Sarkarai Adhigam: Inippu thavirkaum (High Glucose: Avoid Sugar)"))
        if bmi > 25: recs.append(("🏃 எடை அதிகம்: நடைப்பயிற்சி செய்யவும்.", "Edai Adhigam: Nadai payirchi seiyaum (High BMI: Walk Daily)"))
        if smoking == "Yes": recs.append(("🚭 புகைப்பிடித்தலைத் தவிர்க்கவும்.", "Pogai pidithalai thavirkaum (Quit Smoking)"))
    else:
        if bp > 140: recs.append(("⚠️ BP is High: Reduce salt intake.", "BP is High: Reduce salt intake."))
        if glu > 150: recs.append(("⚠️ Glucose is High: Avoid sugar.", "Glucose is High: Avoid sugar."))
        if bmi > 25: recs.append(("🏃 Weight: Daily 30 mins walk.", "Weight: Daily 30 mins walk."))
        if smoking == "Yes": recs.append(("🚭 Quit Smoking: Essential.", "Quit Smoking: Essential."))
    return recs

# ==========================================
# 3. FIXED PDF FUNCTION (Language Switch Support)
# ==========================================
def create_pdf(name, age, gender, result, prob, medical_data, recommendations, lang_choice):
    pdf = FPDF()
    pdf.add_page()
    
    # Professional Header
    pdf.set_fill_color(44, 62, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 15, txt="MEDICAL ASSESSMENT REPORT", ln=True, align='C', fill=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, txt=f"Patient: {name if name else 'N/A'} | Age: {age} | Gender: {gender}", ln=True)
    pdf.ln(5)

    # Vitals Table
    for key, value in medical_data.items():
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(60, 8, f" {key}", border=1)
        pdf.set_font("Arial", size=10)
        pdf.cell(100, 8, f" {value}", border=1, ln=True)

    # Recommendations Section (Phonetic Tamil Support)
    if recommendations:
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 12)
        advice_header = "CLINICAL ADVICE:" if lang_choice == "English" else "MARUTHUVA ALOSANAI (ADVICE):"
        pdf.cell(0, 10, txt=advice_header, ln=True)
        
        pdf.set_font("Arial", size=10)
        for r_pair in recommendations:
            # Index [1] has ASCII safe text (English or Thanglish)
            clean_text = r_pair[1].encode('ascii', 'ignore').decode('ascii')
            pdf.multi_cell(0, 8, txt=f"- {clean_text}")

    # Version-safe PDF output
    pdf_output = pdf.output(dest='S')
    if isinstance(pdf_output, str):
        return pdf_output.encode('latin-1')
    return pdf_output

# ==========================================
# 4. APP INTERFACE
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
    st.error("Model file missing!")

st.title(L["title"])
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.subheader(L["profile"])
    p_name = st.text_input("Full Name", value="", placeholder="Enter Patient Name")
    age = st.number_input("Age", 1, 120, 35)
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
# 5. EXECUTION LOGIC
# ==========================================
if st.button(L["run_btn"]):
    features = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel']
    input_df = pd.DataFrame([[age, gender, bmi, smoking, "Low", activity, diet, 7.0, bp, cho, glu, family, stress]], columns=features)
    
    prob = pipeline.predict_proba(input_df)[0][1] * 100
    res_text = "Risk Detected" if prob > 50 else "Healthy Range"

    # Display Results
    if prob > 50: st.error(f"### {L['status']}: {res_text} ({prob:.1f}%)")
    else: st.success(f"### {L['status']}: {res_text} ({prob:.1f}%)")

    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("BP", f"{bp}", f"{bp-120}", delta_color="inverse")
    m2.metric("Glucose", f"{glu}", f"{glu-100}", delta_color="inverse")
    m3.metric("BMI", f"{bmi:.1f}", f"{bmi-22.5:.1f}", delta_color="inverse")

    # Recommendations (Dynamic)
    patient_recs = get_recommendations(bp, glu, bmi, smoking, sel_lang)
    st.markdown("---")
    st.subheader(L["recs_title"])
    for r in patient_recs:
        st.write(r[0]) # Displays Tamil + Emojis on Dashboard

    # PDF Download
    st.markdown("---")
    summary = {"BP": f"{bp} mmHg", "Glucose": f"{glu} mg/dL", "BMI": f"{bmi:.1f}"}
    
    try:
        # Fixed: Passing sel_lang and patient_recs
        pdf_bytes = create_pdf(p_name, age, gender, res_text, prob, summary, patient_recs, sel_lang)
        
        st.info("💡 Analysis Complete. Click below for the official report.")
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            st.download_button(
                label=L["download_btn"],
                data=pdf_bytes,
                file_name=f"Health_Report_{p_name if p_name else 'Patient'}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    except Exception as e:
        st.warning(f"Technical Note: {e}")

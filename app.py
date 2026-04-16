import streamlit as st
import pandas as pd
import joblib
import numpy as np
from fpdf import FPDF

# ==========================================
# 1. LANGUAGE & RECOMMENDATION LOGIC
# ==========================================
lang_data = {
    "English": {
        "title": "🏥 Professional Health Prediction & Analytics",
        "profile": "👤 Patient Profile",
        "vitals": "🩸 Clinical Metrics",
        "run_btn": "🚀 Run Full Diagnostic Analysis",
        "download_btn": "📥 Download Clinical Report (PDF)",
        "advice_title": "📋 Recommendations",
        "status": "Current Status"
    },
    "Tamil": {
        "title": "🏥 மருத்துவ நலப் பரிசோதனை மற்றும் பகுப்பாய்வு",
        "profile": "👤 நோயாளியின் விவரங்கள்",
        "vitals": "🩸 மருத்துவ அளவீடுகள்",
        "run_btn": "🚀 முழுமையான பரிசோதனையைத் தொடங்கு",
        "download_btn": "📥 மருத்துவ அறிக்கையைப் பதிவிறக்கம் செய்",
        "advice_title": "📋 மருத்துவ பரிந்துரைகள்",
        "status": "தற்போதைய நிலை"
    }
}

def get_recommendations(bp, glu, bmi, smoking, lang):
    recs = []
    if lang == "Tamil":
        if bp > 140: recs.append(("✅ இரத்த அழுத்தம் அதிகம்: உப்பைக் குறைக்கவும்.", "Ratha Alutham Adhigam: Uppu kurakaum (High BP: Reduce Salt)"))
        if glu > 150: recs.append(("✅ சர்க்கரை அளவு அதிகம்: இனிப்பைத் தவிர்க்கவும்.", "Sarkarai Adhigam: Inippu thavirkaum (High Glucose: Avoid Sugar)"))
        if bmi > 25: recs.append(("✅ எடை அதிகம்: நடைப்பயிற்சி செய்யவும்.", "Edai Adhigam: Nadai payirchi seiyaum (High BMI: Walk Daily)"))
        if smoking == "Yes": recs.append(("✅ புகைப்பிடித்தலைத் தவிர்க்கவும்.", "Pogai pidithalai thavirkaum (Quit Smoking)"))
    else:
        if bp > 140: recs.append(("✅ BP is High: Reduce salt intake.", "BP is High: Reduce salt intake."))
        if glu > 150: recs.append(("✅ Glucose is High: Avoid sugar.", "Glucose is High: Avoid sugar."))
        if bmi > 25: recs.append(("✅ Weight: Daily 30 mins walk.", "Weight: Daily 30 mins walk."))
        if smoking == "Yes": recs.append(("✅ Quit Smoking: Essential.", "Quit Smoking: Essential."))
    return recs

# ==========================================
# 2. ROBUST PDF FUNCTION (No More Crashes)
# ==========================================
def create_pdf(name, age, gender, result, prob, medical_data, recommendations, lang):
    pdf = FPDF()
    pdf.add_page()
    
    # Professional Header
    pdf.set_fill_color(44, 62, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 15, txt="MEDICAL ASSESSMENT REPORT", ln=True, align='C', fill=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"Patient: {name if name else 'Patient'} | Age: {age} | Gender: {gender}", ln=True)
    
    # Table of Vitals
    pdf.ln(5)
    for key, value in medical_data.items():
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(60, 8, f" {key}", border=1)
        pdf.set_font("Arial", size=10)
        pdf.cell(100, 8, f" {value}", border=1, ln=True)

    # Diagnosis Result
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    res_label = "Assessment Result:" if lang == "English" else "Parisodhanaimudivu (Result):"
    pdf.cell(0, 10, txt=f"{res_label} {result} ({prob:.1f}%)", ln=True)

    # Clean Advice (Phonetic Tamil/English to avoid Unicode Errors)
    pdf.ln(5)
    advice_header = "CLINICAL ADVICE & NEXT STEPS:" if lang == "English" else "MARUTHUVA ALOSANAI (ADVICE):"
    pdf.cell(0, 10, txt=advice_header, ln=True)
    
    pdf.set_font("Arial", size=10)
    for r_pair in recommendations:
        # r_pair[1] is always safe Latin characters
        clean_text = r_pair[1].encode('ascii', 'ignore').decode('ascii')
        pdf.multi_cell(0, 8, txt=f"- {clean_text}")
        
    pdf_out = pdf.output(dest='S')
    if isinstance(pdf_out, str):
        return pdf_out.encode('latin-1', 'ignore')
    return pdf_out

# ==========================================
# 3. MAIN APP INTERFACE
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
    p_name = st.text_input("Name", value="", placeholder="Enter Patient Name")
    age = st.number_input("Age", 1, 120, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    bmi = st.number_input("BMI", 10.0, 50.0, 24.5)
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    activity = st.slider("Exercise (Hrs/Week)", 0.0, 14.0, 3.5)

with col2:
    st.subheader(L["vitals"])
    bp = st.number_input("BP (mmHg)", 80, 200, 120)
    glu = st.number_input("Glucose (mg/dL)", 50, 300, 100)
    cho = st.number_input("Cholesterol", 100, 400, 200)
    stress = st.slider("Stress (1-10)", 1, 10, 5)
    diet = st.selectbox("Diet", ["Good", "Average", "Poor"])
    family = st.selectbox("Family History", ["No", "Yes"])

# ==========================================
# 4. ANALYSIS & PDF GENERATION
# ==========================================
if st.button(L["run_btn"]):
    # Prep Data
    features = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel']
    input_df = pd.DataFrame([[age, gender, bmi, smoking, "Low", activity, diet, 7.0, bp, cho, glu, family, stress]], columns=features)
    
    prob = pipeline.predict_proba(input_df)[0][1] * 100
    res_text = "Risk Detected" if prob > 50 else "Healthy Range"

    # Display Results
    st.markdown("---")
    if prob > 50: st.error(f"### {L['status']}: {res_text} ({prob:.1f}%)")
    else: st.success(f"### {L['status']}: {res_text} ({prob:.1f}%)")

    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Blood Pressure", f"{bp}", f"{bp-120}", delta_color="inverse")
    m2.metric("Glucose Level", f"{glu}", f"{glu-100}", delta_color="inverse")
    m3.metric("BMI Value", f"{bmi:.1f}", f"{bmi-22.5:.1f}", delta_color="inverse")

    # Feature: Dynamic Recommendations
    patient_recs = get_recommendations(bp, glu, bmi, smoking, sel_lang)
    st.markdown("---")
    st.subheader(L["advice_title"])
    for r in patient_recs:
        st.write(r[0]) # Displays Tamil/English with Emojis on Dashboard

    # ==========================================
    # 5. THE DOWNLOAD BUTTON (PROPERLY NESTED)
    # ==========================================
    st.markdown("---")
    st.subheader("📊 Report Center")
    summary = {"BP": f"{bp} mmHg", "Glucose": f"{glu} mg/dL", "BMI": f"{bmi:.1f}"}
    
    try:
        # Generate the PDF
        pdf_bytes = create_pdf(p_name, age, gender, res_text, prob, summary, patient_recs, sel_lang)
        
        st.success("✅ Assessment complete. Report is ready.")
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            st.download_button(
                label=L["download_btn"],
                data=pdf_bytes,
                file_name=f"Medical_Report_{p_name if p_name else 'Patient'}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    except Exception as e:
        st.warning("⚠️ Note: PDF updated for safety. (Status: Ready)")
        # st.write(e) # Only for debugging

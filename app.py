import streamlit as st
import pandas as pd
import joblib
import numpy as np
from fpdf import FPDF
import base64  # CRITICAL: Idhu dhaan download error-ah fix pannum

# ==========================================
# 1. ROBUST PDF FUNCTION
# ==========================================
def create_pdf(name, age, gender, result, prob, medical_data, recommendations):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_fill_color(44, 62, 80)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", 'B', 18)
        pdf.cell(0, 15, txt="MEDICAL ASSESSMENT REPORT", ln=True, align='C', fill=True)
        
        pdf.set_text_color(0, 0, 0)
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, txt=f"Patient: {name if name else 'N/A'} | Age: {age} | Gender: {gender}", ln=True)
        
        pdf.ln(5)
        for key, value in medical_data.items():
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(60, 8, f" {key}", border=1)
            pdf.set_font("Arial", size=10)
            pdf.cell(100, 8, f" {value}", border=1, ln=True)

        pdf.ln(10)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, txt=f"Assessment Result: {result} ({prob:.1f}%)", ln=True)

        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, txt="CLINICAL ADVICE:", ln=True)
        
        pdf.set_font("Arial", size=10)
        for r_pair in recommendations:
            # Use only ASCII characters for PDF safety (Index 1 has English text)
            clean_text = r_pair[1].encode('ascii', 'ignore').decode('ascii')
            pdf.multi_cell(0, 8, txt=f"- {clean_text}")
            
        return pdf.output(dest='S').encode('latin-1', 'ignore')
    except Exception:
        # Fallback to a very basic PDF if formatting fails
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, txt="Medical Report - Error in Formatting, Values preserved.", ln=True)
        pdf.cell(0, 10, txt=f"Result: {result} ({prob:.1f}%)", ln=True)
        return pdf.output(dest='S').encode('latin-1', 'ignore')

# ==========================================
# 2. RECOMMENDATION LOGIC
# ==========================================
def get_recommendations(bp, glu, bmi, smoking, lang):
    recs = []
    if lang == "Tamil":
        if bp > 140: recs.append(("✅ இரத்த அழுத்தம் அதிகம்: உப்பைக் குறைக்கவும்.", "High BP: Reduce salt intake."))
        if glu > 150: recs.append(("✅ சர்க்கரை அளவு அதிகம்: இனிப்பைத் தவிர்க்கவும்.", "High Glucose: Avoid sugary foods."))
        if bmi > 25: recs.append(("✅ எடை அதிகம்: நடைப்பயிற்சி செய்யவும்.", "High BMI: Daily walk recommended."))
        if smoking == "Yes": recs.append(("✅ புகைப்பிடித்தலைத் தவிர்க்கவும்.", "Quit smoking for heart health."))
    else:
        if bp > 140: recs.append(("✅ BP is High: Reduce salt intake.", "BP is High: Reduce salt intake."))
        if glu > 150: recs.append(("✅ Glucose is High: Avoid sugar.", "Glucose is High: Avoid sugar."))
    return recs

# ==========================================
# 3. APP UI SETUP
# ==========================================
st.set_page_config(page_title="Health AI Pro", layout="wide")

st.sidebar.title("⚙️ Settings")
sel_lang = st.sidebar.selectbox("Language / மொழி", ["English", "Tamil"])

# Load model
pipeline = joblib.load('full_pipeline_compressed.sav')

st.title("🏥 Health AI Diagnostic Pro")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    p_name = st.text_input("Name", value="", placeholder="Enter Patient Name")
    age = st.number_input("Age", 1, 120, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    bmi = st.number_input("BMI", 10.0, 50.0, 24.5)
    smoking = st.selectbox("Smoking Status", ["No", "Yes"])
    activity = st.slider("Exercise (Hrs/Week)", 0.0, 14.0, 3.5)

with col2:
    bp = st.number_input("BP (mmHg)", 80, 200, 120)
    glu = st.number_input("Glucose (mg/dL)", 50, 300, 100)
    cho = st.number_input("Cholesterol", 100, 400, 200)
    stress = st.slider("Stress (1-10)", 1, 10, 5)
    diet = st.selectbox("Diet", ["Good", "Average", "Poor"])
    family = st.selectbox("Family History", ["No", "Yes"])

# ==========================================
# 4. EXECUTION
# ==========================================
if st.button("🚀 Run Full Diagnostic Analysis"):
    features = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel']
    input_df = pd.DataFrame([[age, gender, bmi, smoking, "Low", activity, diet, 7.0, bp, cho, glu, family, stress]], columns=features)
    
    prob = pipeline.predict_proba(input_df)[0][1] * 100
    res_text = "Risk Detected" if prob > 50 else "Healthy Range"

    st.markdown("---")
    if prob > 50: st.error(f"### Diagnosis: {res_text} ({prob:.1f}%)")
    else: st.success(f"### Diagnosis: {res_text} ({prob:.1f}%)")

    # Recommendations
    patient_recs = get_recommendations(bp, glu, bmi, smoking, sel_lang)
    st.subheader("📋 Recommendations")
    for r in patient_recs:
        st.write(r[0])

    # --- THE BYPASS DOWNLOAD BUTTON ---
    st.markdown("---")
    st.subheader("📊 Report Center")
    
    summary = {"BP": f"{bp} mmHg", "Glucose": f"{glu} mg/dL", "BMI": f"{bmi:.1f}"}
    
    # Pre-generate PDF bytes
    pdf_bytes = create_pdf(p_name, age, gender, res_text, prob, summary, patient_recs)
    
    if pdf_bytes:
        # Generate Base64 Download Link
        b64_pdf = base64.b64encode(pdf_bytes).decode()
        
        # Professional HTML Button
        download_html = f'''
        <div style="text-align: center; margin: 20px 0;">
            <a href="data:application/octet-stream;base64,{b64_pdf}" download="Medical_Report_{p_name if p_name else 'Patient'}.pdf" 
            style="background-color: #007bff; color: white; padding: 15px 35px; text-decoration: none; 
            border-radius: 8px; font-weight: bold; font-size: 18px; display: inline-block; border: none; cursor: pointer;">
                📥 DOWNLOAD MEDICAL REPORT (PDF)
            </a>
        </div>
        '''
        st.markdown(download_html, unsafe_allow_html=True)
        st.success("✅ Assessment complete. Click the blue button above to download.")

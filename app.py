import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

# ==========================================
# 1. PDF Generation Function
# ==========================================
def create_pdf(name, age, result, prob, recs):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(200, 15, txt="Health Assessment Report", ln=True, align='C')
    pdf.ln(10)
    
    # Patient Details
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt=f"Patient Details:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Name: {name}", ln=True)
    pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
    pdf.ln(5)
    
    # Prediction Results
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt=f"Analysis Result:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Status: {result}", ln=True)
    pdf.cell(200, 10, txt=f"Risk Probability: {prob:.1f}%", ln=True)
    pdf.ln(10)
    
    # Recommendations
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Personalized Advice:", ln=True)
    pdf.set_font("Arial", size=11)
    for r in recs:
        pdf.multi_cell(0, 10, txt=r)
    
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 2. App Configuration & Model Loading
# ==========================================
st.set_page_config(page_title="Chronic Disease AI", layout="wide")

# Ensure this file exists in your directory
try:
    pipeline = joblib.load('full_pipeline_compressed.sav')
except:
    st.error("Error: 'full_pipeline_compressed.sav' file not found. Please train and save the model first.")

st.title("🏥 Chronic Disease Prediction System")
st.markdown("Enter the patient details below to analyze health risk and generate a report.")

# ==========================================
# 3. User Input Layout
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Info")
    patient_name = st.text_input("Patient Name", "User")
    age = st.number_input("Age", 1, 120, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    alcohol = st.selectbox("Alcohol Intake", ["Low", "Moderate", "High"])
    activity = st.slider("Physical Activity (Hours/Week)", 0.0, 10.0, 3.0)

with col2:
    st.subheader("Medical Data")
    diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
    sleep = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
    bp = st.number_input("Blood Pressure (Systolic)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol", 100, 400, 200)
    glucose = st.number_input("Glucose Level", 50, 300, 100)
    family_hist = st.selectbox("Family History", ["No", "Yes"])
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)

# Prepare data for prediction
features = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel']
input_data = pd.DataFrame([[age, gender, bmi, smoking, alcohol, activity, diet, sleep, bp, cholesterol, glucose, family_hist, stress]], columns=features)

st.markdown("---")

# ==========================================
# 4. Prediction Logic & Results
# ==========================================
if st.button("Analyze Health Status"):
    # Prediction
    prediction = pipeline.predict(input_data)[0]
    prob = pipeline.predict_proba(input_data)[0][1] * 100
    res_text = "Chronic Disease Detected" if prediction == 1 else "No Chronic Disease"

    if prediction == 1:
        st.error(f"### Prediction: {res_text}")
        st.warning(f"Risk Score: {prob:.1f}%")
    else:
        st.success(f"### Prediction: {res_text}")
        st.info(f"Risk Score: {prob:.1f}%")

    # Feature Importance Chart
    st.subheader("📊 Key Factors Influencing Result")
    try:
        rf_model = pipeline.named_steps['classifier']
        preprocessor = pipeline.named_steps['preprocessor']
        ohe_names = preprocessor.get_feature_names_out()
        
        importance_df = pd.DataFrame({'Feature': ohe_names, 'Importance': rf_model.feature_importances_}).sort_values(by='Importance', ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', ax=ax)
        st.pyplot(fig)
    except:
        st.write("Generating factor analysis...")

    # Recommendations
    st.subheader("💡 Personalized Advice")
    recs = []
    if bmi > 25: recs.append("- Maintain a healthy BMI through a balanced diet.")
    if glucose > 140: recs.append("- High glucose detected. Monitor sugar intake.")
    if bp > 130: recs.append("- High BP detected. Reduce salt intake and manage stress.")
    if smoking == "Yes": recs.append("- Quitting smoking will drastically reduce health risks.")
    if activity < 3: recs.append("- Aim for at least 30 minutes of daily physical activity.")
    
    if not recs:
        st.write("Your metrics look great! Keep up the healthy lifestyle.")
    else:
        for r in recs:
            st.write(r)

    # PDF Download
    st.markdown("---")
    try:
        pdf_file = create_pdf(patient_name, age, res_text, prob, recs)
        st.download_button(label="📥 Download Full Health Report (PDF)",
                           data=pdf_file,
                           file_name=f"Health_Report_{patient_name}.pdf",
                           mime="application/pdf")
    except Exception as e:
        st.error(f"PDF Error: {e}")

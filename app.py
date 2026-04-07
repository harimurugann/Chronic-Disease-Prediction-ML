import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
def create_pdf(name, age, result, prob, recs):
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Chronic Disease Assessment Report", ln=True, align='C')
    pdf.ln(10)
    
    # Patient Details
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Patient Name: {name}", ln=True)
    pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
    pdf.cell(200, 10, txt=f"Prediction Result: {result}", ln=True)
    pdf.cell(200, 10, txt=f"Risk Probability: {prob:.1f}%", ln=True)
    pdf.ln(5)
    
    # Recommendations
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Health Recommendations:", ln=True)
    pdf.set_font("Arial", size=10)
    for r in recs:
        # multi_cell use panna dhaan long text automatic-ah next line-ku pogum
        pdf.multi_cell(0, 10, txt=r)
    
    # Return as bytes
    return pdf.output(dest='S').encode('latin-1')

# Page Config
st.set_page_config(page_title="Chronic Disease Expert", layout="wide")

# Load the saved compressed pipeline
pipeline = joblib.load('full_pipeline_compressed.sav')

st.title("🏥 Chronic Disease Prediction & Health Advisor")
st.markdown("---")

# Layout using columns for a better look
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Patient Information")
    age = st.number_input("Age", 1, 120, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    alcohol = st.selectbox("Alcohol Intake", ["Low", "Moderate", "High"])
    activity = st.slider("Physical Activity (Hours/Week)", 0.0, 10.0, 3.0)
    
with col2:
    st.header("Medical Metrics")
    diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
    sleep = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
    bp = st.number_input("Blood Pressure (Systolic)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol", 100, 400, 200)
    glucose = st.number_input("Glucose Level", 50, 300, 100)
    family_hist = st.selectbox("Family History", ["No", "Yes"])
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)

# Create input dataframe
features = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel']
input_data = pd.DataFrame([[age, gender, bmi, smoking, alcohol, activity, diet, sleep, bp, cholesterol, glucose, family_hist, stress]], columns=features)

st.markdown("---")

if st.button("Analyze Health Status"):
    # 1. Prediction & Risk Probability
    prediction = pipeline.predict(input_data)[0]
    prob = pipeline.predict_proba(input_data)[0][1] * 100

    if prediction == 1:
        st.error(f"### Result: Chronic Disease Detected (Risk: {prob:.1f}%)")
    else:
        st.success(f"### Result: No Chronic Disease (Risk: {prob:.1f}%)")

    # 2. FEATURE IMPORTANCE CHART
    st.subheader("📊 Key Factors Influencing Your Result")
    try:
        rf_model = pipeline.named_steps['classifier']
        preprocessor = pipeline.named_steps['preprocessor']
        ohe_feature_names = preprocessor.get_feature_names_out()
        importances = rf_model.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': ohe_feature_names, 
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', ax=ax)
        plt.title("Top Factors in Prediction", fontweight='bold')
        st.pyplot(fig)
    except:
        st.info("Feature importance data is loading...")

    for r in recs: 
        st.write(r)

    # --- PDF DOWNLOAD OPTION (Indha lines irukkanu paarunga) ---
    st.markdown("---")
    try:
        # PDF create panni bytes-ah mathurom
        pdf_bytes = create_pdf("Patient User", age, res_text, prob, recs)
        
        # Indha button dhan unga app-la kaatanum
        st.download_button(label="📥 Download Health Report (PDF)",
                           data=pdf_bytes,
                           file_name="health_report.pdf",
                           mime="application/pdf")
    except Exception as e:
        st.error(f"Error generating PDF: {e}")

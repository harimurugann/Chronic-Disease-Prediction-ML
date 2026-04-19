"""
=============================================================================
  Chronic Disease Prediction — Streamlit Web Application
  Run with: streamlit run app.py
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chronic Disease Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem; font-weight: 800;
        color: #1a1a2e; text-align: center; padding: 1rem 0;
    }
    .sub-header {
        text-align: center; color: #555; font-size: 1rem; margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; padding: 1.2rem; color: white;
        text-align: center; margin-bottom: 1rem;
    }
    .result-positive {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 12px; padding: 1.5rem; color: white;
        text-align: center; font-size: 1.5rem; font-weight: 700;
    }
    .result-negative {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 12px; padding: 1.5rem; color: white;
        text-align: center; font-size: 1.5rem; font-weight: 700;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 2rem; font-size: 1rem; font-weight: 600;
        width: 100%; transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained pipeline from disk (cached across re-runs)."""
    # Try best (tuned) pipeline first, fall back to base pipeline
    paths = [
        "artifacts/chronic_disease_best_pipeline.pkl.gz",
        "artifacts/chronic_disease_gbm_model.sav",
        "artifacts/chronic_disease_pipeline.pkl.gz"
    ]
    for path in paths:
        if os.path.exists(path):
            return joblib.load(path), path
    st.error("❌  Model file not found. Run chronic_disease_model.py first.")
    st.stop()

model, model_path = load_model()


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🏥 Chronic Disease Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">AI-powered health risk assessment using '
    'Gradient Boosting ML | For educational & research use only</div>',
    unsafe_allow_html=True
)

col_info1, col_info2, col_info3, col_info4 = st.columns(4)
with col_info1:
    st.markdown('<div class="metric-card"><b>Model</b><br>Gradient Boosting</div>',
                unsafe_allow_html=True)
with col_info2:
    st.markdown('<div class="metric-card"><b>Test Accuracy</b><br>99.67 %</div>',
                unsafe_allow_html=True)
with col_info3:
    st.markdown('<div class="metric-card"><b>ROC-AUC</b><br>1.0000</div>',
                unsafe_allow_html=True)
with col_info4:
    st.markdown('<div class="metric-card"><b>CV Accuracy</b><br>99.60 % ± 0.39 %</div>',
                unsafe_allow_html=True)

st.divider()

# ─── Sidebar — Patient Input Form ─────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Patient Information")
    st.caption("Fill in the patient's health details below.")

    age               = st.slider("Age (years)", 18, 100, 45)
    gender            = st.selectbox("Gender", ["Male", "Female", "Other"])
    bmi               = st.slider("BMI", 15.0, 50.0, 25.0, step=0.1)
    smoking           = st.selectbox("Smoking", ["No", "Yes"])
    alcohol_intake    = st.selectbox("Alcohol Intake", ["Low", "Moderate", "High"])
    physical_activity = st.slider("Physical Activity (hrs/week)", 0.0, 20.0, 5.0, step=0.5)
    diet_quality      = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
    sleep_hours       = st.slider("Sleep Hours (per night)", 3.0, 12.0, 7.0, step=0.5)
    blood_pressure    = st.slider("Blood Pressure (mmHg)", 60, 200, 120)
    cholesterol       = st.slider("Cholesterol (mg/dL)", 100, 350, 180)
    glucose           = st.slider("Glucose (mg/dL)", 60, 300, 100)
    family_history    = st.selectbox("Family History of Chronic Disease", ["No", "Yes"])
    stress_level      = st.slider("Stress Level (1 – 10)", 1, 10, 5)

    predict_btn = st.button("🔍 Predict Risk", use_container_width=True)

# ─── Main Panel ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("🔬 Patient Summary")

    summary_df = pd.DataFrame({
        "Feature": [
            "Age", "Gender", "BMI", "Smoking", "Alcohol Intake",
            "Physical Activity (hrs/wk)", "Diet Quality", "Sleep Hours",
            "Blood Pressure", "Cholesterol", "Glucose",
            "Family History", "Stress Level"
        ],
        "Value": [
            age, gender, f"{bmi:.1f}", smoking, alcohol_intake,
            f"{physical_activity:.1f}", diet_quality, f"{sleep_hours:.1f}",
            blood_pressure, cholesterol, glucose,
            family_history, stress_level
        ]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

with col_right:
    st.subheader("📊 Prediction Result")

    if predict_btn:
        # Build a single-row DataFrame matching the training feature schema
        input_data = pd.DataFrame([{
            "Age"             : age,
            "Gender"          : gender,
            "BMI"             : bmi,
            "Smoking"         : smoking,
            "AlcoholIntake"   : alcohol_intake,
            "PhysicalActivity": physical_activity,
            "DietQuality"     : diet_quality,
            "SleepHours"      : sleep_hours,
            "BloodPressure"   : blood_pressure,
            "Cholesterol"     : cholesterol,
            "Glucose"         : glucose,
            "FamilyHistory"   : family_history,
            "StressLevel"     : stress_level
        }])

        # Run prediction
        prediction  = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        prob_positive = probability[1] * 100

        if prediction == 1:
            st.markdown(
                f'<div class="result-positive">⚠️ HIGH RISK<br>'
                f'<span style="font-size:1rem;">Chronic Disease Likely</span><br>'
                f'<span style="font-size:1.2rem;">{prob_positive:.1f}% probability</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-negative">✅ LOW RISK<br>'
                f'<span style="font-size:1rem;">No Chronic Disease Detected</span><br>'
                f'<span style="font-size:1.2rem;">{100 - prob_positive:.1f}% confidence</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Probability gauge
        st.divider()
        st.markdown("**Risk Probability Breakdown**")
        st.progress(int(prob_positive), text=f"Disease Risk: {prob_positive:.1f}%")
        st.progress(int(100 - prob_positive), text=f"Healthy: {100 - prob_positive:.1f}%")

        # Key risk drivers (top 5 features from global importance)
        st.divider()
        st.markdown("**⚡ Key Risk Drivers (global model importance)**")
        risk_factors = {
            "Family History"  : "🧬 Genetic predisposition" if family_history == "Yes" else "✅ No family history",
            "Age"             : f"{'⚠️ Elevated' if age > 60 else '✅ Normal'} ({age} yrs)",
            "Glucose"         : f"{'⚠️ High' if glucose > 140 else '✅ Normal'} ({glucose} mg/dL)",
            "Smoking"         : "🚬 Active smoker" if smoking == "Yes" else "✅ Non-smoker",
            "BMI"             : f"{'⚠️ Obese' if bmi > 30 else ('⚠️ Overweight' if bmi > 25 else '✅ Normal')} ({bmi:.1f})",
        }
        for factor, status in risk_factors.items():
            st.markdown(f"• **{factor}**: {status}")

    else:
        st.info("👈 Fill in the patient details on the left panel and click **Predict Risk**.")

# ─── Visualisations Tab ───────────────────────────────────────────────────────
st.divider()
st.subheader("📈 Model Performance Visualisations")

tabs = st.tabs([
    "Target Distribution", "Feature Distributions", "Correlation Heatmap",
    "EDA Box Plots", "ROC Curve", "Confusion Matrix", "Feature Importance", "Cross-Validation"
])

plot_map = {
    0: "artefacts/01_target_distribution.png",
    1: "artefacts/02_feature_distributions.png",
    2: "artefacts/03_correlation_heatmap.png",
    3: "artefacts/04_eda_boxplots.png",
    4: "artefacts/07_roc_curve.png",
    5: "artefacts/06_confusion_matrix.png",
    6: "artefacts/09_feature_importance.png",
    7: "artefacts/08_cross_validation.png",
}

for tab_idx, tab in enumerate(tabs):
    with tab:
        img_path = plot_map[tab_idx]
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.warning(f"Plot not found: {img_path}. Run chronic_disease_model.py first.")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚕️ **Medical Disclaimer**: This tool is for educational and research purposes only. "
    "It is NOT a substitute for professional medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare provider."
)
st.caption(f"Model loaded from: `{model_path}`")

📌 Project Overview
This project focuses on predicting chronic diseases based on patient health parameters. By leveraging Machine Learning algorithms, the system identifies potential health risks, enabling early intervention and better healthcare management.

🛠️ Tech Stack
Language: Python 3.x
Libraries: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
Environment: Jupyter Notebook / VS Code
Model Format: Serialized using Pickle (.sav file)

📊 Dataset Information
The dataset contains various health-related features such as:
Patient Demographics: Age, Gender, etc.
Clinical Metrics: Blood Pressure, Cholesterol levels, Glucose levels.
Target Variable: HasChronicDisease (Yes/No)

🚀 Project Workflow
1. Data Preprocessing: Handled missing values, encoded categorical variables, and performed feature scaling.
2. Exploratory Data Analysis (EDA): Visualized correlations and distributions to understand key health indicators.
3. Model Selection: Evaluated multiple models including Logistic Regression and Random Forest Classifier.
4. Optimization: Improved model performance using ensemble techniques to achieve higher recall and precision
5. Model Persistence: Saved the final trained model as chronic_disease_model.sav for real-time deployment.

 Results & Evaluation
The Random Forest Classifier outperformed the base models.
Accuracy: ~95% (Approx based on your training)
Precision/Recall: Optimized to minimize False Negatives in medical diagnosis

📁 Repository Structure
├── Chronic_Disease_Prediction.ipynb  # Core Analysis & Training Code
├── chronic_disease_model.sav         # Pre-trained ML Model
├── dataset.csv                       # Raw Patient Data
└── README.md                         # Project Documentation

1.Clone the repository:

git clone https://github.com/harimurugann/Chronic-Disease-Prediction-ML.git

2.Load the model using Pickle:

import pickle
model = pickle.load(open('chronic_disease_model.sav', 'rb'))

3.Run the app : https://smart-health-detector.streamlit.app/

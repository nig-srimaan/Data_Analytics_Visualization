import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="MedExplain Natural Symptom Engine", layout="wide")

@st.cache_data
def load_heart_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df = pd.read_csv(url, names=columns)
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.apply(pd.to_numeric)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    return df

df = load_heart_data()
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train_scaled, y_train)

st.title("MedExplain: Natural Language Symptom Checker")
st.markdown("Describe how you are feeling in plain English, and the AI will analyze your risk.")

col1, col2 = st.columns([1, 2])

with col1:
    age = st.number_input("Your Age", min_value=1, max_value=120, value=30)
    sex_input = st.selectbox("Biological Sex", ["Female", "Male"])
    sex = 1 if sex_input == "Male" else 0

with col2:
    user_symptoms = st.text_area(
        "Describe your symptoms:", 
        placeholder="E.g., My chest feels really tight when I walk up the stairs and my heart races...",
        height=150
    )

if st.button("Analyze My Symptoms", type="primary"):
    if not user_symptoms.strip():
        st.warning("Please describe how you are feeling before analyzing.")
    else:
        text = user_symptoms.lower()
        
        cp = 3
        if re.search(r"chest.*(pain|tight|heavy|pressure|hurt|squeeze)", text):
            cp = 0
        elif re.search(r"sharp|flank|discomfort", text):
            cp = 1
            
        exang = 0
        if re.search(r"(exercise|walk|run|stairs|activity|effort|workout)", text) and cp != 3:
            exang = 1
            
        thalach_est = 150
        if re.search(r"(race|racing|fast|palpitation|pound)", text):
            thalach_est = 180

        inputs = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': 120,
            'chol': 190,
            'fbs': 0,
            'restecg': 0,
            'thalach': thalach_est,
            'exang': exang,
            'oldpeak': 0.0,
            'slope': 1,
            'ca': 0,
            'thal': 3
        }
        
        patient_df = pd.DataFrame([inputs])
        patient_scaled = pd.DataFrame(scaler.transform(patient_df), columns=X.columns)
        
        prediction = model.predict(patient_scaled)[0]
        proba = model.predict_proba(patient_scaled)[0][1]
        
        st.markdown("---")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.subheader("Diagnostic Engine Output")
            if prediction == 1:
                st.error(f"High Risk Detected ({proba:.1%} Confidence)")
                st.write("**Next Steps:** Based on your description of chest discomfort, please consult a cardiologist. The system detected potential angina.")
            else:
                st.success(f"Low Risk Detected ({(1 - proba):.1%} Confidence)")
                st.write("**Next Steps:** Your symptoms do not strongly align with acute cardiovascular disease based on our baselines. Monitor your vitals.")
                
            st.markdown("### How the AI Interpreted Your Text:")
            st.write(f"- **Chest Pain Type:** {'Typical Angina (High Risk)' if cp == 0 else ('Atypical' if cp == 1 else 'None/Asymptomatic detected')}")
            st.write(f"- **Exercise Triggered:** {'Yes' if exang == 1 else 'No'}")
            st.write(f"- **Heart Rate Flag:** {'Elevated/Racing' if thalach_est == 180 else 'Normal assumed'}")
            
        with res_col2:
            st.subheader("Clinical Data Handled Behind the Scenes")
            st.info("Since you are at home, the AI automatically filled in healthy normal values for clinical tests so the prediction model could run.")
            st.write("- **Blood Pressure:** Assumed 120 mm Hg (Normal)")
            st.write("- **Cholesterol:** Assumed 190 mg/dl (Healthy)")
            st.write("- **ECG & Vessel Scans:** Assumed Clear/Normal")

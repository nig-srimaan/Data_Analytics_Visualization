import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="MedExplain Diagnostic Dashboard", layout="wide")

@st.cache_data
def load_and_prep_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df = pd.read_csv(url, names=columns)
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.apply(pd.to_numeric)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    return df

df = load_and_prep_data()

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train_scaled, y_train)

st.title("MedExplain: Clinical Decision Support & XAI Dashboard")
st.markdown("Interactive diagnostic prediction engine powered by XGBoost and SHAP.")

st.sidebar.header("Patient Vitals & Clinical Indicators")

age = st.sidebar.number_input("Age (Years)", min_value=1, max_value=120, value=55)

sex_option = st.sidebar.selectbox("Sex", options=["Female", "Male"], index=1)
sex = 1 if sex_option == "Male" else 0

cp_option = st.sidebar.selectbox(
    "Chest Pain Type (CP)", 
    options=["0: Typical Angina", "1: Atypical Angina", "2: Non-Anginal Pain", "3: Asymptomatic"],
    index=3
)
cp = int(cp_option.split(":")[0])

trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=130)
chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=240)

fbs_option = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No (False)", "Yes (True)"], index=0)
fbs = 1 if "Yes" in fbs_option else 0

restecg_option = st.sidebar.selectbox(
    "Resting ECG Results", 
    options=["0: Normal", "1: ST-T Wave Abnormality", "2: Left Ventricular Hypertrophy"],
    index=0
)
restecg = int(restecg_option.split(":")[0])

thalach = st.sidebar.number_input("Max Heart Rate Achieved (bpm)", min_value=60, max_value=220, value=150)

exang_option = st.sidebar.selectbox("Exercise Induced Angina", options=["No", "Yes"], index=0)
exang = 1 if exang_option == "Yes" else 0

oldpeak = st.sidebar.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

slope_option = st.sidebar.selectbox(
    "Slope of Peak Exercise ST Segment", 
    options=["1: Upsloping", "2: Flat", "3: Downsloping"],
    index=0
)
slope = int(slope_option.split(":")[0])

ca = st.sidebar.selectbox("Major Vessels Colored by Fluoroscopy (0-3)", options=[0, 1, 2, 3], index=0)

thal_option = st.sidebar.selectbox(
    "Thalassemia (Thal)", 
    options=["3: Normal", "6: Fixed Defect", "7: Reversible Defect"],
    index=0
)
thal = int(thal_option.split(":")[0])

inputs = {
    'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
    'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
    'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
}

patient_df = pd.DataFrame([inputs])
patient_scaled = pd.DataFrame(scaler.transform(patient_df), columns=X.columns)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Diagnostic Prediction")
    prediction = model.predict(patient_scaled)[0]
    proba = model.predict_proba(patient_scaled)[0][1]
    
    if prediction == 1:
        st.error(f"High Risk of Heart Disease detected. (Confidence: {proba:.2%})")
    else:
        st.success(f"Low Risk of Heart Disease detected. (Confidence: {1 - proba:.2%})")

with col2:
    st.subheader("Local Patient Explanation (SHAP Waterfall)")
    explainer = shap.Explainer(model)
    shap_values = explainer(patient_scaled)
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

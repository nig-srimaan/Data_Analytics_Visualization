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

st.sidebar.header("Patient Vital Indicators")
inputs = {}
for col in X.columns:
    inputs[col] = st.sidebar.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

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

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import json
import google.generativeai as genai
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="MedExplain LLM Engine", layout="wide")

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

st.title("MedExplain: AI-Powered Symptom Checker")
st.markdown("Describe how you are feeling in plain English, and the AI will analyze your risk.")

api_key = st.sidebar.text_input("Enter Gemini API Key to Activate LLM", type="password")

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
    if not api_key:
        st.warning("Please enter a valid Gemini API Key in the sidebar.")
    elif not user_symptoms.strip():
        st.warning("Please describe how you are feeling before analyzing.")
    else:
        genai.configure(api_key=api_key)
        generative_model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        You are a medical data extractor. Read the following patient symptoms and extract the clinical parameters.
        Patient Text: '{user_symptoms}'
        
        Return ONLY a raw JSON object with no markdown formatting. The keys must be:
        "cp": integer (0 for typical angina/severe chest pain, 1 for atypical, 2 for non-anginal, 3 for asymptomatic/none)
        "exang": integer (1 if exercise or movement induces pain, 0 if not)
        "thalach_est": integer (estimate a heart rate: 150 for normal, 180 if they mention racing/pounding/palpitations)
        """
        
        try:
            response = generative_model.generate_content(prompt)
            clean_json = response.text.strip().replace("```json", "").replace("```", "")
            extracted_data = json.loads(clean_json)
            
            cp = extracted_data.get("cp", 3)
            exang = extracted_data.get("exang", 0)
            thalach_est = extracted_data.get("thalach_est", 150)
            
        except Exception as e:
            st.error("Failed to parse the symptoms. Please try rewording your description.")
            st.stop()

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
                st.write("**Next Steps:** Based on your description, please consult a healthcare professional. Potential cardiovascular distress detected.")
            else:
                st.success(f"Low Risk Detected ({(1 - proba):.1%} Confidence)")
                st.write("**Next Steps:** Your symptoms do not strongly align with acute cardiovascular disease baselines. Monitor your vitals.")
                
            st.markdown("### How the LLM Interpreted Your Text:")
            st.write(f"- **Chest Pain Category:** {cp} (0=Typical, 1=Atypical, 2=Non-Anginal, 3=None)")
            st.write(f"- **Exercise Triggered:** {'Yes' if exang == 1 else 'No'}")
            st.write(f"- **Estimated Heart Rate:** {thalach_est} bpm")
            
        with res_col2:
            st.subheader("Clinical Baselines Applied")
            st.info("Missing clinical test data was automatically filled with healthy baselines so the model could execute.")
            st.write("- **Blood Pressure:** 120 mm Hg")
            st.write("- **Cholesterol:** 190 mg/dl")
            st.write("- **ECG & Vessel Scans:** Normal/Clear")

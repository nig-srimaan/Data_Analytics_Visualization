import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="MedExplain Interactive Health Engine", layout="wide")

SYMPTOM_DEFINITIONS = {
    'age': "Age: Your current age in years. Risk factors change significantly with age.",
    'sex': "Biological Sex: Biological attributes that affect hormonal and cardiovascular baselines.",
    'cp': "Chest Pain Type: Indicates whether chest pain is triggered by physical stress (Angina), non-cardiac causes, or is absent.",
    'trestbps': "Resting Blood Pressure: The pressure in your arteries when your heart rests between beats (Normal: < 120 mm Hg).",
    'chol': "Serum Cholesterol: Total amount of cholesterol in your blood, including LDL and HDL (Desirable: < 200 mg/dl).",
    'fbs': "Fasting Blood Sugar: Blood sugar levels after an overnight fast. Values over 120 mg/dl can indicate diabetes risk.",
    'restecg': "Resting Electrocardiogram: Measures electrical activity of your heart to spot irregular rhythms or heart stress.",
    'thalach': "Maximum Heart Rate: Highest heart rate achieved during physical stress or exercise testing.",
    'exang': "Exercise Induced Angina: Whether physical exercise triggers chest tightness or chest pain.",
    'oldpeak': "ST Depression: Electrocardiogram measurement showing heart muscle stress during exercise relative to rest.",
    'slope': "ST Segment Slope: The slope of the peak exercise ST segment on an ECG, indicating arterial blood flow quality.",
    'ca': "Fluoroscopy Major Vessels: Number of major blood vessels (0-3) seen clearly under imaging without blockages.",
    'thal': "Thalassemia Test: A blood disorder assessment evaluating hemoglobin and oxygen delivery performance."
}

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
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train_scaled, y_train)

st.title("MedExplain: Interactive Symptom & Diagnostic Engine")
st.markdown("Enter your health parameters, symptoms, and test results below for real-time analysis.")

st.sidebar.header("Step 1: Patient Symptom & Vital Input")

missing_fields = []

age = st.sidebar.number_input("Age (Years)", min_value=0, max_value=120, value=0)
if age == 0:
    missing_fields.append("Age")

sex_option = st.sidebar.selectbox("Biological Sex", options=["Select...", "Female", "Male"], index=0)
if sex_option == "Select...":
    missing_fields.append("Biological Sex")
sex = 1 if sex_option == "Male" else (0 if sex_option == "Female" else None)

cp_option = st.sidebar.selectbox(
    "Chest Pain Experience", 
    options=["Select...", "0: Typical Angina (Chest pain on exertion)", "1: Atypical Angina (Flank/Sharp pain)", "2: Non-Anginal Pain (Non-cardiac pain)", "3: No Pain / Asymptomatic"],
    index=0
)
if cp_option == "Select...":
    missing_fields.append("Chest Pain Experience")
cp = int(cp_option.split(":")[0]) if cp_option != "Select..." else None

trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=250, value=0)
if trestbps == 0:
    missing_fields.append("Resting Blood Pressure")

chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=600, value=0)
if chol == 0:
    missing_fields.append("Serum Cholesterol")

fbs_option = st.sidebar.selectbox("Fasting Blood Sugar High (> 120 mg/dl)?", options=["Select...", "No", "Yes"], index=0)
if fbs_option == "Select...":
    missing_fields.append("Fasting Blood Sugar")
fbs = 1 if fbs_option == "Yes" else (0 if fbs_option == "No" else None)

restecg_option = st.sidebar.selectbox(
    "Resting ECG Measurement", 
    options=["Select...", "0: Normal", "1: ST-T Wave Abnormality", "2: Left Ventricular Hypertrophy"],
    index=0
)
if restecg_option == "Select...":
    missing_fields.append("Resting ECG")
restecg = int(restecg_option.split(":")[0]) if restecg_option != "Select..." else None

thalach = st.sidebar.number_input("Max Heart Rate Achieved (bpm)", min_value=0, max_value=230, value=0)
if thalach == 0:
    missing_fields.append("Max Heart Rate Achieved")

exang_option = st.sidebar.selectbox("Exercise Induced Angina?", options=["Select...", "No", "Yes"], index=0)
if exang_option == "Select...":
    missing_fields.append("Exercise Induced Angina")
exang = 1 if exang_option == "Yes" else (0 if exang_option == "No" else None)

oldpeak = st.sidebar.number_input("ST Depression Level", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

slope_option = st.sidebar.selectbox(
    "ST Segment Slope", 
    options=["Select...", "1: Upsloping (Normal)", "2: Flat", "3: Downsloping (Unhealthy)"],
    index=0
)
if slope_option == "Select...":
    missing_fields.append("ST Segment Slope")
slope = int(slope_option.split(":")[0]) if slope_option != "Select..." else None

ca_option = st.sidebar.selectbox("Major Vessels Colored by Fluoroscopy", options=["Select...", 0, 1, 2, 3], index=0)
if ca_option == "Select...":
    missing_fields.append("Major Vessels Count")
ca = ca_option if ca_option != "Select..." else None

thal_option = st.sidebar.selectbox(
    "Thalassemia Status", 
    options=["Select...", "3: Normal", "6: Fixed Defect", "7: Reversible Defect"],
    index=0
)
if thal_option == "Select...":
    missing_fields.append("Thalassemia Status")
thal = int(thal_option.split(":")[0]) if thal_option != "Select..." else None

if missing_fields:
    st.warning(f"Engine Warning: The following required inputs are missing: {', '.join(missing_fields)}. Please complete them in the sidebar to run the prediction model.")
else:
    inputs = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    
    patient_df = pd.DataFrame([inputs])
    patient_scaled = pd.DataFrame(scaler.transform(patient_df), columns=X.columns)

    prediction = model.predict(patient_scaled)[0]
    proba = model.predict_proba(patient_scaled)[0][1]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Diagnostic Risk Prediction")
        if prediction == 1:
            st.error(f"High Risk of Cardiovascular Issues Detected ({proba:.1%} Confidence Score)")
        else:
            st.success(f"Low Risk Profile Detected ({(1 - proba):.1%} Confidence Score)")

        st.markdown("---")
        st.subheader("Action Plan: What You Should Do Next")
        
        if prediction == 1:
            st.write("1. **Consult a Specialist:** Schedule an appointment with a cardiologist for secondary testing (Stress Echocardiogram / Angiography).")
            st.write("2. **Immediate Vitals Tracking:** Record your blood pressure twice daily and monitor for sudden shortness of breath.")
            st.write("3. **Emergency Red Flags:** Seek urgent emergency care if you experience crushing chest pain radiating to your jaw or left arm.")
            st.write("4. **Dietary Adjustments:** Reduce sodium intake (< 2,000 mg/day) and avoid trans fats immediately.")
        else:
            st.write("1. **Routine Checkups:** Maintain annual preventative care physicals and regular lipid panel checks.")
            st.write("2. **Active Lifestyle:** Aim for at least 150 minutes of moderate aerobic exercise per week.")
            st.write("3. **Balanced Nutrition:** Maintain a heart-healthy diet rich in fiber, whole grains, and lean proteins.")
            st.write("4. **Monitor Symptoms:** Re-test if you develop new unexplained fatigue or physical exertional tightness.")

    with col2:
        st.subheader("Symptom & Vital Explanations (Bit-by-Bit)")
        explainer = shap.Explainer(model)
        shap_values = explainer(patient_scaled)
        vals = shap_values.values[0]
        
        explanation_df = pd.DataFrame({
            'Raw': X.columns,
            'Value': [inputs[f] for f in X.columns],
            'Impact': vals
        }).sort_values(by='Impact', key=abs, ascending=False)
        
        for idx, row in explanation_df.iterrows():
            feat = row['Raw']
            val = row['Value']
            imp = row['Impact']
            direction = "INCREASED risk (+)" if imp > 0 else "DECREASED risk (-)"
            
            with st.expander(f"{feat.upper()} (Value: {val}) — {direction}"):
                st.write(f"**Medical Definition:** {SYMPTOM_DEFINITIONS[feat]}")
                st.write(f"**Model Impact Score:** `{imp:+.4f}`")

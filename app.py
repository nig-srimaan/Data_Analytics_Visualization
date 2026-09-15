import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import google.generativeai as genai

st.set_page_config(page_title="MedExplain Natural Symptom Engine", layout="wide")

# ---------------------------------------------------------------------------
# 1. DATA
# The UCI heart-disease repository actually has FOUR sources that share the
# same 14 columns: Cleveland, Hungary, Switzerland, and VA Long Beach.
# The original app only used Cleveland (~300 rows). We combine all four here.
#
# Caveat worth knowing: only Cleveland is clean/complete. The other three
# are missing a lot of fields (especially ca, thal, slope). Rather than
# imputing fake values or dropping those rows, we leave the gaps as NaN and
# let XGBoost handle missing values natively (it learns the best default
# split for missing data during training) - that's what the `missing`
# behaviour below is doing.
# ---------------------------------------------------------------------------
DATA_SOURCES = {
    "cleveland":   "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "hungarian":   "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
    "switzerland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
    "va":          "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data",
}

COLUMNS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
FEATURE_COLUMNS = COLUMNS[:-1]


@st.cache_data
def load_heart_data():
    frames = []
    for name, url in DATA_SOURCES.items():
        try:
            part = pd.read_csv(url, names=COLUMNS)
            part["source"] = name
            frames.append(part)
        except Exception as e:
            st.warning(f"Couldn't load the {name} dataset ({e}) - continuing without it.")

    df = pd.concat(frames, ignore_index=True)

    # "?" is this dataset's missing-value marker
    df.replace("?", np.nan, inplace=True)
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # chol == 0 is a known artifact in the Switzerland/VA files (they used 0
    # as a missing-value sentinel for cholesterol, not a real reading)
    df.loc[df["chol"] == 0, "chol"] = np.nan

    # target is 0 (no disease) through 4 (severity) in the raw files -
    # collapse to a binary "has disease" label
    df["target"] = df["target"].apply(lambda x: 1 if pd.notna(x) and x > 0 else 0)

    return df


df = load_heart_data()
X = df[FEATURE_COLUMNS]
y = df["target"]


@st.cache_resource
def train_model(_X, _y):
    # No scaling needed - tree-based models like XGBoost split on raw
    # feature values, they don't care about feature magnitude.
    # missing=np.nan tells XGBoost to treat NaN as "value not available"
    # and learn the best default branch for it, instead of us having to
    # impute a fake number.
    model = xgb.XGBClassifier(eval_metric="logloss", missing=np.nan)
    model.fit(_X, _y)
    return model


model = train_model(X, y)

SAFE_CONFIG = {
    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
}


# ---------------------------------------------------------------------------
# 2. LLM HELPERS
# ---------------------------------------------------------------------------
def get_gemini_model(name="gemini-2.5-flash"):
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(name)


def extract_symptoms(user_symptoms):
    """Ask Gemini to pull out only what's actually stated in the text.
    Anything not mentioned comes back as null - we ask the user for those
    ourselves instead of letting the model guess."""
    prompt = f"""
    You are a medical data extractor. Read the patient's own description of
    their symptoms and extract ONLY what is clearly stated or strongly
    implied. If something is not mentioned, return null for it - never guess.

    Patient text: '{user_symptoms}'

    Return JSON with exactly these keys:
    "cp": 0, 1, 2, 3, or null
        (0 = severe/crushing chest pain, 1 = milder or atypical chest pain,
         2 = discomfort that isn't classic chest pain, 3 = no chest pain mentioned)
    "exang": 0, 1, or null (1 if symptoms are triggered by exercise/exertion)
    "thalach_est": integer or null (ONLY if the patient states an actual heart-rate number)
    "racing_heart": true, false, or null (true if they describe racing/pounding/palpitations without a number)
    "trestbps": integer or null (ONLY if they state an actual blood pressure reading)
    "chol": integer or null (ONLY if they state an actual cholesterol reading)
    "fbs_flag": true, false, or null (true if they mention diabetes or high blood sugar)
    """
    try:
        response = get_gemini_model().generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
            safety_settings=SAFE_CONFIG,
        )
        return json.loads(response.text)
    except Exception as e:
        st.error(f"API error while reading your symptoms: {e}")
        return {}


def generate_advice(user_symptoms, prediction):
    fallback = (
        "Your symptoms indicate some elevated cardiovascular risk. Please consult a doctor soon for a professional checkup."
        if prediction == 1 else
        "Your symptoms do not strongly match acute cardiovascular disease. If you feel unwell, it's still worth resting and seeing a doctor if things don't improve."
    )
    prompt = f"""
    The patient said: "{user_symptoms}"
    The clinical model predicted: {'Elevated Risk of Heart Issues' if prediction == 1 else 'Low Risk of Heart Issues'}.

    Write a 3-paragraph response directly to the patient.
    Paragraph 1: Acknowledge their specific symptoms so they feel heard.
    Paragraph 2: Explain what the result means and whether they should be worried, in plain, comforting English.
    Paragraph 3: Give 2-3 practical, actionable pieces of advice for what to do next.
    """
    try:
        response = get_gemini_model().generate_content(prompt, safety_settings=SAFE_CONFIG)
        try:
            text = response.text.strip()
        except ValueError:
            text = ""
        return text if text else fallback
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# 3. UI - three stages, tracked in session_state: input -> clarify -> result
# ---------------------------------------------------------------------------
st.title("MedExplain: AI-Powered Symptom Checker")
st.caption(
    "Educational prototype trained on a small public research dataset. "
    "This is a screening estimate, not a diagnosis - it doesn't replace seeing a doctor."
)

if "stage" not in st.session_state:
    st.session_state.stage = "input"

# ---- Stage 1: symptom input -------------------------------------------------
col1, col2 = st.columns([1, 2])
with col1:
    age = st.number_input("Your Age", min_value=1, max_value=120, value=30)
    sex_input = st.selectbox("Biological Sex", ["Female", "Male"])
    sex = 1 if sex_input == "Male" else 0
with col2:
    user_symptoms = st.text_area(
        "Describe your symptoms:",
        placeholder="E.g., My chest feels really tight when I walk up the stairs and my heart races...",
        height=150,
    )

if st.button("Analyze My Symptoms", type="primary"):
    if not user_symptoms.strip():
        st.warning("Please describe how you are feeling before analyzing.")
    else:
        st.session_state.age = age
        st.session_state.sex = sex
        st.session_state.user_symptoms = user_symptoms
        st.session_state.extracted = extract_symptoms(user_symptoms)
        st.session_state.stage = "clarify"

# ---- Stage 2: ask for whatever the LLM couldn't find in the text -----------
if st.session_state.stage in ("clarify", "result"):
    extracted = st.session_state.get("extracted", {})

if st.session_state.stage == "clarify":
    st.markdown("---")
    st.subheader("A couple more details")
    st.caption("The AI couldn't tell these from your description - answer what you can, skip what you don't know.")

    with st.form("clarify_form"):
        cp = extracted.get("cp")
        if cp is None:
            cp_label = st.selectbox(
                "How would you describe the discomfort?",
                ["Severe, crushing chest pain",
                 "Chest pain, but milder / not the classic 'crushing' type",
                 "Discomfort that isn't really chest pain (e.g. back, jaw, stomach)",
                 "No chest pain at all"],
            )
            cp = {
                "Severe, crushing chest pain": 0,
                "Chest pain, but milder / not the classic 'crushing' type": 1,
                "Discomfort that isn't really chest pain (e.g. back, jaw, stomach)": 2,
                "No chest pain at all": 3,
            }[cp_label]

        exang = extracted.get("exang")
        if exang is None:
            exang = 1 if st.radio("Does it mainly happen during exercise or exertion?", ["Yes", "No"]) == "Yes" else 0

        thalach_est = extracted.get("thalach_est")
        if thalach_est is None:
            racing = extracted.get("racing_heart")
            if racing is None:
                racing = st.radio("Does your heart feel like it's racing or pounding?", ["Yes", "No"]) == "Yes"
            thalach_est = 180 if racing else 150

        trestbps = extracted.get("trestbps")
        if trestbps is None:
            know_bp = st.checkbox("I know my resting blood pressure")
            trestbps = st.number_input("Resting blood pressure (mm Hg)", 80, 220, 120) if know_bp else np.nan

        chol = extracted.get("chol")
        if chol is None:
            know_chol = st.checkbox("I know my cholesterol level")
            chol = st.number_input("Cholesterol (mg/dl)", 100, 500, 200) if know_chol else np.nan

        fbs_flag = extracted.get("fbs_flag")
        if fbs_flag is None:
            fbs_choice = st.selectbox("Fasting blood sugar over 120 mg/dl?", ["Not sure", "Yes", "No"])
            fbs = 1 if fbs_choice == "Yes" else (0 if fbs_choice == "No" else np.nan)
        else:
            fbs = 1 if fbs_flag else 0

        st.caption(
            "We don't ask about resting ECG, ST-depression, vessel count, or "
            "thalassemia results - almost nobody knows these without a stress "
            "test or angiogram, so the model treats them as unknown rather than guessed."
        )

        submitted = st.form_submit_button("Get My Risk Assessment")

    if submitted:
        st.session_state.inputs = {
            "age": st.session_state.age, "sex": st.session_state.sex,
            "cp": cp, "trestbps": trestbps, "chol": chol, "fbs": fbs,
            "restecg": np.nan, "thalach": thalach_est, "exang": exang,
            "oldpeak": np.nan, "slope": np.nan, "ca": np.nan, "thal": np.nan,
        }
        st.session_state.stage = "result"
        st.rerun()

# ---- Stage 3: prediction + advice ------------------------------------------
if st.session_state.stage == "result":
    inputs = st.session_state.inputs
    patient_df = pd.DataFrame([inputs])[FEATURE_COLUMNS]

    prediction = model.predict(patient_df)[0]
    proba = model.predict_proba(patient_df)[0][1]

    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.subheader("Clinical Result")
        if prediction == 1:
            st.error(f"Elevated Risk ({proba:.1%} Confidence)")
        else:
            st.success(f"Low Risk ({(1 - proba):.1%} Confidence)")

    with res_col2:
        st.subheader("What this means for you")
        st.write(generate_advice(st.session_state.user_symptoms, prediction))

    if st.button("Start Over"):
        for key in ("stage", "extracted", "inputs"):
            st.session_state.pop(key, None)
        st.rerun()

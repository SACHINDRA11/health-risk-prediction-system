import streamlit as st
import joblib
import pandas as pd

# =========================================
# LOAD MODELS
# =========================================
ckd_model = joblib.load("ckd_model_rf.joblib")
icu_model = joblib.load("icu_model.joblib")
feature_names = joblib.load("feature_names.joblib")

st.set_page_config(
    page_title="Health Risk Prediction System",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Health Risk Prediction System")

# =========================================================
# CKD SECTION
# =========================================================
st.header("🧪 CKD Prediction")

age_ckd = st.number_input("Age", 0, 120, 45)

bp = st.number_input("Blood Pressure", 0, 200, 80)
sg = st.number_input("Specific Gravity", 1.0, 1.05, 1.02)
al = st.number_input("Albumin", 0, 5, 0)
su = st.number_input("Sugar", 0, 5, 0)

bgr = st.number_input("Blood Glucose Random", 0, 500, 120)
bu = st.number_input("Blood Urea", 0, 300, 40)
sc = st.number_input("Serum Creatinine", 0.0, 20.0, 1.2)

sod = st.number_input("Sodium", 0.0, 200.0, 135.0)
pot = st.number_input("Potassium", 0.0, 20.0, 4.5)

hemo = st.number_input("Hemoglobin", 0.0, 20.0, 13.0)
pcv = st.number_input("Packed Cell Volume", 0, 60, 40)
wc = st.number_input("White Blood Cell Count", 0, 30000, 8000)
rc = st.number_input("Red Blood Cell Count", 0.0, 10.0, 4.5)

rbc = st.selectbox("RBC", ["normal", "abnormal"])
pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
pcc = st.selectbox("Pus Cell Clumps", ["present", "notpresent"])
ba = st.selectbox("Bacteria", ["present", "notpresent"])

htn = st.selectbox("Hypertension", ["yes", "no"])
dm = st.selectbox("Diabetes", ["yes", "no"])
cad = st.selectbox("Coronary Artery Disease", ["yes", "no"])

appet = st.selectbox("Appetite", ["good", "poor"])
pe = st.selectbox("Pedal Edema", ["yes", "no"])
ane = st.selectbox("Anemia", ["yes", "no"])

# =========================================================
# CKD PREDICTION
# =========================================================
if st.button("Predict CKD"):

    ckd_df = pd.DataFrame({
        'age': [age_ckd],
        'bp': [bp],
        'sg': [sg],
        'al': [al],
        'su': [su],
        'bgr': [bgr],
        'bu': [bu],
        'sc': [sc],
        'sod': [sod],
        'pot': [pot],
        'hemo': [hemo],
        'pcv': [pcv],
        'wc': [wc],
        'rc': [rc],
        'rbc': [rbc],
        'pc': [pc],
        'pcc': [pcc],
        'ba': [ba],
        'htn': [htn],
        'dm': [dm],
        'cad': [cad],
        'appet': [appet],
        'pe': [pe],
        'ane': [ane]
    })

    # =========================================
    # FIX DEPLOYMENT COLUMN ISSUE
    # =========================================
    expected_cols = ckd_model.feature_names_in_

    # add missing columns
    for col in expected_cols:
        if col not in ckd_df.columns:
            ckd_df[col] = 0

    # keep exact order
    ckd_df = ckd_df[expected_cols]

    # prediction
    prob = float(ckd_model.predict_proba(ckd_df)[0][1])

    st.subheader("CKD Prediction")

    st.write(f"CKD Probability: {prob:.2f}")

    if prob >= 0.50:
        st.error(
            f"⚠️ YES - CKD Detected ({prob:.2%})"
        )
    else:
        st.success(
            f"✅ NO - CKD Not Detected ({(1-prob):.2%} confidence)"
        )

# =========================================================
# ICU SECTION
# =========================================================
st.header("🚑 ICU Mortality Prediction")

age = st.number_input("ICU Age", 0, 120, 75)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

urine = st.number_input(
    "Urine Output",
    0.0,
    5000.0,
    300.0
)

hr = st.number_input(
    "Heart Rate",
    20.0,
    220.0,
    120.0
)

temp = st.number_input(
    "Temperature",
    90.0,
    110.0,
    101.0
)

nidiasabp = st.number_input(
    "NIDiasABP",
    20.0,
    150.0,
    50.0
)

sysabp = st.number_input(
    "SysABP",
    50.0,
    250.0,
    85.0
)

diasabp = st.number_input(
    "DiasABP",
    20.0,
    150.0,
    45.0
)

ph = st.number_input(
    "pH",
    6.8,
    8.0,
    7.10
)

paco2 = st.number_input(
    "PaCO2",
    10.0,
    100.0,
    65.0
)

pao2 = st.number_input(
    "PaO2",
    20.0,
    300.0,
    55.0
)

platelets = st.number_input(
    "Platelets",
    0.0,
    1000.0,
    120.0
)

map_val = st.number_input(
    "MAP",
    20.0,
    200.0,
    55.0
)

k = st.number_input(
    "Potassium",
    1.0,
    10.0,
    5.8
)

na = st.number_input(
    "Sodium",
    100.0,
    180.0,
    128.0
)

fio2 = st.number_input(
    "FiO2",
    0.0,
    1.0,
    0.80
)

gcs = st.number_input(
    "GCS",
    3,
    15,
    5
)

icutype = st.selectbox(
    "ICU Type",
    [1, 2, 3, 4],
    format_func=lambda x: {
        1: "Coronary Care Unit",
        2: "Cardiac Surgery Recovery",
        3: "Medical ICU",
        4: "Surgical ICU"
    }[x]
)

# =========================================================
# ICU PREDICTION
# =========================================================
if st.button("Predict Mortality"):

    raw = pd.DataFrame([{
        "Age": age,
        "Gender": 1 if gender == "Male" else 0,
        "Urine": urine,
        "HR": hr,
        "Temp": temp,
        "NIDiasABP": nidiasabp,
        "SysABP": sysabp,
        "DiasABP": diasabp,
        "pH": ph,
        "PaCO2": paco2,
        "PaO2": pao2,
        "Platelets": platelets,
        "MAP": map_val,
        "K": k,
        "Na": na,
        "FiO2": fio2,
        "GCS": gcs,
        "ICUType": icutype
    }])

    # =========================================
    # PREPROCESSING
    # =========================================
    processed = pd.get_dummies(raw, drop_first=True)

    processed = processed.reindex(
        columns=feature_names,
        fill_value=0
    )

    # =========================================
    # BASE ICU MODEL
    # =========================================
    death_prob = float(
        icu_model.predict_proba(processed)[0][1]
    )

    # =========================================
    # CKD RISK ANALYSIS
    # =========================================
    ckd_score = 0

    if urine < 500:
        ckd_score += 1

    if na < 135:
        ckd_score += 1

    if k > 5:
        ckd_score += 1

    if age > 65:
        ckd_score += 1

    if map_val < 60:
        ckd_score += 1

    # =========================================
    # MEDICAL RISK BOOSTING
    # =========================================

    # severe indicators
    if gcs <= 8:
        death_prob += 0.20

    if pao2 < 60:
        death_prob += 0.15

    if ph < 7.2:
        death_prob += 0.15

    if map_val < 60:
        death_prob += 0.10

    if age > 70:
        death_prob += 0.10

    # CKD impact
    if ckd_score >= 3:
        death_prob += 0.25

    elif ckd_score == 2:
        death_prob += 0.15

    # cap
    death_prob = min(death_prob, 0.99)

    survival_prob = 1 - death_prob

    # =========================================
    # OUTPUT
    # =========================================
    st.subheader("Prediction")

    st.write(f"Death Probability: {death_prob:.2f}")
    st.write(f"Survival Probability: {survival_prob:.2f}")

    # =========================================
    # CKD MESSAGE
    # =========================================
    if ckd_score >= 3:
        st.warning(
            "⚠️ Likely CKD Patient Based on ICU Biomarkers"
        )

    elif ckd_score == 2:
        st.info(
            "🟡 Moderate CKD Risk Detected"
        )

    else:
        st.success(
            "✅ Not Likely CKD Patient"
        )

    # =========================================
    # FINAL ICU RESULT
    # =========================================
    if death_prob >= 0.50:
        st.error(
            f"⚠️ HIGH RISK OF DEATH ({death_prob:.2%})"
        )
    else:
        st.success(
            f"✅ LIKELY SURVIVAL ({survival_prob:.2%})"
        )
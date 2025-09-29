import streamlit as st
import pandas as pd
import numpy as np
import joblib, json
from datetime import date

st.set_page_config(page_title="Loan Risk (Live)", layout="centered")
st.title("💳 Loan Risk Panel — Canlı Skor")

# ========= AYAR =========
MODEL_PATH  = "artifacts/model_xgb_cw.pkl"       # Pipeline(pre+clf) KULLAN
SCHEMA_PATH = "artifacts/feature_schema.json"    # train.py’de kaydetmiştik
# Eğer pre ayrıysa ayrıca PRE_PATH yükle ve predict öncesi pre.transform() yap (aşağıda not var)

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    schema_cols = json.load(open(SCHEMA_PATH))["columns"]
    return model, schema_cols

model, schema_cols = load_artifacts()

def to_ts(d):
    return pd.to_datetime(d).value // 10**9

def build_features_single(Principal, terms, age, education, Gender, eff, due):
    eff_ts = to_ts(eff)
    due_ts = to_ts(due)
    planned = (pd.to_datetime(due) - pd.to_datetime(eff)).days
    principal_per_term = (Principal / terms) if terms else np.nan

    df = pd.DataFrame([{
        "Principal": Principal,
        "terms": terms,
        "age": age,
        "education": education,
        "Gender": Gender,
        "effective_date": eff_ts,
        "due_date": due_ts,
        "planned_term_days": planned,
        "principal_per_term": principal_per_term
    }])
    # Eğitimdeki kolon sırasına hizala (fazla/eksik kolonları yönet)
    df = df.reindex(columns=schema_cols, fill_value=np.nan)
    return df

# ========= WIDGET’LAR (FORM YOK -> CANLI) =========
st.subheader("🧍 Tekil Başvuru (Değer değiştikçe anında skor)")

col1, col2 = st.columns(2)
with col1:
    Principal = st.number_input("Principal", min_value=0, value=1000, step=50, key="principal")
    terms     = st.selectbox("terms (gün)", [7, 15, 30], index=2, key="terms")
    age       = st.number_input("age", min_value=18, max_value=80, value=30, key="age")
with col2:
    education = st.selectbox("education", ["High School or Below","college","Bechalor","Master or Above"], key="education")
    Gender    = st.selectbox("Gender", ["male","female"], key="gender")
    effective_date = st.date_input("effective_date", value=date.today(), key="eff")
    due_date       = st.date_input("due_date", value=date.today(), key="due")

# Her çizimde yeniden hesapla (değerlerden herhangi biri değişince Streamlit tekrar render eder)
X = build_features_single(
    Principal=Principal,
    terms=terms,
    age=age,
    education=education,
    Gender=Gender,
    eff=effective_date,
    due=due_date
)

# PIPELINE model ise: direkt predict_proba(X)
proba = float(model.predict_proba(X)[:, 1][0])
st.metric("PAIDOFF Olasılığı", f"{proba:.2%}")
st.caption("Not: Olasılık düşükse risk yüksektir.")

st.divider()

# ========= TOPLU SKOR =========
st.subheader("📂 Toplu Skor (CSV)")
up = st.file_uploader("CSV yükleyin", type=["csv"])
if up is not None:
    data = pd.read_csv(up)

    # Eğitimdeki dönüşümlere hizala
    # Eğer CSV'de effective_date/due_date string ise timestamp'a çevir:
    for c in ["effective_date","due_date"]:
        if c in data.columns and not np.issubdtype(data[c].dtype, np.number):
            data[c] = pd.to_datetime(data[c], errors="coerce").astype("int64") // 10**9

    if {"Principal","terms"}.issubset(data.columns):
        data["principal_per_term"] = data["Principal"] / data["terms"].replace({0: np.nan})

    if set(["effective_date","due_date"]).issubset(data.columns):
        # Eğer tarihleri saniye timestamp olarak tuttuysan gün farkını doğrudan bulamazsın.
        # CSV'de planned_term_days yoksa burada boş bırakmak en güvenlisi:
        pass

    Xbulk = data.reindex(columns=schema_cols, fill_value=np.nan)
    probs = model.predict_proba(Xbulk)[:, 1]
    out = data.copy()
    out["paid_prob"] = probs
    st.dataframe(out.head())
    st.download_button("Sonuçları indir (CSV)", out.to_csv(index=False).encode("utf-8"), file_name="scored.csv")

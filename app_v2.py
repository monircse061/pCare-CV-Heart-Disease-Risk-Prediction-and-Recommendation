"""
CARE-Heart | EarlyMed D1 Explainable Risk Assistant
=====================================================
A Streamlit clinical-decision-support app inspired by the CARE-Cirrhosis
DiCE / SHAP counterfactual-explanation paper (Kumar et al., Intelligent
Pharmacy, 2026), adapted to the D1 "EarlyMed" heart-risk dataset and the
CatBoost model used in the accompanying research pipeline.

Pipeline mirrored from the paper:
  1. Patient data entry (D1 schema: Age + 17 binary clinical/lifestyle/symptom flags)
  2. CatBoost prediction of HeartRisk probability
  3. Global + local SHAP explanation -> Top-10 ranked drivers
  4. Three-tier hybrid recommendation engine:
       - lifestyle-modifiable features found  -> lifestyle advice
       - else moderately-modifiable found      -> clinical/monitoring advice
       - else only symptom/non-modifiable found -> cautious "consult a doctor" advice
  5. Counterfactual "what-if" scenarios (DiCE-style single/paired flips,
     ranked by verified probability reduction) -> Top-3 with bar chart
  6. Visual risk gauge

Run with:
    streamlit run app_v2.py

Notes on data/model:
- On first run the app will try to download the public Kaggle dataset
  "mahatiratusher/heart-disease-risk-prediction-dataset" via `kagglehub`
  (requires Kaggle credentials configured on the host machine) and train
  a CatBoost model matching the fixed hyperparameters used in the D1
  pipeline (iterations=300, depth=6, learning_rate=0.02).
- If internet/Kaggle access is unavailable, the app automatically falls
  back to a synthetic-but-clinically-plausible dataset generator so the
  full application (UI, SHAP, counterfactuals, gauge, recommendations)
  still works end-to-end for demoing / grading purposes.
- If you already have a trained CatBoost model (`model.cbm`) and the
  training dataframe (`train_data.csv`) from your own notebook, drop them
  next to this file and the app will use them directly instead of
  downloading/training anything.
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Page config & global style
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="CARE-Heart | Explainable Personalized Assistant",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Poppins:wght@600;700;800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.4.0/css/all.min.css">
<style>
/* =========================================================
   DESIGN TOKENS — vibrant medical palette
   ========================================================= */
:root {
    --primary: #0d9488;        /* teal-600  */
    --primary-dark: #0f766e;   /* teal-700  */
    --primary-light: #14b8a6;  /* teal-500  */
    --secondary: #4f46e5;      /* indigo-600 */
    --accent: #db2777;         /* pink-600  */
    --danger: #dc2626;         /* red-600   */
    --danger-bg: #fef2f2;
    --warning: #d97706;        /* amber-600 */
    --warning-bg: #fffbeb;
    --success: #16a34a;        /* green-600 */
    --success-bg: #f0fdf4;
    --bg-page: #f4f7fb;
    --bg-card: #ffffff;
    --text: #0f172a;
    --text-muted: #52607a;
    --border: #e2e8f0;
    --radius: 16px;
    --shadow-sm: 0 1px 3px rgba(15, 23, 42, 0.06), 0 1px 2px rgba(15, 23, 42, 0.08);
    --shadow-md: 0 8px 20px rgba(15, 23, 42, 0.08);
}

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }

/* =========================================================
   GLOBAL APP BACKGROUND
   ========================================================= */
html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.main, .block-container {
    background: var(--bg-page) !important;
    color: var(--text) !important;
}

[data-testid="stHeader"] {
    background-color: rgba(244,247,251,0.0) !important;
}

.block-container {
    padding-top: 1.6rem !important;
    max-width: 1300px;
}

/* =========================================================
   TYPOGRAPHY
   ========================================================= */
h1, h2, h3, h4, h5, h6 { font-family: 'Poppins', 'Inter', sans-serif; color: var(--text) !important; }
p, span, label, div, li { color: var(--text); }
[data-testid="stCaptionContainer"], .stCaption, small { color: var(--text-muted) !important; }

/* =========================================================
   SIDEBAR
   ========================================================= */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    color: var(--text) !important;
    min-width: 360px !important;
    max-width: 420px !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] > div, [data-testid="stSidebarContent"] {
    background: transparent !important;
    color: var(--text) !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: var(--primary-dark) !important;
}
[data-testid="stSidebar"] hr { border-color: var(--border); }

.sidebar-header {
    background: #ffffff !important;
    color: var(--text) !important;
    padding: 16px 18px;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    border-left: 5px solid var(--primary);
    margin-bottom: 14px;
    box-shadow: var(--shadow-sm);
}
.sidebar-header h2, .sidebar-header p { margin: 0; }
.sidebar-header h2 { font-size: 1.1rem; font-weight: 800; color: var(--primary-dark) !important; }
.sidebar-header h2 i { color: var(--primary); }
.sidebar-header p { font-size: 0.8rem; color: var(--text-muted) !important; margin-top: 2px; }

.sidebar-section-title {
    font-weight: 700;
    font-size: 0.92rem;
    color: var(--primary-dark) !important;
    margin: 10px 0 4px 0;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* =========================================================
   HERO SECTION
   ========================================================= */
.care-hero {
    background: #ffffff !important;
    padding: 30px 34px;
    border-radius: 20px;
    color: var(--text) !important;
    margin-bottom: 22px;
    border: 1px solid var(--border);
    border-top: 5px solid var(--primary);
    box-shadow: var(--shadow-sm);
    position: relative;
    overflow: hidden;
}
.care-hero::after {
    content: "";
    position: absolute; right: -70px; top: -70px;
    width: 220px; height: 220px; border-radius: 50%;
    background: linear-gradient(135deg, rgba(13,148,136,0.07), rgba(79,70,229,0.07));
}
.care-hero h1 {
    margin: 0;
    font-size: 2.0rem;
    font-weight: 800;
    color: var(--primary-dark) !important;
    display: flex; align-items: center; gap: 12px;
}
.care-hero h1 i { color: var(--primary); }
.care-hero p {
    margin: 10px 0 0 0;
    font-size: 1rem;
    color: var(--text-muted) !important;
    max-width: 780px;
}
.hero-chips { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; }
.hero-chip {
    background: #f0fdfa;
    border: 1px solid #99f6e4;
    color: var(--primary-dark);
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
}

/* =========================================================
   CARDS
   ========================================================= */
.care-card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 22px 24px;
    margin-bottom: 18px;
    color: var(--text) !important;
    box-shadow: var(--shadow-sm);
}
.care-card h3, .care-card .stSubheader, .care-card [data-testid="stMarkdownContainer"] h3 {
    margin-top: 0;
    color: var(--text) !important;
}

/* =========================================================
   RISK BADGES
   ========================================================= */
.care-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 800;
    letter-spacing: 0.02em;
}
.badge-low  { background: var(--success-bg) !important; color: var(--success) !important; border: 1.5px solid var(--success); }
.badge-mod  { background: var(--warning-bg) !important; color: var(--warning) !important; border: 1.5px solid var(--warning); }
.badge-high { background: var(--danger-bg)  !important; color: var(--danger)  !important; border: 1.5px solid var(--danger); }

/* =========================================================
   METRIC TILES (top info bar)
   ========================================================= */
.metric-tile {
    background: #ffffff !important;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 18px;
    box-shadow: var(--shadow-sm);
    color: var(--text) !important;
    height: 100%;
    display: flex; flex-direction: column; gap: 4px;
}
.metric-tile .m-icon { font-size: 1.3rem; }
.metric-tile .m-label { font-size: 0.78rem; font-weight: 700; opacity: 0.75; text-transform: uppercase; letter-spacing: 0.04em; color: var(--text-muted); }
.metric-tile .m-value { font-size: 1.05rem; font-weight: 700; color: var(--text) !important; }
.tile-teal   { border-left: 5px solid var(--primary); }
.tile-indigo { border-left: 5px solid var(--secondary); }
.tile-amber  { border-left: 5px solid var(--warning); }
.tile-teal   .m-value { color: var(--primary-dark) !important; }
.tile-indigo .m-value { color: var(--secondary) !important; }
.tile-amber  .m-value { color: var(--warning) !important; }

/* =========================================================
   RECOMMENDATION ITEMS
   ========================================================= */
.rec-item {
    border-left: 5px solid var(--text-muted);
    background: #f8fafc !important;
    color: var(--text) !important;
    font-size: 0.98rem;
    padding: 12px 16px;
    border-radius: 10px;
    margin-bottom: 10px;
}
.rec-item.doctor    { border-left-color: var(--danger);  background: var(--danger-bg) !important; }
.rec-item.lifestyle { border-left-color: var(--success); background: var(--success-bg) !important; }
.rec-item.moderate  { border-left-color: var(--warning); background: var(--warning-bg) !important; }

/* =========================================================
   COUNTERFACTUAL CARDS
   ========================================================= */
.cf-card {
    background: linear-gradient(135deg, #f0fdfa, #ffffff) !important;
    border-radius: var(--radius);
    padding: 18px 20px;
    border: 1px solid var(--border);
    border-top: 4px solid var(--primary);
    margin-bottom: 12px;
    color: var(--text) !important;
    box-shadow: var(--shadow-sm);
}

/* =========================================================
   INPUT WIDGETS
   ========================================================= */
[data-testid="stTextInput"], [data-testid="stNumberInput"],
[data-testid="stSelectbox"], [data-testid="stMultiSelect"],
[data-testid="stDateInput"], [data-testid="stTextArea"] {
    color: var(--text) !important;
}
input, textarea, select {
    background-color: #ffffff !important;
    color: var(--text) !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] > div > div { background: var(--primary) !important; }

/* Checkbox label text */
[data-testid="stCheckbox"] label p { color: var(--text) !important; font-size: 0.92rem; }
[data-testid="stCheckbox"] { padding: 2px 0; }

/* =========================================================
   BUTTONS
   ========================================================= */
.stButton > button {
    background: var(--primary) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 0.6rem 1rem !important;
    box-shadow: var(--shadow-sm);
    transition: transform 0.08s ease-in-out;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: var(--shadow-md); }
.stButton > button[kind="secondary"], button[kind="secondary"] {
    background: #ffffff !important;
    color: var(--primary-dark) !important;
    border: 1.5px solid var(--primary) !important;
}

/* =========================================================
   TABS
   ========================================================= */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 6px;
    background: #ffffff;
    border: 1px solid var(--border);
    padding: 6px;
    border-radius: 12px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 9px;
    color: var(--text-muted) !important;
    font-weight: 700;
    padding: 10px 16px;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #f0fdfa !important;
    color: var(--primary-dark) !important;
    box-shadow: inset 0 0 0 1.5px var(--primary);
}
[data-testid="stTabs"] [aria-selected="true"] p { color: var(--primary-dark) !important; font-weight: 800; }

/* =========================================================
   EXPANDERS
   ========================================================= */
[data-testid="stExpander"] {
    background-color: #ffffff !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    box-shadow: var(--shadow-sm);
}
[data-testid="stExpanderDetails"] { background-color: #ffffff !important; }
[data-testid="stExpander"] summary p { font-weight: 700 !important; color: var(--primary-dark) !important; }

/* =========================================================
   ALERT / INFO BOXES
   ========================================================= */
[data-testid="stAlert"] {
    border-radius: 12px !important;
}

/* =========================================================
   DATAFRAMES / TABLES
   ========================================================= */
[data-testid="stDataFrame"] { background-color: #ffffff !important; border-radius: 10px; }

/* =========================================================
   PROGRESS BAR
   ========================================================= */
[data-testid="stProgress"] > div > div > div { background: linear-gradient(90deg, var(--success), var(--warning), var(--danger)) !important; }

/* =========================================================
   FOOTER / MENU
   ========================================================= */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }

.app-footer {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.82rem;
    padding: 18px 10px 6px 10px;
}

/* =========================================================
   RESPONSIVE — mobile & tablet
   ========================================================= */
@media (max-width: 900px) {
    .block-container { padding-left: 0.8rem !important; padding-right: 0.8rem !important; }
    .care-hero { padding: 22px 20px; }
    .care-hero h1 { font-size: 1.55rem; }
    .care-hero p { font-size: 0.9rem; }
    .care-card { padding: 16px 16px; }
    [data-testid="stSidebar"] { min-width: 300px !important; }
}
@media (max-width: 600px) {
    .care-hero h1 { font-size: 1.3rem; flex-wrap: wrap; }
    .metric-tile { padding: 12px; }
    .metric-tile .m-value { font-size: 0.92rem; }
}
</style>
"""
st.html(CUSTOM_CSS)

# ----------------------------------------------------------------------------
# D1 schema (matches D1_synthetic_dataset_what-if-moc-nice.ipynb / d1-earlymed
# pipeline: Age + 8 symptom flags + 9 risk/demographic flags)
# ----------------------------------------------------------------------------
TARGET = "HeartRisk"
CONT_COLS = ["Age"]
SYMPTOM_COLS = [
    "ChestPain", "ShortnessofBreath", "Fatigue", "Palpitations",
    "Dizziness", "Swelling", "PainArmsJawBack", "ColdSweatsNausea",
]
RISK_COLS = [
    "HighBP", "HighCholesterol", "Diabetes", "Smoking",
    "Obesity", "SedentaryLifestyle", "FamilyHistory", "ChronicStress", "Gender",
]
BINARY_COLS = SYMPTOM_COLS + RISK_COLS
FEATURE_COLS = CONT_COLS + BINARY_COLS

LIFESTYLE_FEATURES = ["Smoking", "SedentaryLifestyle", "ChronicStress"]
MODERATELY_MODIFIABLE_FEATURES = ["Obesity", "HighCholesterol", "HighBP", "Diabetes"]
SYMPTOM_FEATURES = list(SYMPTOM_COLS)
NON_MODIFIABLE_FEATURES = ["Age", "Gender", "FamilyHistory"]
ALL_MODIFIABLE_FEATURES = LIFESTYLE_FEATURES + MODERATELY_MODIFIABLE_FEATURES + SYMPTOM_FEATURES

FRIENDLY_NAMES = {
    "Age": "Age",
    "ChestPain": "Chest pain",
    "ShortnessofBreath": "Shortness of breath",
    "Fatigue": "Fatigue",
    "Palpitations": "Palpitations",
    "Dizziness": "Dizziness",
    "Swelling": "Swelling (limbs/ankles)",
    "PainArmsJawBack": "Pain in arms / jaw / back",
    "ColdSweatsNausea": "Cold sweats / nausea",
    "HighBP": "High blood pressure",
    "HighCholesterol": "High cholesterol",
    "Diabetes": "Diabetes",
    "Smoking": "Smoking",
    "Obesity": "Obesity",
    "SedentaryLifestyle": "Sedentary lifestyle",
    "FamilyHistory": "Family history of heart disease",
    "ChronicStress": "Chronic stress",
    "Gender": "Gender (Male=1 / Female=0)",
}

# Small clinical/lifestyle icons used to make the sidebar inputs scannable
FEATURE_ICONS = {
    "ChestPain": "💔",
    "ShortnessofBreath": "🫁",
    "Fatigue": "😴",
    "Palpitations": "💓",
    "Dizziness": "🌀",
    "Swelling": "🦵",
    "PainArmsJawBack": "🦴",
    "ColdSweatsNausea": "🥵",
    "HighBP": "🩸",
    "HighCholesterol": "🧈",
    "Diabetes": "💉",
    "Smoking": "🚬",
    "Obesity": "⚖️",
    "SedentaryLifestyle": "🛋️",
    "FamilyHistory": "🧬",
    "ChronicStress": "🧠",
}

# Small clinical/lifestyle icons used to make the sidebar inputs scannable
FEATURE_ICONS = {
    "ChestPain": "💔",
    "ShortnessofBreath": "🫁",
    "Fatigue": "😴",
    "Palpitations": "💓",
    "Dizziness": "🌀",
    "Swelling": "🦵",
    "PainArmsJawBack": "🦴",
    "ColdSweatsNausea": "🥵",
    "HighBP": "🩸",
    "HighCholesterol": "🧈",
    "Diabetes": "💉",
    "Smoking": "🚬",
    "Obesity": "⚖️",
    "SedentaryLifestyle": "🛋️",
    "FamilyHistory": "🧬",
    "ChronicStress": "🧠",
}

LIFESTYLE_ADVICE = {
    "Smoking": "🚭 **Quit smoking.** Smoking cessation is one of the single most effective ways to lower cardiovascular risk — consider nicotine-replacement therapy or a structured cessation program.",
    "SedentaryLifestyle": "🏃 **Increase physical activity.** Aim for at least 150 minutes/week of moderate aerobic exercise (brisk walking, cycling, swimming).",
    "ChronicStress": "🧘 **Manage chronic stress.** Mindfulness, regular sleep, and counseling/therapy can meaningfully reduce stress-related cardiovascular load.",
}
MODERATE_ADVICE = {
    "HighBP": "💊 Discuss antihypertensive management with your physician and monitor blood pressure regularly at home.",
    "HighCholesterol": "🥗 Consider dietary changes (reduce saturated fat) and ask your doctor about statin therapy if indicated.",
    "Diabetes": "🩸 Improve glycemic control through diet, exercise, and medication adherence; regular HbA1c monitoring is advised.",
    "Obesity": "⚖️ Gradual, sustained weight reduction through diet and physical activity can meaningfully lower cardiovascular strain.",
}
SYMPTOM_CAUTION = {
    "ChestPain": "Chest pain is a red-flag cardiac symptom.",
    "PainArmsJawBack": "Pain radiating to the arms/jaw/back can indicate cardiac ischemia.",
    "ColdSweatsNausea": "Cold sweats with nausea can accompany acute cardiac events.",
    "ShortnessofBreath": "Shortness of breath may indicate reduced cardiac output.",
    "Palpitations": "Palpitations can reflect underlying arrhythmia.",
    "Swelling": "Swelling (edema) may indicate fluid retention linked to heart function.",
    "Fatigue": "Unexplained fatigue can be an early, non-specific cardiac symptom.",
    "Dizziness": "Dizziness can reflect reduced perfusion related to cardiac output.",
}

MODEL_DIR = Path("care_heart_artifacts")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "catboost_model.cbm"
TRAIN_DATA_PATH = MODEL_DIR / "train_data.csv"
LOCAL_MODEL_PATH = Path("model.cbm")
LOCAL_TRAIN_PATH = Path("train_data.csv")

CAT_IDX = [FEATURE_COLS.index(c) for c in BINARY_COLS]
FIXED_CATBOOST_PARAMS = dict(iterations=300, depth=6, learning_rate=0.02, random_state=42, verbose=False)


# ----------------------------------------------------------------------------
# Data / model loading (cached)
# ----------------------------------------------------------------------------
def _make_synthetic_dataset(n=8000, seed=42):
    """Clinically-plausible synthetic fallback matching the D1 schema, used
    only when the real Kaggle dataset / a pre-trained model isn't available
    on this machine."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame()
    df["Age"] = rng.randint(25, 85, n)
    for c in RISK_COLS:
        base = 0.35 if c != "Gender" else 0.5
        df[c] = rng.binomial(1, base, n)
    for c in SYMPTOM_COLS:
        df[c] = rng.binomial(1, 0.25, n)

    risk_score = (
        0.03 * (df["Age"] - 25)
        + 1.1 * df["HighBP"] + 1.0 * df["HighCholesterol"] + 1.0 * df["Diabetes"]
        + 1.3 * df["Smoking"] + 0.8 * df["Obesity"] + 0.7 * df["SedentaryLifestyle"]
        + 0.9 * df["FamilyHistory"] + 0.6 * df["ChronicStress"]
        + 1.4 * df["ChestPain"] + 1.2 * df["PainArmsJawBack"] + 1.1 * df["ColdSweatsNausea"]
        + 0.9 * df["Swelling"] + 0.8 * df["Fatigue"] + 0.8 * df["Palpitations"]
        + 0.7 * df["ShortnessofBreath"] + 0.5 * df["Dizziness"]
        + rng.normal(0, 1.5, n)
    )
    prob = 1 / (1 + np.exp(-(risk_score - risk_score.mean()) / 3.0))
    df[TARGET] = rng.binomial(1, prob)
    return df


# The public Kaggle CSV ships snake_case / underscored column names
# (e.g. "Chest_Pain", "Heart_Risk") while the D1 pipeline this app mirrors
# uses camelCase ("ChestPain", "HeartRisk"). Normalize on load so downstream
# code can always rely on FEATURE_COLS / TARGET regardless of source.
_RAW_TO_SCHEMA = {
    "Chest_Pain": "ChestPain",
    "Shortness_of_Breath": "ShortnessofBreath",
    "Fatigue": "Fatigue",
    "Palpitations": "Palpitations",
    "Dizziness": "Dizziness",
    "Swelling": "Swelling",
    "Pain_Arms_Jaw_Back": "PainArmsJawBack",
    "Cold_Sweats_Nausea": "ColdSweatsNausea",
    "High_BP": "HighBP",
    "High_Cholesterol": "HighCholesterol",
    "Diabetes": "Diabetes",
    "Smoking": "Smoking",
    "Obesity": "Obesity",
    "Sedentary_Lifestyle": "SedentaryLifestyle",
    "Family_History": "FamilyHistory",
    "Chronic_Stress": "ChronicStress",
    "Gender": "Gender",
    "Age": "Age",
    "Heart_Risk": TARGET,
}


def _normalize_schema(df):
    """Rename any recognized raw/underscored columns to the app's camelCase
    schema. Leaves already-correct column names untouched."""
    rename_map = {k: v for k, v in _RAW_TO_SCHEMA.items() if k in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


@st.cache_resource(show_spinner="Loading D1 EarlyMed dataset...")
def load_dataset():
    """Try local files -> Kaggle download -> synthetic fallback."""
    if LOCAL_TRAIN_PATH.exists():
        df = _normalize_schema(pd.read_csv(LOCAL_TRAIN_PATH))
        return df, "local file (train_data.csv)"
    if TRAIN_DATA_PATH.exists():
        df = _normalize_schema(pd.read_csv(TRAIN_DATA_PATH))
        return df, "cached artifact"
    try:
        import kagglehub
        path = kagglehub.dataset_download("mahatiratusher/heart-disease-risk-prediction-dataset")
        csv_path = None
        for f in Path(path).rglob("*.csv"):
            csv_path = f
            break
        if csv_path is None:
            raise FileNotFoundError("No CSV found in downloaded dataset")
        df = pd.read_csv(csv_path)
        df = _normalize_schema(df)
        df = df.dropna().drop_duplicates().reset_index(drop=True)
        df.to_csv(TRAIN_DATA_PATH, index=False)
        return df, "Kaggle (mahatiratusher/heart-disease-risk-prediction-dataset)"
    except Exception:
        df = _make_synthetic_dataset()
        df.to_csv(TRAIN_DATA_PATH, index=False)
        return df, "synthetic fallback (no internet/Kaggle credentials found)"


@st.cache_resource(show_spinner="Training / loading CatBoost model...")
def load_model_and_data():
    from catboost import CatBoostClassifier, Pool
    from sklearn.model_selection import train_test_split

    df, source = load_dataset()
    df = df[[c for c in FEATURE_COLS if c in df.columns] + [TARGET]].copy()
    for c in BINARY_COLS:
        if c in df.columns:
            df[c] = df[c].astype(int)
    df["Age"] = df["Age"].astype(float)

    X = df[FEATURE_COLS].copy()
    y = df[TARGET].astype(int).copy()

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    for c in BINARY_COLS:
        Xtr[c] = Xtr[c].astype(int).astype(str)
        Xte[c] = Xte[c].astype(int).astype(str)

    if LOCAL_MODEL_PATH.exists():
        model = CatBoostClassifier()
        model.load_model(str(LOCAL_MODEL_PATH))
    elif MODEL_PATH.exists():
        model = CatBoostClassifier()
        model.load_model(str(MODEL_PATH))
    else:
        model = CatBoostClassifier(cat_features=CAT_IDX, **FIXED_CATBOOST_PARAMS)
        model.fit(Xtr, ytr)
        model.save_model(str(MODEL_PATH))

    test_acc = model.score(Xte, yte)
    return model, Xtr, Xte, ytr, yte, source, test_acc


def to_pool_row(model, row_dict):
    """Turn a dict of raw feature values into a single-row DataFrame matching
    training dtypes (binary cols as string codes, Age as float)."""
    row = {c: row_dict.get(c, 0) for c in FEATURE_COLS}
    df_row = pd.DataFrame([row])
    df_row["Age"] = df_row["Age"].astype(float)
    for c in BINARY_COLS:
        df_row[c] = df_row[c].astype(int).astype(str)
    return df_row


def predict_proba_row(model, df_row):
    return float(model.predict_proba(df_row)[0, 1])


# ----------------------------------------------------------------------------
# SHAP
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_shap_explainer(_model):
    import shap
    return shap.TreeExplainer(_model)


def compute_local_shap(model, df_row):
    explainer = get_shap_explainer(model)
    sv = explainer.shap_values(df_row)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    return np.array(sv).reshape(-1)


def compute_global_shap(model, X_sample):
    explainer = get_shap_explainer(model)
    sv = explainer.shap_values(X_sample)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=X_sample.columns)
    return mean_abs.sort_values(ascending=False)


# ----------------------------------------------------------------------------
# Recommendation engine (3-tier hybrid, mirrors CARE-Cirrhosis rule+gradient logic)
# ----------------------------------------------------------------------------
def build_recommendations(top_features, shap_series):
    """
    Build recommendations only from features whose SHAP values are positive,
    meaning they increase this patient's predicted heart-risk probability.

    Parameters
    ----------
    top_features : list
        Top feature names ranked by absolute SHAP value.
    shap_series : pd.Series
        Signed SHAP values indexed by feature name.
    """
    # IMPORTANT: only positive SHAP features are eligible for recommendations.
    # Negative SHAP features decrease the model-estimated risk and therefore
    # should not trigger recommendations.
    risk_increasing_features = [
        f for f in top_features
        if shap_series.get(f, 0) > 0
    ]

    lifestyle_hits = [
        f for f in risk_increasing_features
        if f in LIFESTYLE_FEATURES
    ]
    moderate_hits = [
        f for f in risk_increasing_features
        if f in MODERATELY_MODIFIABLE_FEATURES
    ]
    symptom_hits = [
        f for f in risk_increasing_features
        if f in SYMPTOM_FEATURES
    ]

    recs = []
    tier = None

    if lifestyle_hits:
        tier = "lifestyle"
        for f in lifestyle_hits:
            recs.append((
                "lifestyle",
                LIFESTYLE_ADVICE.get(
                    f, f"Address {FRIENDLY_NAMES.get(f, f)}."
                )
            ))

        if moderate_hits:
            for f in moderate_hits:
                recs.append((
                    "moderate",
                    MODERATE_ADVICE.get(
                        f, f"Monitor {FRIENDLY_NAMES.get(f, f)}."
                    )
                ))

    elif moderate_hits:
        tier = "moderate"
        for f in moderate_hits:
            recs.append((
                "moderate",
                MODERATE_ADVICE.get(
                    f, f"Monitor {FRIENDLY_NAMES.get(f, f)}."
                )
            ))

    elif symptom_hits:
        tier = "doctor"
        symptom_text = ", ".join(
            FRIENDLY_NAMES.get(f, f) for f in symptom_hits
        )
        recs.append((
            "doctor",
            f"⚠️ Your risk appears driven mainly by reported symptoms ({symptom_text}). "
            "These are not modifiable through lifestyle changes alone — **please consult a "
            "qualified physician promptly** for clinical evaluation (ECG, blood tests, imaging as needed).",
        ))

    else:
        tier = "doctor"
        recs.append((
            "doctor",
            "⚠️ No modifiable or symptom-related features with positive SHAP values "
            "were found among the Top-10 risk-increasing drivers. The remaining "
            "risk contributors may be non-modifiable factors — **please consult a "
            "qualified physician** for a full clinical risk assessment and monitoring plan.",
        ))

    recs.append((
        "doctor",
        "ℹ️ This tool produces a statistical risk estimate, **not a medical diagnosis**. "
        "Always confirm findings with a licensed healthcare professional.",
    ))

    return tier, recs


# ----------------------------------------------------------------------------
# Counterfactuals: rank single/paired plausible flips of top modifiable
# features by verified probability reduction (DiCE-style "what-if" search
# restricted to physiologically plausible single-step transitions, as in the
# CARE-Cirrhosis plausibility-constrained counterfactual module).
# ----------------------------------------------------------------------------
# def generate_counterfactuals(model, patient_dict, top_features, n_best=3):
#     base_row = to_pool_row(model, patient_dict)
#     base_prob = predict_proba_row(model, base_row)

#     candidate_features = [f for f in top_features if f in ALL_MODIFIABLE_FEATURES and f in BINARY_COLS]
#     if not candidate_features:
#         candidate_features = [f for f in ALL_MODIFIABLE_FEATURES if f in BINARY_COLS]

#     scenarios = []
#     # Single-feature flips
#     for f in candidate_features:
#         cur_val = int(patient_dict.get(f, 0))
#         if cur_val != 0:  # only meaningful to flip an active risk factor OFF
#             new_dict = dict(patient_dict)
#             new_dict[f] = 0
#             new_row = to_pool_row(model, new_dict)
#             new_prob = predict_proba_row(model, new_row)
#             reduction = base_prob - new_prob
#             if reduction > 0:
#                 scenarios.append({
#                     "label": f"If {FRIENDLY_NAMES.get(f, f)} = No",
#                     "changed": [f],
#                     "new_prob": new_prob,
#                     "reduction": reduction,
#                 })

#     # Paired flips among the top 2-3 candidates for stronger "best-case" scenarios
#     active_candidates = [f for f in candidate_features if int(patient_dict.get(f, 0)) != 0]
#     for i in range(len(active_candidates)):
#         for j in range(i + 1, len(active_candidates)):
#             f1, f2 = active_candidates[i], active_candidates[j]
#             new_dict = dict(patient_dict)
#             new_dict[f1], new_dict[f2] = 0, 0
#             new_row = to_pool_row(model, new_dict)
#             new_prob = predict_proba_row(model, new_row)
#             reduction = base_prob - new_prob
#             if reduction > 0:
#                 scenarios.append({
#                     "label": f"If {FRIENDLY_NAMES.get(f1, f1)} = No & {FRIENDLY_NAMES.get(f2, f2)} = No",
#                     "changed": [f1, f2],
#                     "new_prob": new_prob,
#                     "reduction": reduction,
#                 })

#     scenarios.sort(key=lambda s: s["reduction"], reverse=True)
#     return base_prob, scenarios[:n_best]

def generate_counterfactuals(model, patient_dict, top_features, n_best=3, threshold=0.50):
    """
    Counterfactual strategy:
    
    1. Try single-feature changes first.
    2. If one or more single-feature changes reduce predicted probability
       below the threshold, return the best valid single-feature CFs.
    3. Otherwise, try combinations of multiple features.
    4. Return valid multi-feature CFs that achieve probability < threshold,
       prioritizing fewer changes and then lower probability.
    """

    from itertools import combinations

    # ---------------------------------------------------------------
    # Original prediction
    # ---------------------------------------------------------------
    base_row = to_pool_row(model, patient_dict)
    base_prob = predict_proba_row(model, base_row)

    # ---------------------------------------------------------------
    # Candidate modifiable features
    # ---------------------------------------------------------------
    candidate_features = [
        f for f in top_features
        if f in ALL_MODIFIABLE_FEATURES and f in BINARY_COLS
    ]

    if not candidate_features:
        candidate_features = [
            f for f in ALL_MODIFIABLE_FEATURES
            if f in BINARY_COLS
        ]

    # Only features currently active (= 1) can be changed to 0
    active_candidates = [
        f for f in candidate_features
        if int(patient_dict.get(f, 0)) == 1
    ]

    # ---------------------------------------------------------------
    # 1. SINGLE-FEATURE COUNTERFACTUALS
    # ---------------------------------------------------------------
    single_valid = []

    for f in active_candidates:

        new_dict = dict(patient_dict)
        new_dict[f] = 0

        new_row = to_pool_row(model, new_dict)
        new_prob = predict_proba_row(model, new_row)

        reduction = base_prob - new_prob

        # Only accept if it changes prediction below threshold
        if new_prob < threshold:
            single_valid.append({
                "label": f"If {FRIENDLY_NAMES.get(f, f)} = No",
                "changed": [f],
                "n_changes": 1,
                "new_prob": new_prob,
                "reduction": reduction,
                "type": "Single-feature"
            })

    # If valid single-feature CFs exist, return them first
    if single_valid:
        single_valid.sort(
            key=lambda s: s["new_prob"]
        )

        return base_prob, single_valid[:n_best]

    # ---------------------------------------------------------------
    # 2. MULTI-FEATURE COUNTERFACTUALS
    # ---------------------------------------------------------------
    multi_valid = []

    # Start from pairs, then triples, etc.
    for n_changes in range(2, len(active_candidates) + 1):

        for combo in combinations(active_candidates, n_changes):

            new_dict = dict(patient_dict)

            # Change all selected active features from 1 -> 0
            for f in combo:
                new_dict[f] = 0

            new_row = to_pool_row(model, new_dict)
            new_prob = predict_proba_row(model, new_row)

            reduction = base_prob - new_prob

            # Accept only valid CFs below threshold
            if new_prob < threshold:

                feature_names = " & ".join(
                    FRIENDLY_NAMES.get(f, f)
                    for f in combo
                )

                multi_valid.append({
                    "label": f"If {feature_names} = No",
                    "changed": list(combo),
                    "n_changes": n_changes,
                    "new_prob": new_prob,
                    "reduction": reduction,
                    "type": f"{n_changes}-feature"
                })

        # If we found valid combinations at this level,
        # stop and return the smallest combination size.
        if multi_valid:

            multi_valid.sort(
                key=lambda s: (
                    s["n_changes"],
                    s["new_prob"]
                )
            )

            return base_prob, multi_valid[:n_best]

    # ---------------------------------------------------------------
    # 3. NO VALID COUNTERFACTUAL FOUND
    # ---------------------------------------------------------------
    return base_prob, []


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
def risk_gauge(prob):
    pct = prob * 100
    if pct < 30:
        color, label = "#16a34a", "LOW RISK"
    elif pct < 60:
        color, label = "#d97706", "MODERATE RISK"
    else:
        color, label = "#dc2626", "HIGH RISK"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 42, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#334155", "tickfont": {"color": "#334155"}},
            "bar": {"color": color, "thickness": 0.35},
            "bgcolor": "#f1f5f9",
            "borderwidth": 1,
            "bordercolor": "#e2e8f0",
            "steps": [
                {"range": [0, 30], "color": "#dcfce7"},
                {"range": [30, 60], "color": "#fef3c7"},
                {"range": [60, 100], "color": "#fee2e2"},
            ],
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        height=280, margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#0f172a"},
    )
    emoji = "🟢" if pct < 30 else ("🟡" if pct < 60 else "🔴")
    return fig, label, color, emoji


def badge_html(label, color_class):
    return f'<span class="care-badge {color_class}">{label}</span>'


# ----------------------------------------------------------------------------
# Sidebar: patient data entry (horizontal 2-column layout, icon-led)
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-header">
            <h2><i class="fa-solid fa-heart-pulse"></i>&nbsp; Patient Data Input</h2>
            <p>Fill in demographics, symptoms &amp; risk factors below.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section-title">🧑‍⚕️ Demographics</div>', unsafe_allow_html=True)
    demo_c1, demo_c2 = st.columns(2)
    with demo_c1:
        age = st.slider("Age (years)", 18, 95, 55)
    with demo_c2:
        gender = st.radio("Gender", ["Female", "Male"], horizontal=True)

    st.markdown(
        '<div class="sidebar-section-title">🫀 Cardiac / Systemic Symptoms</div>',
        unsafe_allow_html=True,
    )
    symptom_inputs = {}
    sym_cols = st.columns(2)
    for i, f in enumerate(SYMPTOM_COLS):
        with sym_cols[i % 2]:
            symptom_inputs[f] = st.checkbox(
                f"{FEATURE_ICONS.get(f, '•')} {FRIENDLY_NAMES[f]}", value=False, key=f"sym_{f}"
            )

    st.markdown(
        '<div class="sidebar-section-title">⚠️ Lifestyle &amp; Clinical Risk Factors</div>',
        unsafe_allow_html=True,
    )
    risk_inputs = {}
    risk_feature_list = [f for f in RISK_COLS if f != "Gender"]
    risk_cols = st.columns(2)
    for i, f in enumerate(risk_feature_list):
        with risk_cols[i % 2]:
            risk_inputs[f] = st.checkbox(
                f"{FEATURE_ICONS.get(f, '•')} {FRIENDLY_NAMES[f]}", value=False, key=f"risk_{f}"
            )

    st.divider()
    run_btn = st.button("🔍 Analyze Patient", use_container_width=True, type="primary")
    st.caption("🔒 Data stays in this session only — nothing is stored or sent to third parties.")

patient_dict = {"Age": age, "Gender": 1 if gender == "Male" else 0}
patient_dict.update({f: int(v) for f, v in symptom_inputs.items()})
patient_dict.update({f: int(v) for f, v in risk_inputs.items()})

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="care-hero">
        <h1><i class="fa-solid fa-heart-circle-check"></i>&nbsp; CARE-Heart Explainable Risk Assistant</h1>
        <p>A CatBoost-powered clinical decision-support tool combining SHAP explainability,
        a hybrid lifestyle recommendation engine, and personalized counterfactual "what-if"
        scenarios to help you understand — and act on — cardiovascular risk.</p>
        <div class="hero-chips">
            <span class="hero-chip"><i class="fa-solid fa-brain"></i>&nbsp; CatBoost ML</span>
            <span class="hero-chip"><i class="fa-solid fa-magnifying-glass-chart"></i>&nbsp; SHAP Explainability</span>
            <span class="hero-chip"><i class="fa-solid fa-list-check"></i>&nbsp; Personalized Advice</span>
            <span class="hero-chip"><i class="fa-solid fa-code-compare"></i>&nbsp; What-If Scenarios</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

model, Xtr, Xte, ytr, yte, data_source, test_acc = load_model_and_data()

top_bar1, top_bar2, top_bar3 = st.columns(3)
with top_bar1:
    st.markdown(
        f'''<div class="metric-tile tile-teal">
            <span class="m-icon">🗄️</span>
            <span class="m-label">Data Source</span>
            <span class="m-value">{data_source}</span>
        </div>''',
        unsafe_allow_html=True,
    )
with top_bar2:
    st.markdown(
        '''<div class="metric-tile tile-indigo">
            <span class="m-icon">🧠</span>
            <span class="m-label">Model</span>
            <span class="m-value">CatBoost (iters=300, depth=6, lr=0.02)</span>
        </div>''',
        unsafe_allow_html=True,
    )
with top_bar3:
    st.markdown(
        f'''<div class="metric-tile tile-amber">
            <span class="m-icon">✅</span>
            <span class="m-label">Held-out Accuracy</span>
            <span class="m-value">{test_acc:.1%}</span>
        </div>''',
        unsafe_allow_html=True,
    )

st.write("")

if not run_btn:
    st.info("👈 Fill in the patient's data in the sidebar and click **Analyze Patient** to generate the risk assessment, SHAP explanation, recommendations, and counterfactuals.")
    st.markdown(
        """
        <div class="care-card">
            <h3>🧭 How this assistant works</h3>
            <ol style="line-height:1.9; font-size:0.98rem;">
                <li><b>Enter patient data</b> in the sidebar — age, gender, symptoms and risk factors.</li>
                <li><b>CatBoost</b> predicts the probability of elevated cardiovascular risk.</li>
                <li><b>SHAP</b> ranks the Top-10 features driving that specific prediction.</li>
                <li>A <b>hybrid recommendation engine</b> prioritizes lifestyle advice, then clinical
                monitoring, then a "consult a doctor" flag — based on which risk-increasing
                features are modifiable.</li>
                <li><b>Counterfactual scenarios</b> show how the risk score would change if key
                modifiable factors were addressed.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ----------------------------------------------------------------------------
# Core computation
# ----------------------------------------------------------------------------
row = to_pool_row(model, patient_dict)
risk_prob = predict_proba_row(model, row)
local_shap = compute_local_shap(model, row)
shap_series = pd.Series(local_shap, index=FEATURE_COLS)
top10 = shap_series.abs().sort_values(ascending=False).head(10)
top10_signed = shap_series.loc[top10.index]
top10_feature_names = top10.index.tolist()

tier, recommendations = build_recommendations(top10_feature_names, shap_series)
base_prob, cf_scenarios = generate_counterfactuals(model, patient_dict, top10_feature_names, n_best=3)

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_risk, tab_shap, tab_rec, tab_cf = st.tabs(
    ["📊 Risk Overview", "🔬 SHAP Explainability", "💡 Recommendations", "🔁 Counterfactuals"]
)

# --- Tab 1: Risk overview -----------------------------------------------
with tab_risk:
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.markdown('<div class="care-card">', unsafe_allow_html=True)
        st.subheader("Predicted Risk Gauge")
        fig_gauge, risk_label, risk_color, risk_emoji = risk_gauge(risk_prob)
        st.plotly_chart(fig_gauge, use_container_width=True)
        badge_class = "badge-low" if risk_label == "LOW RISK" else ("badge-mod" if risk_label == "MODERATE RISK" else "badge-high")
        st.markdown(f"{risk_emoji} {badge_html(risk_label, badge_class)}", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="care-card">', unsafe_allow_html=True)
        st.subheader("Patient Snapshot")

        gender_icon = "🧑" if gender == "Male" else "👩"
        st.markdown(
            f'''
            <div style="display:flex; gap:10px; margin-bottom:14px; flex-wrap:wrap;">
                <span class="hero-chip" style="background:#eef2ff; border-color:#c7d2fe; color:#4338ca;">🎂 Age: {age}</span>
                <span class="hero-chip" style="background:#eef2ff; border-color:#c7d2fe; color:#4338ca;">{gender_icon} {gender}</span>
            </div>
            ''',
            unsafe_allow_html=True,
        )

        active_symptoms = [f for f in SYMPTOM_COLS if patient_dict[f] == 1]
        active_risks = [f for f in RISK_COLS if f != "Gender" and patient_dict[f] == 1]

        st.markdown("**🩹 Reported symptoms**")
        if active_symptoms:
            chips = "".join(
                f'<span class="hero-chip" style="background:#fef2f2; border-color:#fecaca; color:#b91c1c;">'
                f'{FEATURE_ICONS.get(f, "•")} {FRIENDLY_NAMES[f]}</span>'
                for f in active_symptoms
            )
            st.markdown(f'<div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px;">{chips}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:#52607a; margin-bottom:12px;">None reported</p>', unsafe_allow_html=True)

        st.markdown("**⚠️ Active risk / lifestyle factors**")
        if active_risks:
            chips = "".join(
                f'<span class="hero-chip" style="background:#fffbeb; border-color:#fde68a; color:#b45309;">'
                f'{FEATURE_ICONS.get(f, "•")} {FRIENDLY_NAMES[f]}</span>'
                for f in active_risks
            )
            st.markdown(f'<div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px;">{chips}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:#52607a; margin-bottom:12px;">None reported</p>', unsafe_allow_html=True)

        st.progress(min(int(risk_prob * 100), 100), text=f"Model-estimated heart-risk probability: {risk_prob*100:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

# --- Tab 2: SHAP explainability ------------------------------------------
with tab_shap:
    st.markdown('<div class="care-card">', unsafe_allow_html=True)
    st.subheader("Top 10 Ranked Features by |SHAP Value| (this patient)")
    plot_df = pd.DataFrame({
        "Feature": [FRIENDLY_NAMES.get(f, f) for f in top10_signed.index],
        "SHAP value": top10_signed.values,
        "Direction": np.where(top10_signed.values >= 0, "Increases risk", "Decreases risk"),
    }).sort_values("SHAP value")

    fig_shap = px.bar(
        plot_df, x="SHAP value", y="Feature", color="Direction", orientation="h",
        color_discrete_map={"Increases risk": "#ef4444", "Decreases risk": "#22c55e"},
        title=None,
    )
    fig_shap.update_layout(
        height=460,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#111827", size=12),
        xaxis=dict(
            tickfont=dict(color="#111827", size=11),
            title=dict(text="SHAP value", font=dict(color="#111827", size=12)),
        ),
        yaxis=dict(
            tickfont=dict(color="#111827", size=11),
            title=dict(text="Feature", font=dict(color="#111827", size=12)),
        ),
        legend=dict(
            font=dict(color="#111827", size=11),
            title=dict(text="", font=dict(color="#111827")),
        ),
        legend_title_text="",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_shap, use_container_width=True)
    with st.expander("View raw SHAP ranking table"):
        st.dataframe(
            plot_df.rename(columns={"SHAP value": "SHAP value (signed)"}).iloc[::-1].reset_index(drop=True),
            use_container_width=True,
        )
    st.caption("Positive SHAP values push the predicted probability of heart risk up; negative values pull it down.")
    st.markdown("</div>", unsafe_allow_html=True)


# --- Tab 3: Recommendations ------------------------------------------------
with tab_rec:
    st.markdown('<div class="care-card">', unsafe_allow_html=True)
    tier_title = {
        "lifestyle": "🟢 Lifestyle-Modifiable Drivers Found",
        "moderate": "🟡 Moderately-Modifiable Clinical Drivers Found",
        "doctor": "🔴 Non-Modifiable / Symptom Drivers — Clinical Review Advised",
    }[tier]
    st.subheader(tier_title)
    st.caption("Recommendations are generated only from Top-10 features with positive SHAP values (features that increase this patient’s predicted risk), using the lifestyle → moderately-modifiable → symptom priority order.")
    for kind, text in recommendations:
        st.markdown(f'<div class="rec-item {kind}">{text}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Tab 4: Counterfactuals -------------------------------------------------
with tab_cf:
    st.markdown('<div class="care-card">', unsafe_allow_html=True)
    st.subheader("Top-3 Counterfactual 'What-If' Scenarios")
    st.caption("Simulated changes to modifiable risk factors, ranked by verified probability reduction from the trained model.")

    if not cf_scenarios:
        st.info("No modifiable features among this patient's active risk factors could be flipped to produce a meaningful risk reduction — risk appears driven by non-modifiable factors. Please see the Recommendations tab.")
    else:
        cols = st.columns(len(cf_scenarios))
        for i, (col, sc) in enumerate(zip(cols, cf_scenarios), start=1):
            with col:
                st.markdown(
                    f"""
                    <div class="cf-card">
                        <span class="hero-chip" style="background:#f0fdfa; border-color:#99f6e4; color:#0f766e;">🔁 Scenario {i}</span>
                        <div style="margin:10px 0; font-weight:600;">{sc['label']}</div>
                        <div style="font-size:0.95rem;">Risk: <b>{base_prob*100:.1f}%</b> → <b style="color:#16a34a;">{sc['new_prob']*100:.1f}%</b></div>
                        <div style="color:#16a34a; font-weight:700; margin-top:4px;">▼ {sc['reduction']*100:.1f} pp reduction</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        bar_df = pd.DataFrame({
            "Scenario": [sc["label"] for sc in cf_scenarios],
            "Risk reduction (pp)": [sc["reduction"] * 100 for sc in cf_scenarios],
        })
        fig_cf = px.bar(
            bar_df, x="Risk reduction (pp)", y="Scenario", orientation="h",
            color="Risk reduction (pp)", color_continuous_scale="Greens",
        )
        fig_cf.update_layout(
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#111827", size=12),
            xaxis=dict(
                tickfont=dict(color="#111827", size=11),
                title=dict(text="Risk reduction (pp)", font=dict(color="#111827", size=12)),
            ),
            yaxis=dict(
                tickfont=dict(color="#111827", size=11),
                title=dict(text="Scenario", font=dict(color="#111827", size=12)),
            ),
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_cf, use_container_width=True)

        prob_df = pd.DataFrame({
            "Scenario": ["Current"] + [sc["label"] for sc in cf_scenarios],
            "Predicted risk (%)": [base_prob * 100] + [sc["new_prob"] * 100 for sc in cf_scenarios],
        })
        fig_prob = px.bar(
            prob_df, x="Scenario", y="Predicted risk (%)",
            color="Predicted risk (%)", color_continuous_scale="Reds",
        )
        fig_prob.update_layout(
            height=340,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#111827", size=12),
            xaxis=dict(
                tickfont=dict(color="#111827", size=11),
                title=dict(text="Scenario", font=dict(color="#111827", size=12)),
            ),
            yaxis=dict(
                tickfont=dict(color="#111827", size=11),
                title=dict(text="Predicted risk (%)", font=dict(color="#111827", size=12)),
            ),
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    st.caption("⚠️ Counterfactual scenarios are illustrative decision-support aids, not medical guarantees — always confirm feasibility of changes with a clinician.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="app-footer">
        <i class="fa-solid fa-shield-heart" style="color:#0d9488;"></i>
        &nbsp;<b>CARE-Heart</b> — an explainability framework integrating predictive modeling
        and personalized clinical recommendations.<br>
        For research / educational use only. Not a substitute for professional medical advice.
    </div>
    """,
    unsafe_allow_html=True,
)

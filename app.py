"""
ChurnIQ — Customer Churn Intelligence Platform
Streamlit App | Logistic Regression + SMOTE Pipeline
Fully theme-adaptive: works identically in light and dark mode.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    auc as sklearn_auc
)
from imblearn.over_sampling import SMOTE
from scipy.stats import loguniform

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Theme detection (reliable method) ────────────────────────────────────────
try:
    params = st.query_params
    _theme = params.get("theme", "light")
    if isinstance(_theme, list):
        _theme = _theme[0]
except Exception:
    _theme = "light"

DARK = (_theme == "dark")
MPL_BG   = "#1e293b" if DARK else "#ffffff"
MPL_FG   = "#f1f5f9" if DARK else "#0f172a"
MPL_GRID = "#334155" if DARK else "#e2e8f0"
MPL_AX   = "#1e293b" if DARK else "#ffffff"

RANDOM_STATE = 42

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ════════════════════════════════════════════════════════════
   DESIGN TOKENS — light defaults
════════════════════════════════════════════════════════════ */
:root,
.stApp[data-theme="light"],
[data-theme="light"] {
    --bg-card:       #ffffff;
    --bg-card2:      #f8fafc;
    --bg-card3:      #f1f5f9;
    --border:        #e2e8f0;
    --border-soft:   #f1f5f9;
    --text-h:        #0f172a;
    --text-b:        #334155;
    --text-s:        #64748b;
    --text-m:        #94a3b8;
    --accent:        #3b82f6;
    --tab-bg:        #f1f5f9;
    --tab-inactive:  #64748b;
    --badge-bg:      #f0fdf4;
    --badge-br:      #86efac;
    --badge-tx:      #16a34a;
    --pred-idle-bg:  #f8fafc;
    --pred-idle-br:  #e2e8f0;
    --pred-churn-bg: #fff1f2;
    --pred-churn-br: #fca5a5;
    --pred-stay-bg:  #f0fdf4;
    --pred-stay-br:  #86efac;
    --risk-track:    #e2e8f0;
    --risk-lbl:      #94a3b8;
    --prob-col:      #475569;
    --divider:       #e2e8f0;
}

/* ── Dark overrides ── */
.stApp[data-theme="dark"],
[data-theme="dark"] {
    --bg-card:       #1e293b;
    --bg-card2:      #0f172a;
    --bg-card3:      #162032;
    --border:        #334155;
    --border-soft:   #1e293b;
    --text-h:        #f1f5f9;
    --text-b:        #cbd5e1;
    --text-s:        #94a3b8;
    --text-m:        #64748b;
    --accent:        #60a5fa;
    --tab-bg:        #1e293b;
    --tab-inactive:  #94a3b8;
    --badge-bg:      #052e16;
    --badge-br:      #166534;
    --badge-tx:      #4ade80;
    --pred-idle-bg:  #1e293b;
    --pred-idle-br:  #334155;
    --pred-churn-bg: #2d0a0a;
    --pred-churn-br: #991b1b;
    --pred-stay-bg:  #052e16;
    --pred-stay-br:  #166534;
    --risk-track:    #334155;
    --risk-lbl:      #64748b;
    --prob-col:      #94a3b8;
    --divider:       #334155;
}

/* Fallback for prefers-color-scheme */
@media (prefers-color-scheme: dark) {
    :root {
        --bg-card:       #1e293b;
        --bg-card2:      #0f172a;
        --bg-card3:      #162032;
        --border:        #334155;
        --border-soft:   #1e293b;
        --text-h:        #f1f5f9;
        --text-b:        #cbd5e1;
        --text-s:        #94a3b8;
        --text-m:        #64748b;
        --accent:        #60a5fa;
        --tab-bg:        #1e293b;
        --tab-inactive:  #94a3b8;
        --badge-bg:      #052e16;
        --badge-br:      #166534;
        --badge-tx:      #4ade80;
        --pred-idle-bg:  #1e293b;
        --pred-idle-br:  #334155;
        --pred-churn-bg: #2d0a0a;
        --pred-churn-br: #991b1b;
        --pred-stay-bg:  #052e16;
        --pred-stay-br:  #166534;
        --risk-track:    #334155;
        --risk-lbl:      #64748b;
        --prob-col:      #94a3b8;
        --divider:       #334155;
    }
}

/* ════════════════════════════════════════════════════════════
   GLOBAL
════════════════════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }


/* ── Top bar ── */
.top-bar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 24px 0 28px 0;
    border-bottom: 1.5px solid var(--divider);
    margin-bottom: 28px;
}
.brand          { display: flex; align-items: center; gap: 12px; }
.brand-icon     { width:44px; height:44px; background:linear-gradient(135deg,#1e3a5f,#0f172a);
                  border-radius:12px; font-size:22px; display:flex; align-items:center; justify-content:center; }
.brand-title {
    font-size: 48px !important;
    font-weight: 800 !important;
    color: var(--text-h) !important;
    margin: 0;
    letter-spacing: -0.5px;
    line-height: 1.2;
}
.brand-sub      { font-size:12px; color:var(--text-m); margin:0; }
.model-badge    { background:var(--badge-bg); border:1.5px solid var(--badge-br);
                  color:var(--badge-tx); font-size:12px; font-weight:600;
                  padding:6px 14px; border-radius:20px; }

/* ── KPI cards ── */
.kpi-row { display:flex; gap:16px; margin-bottom:28px; flex-wrap:wrap; }
.kpi-card {
    flex:1; min-width:160px;
    background:var(--bg-card); border:1.5px solid var(--border);
    border-radius:14px; padding:20px 22px;
    position:relative; overflow:hidden;
}
.kpi-card::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; }
.kpi-green::before  { background:linear-gradient(90deg,#22c55e,#86efac); }
.kpi-blue::before   { background:linear-gradient(90deg,#3b82f6,#93c5fd); }
.kpi-red::before    { background:linear-gradient(90deg,#ef4444,#fca5a5); }
.kpi-purple::before { background:linear-gradient(90deg,#8b5cf6,#c4b5fd); }
.kpi-val   { font-size:32px; font-weight:700; color:var(--text-h);  margin:0 0 4px 0; }
.kpi-label { font-size:11px; font-weight:600; letter-spacing:.08em;
             color:var(--text-m); text-transform:uppercase; margin:0 0 8px 0; }
.kpi-sub   { font-size:12px; color:var(--accent); font-weight:500; margin:0; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap:0; background:var(--tab-bg);
    border-radius:12px; padding:4px;
    border:1.5px solid var(--border);
    width:fit-content; margin-bottom:24px;
}
.stTabs [data-baseweb="tab"] {
    border-radius:9px; padding:8px 20px;
    font-size:13px; font-weight:500;
    color:var(--tab-inactive) !important;
    background:transparent; border:none;
}
.stTabs [aria-selected="true"] {
    background:#3b82f6 !important; color:#ffffff !important;
}

/* ── Section header ── */
.section-hdr {
    font-size:10px; letter-spacing:.12em; font-weight:700;
    color:var(--text-m); text-transform:uppercase;
    margin:0 0 16px 0; padding-bottom:8px;
    border-bottom:1.5px solid var(--border);
}

/* ── Prediction box ── */
.pred-box {
    background:var(--pred-idle-bg); border:1.5px solid var(--pred-idle-br);
    border-radius:14px; padding:32px; text-align:center;
    min-height:200px; display:flex; flex-direction:column;
    align-items:center; justify-content:center;
}
.pred-churn   { background:var(--pred-churn-bg)!important; border-color:var(--pred-churn-br)!important; }
.pred-nochurn { background:var(--pred-stay-bg) !important; border-color:var(--pred-stay-br) !important; }
.pred-label   { font-size:26px; font-weight:700; margin:8px 0; }
.pred-prob    { font-size:14px; color:var(--prob-col) !important; }
.pred-idle-lbl{ color:var(--text-m) !important; font-size:14px; margin-top:12px; line-height:1.6; }

/* ── Risk meter ── */
.risk-wrap { margin:16px 0; width:100%; }
.risk-lbls  { display:flex; justify-content:space-between; font-size:11px;
               color:var(--risk-lbl) !important; margin-bottom:5px; }
.risk-bg    { height:10px; background:var(--risk-track); border-radius:999px; overflow:hidden; }
.risk-fill  { height:100%; border-radius:999px; }

/* ══════════════════════════════════════════════════════════
   RISK FACTOR TAGS
══════════════════════════════════════════════════════════ */
div.rtag-high,
.stApp div.rtag-high,
[data-theme="dark"] div.rtag-high,
[data-theme="light"] div.rtag-high {
    background-color: rgba(239, 68, 68, 0.20) !important;
    color: #ff6b6b !important;
    border-left: 3px solid #ef4444 !important;
    padding: 9px 13px !important;
    border-radius: 8px !important;
    margin: 4px 0 !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    display: block !important;
}

div.rtag-medium,
.stApp div.rtag-medium,
[data-theme="dark"] div.rtag-medium,
[data-theme="light"] div.rtag-medium {
    background-color: rgba(249, 115, 22, 0.20) !important;
    color: #fb923c !important;
    border-left: 3px solid #f97316 !important;
    padding: 9px 13px !important;
    border-radius: 8px !important;
    margin: 4px 0 !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    display: block !important;
}

div.rtag-ok,
.stApp div.rtag-ok,
[data-theme="dark"] div.rtag-ok,
[data-theme="light"] div.rtag-ok {
    background-color: rgba(34, 197, 94, 0.20) !important;
    color: #4ade80 !important;
    border-left: 3px solid #22c55e !important;
    padding: 9px 13px !important;
    border-radius: 8px !important;
    margin: 4px 0 !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    display: block !important;
}

div.rtag-high *,
div.rtag-medium *,
div.rtag-ok * {
    color: inherit !important;
}

/* ── Info/metric card ── */
.info-card {
    background:var(--bg-card); border:1.5px solid var(--border);
    border-radius:12px; padding:16px 20px; margin-bottom:16px; text-align:center;
}
.info-val   { font-size:22px; font-weight:700; color:var(--text-h) !important; }
.info-icon  { font-size:24px; }
.info-label { font-size:11px; color:var(--text-m) !important; text-transform:uppercase; letter-spacing:.06em; }

/* ── Data table ── */
.stDataFrame { border-radius:12px; overflow:hidden; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  DATA & MODEL PIPELINE (cached)
# ═══════════════════════════════════════════════════════════════
@st.cache_data
def load_and_clean(path="telecom_dataset.csv"):
    df = pd.read_csv(path)
    df.drop(columns=["customerID"], inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = 0
    df["Churn"]  = df["Churn"].map({"Yes": 1, "No": 0})
    df["gender"] = df["gender"].map({"Female": 1, "Male": 0})
    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        if df[col].dtype == object:
            df[col] = df[col].map({"Yes": 1, "No": 0})
    return df


def engineer_features(df, median_charge):
    df = df.copy()
    add_svcs = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]
    for col in ["MultipleLines"] + add_svcs:
        df[col] = df[col].replace({"No phone service": "No", "No internet service": "No"})

    df["IsFirstYear"]      = (df["tenure"] <= 12).astype(int)
    df["AvgMonthlyCharge"] = df.apply(
        lambda x: x["TotalCharges"] / x["tenure"] if x["tenure"] > 0 else x["MonthlyCharges"],
        axis=1
    )

    all_svcs_c = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                  "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

    # FIX: cast to plain numpy int to avoid Arrow-backed dtype arithmetic errors
    svc_num = df[all_svcs_c].apply(
        lambda c: c.map({"Yes": 1, "No": 0}) if c.dtype == object else c
    ).astype(int)

    df["ChargePerService"]      = df["MonthlyCharges"] / (svc_num.sum(axis=1) + 1)
    df["HighCostLowTenure"]     = (
        (df["MonthlyCharges"] > median_charge) & (df["tenure"] < 12)
    ).astype(int)
    df["NumAdditionalServices"] = df[add_svcs].apply(lambda x: (x == "Yes").sum(), axis=1)
    df["HasInternetService"]    = (df["InternetService"] != "No").astype(int)
    df["FiberOpticUser"]        = (df["InternetService"] == "Fiber optic").astype(int)
    df["IsMonthToMonth"]        = (df["Contract"] == "Month-to-month").astype(int)
    df["ChargeContractRisk"]    = df["MonthlyCharges"] * df["IsMonthToMonth"]
    df["PaymentRisk"] = df["PaymentMethod"].map({
        "Electronic check": 3,
        "Mailed check": 2,
        "Bank transfer (automatic)": 1,
        "Credit card (automatic)": 1,
    })
    df["AutoPayment"]            = df["PaymentMethod"].apply(
        lambda x: 1 if "automatic" in str(x).lower() else 0
    )
    df["HasFamily"]              = (
        (df["Partner"] == 1) | (df["Dependents"] == 1)
    ).astype(int)
    df["SeniorAlone"]            = (
        (df["SeniorCitizen"] == 1) & (df["Partner"] == 0) & (df["Dependents"] == 0)
    ).astype(int)
    df["SeniorMonthlyNoSupport"] = (
        (df["SeniorCitizen"] == 1) & (df["IsMonthToMonth"] == 1) & (df["TechSupport"] == "No")
    ).astype(int)
    df["MultipleRiskFactors"]    = (
        df["IsMonthToMonth"]
        + df["FiberOpticUser"]
        + (df["PaymentRisk"] == 3).astype(int)
        + (df["tenure"] <= 12).astype(int)
    )
    return df


@st.cache_resource
def train_pipeline(path="telecom_dataset.csv"):
    df = load_and_clean(path)
    X, y = df.drop("Churn", axis=1), df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    mc = X_train["MonthlyCharges"].median()
    X_train = engineer_features(X_train, mc)
    X_test  = engineer_features(X_test,  mc)

    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test  = pd.get_dummies(X_test,  drop_first=True)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)
    feat_cols = X_train.columns.tolist()

    scaler  = StandardScaler()
    Xtr_sc  = scaler.fit_transform(X_train)
    Xte_sc  = scaler.transform(X_test)

    smote       = SMOTE(random_state=42, k_neighbors=5, sampling_strategy=1.0)
    Xtr_r, ytr_r = smote.fit_resample(Xtr_sc, y_train)

    l1 = LogisticRegression(
        penalty="l1", solver="liblinear", C=0.15, max_iter=1000, random_state=42
    )
    l1.fit(Xtr_r, ytr_r)
    mask      = l1.coef_[0] != 0
    sel_feats = [f for f, m in zip(feat_cols, mask) if m]
    Xtr_sel   = Xtr_r[:, mask]
    Xte_sel   = Xte_sc[:, mask]

    rs = RandomizedSearchCV(
        LogisticRegression(random_state=42),
        param_distributions={
            "C": loguniform(0.01, 10),
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"],
            "max_iter": [500, 1000],
        },
        n_iter=30,
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
        scoring="recall",
        n_jobs=-1,
        random_state=42,
    )
    rs.fit(Xtr_sel, ytr_r)
    best = rs.best_estimator_
    best.fit(Xtr_sel, ytr_r)

    y_prob = best.predict_proba(Xte_sel)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = dict(
        accuracy =accuracy_score(y_test, y_pred),
        recall   =recall_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred),
        f1       =f1_score(y_test, y_pred),
        roc_auc  =roc_auc_score(y_test, y_prob),
    )
    return dict(
        model=best, scaler=scaler,
        feature_cols=feat_cols, l1_mask=mask,
        selected_features=sel_feats,
        median_charge=mc,
        metrics=metrics,
        y_test=y_test, y_pred=y_pred, y_prob=y_prob,
        n_train=len(ytr_r),
        churn_rate=y.mean(), n_rows=len(df),
        best_params=rs.best_params_,
    )


def predict_single(row_dict, p):
    df_in = pd.DataFrame([row_dict])
    df_in = engineer_features(df_in, p["median_charge"])
    df_in = pd.get_dummies(df_in, drop_first=True)
    df_in = df_in.reindex(columns=p["feature_cols"], fill_value=0)
    sc    = p["scaler"].transform(df_in)
    sel   = sc[:, p["l1_mask"]]
    prob  = p["model"].predict_proba(sel)[0, 1]
    return prob, int(prob >= 0.5)


# ═══════════════════════════════════════════════════════════════
#  MATPLOTLIB THEME HELPER
# ═══════════════════════════════════════════════════════════════
def apply_mpl_theme(fig, *axes):
    fig.patch.set_facecolor(MPL_BG)
    for ax in axes:
        ax.set_facecolor(MPL_AX)
        ax.tick_params(colors=MPL_FG, labelsize=10)
        ax.xaxis.label.set_color(MPL_FG)
        ax.yaxis.label.set_color(MPL_FG)
        ax.title.set_color(MPL_FG)
        for spine in ax.spines.values():
            spine.set_edgecolor(MPL_GRID)


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Model Info")
    st.markdown("""
- **Algorithm:** Logistic Regression
- **Imbalance:** SMOTE (1:1)
- **Feature sel.:** L1 (Lasso)
- **Tuning:** RandomizedSearchCV
- **CV:** Stratified 10-Fold
""")

# ═══════════════════════════════════════════════════════════════
#  LOAD PIPELINE
# ═══════════════════════════════════════════════════════════════
with st.spinner("🔄 Training model pipeline… (~15 sec first run)"):
    try:
        pipeline = train_pipeline("telecom_dataset.csv")
    except FileNotFoundError:
        st.error("⚠️ **telecom_dataset.csv not found.** Upload it via the sidebar.")
        st.stop()

m = pipeline["metrics"]

# ═══════════════════════════════════════════════════════════════
#  TOP BAR
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="top-bar">
  <div class="brand">
    <div class="brand-icon">📡</div>
    <div>
      <p class="brand-title">Telecom Customer Churn Predictor</p>
    </div>
  </div>
  <div class="model-badge">✓ MODEL READY</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  KPI CARDS
# ═══════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card kpi-green">
    <p class="kpi-label">Model Accuracy</p>
    <p class="kpi-val">{m['accuracy']*100:.1f}%</p>
    <p class="kpi-sub">↑ Logistic Regression</p>
  </div>
  <div class="kpi-card kpi-blue">
    <p class="kpi-label">ROC-AUC Score</p>
    <p class="kpi-val">{m['roc_auc']:.3f}</p>
    <p class="kpi-sub">↑ Test set</p>
  </div>
  <div class="kpi-card kpi-red">
    <p class="kpi-label">Dataset Churn Rate</p>
    <p class="kpi-val">{pipeline['churn_rate']*100:.1f}%</p>
    <p class="kpi-sub">{int(pipeline['churn_rate']*pipeline['n_rows']):,} customers</p>
  </div>
  <div class="kpi-card kpi-purple">
    <p class="kpi-label">Training Records</p>
    <p class="kpi-val">{pipeline['n_train']:,}</p>
    <p class="kpi-sub">↑ {len(pipeline['selected_features'])} features selected</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Single Prediction", "📋 Batch Prediction", "📈 Model Performance", "📊 Dataset"]
)

# ──────────────────────────────────────────────────────────────
#  TAB 1 — SINGLE PREDICTION
# ──────────────────────────────────────────────────────────────
with tab1:
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.markdown('<p class="section-hdr">Customer Profile</p>', unsafe_allow_html=True)

        with st.expander("👤  DEMOGRAPHICS", expanded=True):
            c1, c2, c3 = st.columns(3)
            gender   = c1.selectbox("Gender",         ["Female", "Male"])
            senior   = c2.selectbox("Senior Citizen", ["No", "Yes"])
            partner  = c3.selectbox("Partner",        ["No", "Yes"])
            c4, c5   = st.columns(2)
            depends  = c4.selectbox("Dependents",     ["No", "Yes"])
            tenure   = c5.number_input("Tenure (months)", 0, 72, 12)

        with st.expander("📞  PHONE SERVICES", expanded=True):
            c1, c2 = st.columns(2)
            phone  = c1.selectbox("Phone Service",  ["Yes", "No"])
            multi  = c2.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

        with st.expander("🌐  INTERNET SERVICES", expanded=True):
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            c1, c2, c3 = st.columns(3)
            sec    = c1.selectbox("Online Security",   ["No", "Yes", "No internet service"])
            backup = c2.selectbox("Online Backup",     ["No", "Yes", "No internet service"])
            devpro = c3.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            c4, c5, c6 = st.columns(3)
            tech   = c4.selectbox("Tech Support",      ["No", "Yes", "No internet service"])
            tv     = c5.selectbox("Streaming TV",      ["No", "Yes", "No internet service"])
            movies = c6.selectbox("Streaming Movies",  ["No", "Yes", "No internet service"])

        with st.expander("💳  BILLING & CONTRACT", expanded=True):
            c1, c2    = st.columns(2)
            contract  = c1.selectbox("Contract",          ["Month-to-month", "One year", "Two year"])
            paperless = c2.selectbox("Paperless Billing",  ["Yes", "No"])
            payment   = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)",
            ])
            c3, c4  = st.columns(2)
            monthly = c3.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
            total   = c4.number_input(
                "Total Charges ($)", 0.0, 9000.0, float(monthly * max(tenure, 1)), step=1.0
            )

        predict_btn = st.button("🔮  Predict Churn", type="primary", use_container_width=True)

    with right_col:
        st.markdown('<p class="section-hdr">Prediction Output</p>', unsafe_allow_html=True)

        if not predict_btn:
            st.markdown("""
            <div class="pred-box">
              <div style="font-size:48px">🔮</div>
              <p class="pred-idle-lbl">Fill in the customer profile<br>and click <strong>Predict Churn</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            row = dict(
                gender=gender,
                SeniorCitizen=1 if senior == "Yes" else 0,
                Partner=partner, Dependents=depends, tenure=tenure,
                PhoneService=phone, MultipleLines=multi, InternetService=internet,
                OnlineSecurity=sec, OnlineBackup=backup, DeviceProtection=devpro,
                TechSupport=tech, StreamingTV=tv, StreamingMovies=movies,
                Contract=contract, PaperlessBilling=paperless,
                PaymentMethod=payment, MonthlyCharges=monthly, TotalCharges=total,
            )
            prob, pred = predict_single(row, pipeline)
            pct = prob * 100

            if pred == 1:
                emoji, label, box_cls, color = "🚨", "LIKELY TO CHURN", "pred-churn", "#ff6b6b"
                bar_grad = "linear-gradient(90deg,#ef4444,#fca5a5)"
            else:
                emoji, label, box_cls, color = "✅", "LIKELY TO STAY", "pred-nochurn", "#4ade80"
                bar_grad = "linear-gradient(90deg,#22c55e,#86efac)"

            st.markdown(f"""
            <div class="pred-box {box_cls}">
              <div style="font-size:52px">{emoji}</div>
              <p class="pred-label" style="color:{color} !important">{label}</p>
              <p class="pred-prob" style="color:var(--prob-col) !important">
                Churn Probability: <strong>{pct:.1f}%</strong>
              </p>
              <div class="risk-wrap">
                <div class="risk-lbls">
                  <span>0%</span><span>Risk Meter</span><span>100%</span>
                </div>
                <div class="risk-bg">
                  <div class="risk-fill" style="width:{pct:.0f}%;background:{bar_grad}"></div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**⚠️ Risk Factor Summary**")

            factors = []
            if contract == "Month-to-month":
                factors.append(("⚠️ Month-to-month contract", "high"))
            if internet == "Fiber optic":
                factors.append(("⚠️ Fiber optic user", "medium"))
            if payment == "Electronic check":
                factors.append(("⚠️ Electronic check payment", "medium"))
            if tenure <= 12:
                factors.append(("⚠️ Low tenure (≤12 months)", "high"))
            if monthly > pipeline["median_charge"] and tenure < 12:
                factors.append(("⚠️ High cost + low tenure", "high"))

            if not factors:
                st.markdown(
                    '<div class="rtag-ok">✅ No major risk factors detected</div>',
                    unsafe_allow_html=True,
                )
            else:
                for txt, lvl in factors:
                    cls = "rtag-high" if lvl == "high" else "rtag-medium"
                    st.markdown(
                        f'<div class="{cls}">{txt}</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("---")
            st.markdown("**💡 Recommended Action**")
            if pred == 1 and prob > 0.7:
                st.error("🔴 **High Priority** — Offer retention deal immediately. Consider a contract upgrade incentive.")
            elif pred == 1:
                st.warning("🟡 **Medium Priority** — Proactive outreach recommended. Monitor over next 30 days.")
            else:
                st.success("🟢 **Low Priority** — Customer appears stable. Regular engagement sufficient.")

# ──────────────────────────────────────────────────────────────
#  TAB 2 — BATCH PREDICTION
# ──────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-hdr">Batch Prediction — Upload Customer File</p>', unsafe_allow_html=True)
    batch_file = st.file_uploader(
        "Upload CSV (same schema as telecom_dataset.csv, Churn column not required)",
        type=["csv"], key="batch",
    )

    if batch_file:
        df_batch = pd.read_csv(batch_file)
        st.markdown(f"**{len(df_batch):,} records loaded.** Preview:")
        st.dataframe(df_batch.head(5), use_container_width=True)

        if st.button("▶️  Run Batch Prediction", type="primary"):
            results, prog = [], st.progress(0)
            for i, row in df_batch.iterrows():
                prob, pred = predict_single(row.to_dict(), pipeline)
                results.append({
                    "customerID":        row.get("customerID", i),
                    "Churn_Probability": round(prob, 4),
                    "Prediction":        "Churn" if pred else "No Churn",
                    "Risk_Level":        "High" if prob > 0.7 else ("Medium" if prob > 0.4 else "Low"),
                })
                prog.progress((i + 1) / len(df_batch))
            prog.empty()

            df_res = pd.DataFrame(results)
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Customers",    f"{len(df_res):,}")
            c2.metric("Predicted Churners", f"{(df_res['Prediction']=='Churn').sum():,}")
            c3.metric("Batch Churn Rate",   f"{(df_res['Prediction']=='Churn').mean()*100:.1f}%")

            def _color_risk(val):
                if val == "High":   return "background-color:rgba(239,68,68,.2);color:#ef4444"
                if val == "Medium": return "background-color:rgba(249,115,22,.2);color:#f97316"
                return "background-color:rgba(34,197,94,.2);color:#22c55e"

            st.dataframe(
                df_res.style.applymap(_color_risk, subset=["Risk_Level"]),
                use_container_width=True, height=400,
            )
            st.download_button(
                "⬇️ Download Results CSV",
                df_res.to_csv(index=False).encode(),
                "churn_predictions.csv",
                "text/csv",
            )
    else:
        st.info("Upload a CSV file with the same columns as the training dataset.")

# ──────────────────────────────────────────────────────────────
#  TAB 3 — MODEL PERFORMANCE
# ──────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-hdr">Model Performance Metrics</p>', unsafe_allow_html=True)

    cols_m = st.columns(5)
    for col, label, val, icon in [
        (cols_m[0], "Accuracy",  m["accuracy"],  "🎯"),
        (cols_m[1], "Recall",    m["recall"],     "📡"),
        (cols_m[2], "Precision", m["precision"],  "🔬"),
        (cols_m[3], "F1 Score",  m["f1"],         "⚖️"),
        (cols_m[4], "ROC-AUC",   m["roc_auc"],    "📈"),
    ]:
        col.markdown(f"""
        <div class="info-card">
          <div class="info-icon">{icon}</div>
          <div class="info-val">{val:.3f}</div>
          <div class="info-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    # ── Three charts ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    apply_mpl_theme(fig, *axes)

    # 1. Confusion Matrix
    cm   = confusion_matrix(pipeline["y_test"], pipeline["y_pred"])
    cmap = "Blues" if not DARK else sns.dark_palette("#60a5fa", as_cmap=True)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap,
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=axes[0], linewidths=1, linecolor=MPL_GRID,
        annot_kws={"color": MPL_FG, "size": 13, "weight": "bold"},
    )
    axes[0].set_title("Confusion Matrix", fontweight="bold", pad=12, color=MPL_FG)
    axes[0].set_xlabel("Predicted", color=MPL_FG)
    axes[0].set_ylabel("Actual",    color=MPL_FG)
    axes[0].tick_params(colors=MPL_FG)

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(pipeline["y_test"], pipeline["y_prob"])
    auc_val      = sklearn_auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color="#60a5fa", lw=2.5, label=f"AUC = {auc_val:.4f}")
    axes[1].fill_between(fpr, tpr, alpha=0.1, color="#60a5fa")
    axes[1].plot([0, 1], [0, 1], "--", color=MPL_GRID, lw=1.2)
    axes[1].set_title("ROC Curve", fontweight="bold", pad=12, color=MPL_FG)
    axes[1].set_xlabel("False Positive Rate", color=MPL_FG)
    axes[1].set_ylabel("True Positive Rate",  color=MPL_FG)
    axes[1].legend(loc="lower right", fontsize=11,
                   facecolor=MPL_BG, edgecolor=MPL_GRID, labelcolor=MPL_FG)
    axes[1].tick_params(colors=MPL_FG)
    for sp in axes[1].spines.values(): sp.set_edgecolor(MPL_GRID)

    # 3. Probability Distribution
    p_no = pipeline["y_prob"][pipeline["y_test"] == 0]
    p_ch = pipeline["y_prob"][pipeline["y_test"] == 1]
    axes[2].hist(p_no, bins=30, alpha=0.65, color="#22c55e", label="No Churn", density=True)
    axes[2].hist(p_ch, bins=30, alpha=0.65, color="#ef4444", label="Churn",    density=True)
    axes[2].axvline(0.5, color="#f59e0b", lw=2, ls="--", label="Threshold 0.5")
    axes[2].set_title("Probability Distribution", fontweight="bold", pad=12, color=MPL_FG)
    axes[2].set_xlabel("Churn Probability", color=MPL_FG)
    axes[2].set_ylabel("Density",           color=MPL_FG)
    axes[2].legend(fontsize=10, facecolor=MPL_BG, edgecolor=MPL_GRID, labelcolor=MPL_FG)
    axes[2].tick_params(colors=MPL_FG)
    for sp in axes[2].spines.values(): sp.set_edgecolor(MPL_GRID)

    plt.tight_layout(pad=2)
    st.pyplot(fig)
    plt.close()

    # ── Feature Importance ──
    st.markdown(
        '<p class="section-hdr" style="margin-top:24px">Feature Importances (L1-selected)</p>',
        unsafe_allow_html=True,
    )
    coefs   = pipeline["model"].coef_[0]
    feat_df = pd.DataFrame({
        "Feature": pipeline["selected_features"], "Coeff": coefs
    }).sort_values("Coeff")
    colors  = ["#ef4444" if c > 0 else "#22c55e" for c in feat_df["Coeff"]]

    fig2, ax2 = plt.subplots(figsize=(10, max(5, len(feat_df) * 0.38)))
    apply_mpl_theme(fig2, ax2)
    ax2.barh(feat_df["Feature"], feat_df["Coeff"], color=colors, alpha=0.85)
    ax2.axvline(0, color=MPL_FG, lw=1.2)
    ax2.set_title(
        "Logistic Regression Coefficients (after L1 selection)",
        fontweight="bold", pad=10, color=MPL_FG,
    )
    ax2.set_xlabel("Coefficient Value", color=MPL_FG)
    ax2.tick_params(colors=MPL_FG)
    for sp in ax2.spines.values(): sp.set_edgecolor(MPL_GRID)
    rp = mpatches.Patch(color="#ef4444", alpha=0.85, label="Increases churn risk")
    gp = mpatches.Patch(color="#22c55e", alpha=0.85, label="Decreases churn risk")
    ax2.legend(handles=[rp, gp], loc="lower right", fontsize=10,
               facecolor=MPL_BG, edgecolor=MPL_GRID, labelcolor=MPL_FG)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # ── Best params ──
    st.markdown(
        '<p class="section-hdr" style="margin-top:16px">Best Hyperparameters (RandomizedSearchCV)</p>',
        unsafe_allow_html=True,
    )
    st.dataframe(
        pd.DataFrame(list(pipeline["best_params"].items()), columns=["Parameter", "Value"]),
        use_container_width=False, width=420,
    )

# ──────────────────────────────────────────────────────────────
#  TAB 4 — DATASET
# ──────────────────────────────────────────────────────────────
with tab4:
    df_raw = load_and_clean("telecom_dataset.csv")
    st.markdown('<p class="section-hdr">Dataset Overview</p>', unsafe_allow_html=True)

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total Rows",     f"{len(df_raw):,}")
    d2.metric("Features",       f"{df_raw.shape[1]-1}")
    d3.metric("Churn Rate",     f"{df_raw['Churn'].mean()*100:.1f}%")
    d4.metric("Missing Values", "0")

    st.markdown("**Sample Data (first 100 rows)**")
    st.dataframe(df_raw.head(100), use_container_width=True, height=320)

    st.markdown(
        '<p class="section-hdr" style="margin-top:20px">Key Distributions</p>',
        unsafe_allow_html=True,
    )

    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 4))
    apply_mpl_theme(fig3, *axes3)

    # Pie
    cc = df_raw["Churn"].value_counts()
    axes3[0].pie(
        [cc.get(0, 0), cc.get(1, 0)],
        labels=["No Churn", "Churn"],
        colors=["#22c55e", "#ef4444"],
        autopct="%1.1f%%", startangle=90, pctdistance=0.75,
        textprops={"color": MPL_FG},
        wedgeprops=dict(linewidth=2, edgecolor=MPL_BG),
    )
    axes3[0].set_title("Churn Distribution", fontweight="bold", pad=10, color=MPL_FG)

    # Tenure
    axes3[1].hist(df_raw["tenure"], bins=30, color="#3b82f6", alpha=0.8, edgecolor=MPL_BG)
    axes3[1].set_title("Tenure Distribution", fontweight="bold", pad=10, color=MPL_FG)
    axes3[1].set_xlabel("Months", color=MPL_FG)
    axes3[1].set_ylabel("Count",  color=MPL_FG)
    axes3[1].tick_params(colors=MPL_FG)
    for sp in axes3[1].spines.values(): sp.set_edgecolor(MPL_GRID)

    # Monthly charges
    axes3[2].hist(
        df_raw[df_raw["Churn"] == 0]["MonthlyCharges"],
        bins=30, alpha=0.65, color="#22c55e", label="No Churn", density=True,
    )
    axes3[2].hist(
        df_raw[df_raw["Churn"] == 1]["MonthlyCharges"],
        bins=30, alpha=0.65, color="#ef4444", label="Churn", density=True,
    )
    axes3[2].set_title("Monthly Charges by Churn", fontweight="bold", pad=10, color=MPL_FG)
    axes3[2].set_xlabel("Monthly Charges ($)", color=MPL_FG)
    axes3[2].set_ylabel("Density",             color=MPL_FG)
    axes3[2].legend(fontsize=10, facecolor=MPL_BG, edgecolor=MPL_GRID, labelcolor=MPL_FG)
    axes3[2].tick_params(colors=MPL_FG)
    for sp in axes3[2].spines.values(): sp.set_edgecolor(MPL_GRID)

    plt.tight_layout(pad=2)
    st.pyplot(fig3)
    plt.close()

"""
============================================================
KOA PAIN LEVEL PREDICTOR — STREAMLIT APP (REDESIGNED)
Pure inference. Loads koa_model.joblib from GitHub.
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import urllib.request
import tempfile
import os
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="KOA Pain Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}

/* ── Page background ── */
.stApp { background: #f4f6fb; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f1e3d;
    border-right: none;
}
[data-testid="stSidebar"] * { color: #e8edf5 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #ffffff !important; }
[data-testid="stSidebar"] .stProgress > div > div {
    background: #3b82f6 !important;
}
[data-testid="stSidebar"] .stProgress {
    background: #1e3a6e !important;
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0f1e3d 0%, #1a3a6b 50%, #1e5799 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    color: white;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: rgba(255,255,255,0.04);
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 40%;
    width: 300px; height: 300px;
    border-radius: 50%;
    background: rgba(255,255,255,0.03);
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    color: rgba(255,255,255,0.9);
}
.hero h1 {
    font-size: 2.1rem;
    font-weight: 700;
    margin: 0 0 0.5rem;
    line-height: 1.2;
}
.hero p {
    font-size: 1rem;
    opacity: 0.8;
    margin: 0;
    font-weight: 300;
}
.hero-stats {
    display: flex;
    gap: 2rem;
    margin-top: 1.5rem;
}
.hero-stat {
    text-align: center;
}
.hero-stat-val {
    font-size: 1.6rem;
    font-weight: 700;
    color: #7dd3fc;
    line-height: 1;
}
.hero-stat-lbl {
    font-size: 0.72rem;
    opacity: 0.7;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 3px;
}

/* ── Stat cards row ── */
.stat-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 1.5rem;
}
.stat-card {
    background: white;
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    border-top: 3px solid #3b82f6;
    text-align: center;
}
.stat-card.green  { border-top-color: #10b981; }
.stat-card.amber  { border-top-color: #f59e0b; }
.stat-card.purple { border-top-color: #8b5cf6; }
.stat-card.blue   { border-top-color: #3b82f6; }
.stat-val {
    font-size: 1.7rem;
    font-weight: 700;
    color: #0f1e3d;
    line-height: 1.1;
}
.stat-lbl {
    font-size: 0.72rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 4px;
    font-weight: 500;
}

/* ── Section cards ── */
.card {
    background: white;
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    margin-bottom: 1.25rem;
}
.card-title {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6b7280;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 6px;
}
.card-title::before {
    content: '';
    display: inline-block;
    width: 3px; height: 14px;
    background: #3b82f6;
    border-radius: 2px;
}

/* ── Input labels ── */
.input-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #374151;
    margin-bottom: 2px;
}
.input-hint {
    font-size: 0.72rem;
    color: #9ca3af;
    margin-top: -2px;
    margin-bottom: 6px;
}

/* ── Sidebar metric rows ── */
.sb-metric {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    font-size: 0.85rem;
}
.sb-metric-val {
    font-weight: 600;
    color: #7dd3fc !important;
}
.sb-model-row {
    background: rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 8px 10px;
    margin: 5px 0;
    font-size: 0.82rem;
}
.sb-model-name { font-weight: 500; }
.sb-model-meta { font-size: 0.72rem; opacity: 0.65; margin-top: 2px; }
.sb-model-best {
    background: rgba(59,130,246,0.2);
    border: 1px solid rgba(59,130,246,0.4);
    border-radius: 8px;
    padding: 8px 10px;
    margin: 5px 0;
    font-size: 0.82rem;
}

/* ── Result panel ── */
.result-panel {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    border: 2px solid;
    position: relative;
    overflow: hidden;
}
.result-panel::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 120px; height: 120px;
    border-radius: 50%;
    background: currentColor;
    opacity: 0.06;
}
.result-level {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.1;
    margin: 0.5rem 0;
}
.result-icon { font-size: 3rem; }
.result-conf {
    font-size: 0.88rem;
    opacity: 0.65;
    margin-top: 0.4rem;
}
.result-note {
    font-size: 0.82rem;
    font-style: italic;
    opacity: 0.7;
    margin-top: 0.75rem;
    line-height: 1.5;
}

/* ── Probability bars ── */
.prob-item {
    margin: 8px 0;
}
.prob-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    margin-bottom: 4px;
    font-weight: 500;
    color: #374151;
}
.prob-track {
    background: #f3f4f6;
    border-radius: 6px;
    height: 10px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.6s ease;
}

/* ── Risk table ── */
.risk-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 9px 0;
    border-bottom: 1px solid #f3f4f6;
    font-size: 0.85rem;
}
.risk-row:last-child { border-bottom: none; }
.risk-feat { color: #374151; font-weight: 500; }
.risk-val  { color: #6b7280; font-size: 0.8rem; }
.risk-badge {
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
}
.risk-high   { background: #fee2e2; color: #dc2626; }
.risk-mod    { background: #fef3c7; color: #d97706; }
.risk-low    { background: #dcfce7; color: #16a34a; }
.risk-long   { background: #fee2e2; color: #dc2626; }
.risk-short  { background: #dcfce7; color: #16a34a; }
.risk-obese  { background: #fee2e2; color: #dc2626; }
.risk-ow     { background: #fef3c7; color: #d97706; }
.risk-normal { background: #dcfce7; color: #16a34a; }

/* ── Disclaimer ── */
.disclaimer {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.8rem;
    color: #78350f;
    margin-top: 1.25rem;
    line-height: 1.6;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: white;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 18px;
    font-size: 0.85rem;
    font-weight: 500;
    color: #6b7280;
}
.stTabs [aria-selected="true"] {
    background: #0f1e3d !important;
    color: white !important;
}

/* ── Predict button ── */
.stButton > button {
    background: linear-gradient(135deg, #0f1e3d 0%, #1e5799 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 14px rgba(15,30,61,0.35) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 20px rgba(15,30,61,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 1.5rem;
    font-size: 0.75rem;
    color: #9ca3af;
    border-top: 1px solid #e5e7eb;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

PAIN_LABELS = {1: "No Pain", 2: "Mild Pain", 3: "Moderate Pain", 4: "Severe Pain"}
PAIN_COLORS = {1: "#10b981", 2: "#f59e0b", 3: "#f97316", 4: "#ef4444"}
PAIN_ICONS  = {1: "✅", 2: "🟡", 3: "🟠", 4: "🔴"}
PAIN_BG     = {1: "#ecfdf5", 2: "#fffbeb", 3: "#fff7ed", 4: "#fef2f2"}
PAIN_BORDER = {1: "#10b981", 2: "#f59e0b", 3: "#f97316", 4: "#ef4444"}

CHANNELS   = ['Fp1', 'Fp2', 'Fz', 'F7', 'F8', 'T7', 'T8', 'Pz']
BANDS      = ['Alpha', 'Beta', 'Theta', 'Delta', 'Gamma']
CONDITIONS = ['Pre', 'Post', 'Standing', 'Sitting']

MODEL_URL = (
    "https://raw.githubusercontent.com/focuzdev/"
    "Pain-Level-Prediction/master/koa_model.joblib"
)

# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def engineer_features(src_dict: dict) -> pd.DataFrame:
    src = pd.DataFrame([src_dict])
    feats = {}

    def safe_log(arr):
        arr = np.array(arr, dtype=float)
        return np.log1p(np.where(arr < 0, 0, arr))

    def col(name):
        return src[name].values if name in src.columns else np.zeros(len(src))

    for cond in CONDITIONS:
        v1 = col(f'Fp1_Alpha_{cond}')
        v2 = col(f'Fp2_Alpha_{cond}')
        feats[f'FAA_{cond}'] = (v1 - v2) / (v1 + v2 + 1e-6)

    for cond in CONDITIONS:
        feats[f'TBR_Fz_{cond}'] = safe_log(
            col(f'Fz_Theta_{cond}') / (col(f'Fz_Beta_{cond}') + 1e-6))

    for cond in CONDITIONS:
        vals = np.stack([col(f'{ch}_Alpha_{cond}') for ch in CHANNELS], axis=1)
        feats[f'Global_Alpha_{cond}'] = safe_log(vals.mean(axis=1))

    for cond in CONDITIONS:
        vals = np.stack([col(f'{ch}_Theta_{cond}') for ch in CHANNELS], axis=1)
        feats[f'Global_Theta_{cond}'] = safe_log(vals.mean(axis=1))

    for cond in CONDITIONS:
        vals = np.stack([col(f'T7_Delta_{cond}'), col(f'T8_Delta_{cond}')], axis=1)
        feats[f'Temporal_Delta_{cond}'] = safe_log(vals.mean(axis=1))

    for cond in CONDITIONS:
        feats[f'Pz_Alpha_{cond}'] = safe_log(col(f'Pz_Alpha_{cond}'))

    for cond in ['Pre', 'Post']:
        vals = np.stack([col(f'Fp1_Gamma_{cond}'), col(f'Fp2_Gamma_{cond}')], axis=1)
        feats[f'Frontal_Gamma_{cond}'] = safe_log(vals.mean(axis=1))

    for cond in ['Pre', 'Post']:
        d = np.stack([col(f'{ch}_Delta_{cond}') for ch in CHANNELS], axis=1).mean(axis=1)
        a = np.stack([col(f'{ch}_Alpha_{cond}') for ch in CHANNELS], axis=1).mean(axis=1)
        feats[f'Delta_Alpha_Ratio_{cond}'] = safe_log(d / (a + 1e-6))

    feats['Alpha_Change_PrePost']  = (np.array(feats['Global_Alpha_Post'])
                                      - np.array(feats['Global_Alpha_Pre']))
    feats['Alpha_Change_StandSit'] = (np.array(feats['Global_Alpha_Standing'])
                                      - np.array(feats['Global_Alpha_Sitting']))

    for c in ['KL_grade', 'McGill', 'Age', 'BMI', 'Duration_of_symptom',
              'PRI', 'WMI', 'PSI', 'QoL']:
        feats[c] = col(c)

    feats['Cognitive_Composite'] = (col('PRI') + col('WMI') + col('PSI')) / 3.0

    return pd.DataFrame(feats)

# ─────────────────────────────────────────────────────────────
# LOAD BUNDLE
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_bundle():
    tmp = tempfile.mktemp(suffix='.joblib')
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        bundle = joblib.load(tmp)
        required = ['scaler', 'model', 'feature_names', 'model_name',
                    'loo_accuracy', 'loo_f1_macro', 'all_model_results']
        missing = [k for k in required if k not in bundle]
        if missing:
            return None, (f"Model file outdated — missing keys: {missing}. "
                          f"Re-run train_and_save.py and re-upload koa_model.joblib.")
        return bundle, None
    except Exception as e:
        return None, str(e)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():

    # ── Load model ───────────────────────────────────────────
    with st.spinner(""):
        bundle, err = load_bundle()

    if err:
        st.markdown("""
        <div style="background:white;border-radius:14px;padding:2rem;
                    border-left:5px solid #ef4444;box-shadow:0 2px 8px rgba(0,0,0,0.08);">
          <h3 style="color:#ef4444;margin:0 0 .5rem;">⚠️ Model could not be loaded</h3>
          <p style="color:#6b7280;font-size:.9rem;margin:0;">
            Please re-run <code>train_and_save.py</code> in Google Colab,
            re-upload <code>koa_model.joblib</code> to GitHub, then reload this page.
          </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"<small style='color:#9ca3af'>Detail: {err}</small>",
                    unsafe_allow_html=True)
        st.stop()

    b = bundle  # shorthand

    # ── SIDEBAR ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="padding:1.25rem 0 1rem;">
          <div style="font-size:1.1rem;font-weight:700;color:white;
                      letter-spacing:-.01em;">🧠 KOA Predictor</div>
          <div style="font-size:0.72rem;color:rgba(255,255,255,0.45);
                      margin-top:2px;text-transform:uppercase;
                      letter-spacing:.06em;">Clinical Decision Support</div>
        </div>
        <hr style="border:none;border-top:1px solid rgba(255,255,255,0.1);margin:0 0 1rem;">
        """, unsafe_allow_html=True)

        # Model performance
        st.markdown("<div style='font-size:.7rem;font-weight:600;"
                    "text-transform:uppercase;letter-spacing:.08em;"
                    "color:rgba(255,255,255,0.45);margin-bottom:.6rem;'>"
                    "Model Performance (LOO-CV)</div>",
                    unsafe_allow_html=True)

        metrics = [
            ("Accuracy",           f"{b['loo_accuracy']*100:.1f}%"),
            ("Balanced Accuracy",  f"{b['loo_balanced_accuracy']*100:.1f}%"),
            ("F1 Macro",           f"{b['loo_f1_macro']*100:.1f}%"),
            ("F1 Weighted",        f"{b['loo_f1_weighted']*100:.1f}%"),
            ("Cohen's Kappa",      f"{b['loo_kappa']:.3f}"),
            ("Training N",         f"{b['n_samples']} patients"),
            ("Features",           f"{b['n_features']}"),
        ]
        rows_html = "".join(
            f"<div class='sb-metric'>"
            f"<span>{lbl}</span>"
            f"<span class='sb-metric-val'>{val}</span>"
            f"</div>"
            for lbl, val in metrics
        )
        st.markdown(rows_html, unsafe_allow_html=True)

        st.markdown("<hr style='border:none;border-top:1px solid "
                    "rgba(255,255,255,0.1);margin:1rem 0;'>",
                    unsafe_allow_html=True)

        # Models comparison
        st.markdown("<div style='font-size:.7rem;font-weight:600;"
                    "text-transform:uppercase;letter-spacing:.08em;"
                    "color:rgba(255,255,255,0.45);margin-bottom:.6rem;'>"
                    "All Models Compared</div>",
                    unsafe_allow_html=True)

        ranked = sorted(b['all_model_results'].items(),
                        key=lambda x: x[1]['f1_macro'], reverse=True)
        for mname, mres in ranked:
            is_best = mname == b['model_name']
            cls = "sb-model-best" if is_best else "sb-model-row"
            badge = (" <span style='background:#3b82f6;color:white;"
                     "font-size:.65rem;padding:1px 7px;border-radius:10px;"
                     "font-weight:600;'>DEPLOYED</span>" if is_best else "")
            st.markdown(
                f"<div class='{cls}'>"
                f"<div class='sb-model-name'>{mname}{badge}</div>"
                f"<div class='sb-model-meta'>"
                f"F1: {mres['f1_macro']*100:.1f}%  ·  "
                f"Acc: {mres['accuracy']*100:.1f}%  ·  "
                f"κ: {mres['kappa']:.3f}"
                f"</div></div>",
                unsafe_allow_html=True
            )

        # Top features
        feat_imp = b.get('feature_importances', {})
        if feat_imp:
            st.markdown("<hr style='border:none;border-top:1px solid "
                        "rgba(255,255,255,0.1);margin:1rem 0;'>",
                        unsafe_allow_html=True)
            st.markdown("<div style='font-size:.7rem;font-weight:600;"
                        "text-transform:uppercase;letter-spacing:.08em;"
                        "color:rgba(255,255,255,0.45);margin-bottom:.6rem;'>"
                        "Top Features</div>",
                        unsafe_allow_html=True)
            top5 = list(feat_imp.items())[:5]
            max_imp = top5[0][1]
            for i, (fname, fimp) in enumerate(top5):
                pct = int(fimp / max_imp * 100)
                st.markdown(
                    f"<div style='margin:5px 0;font-size:.78rem;'>"
                    f"<div style='display:flex;justify-content:space-between;"
                    f"margin-bottom:3px;'>"
                    f"<span style='color:rgba(255,255,255,.75);'>"
                    f"{i+1}. {fname[:24]}</span>"
                    f"<span style='color:#7dd3fc;font-weight:600;'>"
                    f"{fimp:.3f}</span></div>"
                    f"<div style='background:rgba(255,255,255,0.1);"
                    f"border-radius:4px;height:4px;'>"
                    f"<div style='background:#3b82f6;width:{pct}%;"
                    f"height:4px;border-radius:4px;'></div></div></div>",
                    unsafe_allow_html=True
                )

        st.markdown("<hr style='border:none;border-top:1px solid "
                    "rgba(255,255,255,0.1);margin:1rem 0;'>",
                    unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:.7rem;color:rgba(255,255,255,.35);"
            "line-height:1.6;'>"
            "Model trained offline · Fixed weights · "
            "Reproducible predictions · Research use only"
            "</div>",
            unsafe_allow_html=True
        )

    # ── HERO BANNER ──────────────────────────────────────────
    st.markdown(f"""
    <div class="hero">
      <div class="hero-badge">🧬 EEG + Clinical ML System</div>
      <h1>KOA Pain Level Predictor</h1>
      <p>Knee Osteoarthritis pain classification using EEG biomarkers
         and clinical features — powered by {b['model_name']}</p>
      <div class="hero-stats">
        <div class="hero-stat">
          <div class="hero-stat-val">{b['loo_accuracy']*100:.0f}%</div>
          <div class="hero-stat-lbl">Accuracy</div>
        </div>
        <div class="hero-stat">
          <div class="hero-stat-val">{b['loo_f1_macro']*100:.0f}%</div>
          <div class="hero-stat-lbl">F1 Macro</div>
        </div>
        <div class="hero-stat">
          <div class="hero-stat-val">{b['loo_kappa']:.2f}</div>
          <div class="hero-stat-lbl">Kappa</div>
        </div>
        <div class="hero-stat">
          <div class="hero-stat-val">{b['n_samples']}</div>
          <div class="hero-stat-lbl">Patients</div>
        </div>
        <div class="hero-stat">
          <div class="hero-stat-val">{b['n_features']}</div>
          <div class="hero-stat-lbl">Features</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── METRIC CARDS ROW ─────────────────────────────────────
    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-card green">
        <div class="stat-val">{b['loo_accuracy']*100:.1f}%</div>
        <div class="stat-lbl">LOO-CV Accuracy</div>
      </div>
      <div class="stat-card blue">
        <div class="stat-val">{b['loo_balanced_accuracy']*100:.1f}%</div>
        <div class="stat-lbl">Balanced Accuracy</div>
      </div>
      <div class="stat-card amber">
        <div class="stat-val">{b['loo_f1_macro']*100:.1f}%</div>
        <div class="stat-lbl">F1 Macro</div>
      </div>
      <div class="stat-card purple">
        <div class="stat-val">{b['loo_kappa']:.3f}</div>
        <div class="stat-lbl">Cohen's Kappa</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── INPUT TABS ───────────────────────────────────────────
    inputs = {}

    tab1, tab2, tab3, tab4 = st.tabs([
        "🩺  Clinical Features",
        "🧠  EEG Features",
        "📂  Upload CSV",
        "🎯  Predict"
    ])

    # ── TAB 1: CLINICAL ──────────────────────────────────────
    with tab1:
        st.markdown("""
        <div class="card">
          <div class="card-title">Clinical & Demographic Information</div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            inputs['KL_grade'] = st.selectbox(
                "KL Grade — radiological severity",
                options=[0, 1, 2, 3, 4],
                format_func=lambda x: {
                    0: "Grade 0 — Normal",
                    1: "Grade 1 — Doubtful narrowing",
                    2: "Grade 2 — Minimal narrowing",
                    3: "Grade 3 — Moderate narrowing",
                    4: "Grade 4 — Severe, bone-on-bone"
                }[x],
                index=2
            )
            inputs['McGill'] = st.slider(
                "McGill Pain Score", 0, 5, 2,
                help="Total McGill Pain Questionnaire score (0=no pain, 5=severe)"
            )
            inputs['QoL'] = st.slider(
                "Quality of Life Score", 0, 10, 5,
                help="Patient-reported quality of life (0=worst, 10=best)"
            )
        with c2:
            inputs['Age'] = st.number_input(
                "Age (years)", 20, 90, 55, step=1)
            inputs['BMI'] = st.number_input(
                "BMI (kg/m²)", 15.0, 50.0, 27.0, step=0.1, format="%.1f")
            inputs['Duration_of_symptom'] = st.slider(
                "Symptom Duration (years)", 0, 15, 3,
                help="How long the patient has had knee symptoms"
            )
        with c3:
            inputs['PRI'] = st.number_input(
                "PRI — Perceptual Reasoning Index", 0, 200, 90,
                help="WAIS Perceptual Reasoning Index score")
            inputs['WMI'] = st.number_input(
                "WMI — Working Memory Index", 0, 200, 95,
                help="WAIS Working Memory Index score")
            inputs['PSI'] = st.number_input(
                "PSI — Processing Speed Index", 0, 200, 85,
                help="WAIS Processing Speed Index score")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#eff6ff;border:1px solid #bfdbfe;
                    border-left:4px solid #3b82f6;border-radius:8px;
                    padding:.8rem 1rem;font-size:.82rem;color:#1e40af;">
          <b>Key predictors from training data:</b>
          KL Grade & McGill Score (p&lt;0.001) ·
          Symptom Duration (p&lt;0.001) · BMI (p=0.002)
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 2: EEG ───────────────────────────────────────────
    with tab2:
        st.markdown("""
        <div class="card">
          <div class="card-title">EEG Power Spectral Density Values (μV²)</div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#f0fdf4;border:1px solid #bbf7d0;
                    border-left:4px solid #10b981;border-radius:8px;
                    padding:.8rem 1rem;font-size:.82rem;color:#14532d;
                    margin-bottom:1rem;">
          <b>Key EEG biomarkers:</b>
          Frontal Alpha Asymmetry (Fp1 vs Fp2) ·
          Theta/Beta Ratio (Fz) · Temporal Delta Power (T7+T8)
          <br>If only Pre-task data is available, use the same values for all conditions.
        </div>
        """, unsafe_allow_html=True)

        for ch in CHANNELS:
            with st.expander(f"📡  Channel {ch}", expanded=False):
                for band in BANDS:
                    st.markdown(
                        f"<div class='input-label'>{band} band</div>",
                        unsafe_allow_html=True
                    )
                    cols = st.columns(4)
                    for i, cond in enumerate(CONDITIONS):
                        key = f'{ch}_{band}_{cond}'
                        with cols[i]:
                            inputs[key] = st.number_input(
                                cond, min_value=0.0, value=1000.0,
                                step=100.0, key=key, format="%.1f"
                            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 3: CSV ───────────────────────────────────────────
    with tab3:
        st.markdown("""
        <div class="card">
          <div class="card-title">Upload a Patient Data Row</div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <p style="font-size:.85rem;color:#6b7280;margin-bottom:1rem;">
        Upload a single-row CSV file in the same format as
        <code>koa_erd_processed.csv</code>. All column values will be
        used directly, overriding the manual inputs.
        </p>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Choose CSV file", type=["csv"],
            help="Single-row CSV matching koa_erd_processed.csv column format"
        )
        if uploaded:
            try:
                dfup = pd.read_csv(uploaded)
                dfup.columns = (dfup.columns.str.strip()
                                .str.replace(' ', '_')
                                .str.replace('-', '_')
                                .str.replace('Siiting', 'Sitting')
                                .str.replace('Posting', 'Post'))
                row = dfup.iloc[0].to_dict()
                inputs.update({
                    k: float(v)
                    if isinstance(v, (int, float, np.integer, np.floating))
                    else v
                    for k, v in row.items()
                })
                st.success(f"✅ Patient data loaded — {len(row)} columns detected")
                st.dataframe(
                    pd.DataFrame([row]).T.rename(columns={0: "Value"}),
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Could not parse file: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 4: PREDICT ───────────────────────────────────────
    with tab4:

        # Patient summary card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Patient Summary</div>',
                    unsafe_allow_html=True)

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            kl_display = {0:"Normal",1:"Doubtful",2:"Minimal",
                          3:"Moderate",4:"Severe"}
            st.metric("KL Grade",
                      kl_display.get(inputs.get('KL_grade', 2), "—"))
        with p2:
            st.metric("McGill Score", inputs.get('McGill', '—'))
        with p3:
            bmi_v = float(inputs.get('BMI', 0))
            st.metric("BMI", f"{bmi_v:.1f} kg/m²")
        with p4:
            st.metric("Symptom Duration",
                      f"{inputs.get('Duration_of_symptom', '—')} yrs")

        st.markdown('</div>', unsafe_allow_html=True)

        # Predict button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍   Run Prediction", use_container_width=True):
            with st.spinner("Analysing patient profile..."):
                try:
                    scaler     = b['scaler']
                    model      = b['model']
                    feat_names = b['feature_names']

                    X_in = engineer_features(inputs)
                    for feat in feat_names:
                        if feat not in X_in.columns:
                            X_in[feat] = 0.0
                    X_in = X_in[feat_names]
                    X_sc = scaler.transform(X_in)

                    pred  = int(model.predict(X_sc)[0])
                    proba = {
                        int(c): float(p)
                        for c, p in zip(model.classes_,
                                        model.predict_proba(X_sc)[0])
                    }

                    # ── RESULT LAYOUT ─────────────────────
                    st.markdown("<br>", unsafe_allow_html=True)

                    res_col, prob_col = st.columns([1, 1])

                    # Result box
                    with res_col:
                        note = {
                            1: "EEG and clinical markers are within normal range. No significant pain burden detected.",
                            2: "Mild pain indicators present. Consider monitoring KL grade progression and symptom duration.",
                            3: "Moderate pain burden indicated. Significant functional impact is likely.",
                            4: "Severe pain profile detected. Consistent with advanced KOA — clinical review recommended."
                        }
                        conf = proba.get(pred, 0)
                        st.markdown(f"""
                        <div class="result-panel"
                             style="background:{PAIN_BG[pred]};
                                    border-color:{PAIN_BORDER[pred]};
                                    color:{PAIN_COLORS[pred]};">
                          <div class="result-icon">{PAIN_ICONS[pred]}</div>
                          <div class="result-level"
                               style="color:{PAIN_COLORS[pred]};">
                            {PAIN_LABELS[pred]}
                          </div>
                          <div class="result-conf" style="color:#6b7280;">
                            Model confidence: <strong>{conf*100:.1f}%</strong>
                          </div>
                          <div class="result-note" style="color:#6b7280;">
                            {note[pred]}
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Probability bars
                    with prob_col:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown(
                            '<div class="card-title">Probability Distribution</div>',
                            unsafe_allow_html=True
                        )
                        for cls in [1, 2, 3, 4]:
                            p   = proba.get(cls, 0)
                            pct = int(p * 100)
                            active = (
                                "font-weight:700;" if cls == pred else ""
                            )
                            tag = (
                                " <span style='font-size:.68rem;"
                                "background:#0f1e3d;color:white;"
                                "padding:1px 7px;border-radius:10px;"
                                "margin-left:4px;'>predicted</span>"
                                if cls == pred else ""
                            )
                            st.markdown(f"""
                            <div class="prob-item">
                              <div class="prob-header">
                                <span style="{active}">{PAIN_LABELS[cls]}{tag}</span>
                                <span style="{active}color:{PAIN_COLORS[cls]}">
                                  {pct}%
                                </span>
                              </div>
                              <div class="prob-track">
                                <div class="prob-fill"
                                     style="width:{pct}%;
                                            background:{PAIN_COLORS[cls]};"></div>
                              </div>
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Risk factors
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown(
                        '<div class="card-title">Clinical Risk Assessment</div>',
                        unsafe_allow_html=True
                    )

                    kl  = inputs.get('KL_grade', 0)
                    mc  = inputs.get('McGill', 0)
                    dur = inputs.get('Duration_of_symptom', 0)
                    bmi = float(inputs.get('BMI', 0))

                    def badge(cls, text):
                        return (f"<span class='risk-badge risk-{cls}'>"
                                f"{text}</span>")

                    kl_b  = badge('high','High') if kl >= 3 else (badge('mod','Moderate') if kl >= 2 else badge('low','Low'))
                    mc_b  = badge('high','High') if mc >= 4 else (badge('mod','Moderate') if mc >= 2 else badge('low','Low'))
                    dr_b  = badge('long','Long duration') if dur >= 5 else badge('short','Short duration')
                    bm_b  = badge('obese','Obese') if bmi >= 30 else (badge('ow','Overweight') if bmi >= 25 else badge('normal','Normal'))

                    risks = [
                        ("KL Grade", f"Grade {kl}", kl_b),
                        ("McGill Score", f"{mc} / 5", mc_b),
                        ("Symptom Duration", f"{dur} years", dr_b),
                        ("BMI", f"{bmi:.1f} kg/m²", bm_b),
                    ]
                    for feat, val, badge_html in risks:
                        st.markdown(f"""
                        <div class="risk-row">
                          <span class="risk-feat">{feat}</span>
                          <span class="risk-val">{val}</span>
                          {badge_html}
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

                    # Disclaimer
                    st.markdown("""
                    <div class="disclaimer">
                      <strong>⚠️ Clinical Disclaimer</strong><br>
                      This tool is for research and educational purposes only.
                      Predictions are based on a training cohort of 62 patients
                      and must not replace the clinical judgement of a qualified
                      healthcare professional. Always confirm findings through
                      standard clinical assessment.
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.exception(e)

    # ── FOOTER ───────────────────────────────────────────────
    st.markdown(f"""
    <div class="footer">
      KOA Pain Level Predictor &nbsp;·&nbsp;
      Model: {b['model_name']} &nbsp;·&nbsp;
      Validated with Leave-One-Out CV &nbsp;·&nbsp;
      Trained offline — inference only &nbsp;·&nbsp;
      For research use only
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

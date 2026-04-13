"""
=============================================================================
PAIN LEVEL PREDICTION - STREAMLIT APP
=============================================================================
Predicts pain level (No Pain / Mild / Moderate / Severe) for KOA patients
using EEG-derived biomarkers and clinical features.
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="KOA Pain Level Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
    }
    .pain-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2d6a9f;
        margin: 0.5rem 0;
    }
    .section-header {
        color: #1e3a5f;
        font-size: 1.1rem;
        font-weight: 600;
        border-bottom: 2px solid #2d6a9f;
        padding-bottom: 0.3rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .prob-bar-container {
        margin: 0.3rem 0;
    }
    .disclaimer {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        font-size: 0.85rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #1e3a5f;
        color: white;
        border: none;
        padding: 0.75rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #2d6a9f;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================

PAIN_LABELS = {1: "No Pain", 2: "Mild Pain", 3: "Moderate Pain", 4: "Severe Pain"}
PAIN_COLORS = {1: "#27ae60", 2: "#f39c12", 3: "#e67e22", 4: "#e74c3c"}
PAIN_EMOJI  = {1: "✅", 2: "🟡", 3: "🟠", 4: "🔴"}
PAIN_BG     = {1: "#d4efdf", 2: "#fef9e7", 3: "#fdebd0", 4: "#fadbd8"}

# ============================================================
# LOAD MODEL ARTIFACTS (with caching)
# ============================================================

@st.cache_resource
def load_models():
    """Load model artifacts. Tries local first, then downloads & trains."""
    artifact_dir = "model_artifacts"

    if not os.path.exists(artifact_dir) or not os.path.exists(f"{artifact_dir}/ensemble_model.joblib"):
        st.info("🔄 First run detected — training model... (this takes ~30 seconds)")
        try:
            import subprocess, sys
            result = subprocess.run(
                [sys.executable, "train_model.py"],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode != 0:
                st.error(f"Training failed: {result.stderr}")
                return None, None, None
        except Exception as e:
            st.error(f"Could not auto-train: {e}")
            return None, None, None

    try:
        scaler   = joblib.load(f"{artifact_dir}/scaler.joblib")
        model    = joblib.load(f"{artifact_dir}/ensemble_model.joblib")
        metadata = joblib.load(f"{artifact_dir}/metadata.joblib")
        return scaler, model, metadata
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# ============================================================
# FEATURE ENGINEERING (mirrors train_model.py exactly)
# ============================================================

def engineer_features(inputs: dict) -> pd.DataFrame:
    """
    Takes raw EEG band powers + clinical inputs,
    returns a feature vector matching training schema.
    """
    f = {}

    # Helper: safe log1p ratio
    def lrat(a, b): return np.log1p(a / (b + 1e-6))

    conditions = ['Pre', 'Post', 'Standing', 'Sitting']
    bands = ['Alpha', 'Beta', 'Theta', 'Delta', 'Gamma']
    channels = ['Fp1', 'Fp2', 'Fz', 'F7', 'F8', 'T7', 'T8', 'Pz']

    # Build a mock-dataframe row from user inputs
    row = inputs.copy()

    # FAA per condition
    for cond in conditions:
        key1 = f'Fp1_Alpha_{cond}'
        key2 = f'Fp2_Alpha_{cond}'
        v1 = row.get(key1, 0)
        v2 = row.get(key2, 0)
        f[f'FAA_{cond}'] = (v1 - v2) / (v1 + v2 + 1e-6)

    # TBR per condition
    for cond in conditions:
        tk = f'Fz_Theta_{cond}'
        bk = f'Fz_Beta_{cond}'
        f[f'TBR_Fz_{cond}'] = lrat(row.get(tk, 0), row.get(bk, 1))

    # Global Alpha per condition
    for cond in conditions:
        vals = [row.get(f'{ch}_Alpha_{cond}', 0) for ch in channels]
        f[f'Global_Alpha_{cond}'] = np.log1p(np.mean(vals))

    # Global Theta per condition
    for cond in conditions:
        vals = [row.get(f'{ch}_Theta_{cond}', 0) for ch in channels]
        f[f'Global_Theta_{cond}'] = np.log1p(np.mean(vals))

    # Temporal Delta
    for cond in conditions:
        vals = [row.get(f'T7_Delta_{cond}', 0), row.get(f'T8_Delta_{cond}', 0)]
        f[f'Temporal_Delta_{cond}'] = np.log1p(np.mean(vals))

    # Pz Alpha
    for cond in conditions:
        f[f'Pz_Alpha_{cond}'] = np.log1p(row.get(f'Pz_Alpha_{cond}', 0))

    # Frontal Gamma
    for cond in ['Pre', 'Post']:
        vals = [row.get(f'Fp1_Gamma_{cond}', 0), row.get(f'Fp2_Gamma_{cond}', 0)]
        f[f'Frontal_Gamma_{cond}'] = np.log1p(np.mean(vals))

    # Delta/Alpha Ratio
    for cond in ['Pre', 'Post']:
        d_vals = [row.get(f'{ch}_Delta_{cond}', 0) for ch in channels]
        a_vals = [row.get(f'{ch}_Alpha_{cond}', 0) for ch in channels]
        f[f'Delta_Alpha_Ratio_{cond}'] = lrat(np.mean(d_vals), np.mean(a_vals))

    # Pre-Post change
    f['Alpha_Change_PrePost'] = f['Global_Alpha_Post'] - f['Global_Alpha_Pre']
    f['Alpha_Change_StandSit'] = f['Global_Alpha_Standing'] - f['Global_Alpha_Sitting']

    # Clinical features
    for col in ['KL_grade', 'McGill', 'Age', 'BMI', 'Duration_of_symptom',
                'PRI', 'WMI', 'PSI', 'QoL']:
        f[col] = row.get(col, 0)

    f['Cognitive_Composite'] = (row.get('PRI', 0) + row.get('WMI', 0) + row.get('PSI', 0)) / 3

    return pd.DataFrame([f])

# ============================================================
# MAIN APP LAYOUT
# ============================================================

def main():
    # ---- HEADER ----
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size:2rem;">🧠 KOA Pain Level Predictor</h1>
        <p style="margin:0.5rem 0 0; opacity:0.9; font-size:1rem;">
            EEG-based machine learning system for Knee Osteoarthritis pain classification
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ---- LOAD MODELS ----
    scaler, model, metadata = load_models()

    if model is None:
        st.error("⚠️ Model could not be loaded. Please run `python train_model.py` first.")
        st.stop()

    # ---- SIDEBAR: Model Info ----
    with st.sidebar:
        st.markdown("### 📊 Model Performance")
        st.markdown(f"""
        <div class="metric-card">
            <b>Algorithm:</b> Soft Voting Ensemble<br>
            <small>(RF + GBM + SVM + LR)</small>
        </div>
        <div class="metric-card">
            <b>LOO-CV Accuracy:</b> {metadata['loo_accuracy']*100:.1f}%
        </div>
        <div class="metric-card">
            <b>Balanced Accuracy:</b> {metadata['loo_balanced_accuracy']*100:.1f}%
        </div>
        <div class="metric-card">
            <b>F1 Macro:</b> {metadata['loo_f1_macro']*100:.1f}%
        </div>
        <div class="metric-card">
            <b>Training samples:</b> {metadata['n_samples']}
        </div>
        <div class="metric-card">
            <b>Features:</b> {metadata['n_features']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📈 Individual Models (LOO-CV)")
        for mname, mres in metadata['model_results'].items():
            st.markdown(f"**{mname}**")
            st.progress(mres['f1_macro'])
            st.caption(f"F1 Macro: {mres['f1_macro']*100:.1f}%  |  Acc: {mres['accuracy']*100:.1f}%")

        st.markdown("---")
        st.markdown("### 🏆 Top Features")
        fi = metadata['feature_importances']
        top5 = list(fi.items())[:5]
        for fname, fimp in top5:
            bar = "█" * int(fimp * 100)
            st.caption(f"`{fname[:28]}`  \n{bar} {fimp:.3f}")

    # ============================================================
    # INPUT SECTION
    # ============================================================

    tab1, tab2, tab3 = st.tabs(["🩺 Clinical Features", "🧠 EEG Features", "📋 Predict"])

    inputs = {}

    # ---- TAB 1: CLINICAL ----
    with tab1:
        st.markdown('<p class="section-header">Clinical & Demographic Information</p>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            inputs['KL_grade'] = st.selectbox(
                "KL Grade (Radiological)",
                options=[0, 1, 2, 3, 4],
                format_func=lambda x: {0:"0 — Normal", 1:"1 — Doubtful", 2:"2 — Minimal",
                                        3:"3 — Moderate", 4:"4 — Severe"}[x],
                index=2
            )
            inputs['McGill'] = st.slider("McGill Pain Score", 0, 5, 2,
                help="McGill Pain Questionnaire total score (0=no pain, 5=worst)")
            inputs['QoL'] = st.slider("Quality of Life Score", 0, 10, 5)

        with col2:
            inputs['Age'] = st.number_input("Age (years)", 20, 90, 55)
            inputs['BMI'] = st.number_input("BMI (kg/m²)", 15.0, 50.0, 27.0, step=0.1)
            inputs['Duration_of_symptom'] = st.slider("Symptom Duration (years)", 0, 15, 3)

        with col3:
            inputs['PRI'] = st.number_input("PRI Score", 0, 200, 90,
                help="Perceptual Reasoning Index (WAIS)")
            inputs['WMI'] = st.number_input("WMI Score", 0, 200, 95,
                help="Working Memory Index (WAIS)")
            inputs['PSI'] = st.number_input("PSI Score", 0, 200, 85,
                help="Processing Speed Index (WAIS)")

        st.info("💡 KL Grade and McGill score are the strongest predictors in this model (p < 0.001)")

    # ---- TAB 2: EEG ----
    with tab2:
        st.markdown('<p class="section-header">EEG Band Power Inputs</p>', unsafe_allow_html=True)
        st.caption("Enter EEG power spectral density values (μV²) for each channel and condition. "
                   "If you don't have all conditions, use the Pre-task values as defaults.")

        channels_list = ['Fp1', 'Fp2', 'Fz', 'F7', 'F8', 'T7', 'T8', 'Pz']
        bands_list    = ['Alpha', 'Beta', 'Theta', 'Delta', 'Gamma']
        conditions_list = ['Pre', 'Post', 'Standing', 'Sitting']

        # Use expanders per channel to keep UI manageable
        for ch in channels_list:
            with st.expander(f"📡 Channel: {ch}"):
                for band in bands_list:
                    cols = st.columns(4)
                    for i, cond in enumerate(conditions_list):
                        key = f'{ch}_{band}_{cond}'
                        with cols[i]:
                            inputs[key] = st.number_input(
                                f"{cond}",
                                min_value=0.0,
                                value=1000.0,
                                step=100.0,
                                key=key,
                                label_visibility="visible",
                                help=f"{ch} {band} {cond}"
                            )
                    st.caption(f"↑ {band} band (all 4 conditions)")

        st.markdown("---")
        st.info("🔬 **EEG tip**: Frontal Alpha Asymmetry (Fp1 vs Fp2) and Temporal Delta power are key pain biomarkers. "
                "Lower global alpha power and higher theta/beta ratio indicate greater pain burden.")

    # ---- TAB 3: PREDICT ----
    with tab3:
        st.markdown('<p class="section-header">Run Prediction</p>', unsafe_allow_html=True)

        st.markdown("**Review your inputs before predicting:**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"- **KL Grade:** {inputs['KL_grade']}")
            st.markdown(f"- **McGill Score:** {inputs['McGill']}")
            st.markdown(f"- **Age:** {inputs['Age']} years")
            st.markdown(f"- **BMI:** {inputs['BMI']} kg/m²")
        with col2:
            st.markdown(f"- **Symptom Duration:** {inputs['Duration_of_symptom']} years")
            st.markdown(f"- **PRI / WMI / PSI:** {inputs['PRI']} / {inputs['WMI']} / {inputs['PSI']}")
            st.markdown(f"- **QoL Score:** {inputs['QoL']}")
            st.markdown(f"- **EEG channels:** {len(channels_list)} channels × {len(bands_list)} bands × 4 conditions")

        st.markdown("---")

        if st.button("🔍 Predict Pain Level", use_container_width=True):
            with st.spinner("Running prediction..."):
                try:
                    # Engineer features
                    X_input = engineer_features(inputs)

                    # Align with training feature set
                    train_features = metadata['feature_names']
                    for feat in train_features:
                        if feat not in X_input.columns:
                            X_input[feat] = 0
                    X_input = X_input[train_features]

                    # Scale
                    X_scaled = scaler.transform(X_input)

                    # Predict
                    pred_class = model.predict(X_scaled)[0]
                    pred_proba = model.predict_proba(X_scaled)[0]

                    # Get class order
                    classes = model.classes_
                    proba_dict = {int(c): float(p) for c, p in zip(classes, pred_proba)}

                    # ---- RESULT DISPLAY ----
                    st.markdown("---")
                    st.markdown("### 🎯 Prediction Result")

                    result_col, chart_col = st.columns([1, 1])

                    with result_col:
                        pain_label = PAIN_LABELS[pred_class]
                        pain_color = PAIN_COLORS[pred_class]
                        pain_bg    = PAIN_BG[pred_class]
                        pain_emoji = PAIN_EMOJI[pred_class]

                        st.markdown(f"""
                        <div style="background:{pain_bg}; border:2px solid {pain_color};
                                    border-radius:12px; padding:2rem; text-align:center;">
                            <div style="font-size:3rem;">{pain_emoji}</div>
                            <div style="font-size:1.8rem; font-weight:700; color:{pain_color};">
                                {pain_label}
                            </div>
                            <div style="font-size:1rem; color:#555; margin-top:0.5rem;">
                                Confidence: {proba_dict.get(pred_class, 0)*100:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Interpretation
                        interpretations = {
                            1: "Patient profile is consistent with no significant pain burden. EEG and clinical markers are within normal range.",
                            2: "Mild pain indicators detected. Consider monitoring KL grade progression and symptom duration.",
                            3: "Moderate pain burden indicated. EEG biomarkers and clinical scores suggest significant functional impact.",
                            4: "Severe pain profile detected. High KL grade, elevated McGill score, and EEG abnormalities consistent with severe KOA pain."
                        }
                        st.markdown(f"<br><small><i>{interpretations[pred_class]}</i></small>", unsafe_allow_html=True)

                    with chart_col:
                        st.markdown("**Probability Distribution**")
                        for cls in [1, 2, 3, 4]:
                            prob = proba_dict.get(cls, 0)
                            label = PAIN_LABELS[cls]
                            color = PAIN_COLORS[cls]
                            bar_pct = int(prob * 100)
                            is_pred = "🔵 " if cls == pred_class else "    "
                            st.markdown(f"""
                            <div style="margin: 6px 0;">
                                <div style="display:flex; justify-content:space-between; font-size:0.85rem;">
                                    <span>{is_pred}{label}</span>
                                    <span style="font-weight:600;">{prob*100:.1f}%</span>
                                </div>
                                <div style="background:#eee; border-radius:4px; height:10px; margin-top:3px;">
                                    <div style="background:{color}; width:{bar_pct}%; height:10px; border-radius:4px;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Key drivers
                        st.markdown("---")
                        st.markdown("**Key input factors:**")
                        kl_risk = "⚠️ High" if inputs['KL_grade'] >= 3 else ("⬆️ Moderate" if inputs['KL_grade'] >= 2 else "✅ Low")
                        mc_risk = "⚠️ High" if inputs['McGill'] >= 4 else ("⬆️ Moderate" if inputs['McGill'] >= 2 else "✅ Low")
                        dur_risk = "⚠️ Long" if inputs['Duration_of_symptom'] >= 5 else "✅ Short"
                        bmi_risk = "⚠️ Obese" if inputs['BMI'] >= 30 else ("⬆️ Overweight" if inputs['BMI'] >= 25 else "✅ Normal")

                        df_drivers = pd.DataFrame({
                            'Feature': ['KL Grade', 'McGill Score', 'Symptom Duration', 'BMI'],
                            'Value': [inputs['KL_grade'], inputs['McGill'], inputs['Duration_of_symptom'], round(inputs['BMI'],1)],
                            'Risk': [kl_risk, mc_risk, dur_risk, bmi_risk]
                        })
                        st.dataframe(df_drivers, use_container_width=True, hide_index=True)

                    # Disclaimer
                    st.markdown("""
                    <div class="disclaimer">
                        ⚠️ <b>Clinical Disclaimer:</b> This tool is intended for research and educational purposes only.
                        Predictions are based on a training set of 62 patients and should not replace clinical judgement.
                        All predictions must be interpreted by qualified healthcare professionals.
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    st.exception(e)

    # ---- FOOTER ----
    st.markdown("---")
    st.markdown("""
    <small style="color: #888;">
    KOA Pain Level Predictor | EEG + Clinical ML System |
    Model: Soft Voting Ensemble (RF + GBM + SVM + LR) |
    Validation: Leave-One-Out Cross-Validation
    </small>
    """, unsafe_allow_html=True)

# ============================================================
if __name__ == "__main__":
    main()

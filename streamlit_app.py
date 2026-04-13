"""
=============================================================================
KOA PAIN LEVEL PREDICTOR — STREAMLIT APP (SELF-CONTAINED)
=============================================================================
All training logic is embedded here — no external train_model.py needed.
Deploy directly to Streamlit Cloud with just this file + requirements.txt
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
    .metric-card {
        background: #f0f4f8;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #2d6a9f;
        margin: 0.4rem 0;
        font-size: 0.9rem;
    }
    .section-header {
        color: #1e3a5f;
        font-size: 1.05rem;
        font-weight: 600;
        border-bottom: 2px solid #2d6a9f;
        padding-bottom: 0.3rem;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
    }
    .disclaimer {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        font-size: 0.85rem;
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

CHANNELS   = ['Fp1', 'Fp2', 'Fz', 'F7', 'F8', 'T7', 'T8', 'Pz']
BANDS      = ['Alpha', 'Beta', 'Theta', 'Delta', 'Gamma']
CONDITIONS = ['Pre', 'Post', 'Standing', 'Sitting']

DATASET_URL = (
    "https://raw.githubusercontent.com/focuzguy1/Pain-Level-Prediction/"
    "324cd6a74838e471a00f8537ad9376376da1938b/koa_erd_processed.csv"
)

# ============================================================
# FEATURE ENGINEERING  (shared by training + inference)
# ============================================================

def engineer_features(src, is_dataframe=True):
    """
    src: pd.DataFrame (training) or dict (inference).
    Returns a pd.DataFrame of engineered features.
    """
    if not is_dataframe:
        src = pd.DataFrame([src])

    feats = {}

    def safe_log(arr):
        return np.log1p(np.where(np.array(arr) < 0, 0, np.array(arr)))

    def col(name):
        return src[name].values if name in src.columns else np.zeros(len(src))

    # 1. Frontal Alpha Asymmetry (FAA)
    for cond in CONDITIONS:
        v1 = col(f'Fp1_Alpha_{cond}')
        v2 = col(f'Fp2_Alpha_{cond}')
        feats[f'FAA_{cond}'] = (v1 - v2) / (v1 + v2 + 1e-6)

    # 2. Theta/Beta Ratio at Fz
    for cond in CONDITIONS:
        feats[f'TBR_Fz_{cond}'] = safe_log(
            col(f'Fz_Theta_{cond}') / (col(f'Fz_Beta_{cond}') + 1e-6)
        )

    # 3. Global Alpha Power
    for cond in CONDITIONS:
        vals = np.stack([col(f'{ch}_Alpha_{cond}') for ch in CHANNELS], axis=1)
        feats[f'Global_Alpha_{cond}'] = safe_log(vals.mean(axis=1))

    # 4. Global Theta Power
    for cond in CONDITIONS:
        vals = np.stack([col(f'{ch}_Theta_{cond}') for ch in CHANNELS], axis=1)
        feats[f'Global_Theta_{cond}'] = safe_log(vals.mean(axis=1))

    # 5. Temporal Delta Power (T7 + T8)
    for cond in CONDITIONS:
        vals = np.stack([col(f'T7_Delta_{cond}'), col(f'T8_Delta_{cond}')], axis=1)
        feats[f'Temporal_Delta_{cond}'] = safe_log(vals.mean(axis=1))

    # 6. Parietal Alpha (Pz)
    for cond in CONDITIONS:
        feats[f'Pz_Alpha_{cond}'] = safe_log(col(f'Pz_Alpha_{cond}'))

    # 7. Frontal Gamma (Fp1 + Fp2)
    for cond in ['Pre', 'Post']:
        vals = np.stack([col(f'Fp1_Gamma_{cond}'), col(f'Fp2_Gamma_{cond}')], axis=1)
        feats[f'Frontal_Gamma_{cond}'] = safe_log(vals.mean(axis=1))

    # 8. Delta / Alpha Ratio
    for cond in ['Pre', 'Post']:
        d = np.stack([col(f'{ch}_Delta_{cond}') for ch in CHANNELS], axis=1).mean(axis=1)
        a = np.stack([col(f'{ch}_Alpha_{cond}') for ch in CHANNELS], axis=1).mean(axis=1)
        feats[f'Delta_Alpha_Ratio_{cond}'] = safe_log(d / (a + 1e-6))

    # 9. Pre->Post and Stand->Sit Alpha change
    feats['Alpha_Change_PrePost']  = feats['Global_Alpha_Post']    - feats['Global_Alpha_Pre']
    feats['Alpha_Change_StandSit'] = feats['Global_Alpha_Standing'] - feats['Global_Alpha_Sitting']

    # 10. Clinical features
    for c in ['KL_grade', 'McGill', 'Age', 'BMI', 'Duration_of_symptom',
              'PRI', 'WMI', 'PSI', 'QoL']:
        feats[c] = col(c)

    # 11. Cognitive composite
    feats['Cognitive_Composite'] = (col('PRI') + col('WMI') + col('PSI')) / 3.0

    result = pd.DataFrame(feats)
    result = result.fillna(result.median())
    return result

# ============================================================
# TRAINING PIPELINE  (runs once; cached in memory + disk)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_or_train():
    """
    Returns (scaler, model, metadata, error_msg).
    Trains from scratch if no saved artifacts exist.
    Everything runs in-process — no subprocess calls.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import (RandomForestClassifier,
                                   GradientBoostingClassifier,
                                   VotingClassifier)
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.metrics import (accuracy_score, f1_score,
                                  balanced_accuracy_score)

    ART = "model_artifacts"
    SP, MP, DP = f"{ART}/scaler.joblib", f"{ART}/ensemble.joblib", f"{ART}/meta.joblib"

    # Try loading cached artifacts first
    if os.path.exists(MP) and os.path.exists(DP):
        try:
            return (joblib.load(SP), joblib.load(MP), joblib.load(DP), None)
        except Exception:
            pass

    # Fetch dataset
    try:
        df = pd.read_csv(DATASET_URL)
    except Exception as e:
        return None, None, None, f"Cannot fetch dataset: {e}"

    # Clean column names
    df.columns = (df.columns.str.strip()
                  .str.replace(' ', '_')
                  .str.replace('-', '_')
                  .str.replace('Siiting', 'Sitting')
                  .str.replace('Posting', 'Post'))

    if 'Pain_Level' not in df.columns:
        return None, None, None, "Column 'Pain_Level' not found."

    # Feature engineering
    X = engineer_features(df, is_dataframe=True)
    y = df['Pain_Level'].values
    feat_names = list(X.columns)

    # Scale
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # SMOTE
    try:
        from imblearn.over_sampling import SMOTE
        from collections import Counter
        k = max(1, min(3, min(Counter(y).values()) - 1))
        Xr, yr = SMOTE(random_state=42, k_neighbors=k).fit_resample(Xs, y)
    except Exception:
        Xr, yr = Xs, y

    # Define models
    rf  = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_split=3,
                                  min_samples_leaf=2, class_weight='balanced',
                                  random_state=42, n_jobs=-1)
    gb  = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                      learning_rate=0.05, subsample=0.8, random_state=42)
    svm = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced',
              probability=True, random_state=42)
    lr  = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000,
                              random_state=42, multi_class='multinomial')

    loo = LeaveOneOut()
    model_results = {}
    for name, m in [("Random Forest", rf), ("Gradient Boosting", gb),
                    ("SVM", svm), ("Logistic Regression", lr)]:
        yp = cross_val_predict(m, Xs, y, cv=loo)
        model_results[name] = {
            'accuracy':          float(accuracy_score(y, yp)),
            'f1_macro':          float(f1_score(y, yp, average='macro')),
            'balanced_accuracy': float(balanced_accuracy_score(y, yp))
        }

    # Ensemble
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('svm', svm), ('lr', lr)],
        voting='soft'
    )
    yp_e = cross_val_predict(ensemble, Xs, y, cv=loo)
    ens_acc = float(accuracy_score(y, yp_e))
    ens_f1  = float(f1_score(y, yp_e, average='macro'))
    ens_bal = float(balanced_accuracy_score(y, yp_e))

    # Fit on full SMOTE data
    ensemble.fit(Xr, yr)
    rf.fit(Xr, yr)
    fi = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False)

    meta = {
        'feature_names':        feat_names,
        'n_features':           len(feat_names),
        'n_samples':            len(y),
        'loo_accuracy':         ens_acc,
        'loo_f1_macro':         ens_f1,
        'loo_balanced_accuracy':ens_bal,
        'model_results':        model_results,
        'feature_importances':  fi.head(20).to_dict()
    }

    # Persist to disk (best-effort)
    try:
        os.makedirs(ART, exist_ok=True)
        joblib.dump(scaler,   SP)
        joblib.dump(ensemble, MP)
        joblib.dump(meta,     DP)
    except Exception:
        pass

    return scaler, ensemble, meta, None

# ============================================================
# MAIN APP
# ============================================================

def main():

    # HEADER
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0;font-size:2rem;">🧠 KOA Pain Level Predictor</h1>
        <p style="margin:0.5rem 0 0;opacity:0.9;font-size:1rem;">
            EEG-based machine learning system for Knee Osteoarthritis pain classification
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("⏳ Loading model (first run trains automatically — ~30 s)..."):
        scaler, model, meta, err = load_or_train()

    if err:
        st.error(f"❌ {err}")
        st.stop()

    # SIDEBAR
    with st.sidebar:
        st.markdown("### 📊 Model Performance")
        st.markdown(f"""
        <div class="metric-card"><b>Algorithm:</b> Soft Voting Ensemble<br>
        <small>RF · GBM · SVM · Logistic Regression</small></div>
        <div class="metric-card"><b>Validation:</b> Leave-One-Out CV</div>
        <div class="metric-card"><b>LOO-CV Accuracy:</b> {meta['loo_accuracy']*100:.1f}%</div>
        <div class="metric-card"><b>Balanced Accuracy:</b> {meta['loo_balanced_accuracy']*100:.1f}%</div>
        <div class="metric-card"><b>F1 Macro:</b> {meta['loo_f1_macro']*100:.1f}%</div>
        <div class="metric-card"><b>N patients:</b> {meta['n_samples']}</div>
        <div class="metric-card"><b>N features:</b> {meta['n_features']}</div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📈 Individual Models (LOO-CV)")
        for mname, mres in meta['model_results'].items():
            st.markdown(f"**{mname}**")
            st.progress(float(mres['f1_macro']))
            st.caption(f"F1: {mres['f1_macro']*100:.1f}%  |  Acc: {mres['accuracy']*100:.1f}%")

        st.markdown("---")
        st.markdown("### 🏆 Top 5 Features")
        for i, (fname, fimp) in enumerate(list(meta['feature_importances'].items())[:5]):
            bar = "█" * max(1, int(fimp * 80))
            st.caption(f"**{i+1}.** `{fname[:28]}`\n{bar} {fimp:.3f}")

    # INPUT TABS
    tab1, tab2, tab3 = st.tabs(["🩺 Clinical Features", "🧠 EEG Features", "🎯 Predict"])
    inputs = {}

    # TAB 1 — CLINICAL
    with tab1:
        st.markdown('<p class="section-header">Clinical & Demographic Information</p>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            inputs['KL_grade'] = st.selectbox(
                "KL Grade",
                options=[0, 1, 2, 3, 4],
                format_func=lambda x: {0:"0 — Normal", 1:"1 — Doubtful", 2:"2 — Minimal",
                                        3:"3 — Moderate", 4:"4 — Severe"}[x],
                index=2
            )
            inputs['McGill'] = st.slider("McGill Pain Score (0–5)", 0, 5, 2)
            inputs['QoL']    = st.slider("Quality of Life (0–10)", 0, 10, 5)
        with c2:
            inputs['Age'] = st.number_input("Age (years)", 20, 90, 55, step=1)
            inputs['BMI'] = st.number_input("BMI (kg/m²)", 15.0, 50.0, 27.0, step=0.1, format="%.1f")
            inputs['Duration_of_symptom'] = st.slider("Symptom Duration (years)", 0, 15, 3)
        with c3:
            inputs['PRI'] = st.number_input("PRI — Perceptual Reasoning", 0, 200, 90)
            inputs['WMI'] = st.number_input("WMI — Working Memory", 0, 200, 95)
            inputs['PSI'] = st.number_input("PSI — Processing Speed", 0, 200, 85)
        st.info("💡 **Strongest predictors:** KL Grade & McGill (p<0.001), Duration (p<0.001), BMI (p=0.002)")

    # TAB 2 — EEG
    with tab2:
        st.markdown('<p class="section-header">EEG Band Power Values (μV²)</p>',
                    unsafe_allow_html=True)
        st.caption("Enter raw power spectral density values per channel × band × condition.")
        for ch in CHANNELS:
            with st.expander(f"📡  Channel: **{ch}**"):
                for band in BANDS:
                    st.markdown(f"<small><b>{band}</b></small>", unsafe_allow_html=True)
                    cols = st.columns(4)
                    for i, cond in enumerate(CONDITIONS):
                        key = f'{ch}_{band}_{cond}'
                        with cols[i]:
                            inputs[key] = st.number_input(
                                cond, min_value=0.0, value=1000.0,
                                step=100.0, key=key, format="%.1f"
                            )
        st.info("🔬 Key biomarkers: Frontal Alpha Asymmetry (Fp1 vs Fp2), Theta/Beta Ratio (Fz), Temporal Delta (T7+T8).")

    # TAB 3 — PREDICT
    with tab3:
        st.markdown('<p class="section-header">Patient Summary</p>', unsafe_allow_html=True)
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown(f"- **KL Grade:** {inputs['KL_grade']}")
            st.markdown(f"- **McGill:** {inputs['McGill']}  |  **QoL:** {inputs['QoL']}")
            st.markdown(f"- **Age:** {inputs['Age']} yrs  |  **BMI:** {inputs['BMI']:.1f}")
            st.markdown(f"- **Duration:** {inputs['Duration_of_symptom']} yrs")
        with sc2:
            st.markdown(f"- **PRI / WMI / PSI:** {inputs['PRI']} / {inputs['WMI']} / {inputs['PSI']}")
            st.markdown(f"- **EEG:** {len(CHANNELS)} channels × {len(BANDS)} bands × 4 conditions")

        st.markdown("---")

        if st.button("🔍  Predict Pain Level", use_container_width=True):
            with st.spinner("Running prediction..."):
                try:
                    X_in = engineer_features(inputs, is_dataframe=False)
                    for feat in meta['feature_names']:
                        if feat not in X_in.columns:
                            X_in[feat] = 0.0
                    X_in = X_in[meta['feature_names']]
                    Xs   = scaler.transform(X_in)

                    pred  = int(model.predict(Xs)[0])
                    proba = {int(c): float(p) for c, p in zip(model.classes_,
                                                               model.predict_proba(Xs)[0])}

                    st.markdown("---")
                    st.markdown("### 🎯 Prediction Result")
                    r1, r2 = st.columns(2)

                    with r1:
                        st.markdown(f"""
                        <div style="background:{PAIN_BG[pred]};border:2px solid {PAIN_COLORS[pred]};
                             border-radius:12px;padding:2rem;text-align:center;">
                          <div style="font-size:3rem;">{PAIN_EMOJI[pred]}</div>
                          <div style="font-size:1.8rem;font-weight:700;color:{PAIN_COLORS[pred]};">
                            {PAIN_LABELS[pred]}
                          </div>
                          <div style="color:#555;margin-top:0.5rem;">
                            Confidence: {proba.get(pred,0)*100:.1f}%
                          </div>
                        </div>
                        """, unsafe_allow_html=True)
                        note = {
                            1: "EEG and clinical markers within normal range.",
                            2: "Mild indicators — monitor KL grade and symptom progression.",
                            3: "Moderate pain burden — significant functional impact likely.",
                            4: "Severe profile — high KL grade, McGill, and EEG abnormalities."
                        }
                        st.markdown(f"<br><small><i>{note[pred]}</i></small>",
                                    unsafe_allow_html=True)

                    with r2:
                        st.markdown("**Probability distribution**")
                        for cls in [1, 2, 3, 4]:
                            p   = proba.get(cls, 0)
                            pct = int(p * 100)
                            mrk = " ◀ predicted" if cls == pred else ""
                            st.markdown(f"""
                            <div style="margin:6px 0;">
                              <div style="display:flex;justify-content:space-between;font-size:.85rem;">
                                <span>{PAIN_LABELS[cls]}<small style="color:#888;">{mrk}</small></span>
                                <b>{pct}%</b>
                              </div>
                              <div style="background:#eee;border-radius:4px;height:10px;margin-top:3px;">
                                <div style="background:{PAIN_COLORS[cls]};width:{pct}%;height:10px;border-radius:4px;"></div>
                              </div>
                            </div>""", unsafe_allow_html=True)

                        st.markdown("---")
                        st.markdown("**Key risk factors**")
                        kl_r  = "⚠️ High"    if inputs['KL_grade'] >= 3             else ("⬆️ Mod" if inputs['KL_grade'] >= 2             else "✅ Low")
                        mc_r  = "⚠️ High"    if inputs['McGill'] >= 4               else ("⬆️ Mod" if inputs['McGill'] >= 2               else "✅ Low")
                        dr_r  = "⚠️ Long"    if inputs['Duration_of_symptom'] >= 5  else "✅ Short"
                        bm_r  = "⚠️ Obese"   if inputs['BMI'] >= 30                 else ("⬆️ OW"  if inputs['BMI'] >= 25                 else "✅ Normal")
                        st.dataframe(pd.DataFrame({
                            'Feature': ['KL Grade', 'McGill', 'Duration', 'BMI'],
                            'Value':   [inputs['KL_grade'], inputs['McGill'],
                                        inputs['Duration_of_symptom'], round(inputs['BMI'], 1)],
                            'Risk':    [kl_r, mc_r, dr_r, bm_r]
                        }), use_container_width=True, hide_index=True)

                    st.markdown("""
                    <div class="disclaimer">
                      ⚠️ <b>Clinical Disclaimer:</b> Research and educational purposes only.
                      Based on N=62 patients. Not a substitute for clinical judgement.
                    </div>""", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.exception(e)

    st.markdown("---")
    st.caption("KOA Pain Level Predictor  |  Soft Voting Ensemble (RF + GBM + SVM + LR)  |  LOO-CV  |  Research use only")

# ============================================================
if __name__ == "__main__":
    main()

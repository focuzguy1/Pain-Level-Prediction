"""
============================================================
KOA PAIN LEVEL PREDICTOR — STREAMLIT APP
Pure inference. Loads one file: koa_model.joblib from GitHub.
No training. No retraining. New patients never touch training data.
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
    page_title="KOA Pain Level Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
    padding: 2rem; border-radius: 12px; color: white; margin-bottom: 2rem;
}
.metric-card {
    background: #f0f4f8; padding: .65rem 1rem; border-radius: 8px;
    border-left: 4px solid #2d6a9f; margin: .35rem 0; font-size: .88rem;
}
.section-hdr {
    color: #1e3a5f; font-size: 1.05rem; font-weight: 600;
    border-bottom: 2px solid #2d6a9f; padding-bottom: .3rem;
    margin-top: 1rem; margin-bottom: .75rem;
}
.result-box {
    border-radius: 12px; padding: 2rem; text-align: center;
    border-width: 2px; border-style: solid;
}
.disclaimer {
    background: #fff3cd; border: 1px solid #ffc107;
    border-radius: 8px; padding: 1rem; margin-top: 1rem; font-size: .85rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

PAIN_LABELS = {1: "No Pain", 2: "Mild Pain", 3: "Moderate Pain", 4: "Severe Pain"}
PAIN_COLORS = {1: "#27ae60", 2: "#f39c12", 3: "#e67e22", 4: "#e74c3c"}
PAIN_EMOJI  = {1: "✅", 2: "🟡", 3: "🟠", 4: "🔴"}
PAIN_BG     = {1: "#d4efdf", 2: "#fef9e7", 3: "#fdebd0", 4: "#fadbd8"}

CHANNELS   = ['Fp1', 'Fp2', 'Fz', 'F7', 'F8', 'T7', 'T8', 'Pz']
BANDS      = ['Alpha', 'Beta', 'Theta', 'Delta', 'Gamma']
CONDITIONS = ['Pre', 'Post', 'Standing', 'Sitting']

# ── Change this to your repo URL after uploading koa_model.joblib ──
MODEL_URL = (
    "https://raw.githubusercontent.com/focuzdev/"
    "Pain-Level-Prediction/main/koa_model.joblib"
)

# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (identical to train_and_save.py)
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
            col(f'Fz_Theta_{cond}') / (col(f'Fz_Beta_{cond}') + 1e-6)
        )

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
# LOAD MODEL BUNDLE  (download once from GitHub, cache forever)
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_bundle():
    """
    Downloads koa_model.joblib from GitHub.
    Returns the bundle dict or raises a clear error.
    """
    tmp = tempfile.mktemp(suffix='.joblib')
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        bundle = joblib.load(tmp)
        return bundle, None
    except Exception as e:
        return None, str(e)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────

def main():

    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0;font-size:2rem;">🧠 KOA Pain Level Predictor</h1>
        <p style="margin:.5rem 0 0;opacity:.9;font-size:1rem;">
            EEG + clinical machine learning — Knee Osteoarthritis pain classification
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load bundle ──────────────────────────────────────────
    with st.spinner("Loading pre-trained model..."):
        bundle, err = load_bundle()

    if err:
        st.error("❌ Could not load model file.")
        st.markdown(f"""
        **Error:** `{err}`

        **To fix this:**
        1. Run `train_and_save.py` in Google Colab
        2. Upload the output file **`koa_model.joblib`** to your GitHub repo root
        3. Make sure `MODEL_URL` in `streamlit_app.py` points to the correct raw URL
        4. Reload this app
        """)
        st.stop()

    # Unpack bundle
    scaler       = bundle['scaler']
    model        = bundle['model']
    feat_names   = bundle['feature_names']
    model_name   = bundle['model_name']
    all_results  = bundle['all_model_results']
    feat_imp     = bundle.get('feature_importances', {})

    st.success(f"✅ Model loaded: **{model_name}** (trained on {bundle['n_samples']} patients)")

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📊 Model Performance (LOO-CV)")
        st.markdown(f"""
        <div class="metric-card">
            <b>Deployed model:</b> {model_name}
        </div>
        <div class="metric-card">
            <b>Accuracy:</b> {bundle['loo_accuracy']*100:.1f}%
        </div>
        <div class="metric-card">
            <b>Balanced Accuracy:</b> {bundle['loo_balanced_accuracy']*100:.1f}%
        </div>
        <div class="metric-card">
            <b>F1 Macro:</b> {bundle['loo_f1_macro']*100:.1f}%
        </div>
        <div class="metric-card">
            <b>F1 Weighted:</b> {bundle['loo_f1_weighted']*100:.1f}%
        </div>
        <div class="metric-card">
            <b>Cohen's Kappa:</b> {bundle['loo_kappa']:.3f}
        </div>
        <div class="metric-card">
            <b>Sensitivity (macro):</b> {bundle.get('loo_macro_sensitivity', 0)*100:.1f}%
        </div>
        <div class="metric-card">
            <b>Specificity (macro):</b> {bundle.get('loo_macro_specificity', 0)*100:.1f}%
        </div>
        <div class="metric-card">
            <b>Training N:</b> {bundle['n_samples']} patients
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📈 All Models Compared")
        for mname, mres in all_results.items():
            is_deployed = "🟢 " if mname == model_name else "    "
            st.markdown(f"**{is_deployed}{mname}**")
            st.progress(float(mres['f1_macro']))
            st.caption(
                f"F1: {mres['f1_macro']*100:.1f}%  "
                f"Acc: {mres['accuracy']*100:.1f}%  "
                f"κ: {mres['kappa']:.3f}  |  "
                f"Sens: {mres.get('macro_sensitivity',0)*100:.1f}%  "
                f"Spec: {mres.get('macro_specificity',0)*100:.1f}%"
            )

        if feat_imp:
            st.markdown("---")
            st.markdown("### 🏆 Top 5 Features")
            for i, (fname, fimp) in enumerate(list(feat_imp.items())[:5]):
                bar = "█" * max(1, int(fimp * 80))
                st.caption(f"**{i+1}.** `{fname[:28]}`\n{bar} {fimp:.3f}")

        st.markdown("---")
        st.markdown("### 🗂 Class Distribution (training)")
        dist = bundle.get('class_dist', {})
        for k in sorted(dist):
            st.caption(f"Class {k} ({PAIN_LABELS[k]}): n={dist[k]}")

    # ── Input tabs ───────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🩺 Clinical Features",
        "🧠 EEG Features",
        "📂 Upload CSV Row",
        "🎯 Predict"
    ])
    inputs = {}

    # TAB 1 — CLINICAL
    with tab1:
        st.markdown('<p class="section-hdr">Clinical & Demographic Information</p>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            inputs['KL_grade'] = st.selectbox(
                "KL Grade (radiological severity)",
                options=[0, 1, 2, 3, 4],
                format_func=lambda x: {
                    0: "0 — Normal", 1: "1 — Doubtful", 2: "2 — Minimal",
                    3: "3 — Moderate", 4: "4 — Severe"
                }[x],
                index=2
            )
            inputs['McGill'] = st.slider("McGill Pain Score (0–5)", 0, 5, 2)
            inputs['QoL']    = st.slider("Quality of Life (0–10)", 0, 10, 5)
        with c2:
            inputs['Age'] = st.number_input("Age (years)", 20, 90, 55, step=1)
            inputs['BMI'] = st.number_input("BMI (kg/m²)", 15.0, 50.0, 27.0,
                                             step=0.1, format="%.1f")
            inputs['Duration_of_symptom'] = st.slider("Symptom Duration (years)", 0, 15, 3)
        with c3:
            inputs['PRI'] = st.number_input("PRI — Perceptual Reasoning", 0, 200, 90)
            inputs['WMI'] = st.number_input("WMI — Working Memory",        0, 200, 95)
            inputs['PSI'] = st.number_input("PSI — Processing Speed",      0, 200, 85)

        st.info("💡 **Strongest predictors (from training data):** "
                "KL Grade & McGill Score (p<0.001), Symptom Duration (p<0.001), BMI (p=0.002)")

    # TAB 2 — EEG
    with tab2:
        st.markdown('<p class="section-hdr">EEG Band Power Values (μV²)</p>',
                    unsafe_allow_html=True)
        st.caption(
            "Enter raw power spectral density values. "
            "If only Pre-task recordings are available, copy those values to all conditions."
        )
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
        st.info("🔬 Key EEG biomarkers: Frontal Alpha Asymmetry (Fp1 vs Fp2), "
                "Theta/Beta Ratio (Fz), Temporal Delta (T7+T8).")

    # TAB 3 — CSV UPLOAD
    with tab3:
        st.markdown('<p class="section-hdr">Upload a Patient Row (CSV)</p>',
                    unsafe_allow_html=True)
        st.markdown(
            "Upload a single-row CSV matching your dataset column format. "
            "Column names should follow `koa_erd_processed.csv` conventions. "
            "Values will override the manual inputs above."
        )
        uploaded = st.file_uploader("Choose CSV", type=["csv"])
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
                    k: float(v) if isinstance(v, (int, float, np.integer, np.floating))
                    else v
                    for k, v in row.items()
                })
                st.success(f"✅ Loaded {len(row)} columns from uploaded file.")
                st.dataframe(
                    pd.DataFrame([row]).T.rename(columns={0: "Value"}),
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")

    # TAB 4 — PREDICT
    with tab4:
        st.markdown('<p class="section-hdr">Patient Summary</p>', unsafe_allow_html=True)

        s1, s2 = st.columns(2)
        with s1:
            st.markdown(f"- **KL Grade:** {inputs.get('KL_grade', '—')}")
            st.markdown(f"- **McGill:** {inputs.get('McGill', '—')}  |  "
                        f"**QoL:** {inputs.get('QoL', '—')}")
            bmi_val = inputs.get('BMI', 0)
            st.markdown(f"- **Age:** {inputs.get('Age', '—')} yrs  |  "
                        f"**BMI:** {float(bmi_val):.1f} kg/m²")
            st.markdown(f"- **Duration:** {inputs.get('Duration_of_symptom', '—')} yrs")
        with s2:
            st.markdown(f"- **PRI / WMI / PSI:** "
                        f"{inputs.get('PRI','—')} / "
                        f"{inputs.get('WMI','—')} / "
                        f"{inputs.get('PSI','—')}")
            st.markdown(f"- **EEG inputs:** "
                        f"{len(CHANNELS)} channels × {len(BANDS)} bands × 4 conditions")

        st.markdown("---")

        if st.button("🔍  Predict Pain Level", use_container_width=True):
            with st.spinner("Predicting..."):
                try:
                    # 1. Engineer features from raw inputs
                    X_in = engineer_features(inputs)

                    # 2. Align to exact training feature order
                    for feat in feat_names:
                        if feat not in X_in.columns:
                            X_in[feat] = 0.0
                    X_in = X_in[feat_names]

                    # 3. Scale using the saved (training-fitted) scaler
                    X_sc = scaler.transform(X_in)

                    # 4. Predict with the saved model
                    pred  = int(model.predict(X_sc)[0])
                    proba = {
                        int(c): float(p)
                        for c, p in zip(model.classes_,
                                        model.predict_proba(X_sc)[0])
                    }

                    # ── Display results ───────────────────────
                    st.markdown("---")
                    st.markdown("### 🎯 Prediction Result")

                    r1, r2 = st.columns(2)

                    with r1:
                        st.markdown(f"""
                        <div class="result-box"
                             style="background:{PAIN_BG[pred]};
                                    border-color:{PAIN_COLORS[pred]};">
                          <div style="font-size:3rem;">{PAIN_EMOJI[pred]}</div>
                          <div style="font-size:1.8rem;font-weight:700;
                               color:{PAIN_COLORS[pred]};">
                            {PAIN_LABELS[pred]}
                          </div>
                          <div style="color:#555;margin-top:.5rem;">
                            Model confidence: {proba.get(pred,0)*100:.1f}%
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                        clinical_note = {
                            1: "Clinical and EEG markers within normal range.",
                            2: "Mild indicators present. Monitor KL grade and symptom progression.",
                            3: "Moderate pain burden. Significant functional impact likely.",
                            4: "Severe pain profile. Consistent with advanced KOA."
                        }
                        st.markdown(
                            f"<br><small><i>{clinical_note[pred]}</i></small>",
                            unsafe_allow_html=True
                        )

                    with r2:
                        st.markdown("**Probability per class**")
                        for cls in [1, 2, 3, 4]:
                            p   = proba.get(cls, 0)
                            pct = int(p * 100)
                            mark = " ◀ predicted" if cls == pred else ""
                            st.markdown(f"""
                            <div style="margin:7px 0;">
                              <div style="display:flex;justify-content:space-between;
                                   font-size:.85rem;">
                                <span>{PAIN_LABELS[cls]}
                                  <small style="color:#999;">{mark}</small>
                                </span>
                                <b>{pct}%</b>
                              </div>
                              <div style="background:#e8e8e8;border-radius:4px;
                                   height:10px;margin-top:3px;">
                                <div style="background:{PAIN_COLORS[cls]};
                                     width:{pct}%;height:10px;
                                     border-radius:4px;"></div>
                              </div>
                            </div>""", unsafe_allow_html=True)

                        st.markdown("---")
                        st.markdown("**Sensitivity & specificity for predicted class**")
                        per_cls_ss = bundle.get('loo_per_class_sens_spec', {})
                        if per_cls_ss and str(pred) in [str(k) for k in per_cls_ss]:
                            cls_key = pred
                            cls_ss  = per_cls_ss.get(cls_key, per_cls_ss.get(str(cls_key), {}))
                            if cls_ss:
                                sens_val = cls_ss.get('sensitivity', 0)
                                spec_val = cls_ss.get('specificity', 0)
                                s1, s2 = st.columns(2)
                                with s1:
                                    st.metric(
                                        label=f"Sensitivity — {PAIN_LABELS[pred]}",
                                        value=f"{sens_val*100:.1f}%",
                                        help="Of all patients truly in this class, % correctly identified"
                                    )
                                with s2:
                                    st.metric(
                                        label=f"Specificity — {PAIN_LABELS[pred]}",
                                        value=f"{spec_val*100:.1f}%",
                                        help="Of all patients NOT in this class, % correctly excluded"
                                    )

                        st.markdown("**All classes — sensitivity & specificity (LOO-CV)**")
                        ss_rows = []
                        for cls in [1, 2, 3, 4]:
                            cls_data = per_cls_ss.get(cls, per_cls_ss.get(str(cls), {}))
                            marker = " ◀" if cls == pred else ""
                            ss_rows.append({
                                'Pain level': PAIN_LABELS[cls] + marker,
                                'Sensitivity': f"{cls_data.get('sensitivity',0)*100:.1f}%",
                                'Specificity': f"{cls_data.get('specificity',0)*100:.1f}%",
                                'TP': cls_data.get('TP', '—'),
                                'FP': cls_data.get('FP', '—'),
                                'TN': cls_data.get('TN', '—'),
                                'FN': cls_data.get('FN', '—'),
                            })
                        st.dataframe(
                            pd.DataFrame(ss_rows),
                            use_container_width=True,
                            hide_index=True
                        )

                        st.markdown("---")
                        st.markdown("**Key clinical risk factors**")

                        kl  = inputs.get('KL_grade', 0)
                        mc  = inputs.get('McGill', 0)
                        dur = inputs.get('Duration_of_symptom', 0)
                        bmi = float(inputs.get('BMI', 0))

                        kl_r  = "⚠️ High" if kl  >= 3 else ("⬆️ Moderate" if kl  >= 2 else "✅ Low")
                        mc_r  = "⚠️ High" if mc  >= 4 else ("⬆️ Moderate" if mc  >= 2 else "✅ Low")
                        dr_r  = "⚠️ Long" if dur >= 5 else "✅ Short"
                        bm_r  = "⚠️ Obese" if bmi >= 30 else ("⬆️ Overweight" if bmi >= 25 else "✅ Normal")

                        st.dataframe(
                            pd.DataFrame({
                                'Feature': ['KL Grade', 'McGill', 'Duration (yrs)', 'BMI'],
                                'Value':   [kl, mc, dur, round(bmi, 1)],
                                'Risk':    [kl_r, mc_r, dr_r, bm_r]
                            }),
                            use_container_width=True,
                            hide_index=True
                        )

                    st.markdown("""
                    <div class="disclaimer">
                      ⚠️ <b>Clinical Disclaimer:</b> For research and educational
                      purposes only. Based on N=62 patients. Not a substitute for
                      clinical judgement. Consult a qualified healthcare professional.
                    </div>""", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.exception(e)

    st.markdown("---")
    st.caption(
        f"KOA Pain Level Predictor  |  Model: {model_name}  |  "
        f"LOO-CV Validated  |  Inference only — model trained offline  |  Research use only"
    )

# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

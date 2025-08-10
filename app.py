# app.py
import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ------------------ Page & Styles ------------------
st.set_page_config(page_title="CardioCheck — Heart Disease Risk", page_icon="❤️", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.2rem;padding-bottom:1.6rem;}
.topbar{display:flex;justify-content:space-between;align-items:center;padding:1rem 0;border-bottom:1px solid rgba(255,255,255,.08)   ; line-height: 2.4; /* make text breathe more */
;}
.brand{font-weight:800;font-size:1.2rem;}
.nav a{margin:0 .8rem;text-decoration:none;color:rgba(255,255,255,.85);} .nav a:hover{color:#fff;}
.hero{background:#0f172a;border-radius:18px;padding:2rem;}
.hero h1{font-size:2.4rem;line-height:1.15;margin:0 0 .8rem 0;}
.hero p{opacity:.9;margin:.2rem 0 1.2rem 0;}
.btn{display:inline-block;padding:.8rem 1.2rem;font-weight:700;border-radius:12px;border:0;background:linear-gradient(90deg,#2563eb,#7c3aed);color:#fff;text-decoration:none;}
.btn.secondary{background:transparent;border:1px solid rgba(255,255,255,.18);}
.card{background:#111418;border:1px solid #262b33;border-radius:16px;padding:1rem;}
div[data-testid="stForm"]{background:#111418;border:1px solid #262b33;border-radius:16px;padding:1rem;}
div.stButton>button:first-child{width:100%;padding:.9rem 0;font-weight:700;border-radius:10px;}
.footer{opacity:.75;font-size:.9rem;padding:1.2rem 0;border-top:1px solid rgba(255,255,255,.08);}
</style>
""", unsafe_allow_html=True)

# ------------------ Load Model Bundle ------------------
@st.cache_resource
def load_model_bundle():
    model = joblib.load("heart_model.pkl")
    features = joblib.load("feature_names.pkl")
    return model, features

@st.cache_data
def load_sample():
    return pd.read_csv("X_test_sample.csv")

model, FEATURES = load_model_bundle()
X_sample = load_sample()

if st.sidebar.button("Reload latest model"):
    st.cache_resource.clear()
    st.rerun()

# ------------------ Mappings & Helpers ------------------
CP_MAP   = {"Typical Angina":0, "Atypical Angina":1, "Non-Anginal Pain":2, "Asymptomatic":3}
ECG_MAP  = {"Normal":0, "ST-T Wave Abnormality":1, "Left Ventricular Hypertrophy":2}
SLOPE_MAP= {"Upsloping":0, "Flat":1, "Downsloping":2}
THAL_MAP = {"Normal":0, "Fixed Defect":1, "Reversible Defect":2, "Unknown":3}

def encode_inputs(ui):
    return pd.DataFrame([{
        "age": ui["age"],
        "sex": 1 if ui["sex"] == "Male" else 0,
        "cp": CP_MAP[ui["cp"]],
        "trestbps": ui["trestbps"],
        "chol": ui["chol"],
        "fbs": 1 if ui["fbs"] == "Yes" else 0,
        "restecg": ECG_MAP[ui["restecg"]],
        "thalach": ui["thalach"],
        "exang": 1 if ui["exang"] == "Yes" else 0,
        "oldpeak": ui["oldpeak"],
        "slope": SLOPE_MAP[ui["slope"]],
        "ca": int(ui["ca"]),
        "thal": THAL_MAP[ui["thal"]],
    }])[FEATURES]

def shap_beeswarm_block(model, X_background):
    rf = model.named_steps["rf"]; scaler = model.named_steps["scale"]
    X_bg_sc = pd.DataFrame(scaler.transform(X_background), columns=FEATURES, index=X_background.index)
    explainer = shap.TreeExplainer(rf)

    def _render(save_path=None):
        fig = plt.gcf()
        st.pyplot(fig, clear_figure=True)
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    try:
        exp = explainer(X_bg_sc)
        shap.plots.beeswarm(exp, max_display=12, show=False)
        _render("shap_beeswarm.png")
        return
    except Exception:
        pass

    try:
        sv = explainer.shap_values(X_bg_sc)
        if isinstance(sv, list) and len(sv) > 1:
            shap.summary_plot(sv[1], X_bg_sc, feature_names=FEATURES, show=False)
        else:
            shap.summary_plot(sv if not isinstance(sv, list) else sv[0],
                              X_bg_sc, feature_names=FEATURES, show=False)
        _render("shap_beeswarm.png")
    except Exception as e:
        st.info(f"SHAP summary unavailable. ({e})")

# ------------------ Topbar (with anchors) ------------------
st.markdown(
    "<div class='topbar'><div class='brand'>CardioCheck</div>"
    "<div class='nav'><a href='#home'>Home</a>"
    "<a href='#predictions'>Predictions</a>"
    "<a href='#about'>About</a></div></div>",
    unsafe_allow_html=True
)

# ------------------ Hero ------------------
st.markdown("<a id='home'></a>", unsafe_allow_html=True)
L, R = st.columns([1.15, .85], vertical_alignment="center")
with L:
    st.markdown("<div class='hero'>", unsafe_allow_html=True)
    st.markdown("<h1>Empowering You to Take<br/>Charge of Your Heart Health</h1>", unsafe_allow_html=True)
    st.markdown("<p>Estimate heart-disease risk from routine clinical data — with clear SHAP explanations for clinicians.</p>", unsafe_allow_html=True)
    st.markdown("<a class='btn' href='#predictions'>Get Started</a> &nbsp; <a class='btn secondary' href='#about'>Learn More</a>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with R:
    hero_path = "assets/hero.webp"
    if os.path.exists(hero_path):
        st.image(hero_path, use_container_width=True)          # <-- fixed
    else:
        st.image(
            "https://images.unsplash.com/photo-1584013515353-5f6f0f8b52c7?q=80&w=1200&auto=format&fit=crop",
            use_container_width=True                            # <-- fixed
        )

st.write("")

# ------------------ Predictions (Wide Form) ------------------
st.markdown("<a id='predictions'></a>", unsafe_allow_html=True)
st.markdown("### Predictions")
with st.form("hd_form", clear_on_submit=False):
    def dflt(col, cast=int, fallback=0):
        return cast(X_sample[col].mean()) if col in X_sample else fallback

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        age = st.number_input("Age", 20, 100, dflt("age"))
        sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
        cp = st.selectbox("Chest Pain Type", list(CP_MAP.keys()))
    with c2:
        trestbps = st.number_input("Resting BP (mmHg)", 80, 220, dflt("trestbps"))
        chol = st.number_input("Cholesterol (mg/dL)", 100, 600, dflt("chol"))
        thalach = st.number_input("Max Heart Rate", 70, 220, dflt("thalach"))
    with c3:
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dL", ["Yes", "No"], horizontal=True)
        exang = st.radio("Exercise Induced Angina", ["Yes", "No"], horizontal=True)
        restecg = st.selectbox("Resting ECG", list(ECG_MAP.keys()))
    with c4:
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.5,
                                  float(round(X_sample.get("oldpeak", pd.Series([1.0])).mean(), 1)), step=0.1)
        slope = st.selectbox("ST Slope", list(SLOPE_MAP.keys()))
        ca = st.selectbox("No. of Major Vessels (0–3)", [0,1,2,3])
        thal = st.selectbox("Thalassemia", list(THAL_MAP.keys()))

    submitted = st.form_submit_button("Predict Heart Disease")

if submitted:
    ui = {"age":age,"sex":sex,"cp":cp,"trestbps":trestbps,"chol":chol,"fbs":fbs,
          "restecg":restecg,"thalach":thalach,"exang":exang,"oldpeak":oldpeak,
          "slope":slope,"ca":ca,"thal":thal}
    X_user = encode_inputs(ui)
    prob = float(model.predict_proba(X_user)[0,1])
    pred = "Disease" if prob >= 0.5 else "No Disease"

    m1, m2, m3 = st.columns(3)
    with m1: st.success(f"Estimated Risk: **{prob*100:.1f}%**")
    with m2: st.info(f"Predicted Class: **{pred}**")
    with m3: st.caption("Threshold = 0.50 (adjust in code if needed)")
    # Patient-friendly explanation
    if pred == "No Disease":
        st.write(
        "✅ Based on the information provided, patient's likelihood of having heart disease is **low to moderate**. "
        "This is not a diagnosis, but it suggests that the current risk is not high. "
        "NB: Maintain healthy habits and continue regular check-ups."
    )
    else:
        st.write(
        "⚠️ Based on the information provided, patient's likelihood of having heart disease is **high**. "
        "This is not a diagnosis, but patient should speak with a healthcare professional promptly "
        "for further testing and advice."
    )


    with st.expander("Show encoded inputs"):
        st.dataframe(X_user.T.rename(columns={0:"value"}))

    with st.expander("Why this prediction? (SHAP)"):
        st.markdown("""
        **How to read:**  
        - **Red** = higher feature values; **Blue** = lower feature values  
        - Points to the **right** increase risk; to the **left** decrease risk  
        - **Age/Sex**: red ≈ older age / male (sex=1); blue ≈ younger age / female (sex=0)
        """)
        bg = X_sample.sample(n=min(200, len(X_sample)), random_state=42)
        shap_beeswarm_block(model, bg)
       

# ------------------ About & Footer ------------------
st.markdown("<a id='about'></a>", unsafe_allow_html=True)
st.markdown("### About")
st.markdown("<div class='card'>This prototype uses a pre-trained Random Forest and SHAP explanations to make risk predictions understandable. It is intended for educational decision support and should not replace clinical judgement.</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>© CardioCheck · ML-powered risk estimates with interpretability</div>", unsafe_allow_html=True)


# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

# -----------------------------
# Load model, features, samples
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("heart_model.pkl")          # Pipeline(StandardScaler, RandomForest)
    features = joblib.load("feature_names.pkl")     # list of column names
    return model, features

@st.cache_data
def load_sample():
    return pd.read_csv("X_test_sample.csv")

model, FEATURES = load_model()
X_sample = load_sample()

# -----------------------------
# Sidebar inputs (doctor-friendly labels)
# -----------------------------
st.sidebar.header("Enter Patient Data")

def widget_for_feature(f):
    if f == "sex":
        choice = st.sidebar.radio("Sex", ["Male", "Female"], help="1 = Male, 0 = Female")
        return 1 if choice == "Male" else 0

    if f == "fbs":
        choice = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dL", ["Yes", "No"], help="1 = Yes, 0 = No")
        return 1 if choice == "Yes" else 0

    if f == "exang":
        choice = st.sidebar.radio("Exercise Induced Angina", ["Yes", "No"], help="1 = Yes, 0 = No")
        return 1 if choice == "Yes" else 0

    if f == "cp":
        mapping = {
            "Typical Angina": 0,
            "Atypical Angina": 1,
            "Non-Anginal Pain": 2,
            "Asymptomatic": 3
        }
        choice = st.sidebar.selectbox("Chest Pain Type", list(mapping.keys()), help="Type of chest pain experienced")
        return mapping[choice]

    if f == "restecg":
        mapping = {
            "Normal": 0,
            "ST-T Wave Abnormality": 1,
            "Left Ventricular Hypertrophy": 2
        }
        choice = st.sidebar.selectbox("Resting ECG Results", list(mapping.keys()), help="Electrocardiographic results at rest")
        return mapping[choice]

    if f == "slope":
        mapping = {
            "Upsloping": 0,
            "Flat": 1,
            "Downsloping": 2
        }
        choice = st.sidebar.selectbox("Slope of Peak Exercise ST Segment", list(mapping.keys()))
        return mapping[choice]

    if f == "thal":
        mapping = {
            "Normal": 0,
            "Fixed Defect": 1,
            "Reversible Defect": 2,
            "Unknown": 3
        }
        choice = st.sidebar.selectbox("Thalassemia", list(mapping.keys()))
        return mapping[choice]

    if f == "oldpeak":
        lo = float(np.floor(X_sample[f].min())) if f in X_sample else 0.0
        hi = float(np.ceil(X_sample[f].max())) if f in X_sample else 6.5
        default = float(X_sample[f].mean()) if f in X_sample else (lo + hi) / 2
        return float(st.sidebar.slider("ST Depression (oldpeak)", lo, hi, default, 0.1))

    # Default numeric (age, trestbps, chol, thalach, ca)
    lo = int(np.floor(X_sample[f].min())) if f in X_sample else 0
    hi = int(np.ceil(X_sample[f].max())) if f in X_sample else 300
    default = int(round(X_sample[f].mean())) if f in X_sample else int((lo + hi) / 2)
    return int(st.sidebar.slider(f.replace('_', ' ').title(), lo, hi, default))

user_dict = {f: widget_for_feature(f) for f in FEATURES}
X_user = pd.DataFrame([user_dict])[FEATURES]

# -----------------------------
# Prediction
# -----------------------------
st.title("Enhanced Heart Disease Prediction System")
st.write("Pre-trained Random Forest with SHAP explanations for interpretability.")

prob = model.predict_proba(X_user)[0, 1]
pred = int(prob >= 0.5)

c1, c2 = st.columns([1, 2])
with c1:
    st.subheader("Prediction")
    st.write(f"**Predicted risk:** {prob*100:.1f}%")
    st.write(f"**Class:** {'Disease' if pred == 1 else 'No Disease'}")
    st.progress(min(max(prob, 0), 1))

with c2:
    st.subheader("Your Input")
    st.dataframe(X_user.T.rename(columns={0: "Value"}))

st.divider()

# -----------------------------
# SHAP helpers (safe across versions)
# -----------------------------
def get_shap_matrix(explainer, X):
    try:
        return explainer(X)  # New API returns shap.Explanation
    except Exception:
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            return sv[1] if len(sv) > 1 else sv[0]
        if isinstance(sv, np.ndarray):
            if sv.ndim == 2:
                return sv
            if sv.ndim == 3:
                return sv[1] if sv.shape[0] > 1 else sv[0]
        raise RuntimeError(f"Unexpected SHAP output shape: {getattr(sv, 'shape', None)}")




st.subheader("Model Interpretability (SHAP Summary)")
st.markdown("""
**Figure Explanation:**  
- **Red dots** = higher values (older age, male sex)  
- **Blue dots** = lower values (younger age, female sex)  
- Horizontal position = how much the age–sex interaction changes the prediction.  
- Male + older → higher predicted risk  
- Female + younger → lower predicted risk  
""")


rf = model.named_steps["rf"]
scaler = model.named_steps["scale"]

# sample & scale
X_shap_raw = X_sample.sample(n=min(200, len(X_sample)), random_state=42)
X_scaled = scaler.transform(X_shap_raw)
X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURES, index=X_shap_raw.index)

explainer = shap.TreeExplainer(rf)

def _render_current_figure(save_path):
    fig = plt.gcf()                # <-- get the figure SHAP actually drew on
    st.pyplot(fig, clear_figure=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)                 # <-- prevent blank next time

# NEW API
try:
    exp = explainer(X_scaled_df)
    shap.plots.beeswarm(exp, max_display=12, show=False)
    _render_current_figure("shap_beeswarm.png")

# LEGACY API
except Exception:
    sv = explainer.shap_values(X_scaled_df)
    if isinstance(sv, list) and len(sv) > 1:
        shap.summary_plot(sv[1], X_scaled_df, feature_names=FEATURES, show=False)
    else:
        shap.summary_plot(sv if not isinstance(sv, list) else sv[0],
                          X_scaled_df, feature_names=FEATURES, show=False)
    _render_current_figure("shap_beeswarm.png")

# download button
with open("shap_beeswarm.png", "rb") as f:
    st.download_button("Download SHAP Beeswarm (PNG)", f, file_name="shap_beeswarm.png")

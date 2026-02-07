"""Multi-Cloud Cost Prediction — attractive, modern UI."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import requests
import streamlit as st

st.set_page_config(
    page_title="Cloud Cost Predictor",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ——— Attractive modern styling ———
st.markdown("""
<style>
    /* Import a clean font */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    .stApp, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
        background: #0f0f23 !important;
    }
    
    /* Hide default padding and make full bleed */
    .block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; max-width: 1100px !important; }
    
    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.25);
        text-align: center;
    }
    .hero h1 { color: white !important; font-size: 2rem !important; font-weight: 700 !important; margin: 0 !important; letter-spacing: -0.02em; }
    .hero p { color: rgba(255,255,255,0.9) !important; margin: 0.5rem 0 0 0 !important; font-size: 1rem !important; }
    
    /* Input card */
    .input-card {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 1.75rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    .input-card h3 { color: #e2e8f0 !important; font-size: 1.1rem !important; margin-bottom: 1rem !important; font-weight: 600 !important; }
    
    /* Result highlight — eye-catching */
    .result-box {
        background: linear-gradient(135deg, #059669 0%, #10b981 50%, #34d399 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 20px 40px rgba(5, 150, 105, 0.3);
        border: 1px solid rgba(255,255,255,0.15);
    }
    .result-box .value { font-size: 2.75rem !important; font-weight: 800 !important; letter-spacing: -0.03em !important; margin: 0 !important; }
    .result-box .label { font-size: 0.95rem !important; opacity: 0.95 !important; margin-top: 0.5rem !important; font-weight: 500 !important; }
    
    /* Evaluation cards — colorful */
    .metric-card {
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        text-align: center;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.08);
    }
    .metric-card.mae { background: linear-gradient(145deg, #1e3a5f 0%, #0f172a 100%); color: #93c5fd; }
    .metric-card.rmse { background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%); color: #86efac; }
    .metric-card.r2 { background: linear-gradient(145deg, #312e81 0%, #1e1b4b 100%); color: #c4b5fd; }
    .metric-card .metric-value { font-size: 1.75rem; font-weight: 700; margin: 0.25rem 0; }
    .metric-card .metric-label { font-size: 0.8rem; opacity: 0.9; }
    
    /* Section titles */
    h2, h3 { font-family: 'Plus Jakarta Sans', sans-serif !important; }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0f23 100%) !important;
        border-right: 1px solid rgba(99, 102, 241, 0.15) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4) !important;
    }
    .stButton > button:hover {
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
    }
    
    /* Number inputs / select — dark theme friendly */
    [data-testid="stNumberInput"] input, [data-testid="stSelectbox"] div {
        background: #1e1e3f !important;
        border-color: rgba(99, 102, 241, 0.3) !important;
        color: #e2e8f0 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader { background: #1a1a2e !important; color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ——— Session state ———
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = None


def get_base_url() -> str:
    return os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


def fetch_prediction(base: str, payload: dict) -> float:
    resp = requests.post(f"{base}/predict", json=payload, timeout=10)
    resp.raise_for_status()
    return float(resp.json()["predicted_monthly_cloud_cost"])


def _load_evaluation_from_artifacts() -> dict | None:
    """Fallback: load metrics from artifacts/training_meta.pkl when API /evaluation returns 404."""
    try:
        # __file__ = .../frontend/app.py → parents[1] = project root
        project_root = Path(__file__).resolve().parents[1]
        meta_path = project_root / "artifacts" / "training_meta.pkl"
        if not meta_path.exists():
            return None
        meta = joblib.load(meta_path)
        return meta.get("metrics")
    except Exception:
        return None


def fetch_evaluation(base: str) -> tuple[dict | None, str | None]:
    """Fetch evaluation metrics. Returns (data, None) on success, (None, error_message) on failure.
    If API returns 404, tries to load from artifacts/training_meta.pkl as fallback."""
    try:
        url = f"{base.rstrip('/')}/evaluation"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return resp.json(), None
        if resp.status_code == 404:
            # Backend may be old (no /evaluation route). Use local artifacts if available.
            local = _load_evaluation_from_artifacts()
            if local:
                return local, None
        return None, f"API returned {resp.status_code}: {resp.text[:200] if resp.text else 'No details'}"
    except requests.exceptions.ConnectionError:
        local = _load_evaluation_from_artifacts()
        if local:
            return local, None
        return None, "Cannot reach API. Is the backend running on the API URL?"
    except requests.exceptions.Timeout:
        local = _load_evaluation_from_artifacts()
        if local:
            return local, None
        return None, "Request timed out."
    except Exception as e:
        local = _load_evaluation_from_artifacts()
        if local:
            return local, None
        return None, str(e)


# ——— Sidebar ———
with st.sidebar:
    st.markdown("**Connection**")
    base_url = st.text_input("API URL", value=get_base_url(), label_visibility="collapsed").rstrip("/") or get_base_url()

# ——— Hero ———
st.markdown("""
<div class="hero">
    <h1>☁️ Multi-Cloud Cost Prediction</h1>
    <p>Predict monthly cloud cost from resource usage and provider — AWS, Azure, GCP</p>
</div>
""", unsafe_allow_html=True)

# ——— Inputs ———
st.markdown("### 📊 Resource usage")
c1, c2, c3 = st.columns(3)

with c1:
    cloud_provider = st.selectbox("Cloud provider", ["AWS", "Azure", "GCP"], key="provider")
    cpu_hours = st.number_input("CPU hours (monthly)", min_value=0.0, value=200.0, step=10.0, key="cpu")
    storage_gb = st.number_input("Storage (GB)", min_value=0.0, value=1000.0, step=50.0, key="storage")

with c2:
    bandwidth_gb = st.number_input("Bandwidth (GB)", min_value=0.0, value=800.0, step=50.0, key="bw")
    active_users = st.number_input("Active users", min_value=0.0, value=150.0, step=10.0, key="users")
    uptime_hours = st.number_input("Uptime hours (monthly)", min_value=0.0, value=720.0, step=24.0, key="uptime")

with c3:
    st.markdown("<br>", unsafe_allow_html=True)  # spacer
    predict_clicked = st.button("🚀 Predict cost", type="primary", use_container_width=True)

payload = {
    "cloud_provider": cloud_provider,
    "cpu_hours": cpu_hours,
    "storage_gb": storage_gb,
    "bandwidth_gb": bandwidth_gb,
    "active_users": active_users,
    "uptime_hours": uptime_hours,
}

# ——— Prediction ———
if predict_clicked:
    with st.spinner("Predicting…"):
        try:
            pred = fetch_prediction(base_url, payload)
            st.session_state.last_prediction = pred
            st.session_state.last_inputs = payload
        except requests.RequestException as e:
            st.error(f"API error: {e}")
            st.session_state.last_prediction = None

if st.session_state.last_prediction is not None:
    st.markdown(
        f'<div class="result-box">'
        f'<div class="value">${st.session_state.last_prediction:,.0f}</div>'
        f'<div class="label">Predicted monthly cloud cost</div></div>',
        unsafe_allow_html=True,
    )
    with st.expander("How is this computed?"):
        st.markdown(
            "Inputs are sent to the API. The backend **one-hot encodes** the provider (AWS = baseline), "
            "**standardizes** numeric usage (z-score), then applies **Linear Regression** to estimate cost."
        )

# ——— Evaluation ———
st.markdown("---")
st.subheader("📈 Model evaluation")

eval_data, eval_error = fetch_evaluation(base_url)
if eval_error:
    st.warning(f"Evaluation: {eval_error}")
elif eval_data:
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f'<div class="metric-card mae">'
            f'<div class="metric-label">MAE</div>'
            f'<div class="metric-value">${eval_data.get("mae", 0):,.2f}</div>'
            f'<div class="metric-label">Mean Absolute Error</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card rmse">'
            f'<div class="metric-label">RMSE</div>'
            f'<div class="metric-value">${eval_data.get("rmse", 0):,.2f}</div>'
            f'<div class="metric-label">Root Mean Squared Error</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        r2 = eval_data.get("r2", 0)
        st.markdown(
            f'<div class="metric-card r2">'
            f'<div class="metric-label">R²</div>'
            f'<div class="metric-value">{r2:.1%}</div>'
            f'<div class="metric-label">Variance explained</div></div>',
            unsafe_allow_html=True,
        )
else:
    st.info("No evaluation data. Train the model (run_train.ps1) and ensure the backend is running.")

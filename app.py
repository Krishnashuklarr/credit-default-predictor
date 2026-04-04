"""
app.py - Credit Default Predictor (Modern Dark UI)
Run: streamlit run app.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import joblib
import pandas as pd
import streamlit as st

from src.config import MODEL_PATH, PREPROCESSOR_PATH, THRESHOLD_PATH
from src.feature_engineering import add_features

st.set_page_config(
    page_title="Credit Default Predictor",
    page_icon="💳",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    background-color: #060612 !important;
    color: #c8d8f0 !important;
    font-family: 'Rajdhani', sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem !important; max-width: 1200px !important; }

.hero { text-align: center; padding: 2.5rem 0 1.5rem 0; }
.hero h1 {
    font-family: 'Orbitron', monospace;
    font-size: 2.6rem;
    font-weight: 900;
    background: linear-gradient(135deg, #00f5ff 0%, #7c3aed 50%, #00f5ff 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 3s linear infinite;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}
.hero p { color: #6b7fa8; font-size: 1.05rem; letter-spacing: 0.05em; }
@keyframes shimmer {
    0%   { background-position: 0% center; }
    100% { background-position: 200% center; }
}

.glow-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #00f5ff, #7c3aed, #00f5ff, transparent);
    margin: 1.5rem 0;
    box-shadow: 0 0 10px #00f5ff66;
}

.section-card {
    background: linear-gradient(135deg, #0d1528 0%, #0a0f1e 100%);
    border: 1px solid #1a2a4a;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 0 30px #00f5ff0d, inset 0 0 30px #00000033;
    position: relative;
    overflow: hidden;
}
.section-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00f5ff, transparent);
}
.section-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.2em;
    color: #00f5ff;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}

.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background: #0a1628 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    color: #c8d8f0 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1rem !important;
}
label {
    color: #7a9cc4 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    font-family: 'Rajdhani', sans-serif !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: #0a0f1e !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid #1a2a4a !important;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #4a6080 !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.2rem !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #00f5ff22, #7c3aed22) !important;
    color: #00f5ff !important;
    border: 1px solid #00f5ff44 !important;
    box-shadow: 0 0 15px #00f5ff22 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #00b4d8, #7c3aed) !important;
    color: white !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.15em !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.8rem 3rem !important;
    width: 100% !important;
    box-shadow: 0 0 20px #00b4d855 !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 0 40px #00f5ff88 !important;
    transform: translateY(-2px) !important;
}

.result-safe {
    background: linear-gradient(135deg, #002a1a, #001a10);
    border: 1px solid #00ff8844;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 40px #00ff4422;
    animation: fadeIn 0.6s ease;
}
.result-danger {
    background: linear-gradient(135deg, #2a0010, #1a000a);
    border: 1px solid #ff004444;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 40px #ff002222;
    animation: fadeIn 0.6s ease;
}
.result-label-safe  { font-family:'Orbitron',monospace; font-size:1.8rem; font-weight:900; color:#00ff88; text-shadow:0 0 20px #00ff8888; letter-spacing:0.08em; }
.result-label-danger{ font-family:'Orbitron',monospace; font-size:1.8rem; font-weight:900; color:#ff4466; text-shadow:0 0 20px #ff446688; letter-spacing:0.08em; }
.prob-value { font-family:'Orbitron',monospace; font-size:3.5rem; font-weight:900; letter-spacing:0.05em; margin:0.5rem 0; }

.gauge-container {
    background: #0a1628;
    border-radius: 50px;
    height: 18px;
    width: 100%;
    border: 1px solid #1a2a4a;
    overflow: hidden;
    margin: 1rem 0;
}
.gauge-fill-safe   { height:100%; border-radius:50px; background:linear-gradient(90deg,#00ff88,#00b4d8); box-shadow:0 0 15px #00ff8888; }
.gauge-fill-danger { height:100%; border-radius:50px; background:linear-gradient(90deg,#ffaa00,#ff4466); box-shadow:0 0 15px #ff446888; }
.gauge-labels { display:flex; justify-content:space-between; font-size:0.72rem; color:#4a6080; letter-spacing:0.05em; margin-top:0.3rem; }

.metric-card { background:#0a1628; border:1px solid #1a2a4a; border-radius:12px; padding:1rem 1.5rem; text-align:center; }
.metric-label { font-size:0.68rem; color:#4a6080; letter-spacing:0.12em; text-transform:uppercase; font-family:'Orbitron',monospace; }
.metric-value { font-family:'Orbitron',monospace; font-size:1.4rem; font-weight:700; color:#00f5ff; margin-top:0.3rem; }

@keyframes fadeIn {
    from { opacity:0; transform:translateY(16px); }
    to   { opacity:1; transform:translateY(0); }
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model        = joblib.load(MODEL_PATH)
    threshold    = joblib.load(THRESHOLD_PATH)
    return preprocessor, model, threshold

try:
    preprocessor, model, threshold = load_artifacts()
except FileNotFoundError:
    st.error("❌ Model not found. Run `python scripts/train.py` first.")
    st.stop()

st.markdown("""
<div class="hero">
    <h1>💳 CREDIT DEFAULT PREDICTOR</h1>
    <p>AI-powered risk assessment &nbsp;·&nbsp; LightGBM &nbsp;·&nbsp; Real-time prediction</p>
</div>
<div class="glow-divider"></div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["👤  PERSONAL", "📅  PAYMENT HISTORY", "💰  FINANCIALS"])

month_labels = ["Sep", "Aug", "Jul", "Jun", "May", "Apr"]

with tab1:
    st.markdown('<div class="section-card"><div class="section-title">⬡ Customer Profile</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        limit_bal = st.number_input("Credit Limit (NT$)", 0, 1_000_000, 50_000, 5000)
        age       = st.number_input("Age", 18, 100, 30)
    with c2:
        sex      = st.selectbox("Sex", [1, 2], format_func=lambda x: "Male" if x==1 else "Female")
        marriage = st.selectbox("Marital Status", [1,2,3], format_func=lambda x: {1:"Married",2:"Single",3:"Other"}[x])
    with c3:
        education = st.selectbox("Education", [1,2,3,4], format_func=lambda x: {1:"Graduate School",2:"University",3:"High School",4:"Other"}[x])
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-card"><div class="section-title">⬡ Repayment Status · Last 6 Months</div>', unsafe_allow_html=True)
    st.caption("**-1** = Paid in full &nbsp;|&nbsp; **0** = Minimum paid &nbsp;|&nbsp; **1–9** = Months delayed")
    pay_keys = ["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
    pay_vals = {}
    for col, label, key in zip(st.columns(6), month_labels, pay_keys):
        with col:
            pay_vals[key] = st.number_input(label, -2, 9, 0, key=f"pay_{key}")
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section-card"><div class="section-title">⬡ Bill Amounts · Last 6 Months (NT$)</div>', unsafe_allow_html=True)
    bill_keys = [f"BILL_AMT{i}" for i in range(1,7)]
    bill_vals = {}
    for col, label, key in zip(st.columns(6), month_labels, bill_keys):
        with col:
            bill_vals[key] = st.number_input(label, 0, 1_000_000, 10_000, 500, key=f"b_{key}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">⬡ Payment Amounts · Last 6 Months (NT$)</div>', unsafe_allow_html=True)
    pamt_keys = [f"PAY_AMT{i}" for i in range(1,7)]
    pamt_vals = {}
    for col, label, key in zip(st.columns(6), month_labels, pamt_keys):
        with col:
            pamt_vals[key] = st.number_input(label, 0, 500_000, 1_000, 500, key=f"p_{key}")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div class='glow-divider'></div>", unsafe_allow_html=True)
predict_clicked = st.button("⚡  RUN PREDICTION")

if predict_clicked:
    row = {"ID":0,"LIMIT_BAL":limit_bal,"SEX":sex,"EDUCATION":education,
           "MARRIAGE":marriage,"AGE":age,**pay_vals,**bill_vals,**pamt_vals}
    df     = pd.DataFrame([row])
    df_eng = add_features(df)
    X_proc = preprocessor.transform(df_eng)
    proba  = float(model.predict_proba(X_proc)[:,1][0])
    is_default = proba >= threshold
    pct    = round(proba * 100, 1)
    width  = round(proba * 100, 1)

    st.markdown("<div class='glow-divider'></div>", unsafe_allow_html=True)

    if is_default:
        st.markdown(f"""
        <div class="result-danger">
            <div class="result-label-danger">⚠ HIGH DEFAULT RISK</div>
            <div class="prob-value" style="color:#ff4466;text-shadow:0 0 30px #ff446899;">{pct}%</div>
            <div style="color:#ff6688;font-size:0.9rem;letter-spacing:0.08em;">PROBABILITY OF DEFAULT</div>
            <div class="gauge-container" style="margin-top:1.5rem;">
                <div class="gauge-fill-danger" style="width:{width}%;"></div>
            </div>
            <div class="gauge-labels"><span>LOW RISK</span><span>THRESHOLD {round(threshold*100,1)}%</span><span>HIGH RISK</span></div>
            <p style="color:#ff4466aa;margin-top:1rem;font-size:0.9rem;">
                🔴 This customer is likely to default next month. Immediate review recommended.
            </p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-safe">
            <div class="result-label-safe">✓ LOW DEFAULT RISK</div>
            <div class="prob-value" style="color:#00ff88;text-shadow:0 0 30px #00ff8899;">{pct}%</div>
            <div style="color:#00cc66;font-size:0.9rem;letter-spacing:0.08em;">PROBABILITY OF DEFAULT</div>
            <div class="gauge-container" style="margin-top:1.5rem;">
                <div class="gauge-fill-safe" style="width:{width}%;"></div>
            </div>
            <div class="gauge-labels"><span>LOW RISK</span><span>THRESHOLD {round(threshold*100,1)}%</span><span>HIGH RISK</span></div>
            <p style="color:#00ff8877;margin-top:1rem;font-size:0.9rem;">
                🟢 This customer is unlikely to default next month. Profile looks stable.
            </p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    risk  = "LOW" if pct < 30 else ("MEDIUM" if not is_default else "HIGH")
    color = "#00ff88" if risk=="LOW" else ("#ffaa00" if risk=="MEDIUM" else "#ff4466")
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Default Probability</div><div class="metric-value">{pct}%</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Decision Threshold</div><div class="metric-value">{round(threshold*100,1)}%</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Risk Level</div><div class="metric-value" style="color:{color};">{risk}</div></div>', unsafe_allow_html=True)
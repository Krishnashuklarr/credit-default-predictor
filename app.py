"""
app.py - Credit Risk Analyzer v3
Run: streamlit run app.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import MODEL_PATH, PREPROCESSOR_PATH, THRESHOLD_PATH
from src.feature_engineering import add_features
from src.logger import log_prediction, load_log
from src.styles import DARK_CSS

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Credit Risk Analyzer", page_icon="💳", layout="wide")
st.markdown(DARK_CSS, unsafe_allow_html=True)

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0f1e",
    font=dict(family="Rajdhani", color="#c8d8f0", size=13),
    xaxis=dict(gridcolor="#1a2a4a", zerolinecolor="#1a2a4a", showgrid=True),
    yaxis=dict(gridcolor="#1a2a4a", zerolinecolor="#1a2a4a", showgrid=True),
    margin=dict(t=20, b=30, l=10, r=10),
)

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    return (
        joblib.load(PREPROCESSOR_PATH),
        joblib.load(MODEL_PATH),
        joblib.load(THRESHOLD_PATH),
    )

try:
    preprocessor, model, threshold = load_artifacts()
except FileNotFoundError:
    st.error("❌ Model not found. Run `python scripts/train.py` first.")
    st.stop()

# ── Helpers ───────────────────────────────────────────────────────────────────
def risk_level(proba):
    if proba < 0.3:         return "LOW",    "#00ff88", "badge-low"
    elif proba < threshold:  return "MEDIUM", "#ffaa00", "badge-medium"
    else:                    return "HIGH",   "#ff4466", "badge-high"

def gauge_class(proba):
    if proba < 0.3:        return "gauge-fill-safe"
    elif proba < threshold: return "gauge-fill-medium"
    else:                  return "gauge-fill-danger"

def result_class(proba):
    if proba < 0.3:        return "result-safe",   "result-label-safe",   "✓ LOW RISK"
    elif proba < threshold: return "result-medium", "result-label-medium", "⚠ MODERATE RISK"
    else:                  return "result-danger",  "result-label-danger", "⚠ HIGH RISK"

def predict_single(row_dict):
    df = add_features(pd.DataFrame([row_dict]))
    return float(model.predict_proba(preprocessor.transform(df))[:, 1][0])

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1.8rem 0 1.2rem 0;'>
        <div style='font-size:2rem;'>💳</div>
        <div style='font-family:Orbitron,monospace;font-size:1rem;font-weight:900;
            background:linear-gradient(135deg,#00f5ff,#7c3aed);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            background-clip:text;letter-spacing:0.1em;margin-top:0.4rem;'>
            CREDIT RISK
        </div>
        <div style='color:#4a6080;font-size:0.7rem;letter-spacing:0.12em;margin-top:0.2rem;'>
            ANALYZER · v3.0
        </div>
    </div>
    <div class='glow-divider'></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-nav-title">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("nav", [
        "⚡  Predict",
        "📊  Dashboard",
        "📂  Batch Review",
        "ℹ️  Model Info",
    ], label_visibility="collapsed")

    st.markdown("""
    <div style='height:1px;background:linear-gradient(90deg,transparent,#1a2a4a,transparent);margin:1.5rem 0;'></div>
    <div style='font-family:Orbitron,monospace;font-size:0.6rem;color:#4a6080;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.6rem;'>
        Model Stats
    </div>
    """, unsafe_allow_html=True)

    log_df = load_log()
    total_preds = len(log_df)
    high_count  = int((log_df["risk_level"] == "HIGH").sum()) if not log_df.empty else 0

    st.markdown(f"""
    <div style='font-size:0.82rem;color:#6b7fa8;line-height:2;'>
        Algorithm &nbsp;<span style='color:#00f5ff;float:right;'>LightGBM</span><br>
        ROC-AUC &nbsp;<span style='color:#00f5ff;float:right;'>0.7744</span><br>
        Accuracy &nbsp;<span style='color:#00f5ff;float:right;'>~79%</span><br>
        Threshold &nbsp;<span style='color:#00f5ff;float:right;'>{round(threshold*100,1)}%</span><br>
        Total Reviewed &nbsp;<span style='color:#00f5ff;float:right;'>{total_preds}</span><br>
        High Risk Found &nbsp;<span style='color:#ff4466;float:right;'>{high_count}</span>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if page == "⚡  Predict":

    st.markdown("""
    <div class="hero">
        <h1>💳 CREDIT RISK PREDICTOR</h1>
        <p>Assess individual customer default probability in real time</p>
    </div>
    <div class="glow-divider"></div>
    """, unsafe_allow_html=True)

    month_labels = ["Sep","Aug","Jul","Jun","May","Apr"]

    # ── Input form ─────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["👤  PERSONAL","📅  PAYMENT HISTORY","💰  FINANCIALS"])

    with tab1:
        st.markdown('<div class="section-card"><div class="section-title">⬡ Customer Profile</div>', unsafe_allow_html=True)
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        with r1c1: customer_id = st.text_input("Customer ID", placeholder="CUST-0001")
        with r1c2: limit_bal   = st.number_input("Credit Limit (NT$)", 0, 1_000_000, 50_000, 5000)
        with r1c3: age         = st.number_input("Age", 18, 100, 30)
        with r1c4: sex         = st.selectbox("Sex", [1,2], format_func=lambda x: "Male" if x==1 else "Female")

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1: education = st.selectbox("Education", [1,2,3,4], format_func=lambda x:{1:"Graduate School",2:"University",3:"High School",4:"Other"}[x])
        with r2c2: marriage  = st.selectbox("Marital Status", [1,2,3], format_func=lambda x:{1:"Married",2:"Single",3:"Other"}[x])
        with r2c3: st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-card"><div class="section-title">⬡ Repayment Status · Last 6 Months</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">💡 <b>-2</b> No consumption &nbsp;·&nbsp; <b>-1</b> Paid in full &nbsp;·&nbsp; <b>0</b> Minimum paid &nbsp;·&nbsp; <b>1–9</b> Months delayed</div>', unsafe_allow_html=True)
        pay_keys = ["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
        pay_vals = {}
        for col, label, key in zip(st.columns(6), month_labels, pay_keys):
            with col: pay_vals[key] = st.number_input(label, -2, 9, 0, key=f"pay_{key}")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-card"><div class="section-title">⬡ Bill Amounts · Last 6 Months (NT$)</div>', unsafe_allow_html=True)
        bill_keys = [f"BILL_AMT{i}" for i in range(1,7)]
        bill_vals = {}
        for col, label, key in zip(st.columns(6), month_labels, bill_keys):
            with col: bill_vals[key] = st.number_input(label, 0, 1_000_000, 10_000, 500, key=f"b_{key}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card"><div class="section-title">⬡ Payment Amounts · Last 6 Months (NT$)</div>', unsafe_allow_html=True)
        pamt_keys = [f"PAY_AMT{i}" for i in range(1,7)]
        pamt_vals = {}
        for col, label, key in zip(st.columns(6), month_labels, pamt_keys):
            with col: pamt_vals[key] = st.number_input(label, 0, 500_000, 1_000, 500, key=f"p_{key}")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Live customer summary card ─────────────────────────────────────────
    st.markdown("<div class='glow-divider'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-card"><div class="section-title">⬡ Customer Summary</div>', unsafe_allow_html=True)
    s1,s2,s3,s4,s5,s6 = st.columns(6)
    avg_bill_preview = sum(bill_vals.values()) / 6 if bill_vals else 0
    avg_pay_preview  = sum(pamt_vals.values()) / 6 if pamt_vals else 0
    util_preview     = bill_vals.get("BILL_AMT1",0) / (limit_bal+1) * 100

    for col, label, val in zip([s1,s2,s3,s4,s5,s6], [
        "Customer ID","Credit Limit","Age","Avg Bill","Avg Payment","Utilization"
    ],[
        customer_id or "—",
        f"NT$ {limit_bal:,}",
        f"{age} yrs",
        f"NT$ {avg_bill_preview:,.0f}",
        f"NT$ {avg_pay_preview:,.0f}",
        f"{min(util_preview,100):.1f}%",
    ]):
        with col:
            st.markdown(f"""
            <div style='text-align:center;'>
                <div style='font-size:0.62rem;color:#4a6080;letter-spacing:0.1em;text-transform:uppercase;font-family:Orbitron,monospace;'>{label}</div>
                <div style='font-size:1rem;font-weight:600;color:#c8d8f0;margin-top:0.3rem;'>{val}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Predict button ─────────────────────────────────────────────────────
    if st.button("⚡  RUN RISK ASSESSMENT"):
        row = {"ID":0,"LIMIT_BAL":limit_bal,"SEX":sex,"EDUCATION":education,
               "MARRIAGE":marriage,"AGE":age,**pay_vals,**bill_vals,**pamt_vals}

        with st.spinner("Analyzing customer profile…"):
            proba = predict_single(row)

        pct = round(proba * 100, 1)
        rlevel, rcolor, rbadge = risk_level(proba)
        rcard, rlabel, rtitle  = result_class(proba)
        gclass = gauge_class(proba)

        log_prediction(row, proba, threshold, customer_id or "—")

        st.markdown("<div class='glow-divider'></div>", unsafe_allow_html=True)

        # Result card
        st.markdown(f"""
        <div class="{rcard}">
            <div class="{rlabel}">{rtitle}</div>
            <div class="prob-value" style="color:{rcolor};text-shadow:0 0 30px {rcolor}77;">{pct}%</div>
            <div style="color:{rcolor}99;font-size:0.85rem;letter-spacing:0.1em;margin-bottom:1rem;">
                PROBABILITY OF DEFAULT NEXT MONTH
            </div>
            <div class="gauge-wrap">
                <div class="gauge-bar"><div class="{gclass}" style="width:{pct}%;"></div></div>
                <div class="gauge-labels">
                    <span>0%</span>
                    <span style="color:#00f5ff;">▲ Threshold {round(threshold*100,1)}%</span>
                    <span>100%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Metric row
        m1,m2,m3,m4,m5 = st.columns(5)
        pay_avg = sum(pay_vals.values())/6
        with m1: st.markdown(f'<div class="metric-card"><div class="metric-label">Default Probability</div><div class="metric-value">{pct}%</div></div>', unsafe_allow_html=True)
        with m2: st.markdown(f'<div class="metric-card"><div class="metric-label">Risk Level</div><div class="metric-value" style="color:{rcolor};">{rlevel}</div></div>', unsafe_allow_html=True)
        with m3: st.markdown(f'<div class="metric-card"><div class="metric-label">Credit Utilization</div><div class="metric-value">{min(util_preview,100):.1f}%</div></div>', unsafe_allow_html=True)
        with m4: st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Pay Status</div><div class="metric-value">{pay_avg:.1f}</div></div>', unsafe_allow_html=True)
        with m5: st.markdown(f'<div class="metric-card"><div class="metric-label">Customer ID</div><div class="metric-value" style="font-size:0.95rem;">{customer_id or "—"}</div></div>', unsafe_allow_html=True)

        # Recommendation box
        st.markdown("<br>", unsafe_allow_html=True)
        if rlevel == "LOW":
            st.markdown('<div class="info-box" style="border-left-color:#00ff88;">✅ <b>Recommendation:</b> Customer profile is stable. Standard approval process applies.</div>', unsafe_allow_html=True)
        elif rlevel == "MEDIUM":
            st.markdown('<div class="info-box" style="border-left-color:#ffaa00;">⚠️ <b>Recommendation:</b> Moderate risk detected. Consider requesting additional documentation or reducing credit limit.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box" style="border-left-color:#ff4466;">🔴 <b>Recommendation:</b> High default risk. Recommend rejection or escalation to senior credit officer for manual review.</div>', unsafe_allow_html=True)

        st.caption("✅ Assessment saved to dashboard log.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Dashboard":

    st.markdown("""
    <div class="hero"><h1>📊 RISK DASHBOARD</h1>
    <p>Live overview of all customer assessments · Updates after every prediction</p></div>
    <div class="glow-divider"></div>""", unsafe_allow_html=True)

    df = load_log()

    if df.empty:
        st.markdown("""
        <div class="section-card" style="text-align:center;padding:4rem;">
            <div style="font-family:'Orbitron',monospace;color:#4a6080;font-size:1rem;letter-spacing:0.1em;">NO DATA YET</div>
            <div style="color:#4a6080;margin-top:1rem;font-size:0.9rem;">
                Go to <b style="color:#00f5ff;">⚡ Predict</b> and assess some customers first.
            </div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    total     = len(df)
    n_default = int(df["prediction"].sum())
    n_safe    = total - n_default
    avg_prob  = df["default_probability"].mean()
    high_risk = int((df["risk_level"] == "HIGH").sum())
    medium    = int((df["risk_level"] == "MEDIUM").sum())

    # ── KPI row ──────────────────────────────────────────────────────────
    k1,k2,k3,k4,k5 = st.columns(5)
    for col,(label,val,color,sub) in zip([k1,k2,k3,k4,k5],[
        ("TOTAL ASSESSED",   total,                     "#00f5ff", "all time"),
        ("HIGH RISK",        high_risk,                 "#ff4466", f"{round(high_risk/total*100,1)}% of total"),
        ("MEDIUM RISK",      medium,                    "#ffaa00", f"{round(medium/total*100,1)}% of total"),
        ("FLAGGED DEFAULT",  n_default,                 "#ff6644", f"{round(n_default/total*100,1)}% rate"),
        ("AVG PROBABILITY",  f"{round(avg_prob*100,1)}%","#7c3aed","mean score"),
    ]):
        with col:
            st.markdown(f"""
            <div class="stat-card" style="--accent:{color};">
                <div class="stat-label">{label}</div>
                <div class="stat-value" style="color:{color};">{val}</div>
                <div class="stat-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row 1 ──────────────────────────────────────────────────────
    ca, cb = st.columns(2)

    with ca:
        st.markdown('<div class="section-card"><div class="section-title">⬡ Risk Distribution</div>', unsafe_allow_html=True)
        rc = df["risk_level"].value_counts().reindex(["LOW","MEDIUM","HIGH"], fill_value=0)
        fig = go.Figure(go.Pie(
            labels=rc.index, values=rc.values,
            hole=0.6,
            marker=dict(colors=["#00ff88","#ffaa00","#ff4466"],
                        line=dict(color="#060612", width=3)),
            textinfo="label+percent",
            textfont=dict(family="Orbitron", size=11),
        ))
        fig.add_annotation(text=f"<b>{total}</b><br><span style='font-size:10px'>Total</span>",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=18, color="#c8d8f0", family="Orbitron"))
        fig.update_layout(**PLOTLY_BASE, height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with cb:
        st.markdown('<div class="section-card"><div class="section-title">⬡ Default Probability Distribution</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=df["default_probability"], nbinsx=25,
            marker=dict(color="#7c3aed", line=dict(color="#00f5ff22", width=1)),
            opacity=0.9, name="Customers"))
        fig2.add_vline(x=threshold, line_dash="dash", line_color="#00f5ff", line_width=2,
            annotation_text=f"Threshold {round(threshold*100,1)}%",
            annotation_font=dict(color="#00f5ff", family="Orbitron", size=11))
        fig2.update_layout(**PLOTLY_BASE, height=300, showlegend=False,
            xaxis_title="Default Probability", yaxis_title="No. of Customers")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Charts row 2 ──────────────────────────────────────────────────────
    cc, cd = st.columns(2)

    with cc:
        st.markdown('<div class="section-card"><div class="section-title">⬡ Assessments Over Time</div>', unsafe_allow_html=True)
        if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["date"] = df["timestamp"].dt.date
            daily = df.groupby(["date","risk_level"]).size().reset_index(name="count")
            fig3 = px.bar(daily, x="date", y="count", color="risk_level", barmode="stack",
                color_discrete_map={"LOW":"#00ff88","MEDIUM":"#ffaa00","HIGH":"#ff4466"})
            fig3.update_layout(**PLOTLY_BASE, height=280,
                legend=dict(orientation="h", y=1.1, font=dict(size=11, family="Orbitron")))
            st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with cd:
        st.markdown('<div class="section-card"><div class="section-title">⬡ Default Rate by Education</div>', unsafe_allow_html=True)
        edu = df.groupby("education")["prediction"].agg(["sum","count"]).reset_index()
        edu["rate"] = (edu["sum"] / edu["count"] * 100).round(1)
        fig4 = go.Figure(go.Bar(
            x=edu["education"], y=edu["rate"],
            marker=dict(color=edu["rate"],
                        colorscale=[[0,"#00ff88"],[0.5,"#ffaa00"],[1,"#ff4466"]],
                        showscale=False),
            text=[f"{v}%" for v in edu["rate"]],
            textposition="outside",
            textfont=dict(family="Orbitron", size=11, color="#c8d8f0"),
        ))
        fig4.update_layout(**PLOTLY_BASE, height=280,
            showlegend=False, bargap=0.4, yaxis_title="Default Rate %")
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── High risk table ──────────────────────────────────────────────────
    st.markdown('<div class="section-card"><div class="section-title">⬡ 🔴 High Risk Customers — Action Required</div>', unsafe_allow_html=True)
    high_df = df[df["risk_level"]=="HIGH"].sort_values("default_probability", ascending=False)
    if high_df.empty:
        st.markdown("<div style='color:#4a6080;text-align:center;padding:1.5rem;'>No high-risk customers yet.</div>", unsafe_allow_html=True)
    else:
        disp = high_df[["timestamp","customer_id","limit_bal","age","education","avg_pay_status","avg_bill","default_probability"]].copy()
        disp["default_probability"] = (disp["default_probability"]*100).round(1).astype(str) + "%"
        disp["avg_pay_status"] = disp["avg_pay_status"].round(2)
        disp["avg_bill"] = disp["avg_bill"].apply(lambda x: f"NT$ {x:,.0f}")
        disp["limit_bal"] = disp["limit_bal"].apply(lambda x: f"NT$ {x:,.0f}")
        disp.columns = ["Time","Customer ID","Credit Limit","Age","Education","Avg Pay Status","Avg Bill","Default Prob"]
        st.dataframe(disp, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Recent predictions ────────────────────────────────────────────────
    st.markdown('<div class="section-card"><div class="section-title">⬡ Recent Assessments</div>', unsafe_allow_html=True)
    recent = df.sort_values("timestamp", ascending=False).head(10).copy()
    recent["default_probability"] = (recent["default_probability"]*100).round(1).astype(str)+"%"
    recent["limit_bal"] = recent["limit_bal"].apply(lambda x: f"NT$ {x:,.0f}")
    recent = recent[["timestamp","customer_id","limit_bal","age","education","default_probability","risk_level"]]
    recent.columns = ["Time","Customer ID","Credit Limit","Age","Education","Default Prob","Risk Level"]
    st.dataframe(recent, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇  Export Full Assessment Log as CSV",
                       data=csv, file_name="prediction_log.csv", mime="text/csv")
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BATCH REVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📂  Batch Review":

    st.markdown("""
    <div class="hero"><h1>📂 BATCH REVIEW</h1>
    <p>Upload a CSV of customers and score them all at once</p></div>
    <div class="glow-divider"></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">⬡ Upload Customer Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">📋 Upload a CSV with the same columns as <b>UCI_Credit_Card.csv</b> (without the target column). The model will score each row automatically.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop your CSV here", type=["csv"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded:
        try:
            raw = pd.read_csv(uploaded)
            st.markdown(f'<div class="section-card"><div class="section-title">⬡ Preview — {len(raw)} customers loaded</div>', unsafe_allow_html=True)
            st.dataframe(raw.head(5), use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("⚡  SCORE ALL CUSTOMERS"):
                with st.spinner(f"Scoring {len(raw)} customers…"):
                    df_eng = add_features(raw.copy())
                    X_proc = preprocessor.transform(df_eng)
                    probas = model.predict_proba(X_proc)[:, 1]

                results = raw.copy()
                results["Default_Probability"] = (probas * 100).round(1)
                results["Prediction"]          = (probas >= threshold).astype(int)
                results["Risk_Level"]          = pd.cut(
                    probas,
                    bins=[-0.001, 0.3, threshold, 1.001],
                    labels=["LOW","MEDIUM","HIGH"]
                )

                # Summary
                st.markdown("<div class='glow-divider'></div>", unsafe_allow_html=True)
                st.markdown('<div class="section-card"><div class="section-title">⬡ Batch Summary</div>', unsafe_allow_html=True)

                b1,b2,b3,b4 = st.columns(4)
                rc = results["Risk_Level"].value_counts()
                for col,(label,val,color) in zip([b1,b2,b3,b4],[
                    ("Total Scored",   len(results),         "#00f5ff"),
                    ("Low Risk",       rc.get("LOW",0),      "#00ff88"),
                    ("Medium Risk",    rc.get("MEDIUM",0),   "#ffaa00"),
                    ("High Risk",      rc.get("HIGH",0),     "#ff4466"),
                ]):
                    with col:
                        st.markdown(f'<div class="stat-card" style="--accent:{color};"><div class="stat-label">{label}</div><div class="stat-value" style="color:{color};">{val}</div></div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-card"><div class="section-title">⬡ Results</div>', unsafe_allow_html=True)

                # Color-code risk column
                display_cols = ["ID","LIMIT_BAL","AGE","Default_Probability","Prediction","Risk_Level"] if "ID" in results.columns else list(results.columns[-5:])
                st.dataframe(results[display_cols], use_container_width=True, hide_index=True)

                csv_out = results.to_csv(index=False).encode("utf-8")
                st.download_button("⬇  Download Scored Results", data=csv_out,
                                   file_name="batch_results.csv", mime="text/csv")
                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error processing file: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️  Model Info":

    st.markdown("""
    <div class="hero"><h1>ℹ️ MODEL INFORMATION</h1>
    <p>How the model works, what it was trained on, and how to interpret results</p></div>
    <div class="glow-divider"></div>""", unsafe_allow_html=True)

    # ── Model stats ───────────────────────────────────────────────────────
    st.markdown('<div class="section-card"><div class="section-title">⬡ Model Performance</div>', unsafe_allow_html=True)
    p1,p2,p3,p4 = st.columns(4)
    for col,(label,val,color) in zip([p1,p2,p3,p4],[
        ("ROC-AUC",    "0.7744", "#00f5ff"),
        ("Accuracy",   "~79%",   "#00ff88"),
        ("Threshold",  f"{round(threshold*100,1)}%","#7c3aed"),
        ("Train Size", "24,000", "#ffaa00"),
    ]):
        with col:
            st.markdown(f'<div class="stat-card" style="--accent:{color};"><div class="stat-label">{label}</div><div class="stat-value" style="color:{color};">{val}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Feature importance ────────────────────────────────────────────────
    st.markdown('<div class="section-card"><div class="section-title">⬡ Top Feature Importances</div>', unsafe_allow_html=True)
    try:
        feat_names = preprocessor.get_feature_names_out()
        importances = model.feature_importances_
        fi = pd.DataFrame({"Feature": feat_names, "Importance": importances})
        fi = fi.sort_values("Importance", ascending=True).tail(15)
        fi["Feature"] = fi["Feature"].str.replace("num__","").str.replace("cat__","")

        fig_fi = go.Figure(go.Bar(
            x=fi["Importance"], y=fi["Feature"],
            orientation="h",
            marker=dict(
                color=fi["Importance"],
                colorscale=[[0,"#1a2a4a"],[0.5,"#7c3aed"],[1,"#00f5ff"]],
                showscale=False,
            ),
            text=fi["Importance"].round(1),
            textposition="outside",
            textfont=dict(color="#c8d8f0", family="Orbitron", size=10),
        ))
        fig_fi.update_layout(**PLOTLY_BASE, height=420,
            xaxis_title="Importance Score", yaxis_title="",
            yaxis=dict(tickfont=dict(size=11, family="Rajdhani")))
        st.plotly_chart(fig_fi, use_container_width=True)
    except Exception:
        st.info("Feature importance chart unavailable.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── How it works ──────────────────────────────────────────────────────
    st.markdown('<div class="section-card"><div class="section-title">⬡ How The Model Works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <b>Algorithm:</b> LightGBM (Light Gradient Boosting Machine) — an ensemble of decision trees trained to minimize prediction error.
    </div>
    <div class="info-box">
        <b>Training data:</b> 30,000 credit card holders from a Taiwan bank (UCI dataset). 80% used for training, 20% for testing.
    </div>
    <div class="info-box">
        <b>Class imbalance:</b> Only ~22% of customers defaulted, so the model uses <code>class_weight='balanced'</code> to avoid always predicting "no default".
    </div>
    <div class="info-box">
        <b>Threshold:</b> Instead of defaulting to 50%, the model uses an F1-optimal threshold found on the test set. This balances catching defaults without too many false alarms.
    </div>
    <div class="info-box">
        <b>Risk levels:</b> Below 30% = LOW · Between 30% and threshold = MEDIUM · Above threshold = HIGH.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Engineered features ───────────────────────────────────────────────
    st.markdown('<div class="section-card"><div class="section-title">⬡ Engineered Features</div>', unsafe_allow_html=True)
    feat_df = pd.DataFrame({
        "Feature":     ["debt_to_age","debt_to_limit","payment_ratio","avg_bill","pay_ratio","bill_to_limit","pay_ratio_2","avg_pay_ratio","bill_amnt_diff"],
        "Formula":     ["LIMIT_BAL / (AGE+1)","BILL_AMT1 / (LIMIT_BAL+1)","PAY_AMT1 / (BILL_AMT1+1)","Mean of BILL_AMT1–6","Mean of PAY_0,2,3","(BILL_AMT1+2+3) / LIMIT_BAL","PAY_AMT2 / (BILL_AMT2+1)","Mean of pay_ratio & pay_ratio_2","BILL_AMT1 − BILL_AMT2"],
        "What it captures": ["Credit per year of life","How much of the limit is used","Fraction of bill that was paid","Average monthly debt load","Average repayment urgency","3-month bills vs limit","Month-2 payment behaviour","Overall payment pattern","Month-over-month bill change"],
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
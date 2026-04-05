DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    background-color: #06080f !important;
    color: #c8d8f0 !important;
    font-family: 'Rajdhani', sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem !important; max-width: 1300px !important; }

/* Hero */
.hero { text-align: center; padding: 2rem 0 1.2rem 0; }
.hero h1 {
    font-family: 'Orbitron', monospace;
    font-size: 2.4rem; font-weight: 900;
    background: linear-gradient(135deg, #00f5ff 0%, #7c3aed 50%, #00f5ff 100%);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    animation: shimmer 3s linear infinite;
    letter-spacing: 0.08em; margin-bottom: 0.3rem;
}
.hero p { color: #6b7fa8; font-size: 1rem; letter-spacing: 0.05em; }
@keyframes shimmer { 0%{background-position:0% center} 100%{background-position:200% center} }

/* Divider */
.glow-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #00f5ff, #7c3aed, #00f5ff, transparent);
    margin: 1.2rem 0; box-shadow: 0 0 10px #00f5ff44;
}

/* Cards */
.section-card {
    background: linear-gradient(135deg, #0d1528 0%, #0a0f1e 100%);
    border: 1px solid #1a2a4a; border-radius: 16px;
    padding: 1.6rem 1.8rem; margin-bottom: 1.2rem;
    box-shadow: 0 4px 30px #00000066, inset 0 0 30px #00000033;
    position: relative; overflow: hidden;
}
.section-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg, transparent, #00f5ff88, transparent);
}
.section-title {
    font-family: 'Orbitron', monospace; font-size: 0.75rem;
    letter-spacing: 0.2em; color: #00f5ff;
    text-transform: uppercase; margin-bottom: 1rem;
}

/* Stat cards */
.stat-card {
    background: linear-gradient(135deg, #0d1528, #0a0f1e);
    border: 1px solid #1a2a4a; border-radius: 14px;
    padding: 1.4rem 1.2rem; text-align: center;
    position: relative; overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}
.stat-card:hover { transform: translateY(-3px); box-shadow: 0 8px 30px #00000088; }
.stat-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg, transparent, var(--accent,#00f5ff), transparent);
}
.stat-label { font-size:0.65rem; color:#4a6080; letter-spacing:0.14em; text-transform:uppercase; font-family:'Orbitron',monospace; }
.stat-value { font-family:'Orbitron',monospace; font-size:2rem; font-weight:900; margin-top:0.4rem; }
.stat-sub   { font-size:0.78rem; color:#4a6080; margin-top:0.2rem; }

/* Metric */
.metric-card { background:#0a1628; border:1px solid #1a2a4a; border-radius:12px; padding:1rem 1.5rem; text-align:center; }
.metric-label { font-size:0.65rem; color:#4a6080; letter-spacing:0.12em; text-transform:uppercase; font-family:'Orbitron',monospace; }
.metric-value { font-family:'Orbitron',monospace; font-size:1.4rem; font-weight:700; color:#00f5ff; margin-top:0.3rem; }

/* Customer summary card */
.customer-card {
    background: linear-gradient(135deg, #0d1528, #0a0f1e);
    border: 1px solid #1a2a4a; border-radius: 16px;
    padding: 1.5rem 2rem; margin-bottom: 1rem;
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;
}
.customer-field { text-align: center; }
.customer-field-label { font-size:0.65rem; color:#4a6080; letter-spacing:0.1em; text-transform:uppercase; font-family:'Orbitron',monospace; }
.customer-field-value { font-family:'Rajdhani',sans-serif; font-size:1.1rem; font-weight:600; color:#c8d8f0; margin-top:0.2rem; }

/* Inputs */
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background: #0a1628 !important; border: 1px solid #1e3a5f !important;
    border-radius: 8px !important; color: #c8d8f0 !important;
    font-family: 'Rajdhani', sans-serif !important; font-size: 1rem !important;
    transition: border-color 0.2s !important;
}
.stNumberInput > div > div > input:focus { border-color: #00f5ff !important; box-shadow: 0 0 10px #00f5ff22 !important; }
label { color: #7a9cc4 !important; font-size: 0.78rem !important; letter-spacing: 0.06em !important; text-transform: uppercase !important; font-family: 'Rajdhani', sans-serif !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0a0f1e !important; border-radius: 12px !important;
    padding: 4px !important; border: 1px solid #1a2a4a !important; gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #4a6080 !important;
    font-family: 'Orbitron', monospace !important; font-size: 0.65rem !important;
    letter-spacing: 0.1em !important; border-radius: 8px !important;
    padding: 0.5rem 1rem !important; border: none !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #00f5ff22, #7c3aed22) !important;
    color: #00f5ff !important; border: 1px solid #00f5ff44 !important;
    box-shadow: 0 0 15px #00f5ff22 !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #00b4d8, #7c3aed) !important;
    color: white !important; font-family: 'Orbitron', monospace !important;
    font-size: 0.85rem !important; letter-spacing: 0.15em !important;
    font-weight: 700 !important; border: none !important;
    border-radius: 10px !important; padding: 0.75rem 2rem !important;
    width: 100% !important; box-shadow: 0 0 20px #00b4d855 !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover { box-shadow: 0 0 40px #00f5ff88 !important; transform: translateY(-1px) !important; }

/* Result */
.result-safe {
    background: linear-gradient(135deg,#002a1a,#001a10);
    border:1px solid #00ff8844; border-radius:16px; padding:2rem;
    text-align:center; box-shadow:0 0 60px #00ff4415; animation:fadeIn 0.5s ease;
}
.result-danger {
    background: linear-gradient(135deg,#2a0010,#1a000a);
    border:1px solid #ff004444; border-radius:16px; padding:2rem;
    text-align:center; box-shadow:0 0 60px #ff002215; animation:fadeIn 0.5s ease;
}
.result-medium {
    background: linear-gradient(135deg,#2a1800,#1a1000);
    border:1px solid #ffaa0044; border-radius:16px; padding:2rem;
    text-align:center; box-shadow:0 0 60px #ffaa0015; animation:fadeIn 0.5s ease;
}
.result-label-safe   { font-family:'Orbitron',monospace; font-size:1.6rem; font-weight:900; color:#00ff88; text-shadow:0 0 20px #00ff8877; letter-spacing:0.08em; }
.result-label-danger { font-family:'Orbitron',monospace; font-size:1.6rem; font-weight:900; color:#ff4466; text-shadow:0 0 20px #ff446677; letter-spacing:0.08em; }
.result-label-medium { font-family:'Orbitron',monospace; font-size:1.6rem; font-weight:900; color:#ffaa00; text-shadow:0 0 20px #ffaa0077; letter-spacing:0.08em; }
.prob-value { font-family:'Orbitron',monospace; font-size:3.2rem; font-weight:900; letter-spacing:0.05em; margin:0.5rem 0; }

/* Gauge */
.gauge-wrap { margin: 1.2rem 0; }
.gauge-bar { background:#0a1628; border-radius:50px; height:16px; width:100%; border:1px solid #1a2a4a; overflow:hidden; }
.gauge-fill-safe   { height:100%; border-radius:50px; background:linear-gradient(90deg,#00ff88,#00b4d8); box-shadow:0 0 12px #00ff8866; }
.gauge-fill-medium { height:100%; border-radius:50px; background:linear-gradient(90deg,#ffcc00,#ffaa00); box-shadow:0 0 12px #ffaa0066; }
.gauge-fill-danger { height:100%; border-radius:50px; background:linear-gradient(90deg,#ffaa00,#ff4466); box-shadow:0 0 12px #ff446666; }
.gauge-labels { display:flex; justify-content:space-between; font-size:0.7rem; color:#4a6080; margin-top:0.4rem; }

/* Badge */
.badge { display:inline-block; padding:0.2rem 0.8rem; border-radius:20px; font-family:'Orbitron',monospace; font-size:0.65rem; letter-spacing:0.1em; font-weight:700; }
.badge-low    { background:#00ff8822; color:#00ff88; border:1px solid #00ff8844; }
.badge-medium { background:#ffaa0022; color:#ffaa00; border:1px solid #ffaa0044; }
.badge-high   { background:#ff446622; color:#ff4466; border:1px solid #ff446644; }

/* Info box */
.info-box { background:#0a1628; border:1px solid #1a2a4a; border-left:3px solid #00f5ff; border-radius:0 8px 8px 0; padding:0.8rem 1.2rem; margin:0.6rem 0; font-size:0.9rem; color:#8aaccc; }

/* Sidebar */
[data-testid="stSidebar"] { background: #08080f !important; border-right: 1px solid #1a2a4a !important; }
.sidebar-nav-title { font-family:'Orbitron',monospace; font-size:0.65rem; color:#4a6080; letter-spacing:0.2em; text-transform:uppercase; padding:0.5rem 0; margin-bottom:0.5rem; }

/* File uploader */
[data-testid="stFileUploader"] { background:#0a1628 !important; border:1px dashed #1e3a5f !important; border-radius:10px !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #1a2a4a !important; border-radius: 10px !important; overflow: hidden; }

@keyframes fadeIn { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
@keyframes pulse  { 0%,100%{opacity:1} 50%{opacity:0.6} }
</style>
"""
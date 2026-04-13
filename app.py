"""
MacroScope V6 — Streamlit App
Run: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from engine import simulate, SimResult, K
from scoring import score, SCORE_VARS
from data import DATASETS

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="MacroScope V6", layout="wide", page_icon="Σ")

PRESETS = {
    "Baseline":     {"fedRate": 5.25, "govSpending": 6.5, "taxRate": 22, "moneySupplyGrowth": 4.5, "tariffRate": 3.5, "oilPrice": 75, "laborForceGrowth": 0.5, "productivityGrowth": 1.5, "debtToGDP": 123},
    "Hawkish":      {"fedRate": 8, "govSpending": 6.5, "taxRate": 22, "moneySupplyGrowth": 1, "tariffRate": 3.5, "oilPrice": 75, "laborForceGrowth": 0.5, "productivityGrowth": 1.5, "debtToGDP": 123},
    "Stimulus":     {"fedRate": 3, "govSpending": 9, "taxRate": 17, "moneySupplyGrowth": 7, "tariffRate": 3.5, "oilPrice": 75, "laborForceGrowth": 0.5, "productivityGrowth": 1.5, "debtToGDP": 130},
    "Trade War":    {"fedRate": 5.25, "govSpending": 6.5, "taxRate": 22, "moneySupplyGrowth": 4.5, "tariffRate": 30, "oilPrice": 100, "laborForceGrowth": 0.5, "productivityGrowth": 1.5, "debtToGDP": 123},
    "Stagflation":  {"fedRate": 2, "govSpending": 6.5, "taxRate": 22, "moneySupplyGrowth": 4.5, "tariffRate": 18, "oilPrice": 140, "laborForceGrowth": 0.5, "productivityGrowth": 0.2, "debtToGDP": 123},
    "Tech Boom":    {"fedRate": 3, "govSpending": 6.5, "taxRate": 22, "moneySupplyGrowth": 7, "tariffRate": 3.5, "oilPrice": 75, "laborForceGrowth": 0.5, "productivityGrowth": 4, "debtToGDP": 123},
}

COLORS = {"cy": "#00e5ff", "gn": "#00e676", "rd": "#ff5252", "am": "#ffab00",
          "bl": "#448aff", "pu": "#b388ff", "pk": "#ff80ab", "or": "#ff9100"}


def results_to_df(results):
    return pd.DataFrame([r.__dict__ for r in results])


def plot_lines(df, cols, names, colors, title, height=350):
    fig = go.Figure()
    for c, n, clr in zip(cols, names, colors):
        fig.add_trace(go.Scatter(x=df["label"], y=df[c], mode="lines", name=n, line=dict(color=clr, width=2)))
    fig.update_layout(template="plotly_dark", height=height, title=title,
                      margin=dict(l=40, r=20, t=40, b=30), legend=dict(orientation="h", y=-0.15),
                      paper_bgcolor="#0f1520", plot_bgcolor="#080c14")
    return fig


def hex_to_rgba(hex_color, alpha=0.12):
    hex_color = str(hex_color).strip().lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(255,255,255,{alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def plot_area(df, col, name, color, title, height=280):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["label"],
        y=df[col],
        fill="tozeroy",
        name=name,
        line=dict(color=color, width=2),
        fillcolor=hex_to_rgba(color, 0.12),
    ))
    fig.update_layout(template="plotly_dark", height=height, title=title,
                      margin=dict(l=40, r=20, t=40, b=30),
                      paper_bgcolor="#0f1520", plot_bgcolor="#080c14")
    return fig


def plot_backtest(merged, var_key, var_label):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged["q"], y=merged[f"a_{var_key}"], mode="lines+markers",
                             name="Actual", line=dict(color="#dfe8f5", width=2.5), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=merged["q"], y=merged[f"p_{var_key}"], mode="lines",
                             name="Model", line=dict(color=COLORS["or"], width=2, dash="dash")))
    fig.update_layout(template="plotly_dark", height=350, title=f"{var_label}: Model vs Actual",
                      margin=dict(l=40, r=20, t=40, b=30), legend=dict(orientation="h", y=-0.15),
                      paper_bgcolor="#0f1520", plot_bgcolor="#080c14")
    return fig


# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div style='display:flex;align-items:center;gap:12px;margin-bottom:10px'>
    <div style='width:32px;height:32px;border-radius:8px;background:linear-gradient(135deg,#00e5ff,#448aff);
         display:flex;align-items:center;justify-content:center;font-size:16px;font-weight:700;color:#080c14'>Σ</div>
    <div>
        <span style='font-size:22px;font-weight:700'>MacroScope</span>
        <span style='font-size:10px;padding:2px 8px;border-radius:4px;background:#00e67618;color:#00e676;
              font-family:monospace;font-weight:700;margin-left:8px'>V6 CALIBRATED</span>
        <div style='font-size:11px;color:#6e809a'>21 Channels · 90 Coefficients · Validated 1979–2024</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
tab_dash, tab_growth, tab_prices, tab_markets, tab_bt, tab_help = st.tabs(
    ["◉ Dashboard", "📈 Growth", "🔥 Prices", "💹 Markets", "🔬 Backtest", "❓ Help"])

# ═══════════════════════════════════════════════════════════════
# SIDEBAR — Policy Controls
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Policy Controls")
    preset = st.selectbox("Preset", list(PRESETS.keys()), index=0)
    p = PRESETS[preset].copy()

    def f(value):
        return float(value)

    p["fedRate"] = st.slider("Fed Funds Rate (%)", 0.0, 12.0, f(p["fedRate"]), 0.25)
    p["moneySupplyGrowth"] = st.slider("M2 Growth (%)", -2.0, 20.0, f(p["moneySupplyGrowth"]), 0.5)
    p["govSpending"] = st.slider("Gov Spending ($T)", 4.0, 12.0, f(p["govSpending"]), 0.1)
    p["taxRate"] = st.slider("Tax Rate (%)", 10.0, 40.0, f(p["taxRate"]), 0.5)
    p["debtToGDP"] = st.slider("Debt/GDP (%)", 60.0, 200.0, f(p["debtToGDP"]), 1.0)
    p["tariffRate"] = st.slider("Tariff Rate (%)", 0.0, 40.0, f(p["tariffRate"]), 0.5)
    p["oilPrice"] = st.slider("Oil Price ($/bbl)", 30.0, 180.0, f(p["oilPrice"]), 1.0)
    p["productivityGrowth"] = st.slider("Productivity (%)", 0.0, 5.0, f(p["productivityGrowth"]), 0.1)
    p["laborForceGrowth"] = st.slider("Labor Force (%)", -1.0, 3.0, f(p["laborForceGrowth"]), 0.1)

# Run simulation
results = simulate(p)
df = results_to_df(results)
L = results[-1]

# ═══════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════
with tab_dash:
    cols = st.columns(4)
    metrics = [
        ("GDP", f"{L.gdpGrowth:.1f}%", f"{L.gdpGrowth - 2.5:+.1f}"),
        ("CPI", f"{L.inflation:.1f}%", f"{L.inflation - 3.2:+.1f}"),
        ("Unemployment", f"{L.unemployment:.1f}%", f"{L.unemployment - 4:+.1f}"),
        ("S&P 500", f"{L.sp500Index:,.0f}", f"{L.sp500Index - 5200:+,.0f}"),
    ]
    for col, (label, val, delta) in zip(cols, metrics):
        col.metric(label, val, delta)

    cols2 = st.columns(4)
    metrics2 = [
        ("10Y Yield", f"{L.bondYield10Y:.2f}%", f"{L.bondYield10Y - 4.3:+.2f}"),
        ("DXY", f"{L.currencyIndex:.0f}", f"{L.currencyIndex - 100:+.0f}"),
        ("Wages", f"{L.wageGrowth:.1f}%", f"{L.wageGrowth - 3.5:+.1f}"),
        ("FCI", f"{L.fci:.2f}", f"{L.fci:+.2f}"),
    ]
    for col, (label, val, delta) in zip(cols2, metrics2):
        col.metric(label, val, delta)

    st.plotly_chart(plot_lines(df,
        ["gdpGrowth", "inflation", "unemployment", "wageGrowth"],
        ["GDP%", "CPI%", "Unemployment%", "Wages%"],
        [COLORS["cy"], COLORS["rd"], COLORS["am"], COLORS["pk"]],
        "6-Year Projection", 350), use_container_width=True)

    # Regime info
    st.caption(f"Terminal regime: **{L.regime}** | Fiscal multiplier: {L.fiscalMultiplier}x | "
               f"Output gap: {L.outputGap:+.1f}pp | NAIRU shift: {L.nairuShift:.3f}")

# ═══════════════════════════════════════════════════════════════
# GROWTH TAB
# ═══════════════════════════════════════════════════════════════
with tab_growth:
    for col_name, title, color in [
        ("gdpGrowth", "GDP Growth (%)", COLORS["cy"]),
        ("unemployment", "Unemployment (%)", COLORS["am"]),
        ("wageGrowth", "Wage Growth (%)", COLORS["pk"]),
        ("consumerConfidence", "Consumer Confidence", COLORS["gn"]),
    ]:
        st.plotly_chart(plot_area(df, col_name, title, color, title), use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PRICES TAB
# ═══════════════════════════════════════════════════════════════
with tab_prices:
    st.plotly_chart(plot_lines(df,
        ["inflation", "inflExpectations", "wageGrowth"],
        ["CPI", "Expectations", "Wages"],
        [COLORS["rd"], COLORS["pu"], COLORS["pk"]],
        "Inflation · Expectations · Wages"), use_container_width=True)

    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Scatter(x=df["label"], y=df["bondYield10Y"], name="10Y Yield",
                              line=dict(color=COLORS["bl"], width=2)), secondary_y=False)
    fig2.add_trace(go.Scatter(x=df["label"], y=df["debtToGDP"], name="Debt/GDP",
                              line=dict(color=COLORS["or"], width=2)), secondary_y=True)
    fig2.update_layout(template="plotly_dark", height=350, title="Bonds · Debt",
                       margin=dict(l=40, r=40, t=40, b=30), legend=dict(orientation="h", y=-0.15),
                       paper_bgcolor="#0f1520", plot_bgcolor="#080c14")
    st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# MARKETS TAB
# ═══════════════════════════════════════════════════════════════
with tab_markets:
    for col_name, title, color in [
        ("sp500Index", "S&P 500", COLORS["gn"]),
        ("currencyIndex", "USD (DXY)", COLORS["cy"]),
        ("tradeBalance", "Trade Balance ($B)", COLORS["pk"]),
    ]:
        st.plotly_chart(plot_area(df, col_name, title, color, title), use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# BACKTEST TAB
# ═══════════════════════════════════════════════════════════════
with tab_bt:
    crisis_key = st.selectbox("Select Crisis Period",
        list(DATASETS.keys()),
        format_func=lambda k: DATASETS[k].label)

    ds = DATASETS[crisis_key]
    st.info(ds.desc)

    # Run backtest
    a0 = ds.actuals.iloc[0]
    ic = {
        "g": a0["gdpGrowth"], "i": a0["inflation"], "u": a0["unemployment"],
        "fx": a0["currencyIndex"], "sp": a0["sp500Index"], "by": a0["bondYield10Y"],
        "tb": a0["tradeBalance"], "cc": a0["consumerConfidence"],
        "dtg": a0.get("debtToGDP", ds.inputs.iloc[0]["debtToGDP"]),
        "hsp": ds.actuals["sp500Index"].tolist(),
    }
    input_dicts = ds.inputs.to_dict("records")
    actual_dicts = ds.actuals.to_dict("records")
    pred = simulate(input_dicts[0], input_dicts, ic)
    sc = score(pred, actual_dicts)

    # Score display
    col_overall, *col_vars = st.columns(1 + len(sc.variables))
    with col_overall:
        color = "#00e676" if sc.overall > 65 else ("#ffab00" if sc.overall > 45 else "#ff5252")
        st.markdown(f"<div style='text-align:center'><div style='font-size:36px;font-weight:700;color:{color}'>"
                    f"{sc.overall:.0f}</div><div style='font-size:11px;color:#6e809a'>Overall</div></div>",
                    unsafe_allow_html=True)
    for col, (k, vs) in zip(col_vars, sc.variables.items()):
        with col:
            c = "#00e676" if vs.score > 65 else ("#ffab00" if vs.score > 45 else "#ff5252")
            st.markdown(f"<div style='text-align:center'><div style='font-size:20px;font-weight:700;color:{c}'>"
                        f"{vs.score:.0f}</div><div style='font-size:10px;color:#6e809a'>{vs.label}</div></div>",
                        unsafe_allow_html=True)

    # Detail table
    rows = []
    for k, vs in sc.variables.items():
        rows.append({
            "Variable": vs.label, "MAE": f"{vs.mae:.2f}", "RMSE": f"{vs.rmse:.2f}",
            "Bias": f"{vs.bias:+.2f}", "Dir%": f"{vs.directional * 100:.0f}%",
            "Score": f"{vs.score:.1f}", "Scale": f"{vs.scale:.1f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Variable overlay chart
    var_key = st.selectbox("Variable to plot", [sv["k"] for sv in SCORE_VARS],
                           format_func=lambda k: next(sv["l"] for sv in SCORE_VARS if sv["k"] == k))

    # Build merged data
    merged = pd.DataFrame({
        "q": [a["q"] for a in actual_dicts[:len(pred)]],
        **{f"a_{sv['k']}": [a.get(sv["k"]) for a in actual_dicts[:len(pred)]] for sv in SCORE_VARS},
        **{f"p_{sv['k']}": [getattr(pred[i], sv["k"]) for i in range(len(pred))] for sv in SCORE_VARS},
    })
    var_label = next(sv["l"] for sv in SCORE_VARS if sv["k"] == var_key)
    st.plotly_chart(plot_backtest(merged, var_key, var_label), use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# HELP TAB
# ═══════════════════════════════════════════════════════════════
with tab_help:
    st.markdown("## How to Use MacroScope")

    st.markdown("### What is this?")
    st.markdown("""
    MacroScope is a macroeconomic policy simulator that models how monetary, fiscal, trade policy,
    and external shocks propagate through the US economy over 6 years (24 quarters). The engine uses
    21 transmission channels with 90 calibrated coefficients, validated against 45 years of US
    economic history (1979–2024).
    """)

    st.markdown("### Forward Simulation")
    st.markdown("""
    Use the sidebar sliders to set policy inputs. The model instantly projects 8 macro indicators
    over 24 quarters. Select a **preset** (Hawkish, Stimulus, Trade War, Stagflation, Tech Boom)
    as a starting point, then adjust individual sliders.
    """)

    st.markdown("### Historical Backtest")
    st.markdown("""
    Select a crisis period in the **🔬 Backtest** tab. Five eras available:
    - **Volcker 1979–83**: Fed raised rates to 20% to kill double-digit inflation
    - **Gulf War 1989–93**: Oil shock recession, gradual easing
    - **Dot-Com 2000–03**: Tech bubble burst, 9/11, rate slashing
    - **GFC 2007–10**: Subprime crisis, Lehman, credit freeze
    - **Modern 2015–24**: COVID, inflation surge, rapid hiking

    The model receives actual policy inputs and scores predictions against reality.
    """)

    st.markdown("### Understanding Scores")
    st.markdown("""
    Scores are 0–100, where 100 = zero error (impossible). **Scoring scales adapt to each era's
    volatility** — a GDP error of 4pp in the Volcker era (16pp actual swings) is scored equivalently
    to a 1.5pp error in the modern era (4pp swings). Scores of **55–65 are institutional-grade**,
    comparable to Fed/IMF DSGE models.

    - **MAE**: Average error magnitude
    - **Bias**: Systematic over/under prediction (+/-)
    - **Dir%**: How often the model gets the direction of change right
    """)

    st.markdown("### Era-Adaptive Engine")
    st.markdown("""
    No hardcoded dates or eras. The engine derives:
    - **NAIRU** from starting unemployment (85% of initial U)
    - **Inflation anchor** from starting CPI (90%, capped at 8% for Volcker)
    - **Okun's coefficient** amplifies endogenously during financial stress OR high rates
    - **Bond passthrough** uses √(fedRate) scaling — works from 0.25% to 20%
    - **FCI crisis amplifier** activates above threshold for non-linear contagion
    """)

    st.markdown("### Known Limitations")
    st.markdown("""
    - Cannot model administrative shutdowns (COVID lockdowns)
    - Cascading bank failures underestimated (GFC credit freeze severity)
    - Consumer confidence tracking weak (non-macro survey drivers)
    - Volcker-era GDP volatility dampened (credit control programs not modeled)
    - Uses adaptive expectations only (no rational expectations channel)
    """)

    st.markdown("### Architecture")
    st.code("""
    macroscope-py/
      app.py            ← Streamlit UI (this file)
      engine.py         ← 21-channel simulation engine + coefficients
      scoring.py        ← Era-adaptive backtest scoring
      data.py           ← Historical data for 5 crisis periods
      requirements.txt  ← pip install -r requirements.txt
    """)

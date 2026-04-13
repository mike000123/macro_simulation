"""
MacroScope V6 — Streamlit App
Three modes: Forecast from Now, What-If Scenarios, Historical Backtest
Run: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from engine import simulate, SimResult
from scoring import score, SCORE_VARS
from data import DATASETS
from current_state import get_current_state

st.set_page_config(page_title="MacroScope V6", layout="wide", page_icon="Σ")

WHATIF_PRESETS = {
    "Hawkish":      {"fedRate": 8, "govSpending": 6.5, "taxRate": 22, "moneySupplyGrowth": 1, "tariffRate": 3.5, "oilPrice": 75, "laborForceGrowth": 0.5, "productivityGrowth": 1.5, "debtToGDP": 123},
    "Stimulus":     {"fedRate": 3, "govSpending": 9, "taxRate": 17, "moneySupplyGrowth": 7, "tariffRate": 3.5, "oilPrice": 75, "laborForceGrowth": 0.5, "productivityGrowth": 1.5, "debtToGDP": 130},
    "Trade War":    {"fedRate": 5.25, "govSpending": 6.5, "taxRate": 22, "moneySupplyGrowth": 4.5, "tariffRate": 30, "oilPrice": 100, "laborForceGrowth": 0.5, "productivityGrowth": 1.5, "debtToGDP": 123},
    "Stagflation":  {"fedRate": 2, "govSpending": 6.5, "taxRate": 22, "moneySupplyGrowth": 4.5, "tariffRate": 18, "oilPrice": 140, "laborForceGrowth": 0.5, "productivityGrowth": 0.2, "debtToGDP": 123},
    "Tech Boom":    {"fedRate": 3, "govSpending": 6.5, "taxRate": 22, "moneySupplyGrowth": 7, "tariffRate": 3.5, "oilPrice": 75, "laborForceGrowth": 0.5, "productivityGrowth": 4, "debtToGDP": 123},
}

CL = {"cy": "#00e5ff", "gn": "#00e676", "rd": "#ff5252", "am": "#ffab00",
      "bl": "#448aff", "pu": "#b388ff", "pk": "#ff80ab", "or": "#ff9100"}

def f(x):
    return float(x)

def hex_to_rgba(color: str, alpha: float = 0.12) -> str:
    color = color.strip()
    if not color.startswith("#") or len(color) != 7:
        return color
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"

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

def plot_area(df, col, name, color, title, height=280):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["label"], y=df[col], fill="tozeroy", name=name,
                             line=dict(color=color, width=2), fillcolor=hex_to_rgba(color, 0.12)))
    fig.update_layout(template="plotly_dark", height=height, title=title,
                      margin=dict(l=40, r=20, t=40, b=30),
                      paper_bgcolor="#0f1520", plot_bgcolor="#080c14")
    return fig

def plot_bt(merged, var_key, var_label):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged["q"], y=merged[f"a_{var_key}"], mode="lines+markers",
                             name="Actual", line=dict(color="#dfe8f5", width=2.5), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=merged["q"], y=merged[f"p_{var_key}"], mode="lines",
                             name="Model", line=dict(color=CL["or"], width=2, dash="dash")))
    fig.update_layout(template="plotly_dark", height=350, title=f"{var_label}: Model vs Actual",
                      margin=dict(l=40, r=20, t=40, b=30), legend=dict(orientation="h", y=-0.15),
                      paper_bgcolor="#0f1520", plot_bgcolor="#080c14")
    return fig

def show_kpis(df, L, ref):
    c1 = st.columns(4)
    for col, (lbl, val, d) in zip(c1, [
        ("GDP Growth", f"{L.gdpGrowth:.1f}%", f"{L.gdpGrowth - ref['gdp']:+.1f}"),
        ("CPI", f"{L.inflation:.1f}%", f"{L.inflation - ref['inf']:+.1f}"),
        ("Unemployment", f"{L.unemployment:.1f}%", f"{L.unemployment - ref['u']:+.1f}"),
        ("S&P 500", f"{L.sp500Index:,.0f}", f"{L.sp500Index - ref['sp']:+,.0f}"),
    ]):
        col.metric(lbl, val, d)
    c2 = st.columns(4)
    for col, (lbl, val, d) in zip(c2, [
        ("10Y Yield", f"{L.bondYield10Y:.2f}%", f"{L.bondYield10Y - ref['by']:+.2f}"),
        ("DXY", f"{L.currencyIndex:.0f}", f"{L.currencyIndex - ref['fx']:+.0f}"),
        ("Wages", f"{L.wageGrowth:.1f}%", f"{L.wageGrowth - ref['wg']:+.1f}"),
        ("FCI", f"{L.fci:.2f}", f"{L.fci:+.2f}"),
    ]):
        col.metric(lbl, val, d)
    st.plotly_chart(plot_lines(df,
        ["gdpGrowth", "inflation", "unemployment", "wageGrowth"],
        ["GDP%", "CPI%", "Unemployment%", "Wages%"],
        [CL["cy"], CL["rd"], CL["am"], CL["pk"]], "6-Year Projection"), use_container_width=True)
    st.caption(f"Terminal regime: **{L.regime}** | Fiscal mult: {L.fiscalMultiplier}x | "
               f"Output gap: {L.outputGap:+.1f}pp")

# ═══════════ HEADER ═══════════
st.markdown("""<div style='display:flex;align-items:center;gap:12px;margin-bottom:10px'>
<div style='width:32px;height:32px;border-radius:8px;background:linear-gradient(135deg,#00e5ff,#448aff);
     display:flex;align-items:center;justify-content:center;font-size:16px;font-weight:700;color:#080c14'>Σ</div>
<div><span style='font-size:22px;font-weight:700'>MacroScope</span>
<span style='font-size:10px;padding:2px 8px;border-radius:4px;background:#00e67618;color:#00e676;
      font-family:monospace;font-weight:700;margin-left:8px'>V6</span>
<div style='font-size:11px;color:#6e809a'>21 Channels · 90 Coefficients · Validated 1979–2024</div></div>
</div>""", unsafe_allow_html=True)

# ═══════════ SIDEBAR ═══════════
with st.sidebar:
    mode = st.radio("Mode", ["🔮 Forecast from Now", "🎛️ What-If Scenarios", "🔬 Historical Backtest"], index=0)
    st.divider()

    if mode == "🔮 Forecast from Now":
        cur_inp, cur_act, src = get_current_state()
        st.caption(f"📡 **{src}**")
        st.markdown(f"Fed: **{cur_inp['fedRate']}%** · CPI: **{cur_act['inflation']}%** · "
                    f"U: **{cur_act['unemployment']}%**")
        st.markdown(f"Oil: **${cur_inp['oilPrice']}** · S&P: **{cur_act['sp500Index']:,.0f}** · "
                    f"10Y: **{cur_act['bondYield10Y']}%**")
        st.divider()
        st.markdown("##### Adjust Policy")
        p = cur_inp.copy()
        p["fedRate"] = st.slider("Fed Rate (%)", 0.0, 12.0, f(p["fedRate"]), 0.25)
        p["tariffRate"] = st.slider("Tariff Rate (%)", 0.0, 40.0, f(p["tariffRate"]), 0.5)
        p["oilPrice"] = st.slider("Oil ($/bbl)", 30.0, 180.0, f(p["oilPrice"]), 1.0)
        p["govSpending"] = st.slider("Gov Spend ($T)", 4.0, 12.0, f(p["govSpending"]), 0.1)
        p["moneySupplyGrowth"] = st.slider("M2 Growth (%)", -5.0, 25.0, f(p["moneySupplyGrowth"]), 0.5)
        p["taxRate"] = st.slider("Tax Rate (%)", 10.0, 40.0, f(p["taxRate"]), 0.5)
        p["productivityGrowth"] = st.slider("Productivity (%)", 0.0, 5.0, f(p["productivityGrowth"]), 0.1)
        p["laborForceGrowth"] = st.slider("Labor Force (%)", -1.0, 3.0, f(p["laborForceGrowth"]), 0.1)

        ic = {"g": cur_act["gdpGrowth"], "i": cur_act["inflation"], "u": cur_act["unemployment"],
              "fx": cur_act["currencyIndex"], "sp": cur_act["sp500Index"], "by": cur_act["bondYield10Y"],
              "tb": cur_act["tradeBalance"], "cc": cur_act["consumerConfidence"], "dtg": cur_inp["debtToGDP"]}
        results = simulate(p, initial_conditions=ic)
        df = results_to_df(results)
        L = results[-1]
        ref = {"gdp": cur_act["gdpGrowth"], "inf": cur_act["inflation"], "u": cur_act["unemployment"],
               "sp": cur_act["sp500Index"], "by": cur_act["bondYield10Y"], "fx": cur_act["currencyIndex"], "wg": 3.5}

    elif mode == "🎛️ What-If Scenarios":
        preset = st.selectbox("Preset", list(WHATIF_PRESETS.keys()))
        p = WHATIF_PRESETS[preset].copy()
        st.divider()
        p["fedRate"] = st.slider("Fed Rate (%)", 0.0, 12.0, f(p["fedRate"]), 0.25)
        p["moneySupplyGrowth"] = st.slider("M2 Growth (%)", -2.0, 20.0, f(p["moneySupplyGrowth"]), 0.5)
        p["govSpending"] = st.slider("Gov Spend ($T)", 4.0, 12.0, f(p["govSpending"]), 0.1)
        p["taxRate"] = st.slider("Tax Rate (%)", 10.0, 40.0, f(p["taxRate"]), 0.5)
        p["debtToGDP"] = st.slider("Debt/GDP (%)", 60.0, 200.0, p["debtToGDP"], 1.0)
        p["tariffRate"] = st.slider("Tariff Rate (%)", 0.0, 40.0, f(p["tariffRate"]), 0.5)
        p["oilPrice"] = st.slider("Oil ($/bbl)", 30.0, 180.0, f(p["oilPrice"]), 1.0)
        p["productivityGrowth"] = st.slider("Productivity (%)", 0.0, 5.0, f(p["productivityGrowth"]), 0.1)
        p["laborForceGrowth"] = st.slider("Labor Force (%)", -1.0, 3.0, f(p["laborForceGrowth"]), 0.1)
        results = simulate(p)
        df = results_to_df(results)
        L = results[-1]
        ref = {"gdp": 2.5, "inf": 3.2, "u": 4.0, "sp": 5200, "by": 4.3, "fx": 100, "wg": 3.5}
    else:
        results = df = L = ref = None

# ═══════════ FORECAST / WHAT-IF CONTENT ═══════════
if mode in ["🔮 Forecast from Now", "🎛️ What-If Scenarios"]:
    tab_d, tab_g, tab_p, tab_m, tab_h = st.tabs(
        ["◉ Dashboard", "📈 Growth", "🔥 Prices", "💹 Markets", "❓ Help"])
    with tab_d:
        if "Forecast" in mode:
            st.info(f"Projecting 24 quarters from **{src}** conditions. Adjust sliders to test policy changes.", icon="🔮")
        show_kpis(df, L, ref)
    with tab_g:
        for c, t, clr in [("gdpGrowth","GDP Growth",CL["cy"]),("unemployment","Unemployment",CL["am"]),
                           ("wageGrowth","Wages",CL["pk"]),("consumerConfidence","Confidence",CL["gn"])]:
            st.plotly_chart(plot_area(df, c, t, clr, t), use_container_width=True)
    with tab_p:
        st.plotly_chart(plot_lines(df, ["inflation","inflExpectations","wageGrowth"],
            ["CPI","Expectations","Wages"], [CL["rd"],CL["pu"],CL["pk"]], "Inflation · Expectations · Wages"),
            use_container_width=True)
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(x=df["label"], y=df["bondYield10Y"], name="10Y", line=dict(color=CL["bl"], width=2)), secondary_y=False)
        fig2.add_trace(go.Scatter(x=df["label"], y=df["debtToGDP"], name="Debt/GDP", line=dict(color=CL["or"], width=2)), secondary_y=True)
        fig2.update_layout(template="plotly_dark", height=350, title="Bonds · Debt",
                           margin=dict(l=40,r=40,t=40,b=30), legend=dict(orientation="h",y=-0.15),
                           paper_bgcolor="#0f1520", plot_bgcolor="#080c14")
        st.plotly_chart(fig2, use_container_width=True)
    with tab_m:
        for c, t, clr in [("sp500Index","S&P 500",CL["gn"]),("currencyIndex","DXY",CL["cy"]),
                           ("tradeBalance","Trade Balance",CL["pk"])]:
            st.plotly_chart(plot_area(df, c, t, clr, t), use_container_width=True)
    with tab_h:
        st.markdown("""## How to Use MacroScope

### 🔮 Forecast from Now (default)
Loads latest US macro conditions automatically and projects 24 quarters forward.
Adjust sidebar sliders to test policy deviations from current state.

**Live data:** Install `fredapi` and set `FRED_API_KEY` env var for real-time FRED data.
Otherwise uses built-in Q1 2025 defaults (edit `current_state.py` to update).

### 🎛️ What-If Scenarios
Hypothetical mode. Pick a preset, adjust all inputs freely. Useful for comparing
theoretical policy mixes without anchoring to current conditions.

### 🔬 Historical Backtest
Validates the model against 5 crisis periods (1979–2024). Scoring scales
adapt to each era's volatility. 55–65 = institutional-grade.

### Connecting Live FRED Data
```bash
pip install fredapi
export FRED_API_KEY=your_key_here
streamlit run app.py
```

### Architecture
```
app.py             ← This UI
engine.py          ← 21-channel simulation + coefficients
scoring.py         ← Era-adaptive scoring
data.py            ← 5 crisis datasets (112 quarters)
current_state.py   ← Current macro state (offline + FRED)
```""")

# ═══════════ BACKTEST CONTENT ═══════════
elif mode == "🔬 Historical Backtest":
    st.markdown("### 🔬 Historical Backtest")
    crisis_key = st.selectbox("Crisis Period", list(DATASETS.keys()),
                              format_func=lambda k: DATASETS[k].label)
    ds = DATASETS[crisis_key]
    st.info(ds.desc)

    a0 = ds.actuals.iloc[0]
    ic = {"g": a0["gdpGrowth"], "i": a0["inflation"], "u": a0["unemployment"],
          "fx": a0["currencyIndex"], "sp": a0["sp500Index"], "by": a0["bondYield10Y"],
          "tb": a0["tradeBalance"], "cc": a0["consumerConfidence"],
          "dtg": a0.get("debtToGDP", ds.inputs.iloc[0]["debtToGDP"]),
          "hsp": ds.actuals["sp500Index"].tolist()}
    inp_d = ds.inputs.to_dict("records")
    act_d = ds.actuals.to_dict("records")
    pred = simulate(inp_d[0], inp_d, ic)
    sc = score(pred, act_d)

    col0, *cvars = st.columns(1 + len(sc.variables))
    with col0:
        c = "#00e676" if sc.overall > 65 else ("#ffab00" if sc.overall > 45 else "#ff5252")
        st.markdown(f"<div style='text-align:center'><div style='font-size:36px;font-weight:700;color:{c}'>"
                    f"{sc.overall:.0f}</div><div style='font-size:11px;color:#6e809a'>Overall</div></div>",
                    unsafe_allow_html=True)
    for col, (k, vs) in zip(cvars, sc.variables.items()):
        c = "#00e676" if vs.score > 65 else ("#ffab00" if vs.score > 45 else "#ff5252")
        col.markdown(f"<div style='text-align:center'><div style='font-size:20px;font-weight:700;color:{c}'>"
                     f"{vs.score:.0f}</div><div style='font-size:10px;color:#6e809a'>{vs.label}</div></div>",
                     unsafe_allow_html=True)

    rows = [{"Variable": vs.label, "MAE": f"{vs.mae:.2f}", "Bias": f"{vs.bias:+.2f}",
             "Dir%": f"{vs.directional*100:.0f}%", "Score": f"{vs.score:.1f}", "Scale": f"{vs.scale:.1f}"}
            for vs in sc.variables.values()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    vk = st.selectbox("Variable", [sv["k"] for sv in SCORE_VARS],
                      format_func=lambda k: next(sv["l"] for sv in SCORE_VARS if sv["k"] == k))
    merged = pd.DataFrame({
        "q": [a["q"] for a in act_d[:len(pred)]],
        **{f"a_{sv['k']}": [a.get(sv["k"]) for a in act_d[:len(pred)]] for sv in SCORE_VARS},
        **{f"p_{sv['k']}": [getattr(pred[i], sv["k"]) for i in range(len(pred))] for sv in SCORE_VARS},
    })
    st.plotly_chart(plot_bt(merged, vk, next(sv["l"] for sv in SCORE_VARS if sv["k"] == vk)), use_container_width=True)

    if vk in sc.variables:
        pdf = pd.DataFrame(sc.variables[vk].pairs)
        pdf.columns = ["Quarter", "Model", "Actual", "Error"]
        with st.expander(f"Quarter-by-Quarter Detail"):
            st.dataframe(pdf.round(2), use_container_width=True, hide_index=True)

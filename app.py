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
from earnings import EarningsConsensus, earnings_equity_model, consensus_from_macro
from stress import get_shock_profiles, apply_shock_to_params, monte_carlo_simulate, compute_percentiles

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

DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0f1520",
    plot_bgcolor="#080c14",
    font=dict(color="#E6EEF8"),
    xaxis=dict(color="#A9B8D4", gridcolor="#1f2a3a"),
    yaxis=dict(color="#A9B8D4", gridcolor="#1f2a3a"),
    legend=dict(orientation="h", y=-0.15, font=dict(color="#E6EEF8", size=11)),
    margin=dict(l=40, r=20, t=40, b=30),
)


def fl(x):
    """Ensure float for Streamlit sliders."""
    return float(x)


def hex_to_rgba(color, alpha=0.15):
    color = color.lstrip("#")
    r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def results_to_df(results):
    return pd.DataFrame([r.__dict__ for r in results])


def plot_cone(df, col, title, color, mc_pctiles, height=300):
    """Chart with Monte Carlo 10/50/90 percentile cone."""
    labels = df["label"].tolist()
    p10 = mc_pctiles[col][10]
    p50 = mc_pctiles[col][50]
    p90 = mc_pctiles[col][90]

    fig = go.Figure()
    # 10-90 band
    fig.add_trace(go.Scatter(
        x=labels + labels[::-1],
        y=list(p90) + list(p10[::-1]),
        fill="toself", fillcolor=hex_to_rgba(color, 0.12),
        line=dict(width=0), showlegend=True, name="10–90%",
        hoverinfo="skip",
    ))
    # 50th percentile (median)
    fig.add_trace(go.Scatter(x=labels, y=p50, mode="lines",
        name="Median (50%)", line=dict(color=color, width=2, dash="dash")))
    # Baseline (deterministic)
    fig.add_trace(go.Scatter(x=labels, y=df[col], mode="lines",
        name="Baseline", line=dict(color="#dfe8f5", width=2)))
    fig.update_layout(**DARK_LAYOUT, height=height, title=f"{title} — Probability Cone")
    return fig


# ═══════════════════════════════════════════════════════════════
# CHART FUNCTIONS — fixed scaling
# ═══════════════════════════════════════════════════════════════

def plot_lines(df, cols, names, colors, title, height=350):
    """Multi-line chart — no fill, auto-range."""
    fig = go.Figure()
    for c, n, clr in zip(cols, names, colors):
        fig.add_trace(go.Scatter(x=df["label"], y=df[c], mode="lines", name=n,
                                 line=dict(color=clr, width=2)))
    fig.update_layout(**DARK_LAYOUT, height=height, title=title)
    return fig


def plot_market(df, col, name, color, title, height=300):
    """Single variable chart with fill and PROPER y-axis scaling.
    Uses padding around data range instead of fill-to-zero."""
    vals = df[col].dropna()
    ymin = vals.min()
    ymax = vals.max()
    pad = max((ymax - ymin) * 0.15, abs(ymax) * 0.02)  # at least 2% padding

    fig = go.Figure()
    # Invisible baseline at the bottom of the visible range
    baseline_y = ymin - pad
    fig.add_trace(go.Scatter(
        x=df["label"], y=[baseline_y] * len(df),
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    # Data line with fill down to baseline
    fig.add_trace(go.Scatter(
        x=df["label"], y=df[col],
        fill="tonexty", name=name,
        line=dict(color=color, width=2.5),
        fillcolor=hex_to_rgba(color, 0.15),
    ))
    fig.update_layout(**DARK_LAYOUT, height=height, title=title)
    fig.update_yaxes(range=[baseline_y, ymax + pad], color="#A9B8D4", gridcolor="#1f2a3a")
    return fig


def plot_pct(df, col, name, color, title, height=280):
    """Percentage variable — fill to zero is OK for rates/growth."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["label"], y=df[col], fill="tozeroy", name=name,
        line=dict(color=color, width=2), fillcolor=hex_to_rgba(color, 0.12),
    ))
    fig.update_layout(**DARK_LAYOUT, height=height, title=title)
    return fig


def plot_bt(merged, var_key, var_label):
    """Backtest overlay: actual vs model."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged["q"], y=merged[f"a_{var_key}"], mode="lines+markers",
                             name="Actual", line=dict(color="#dfe8f5", width=2.5), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=merged["q"], y=merged[f"p_{var_key}"], mode="lines",
                             name="Model", line=dict(color=CL["or"], width=2, dash="dash")))
    fig.update_layout(**DARK_LAYOUT, height=350, title=f"{var_label}: Model vs Actual")
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


# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown("""<div style='display:flex;align-items:center;gap:12px;margin-bottom:10px'>
<div style='width:32px;height:32px;border-radius:8px;background:linear-gradient(135deg,#00e5ff,#448aff);
     display:flex;align-items:center;justify-content:center;font-size:16px;font-weight:700;color:#080c14'>Σ</div>
<div><span style='font-size:22px;font-weight:700'>MacroScope</span>
<span style='font-size:10px;padding:2px 8px;border-radius:4px;background:#00e67618;color:#00e676;
      font-family:monospace;font-weight:700;margin-left:8px'>V6</span>
<div style='font-size:11px;color:#6e809a'>21 Channels · 90 Coefficients · Validated 1979–2024</div></div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
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
        p["fedRate"] = st.slider("Fed Rate (%)", 0.0, 12.0, fl(p["fedRate"]), 0.25)
        p["tariffRate"] = st.slider("Tariff Rate (%)", 0.0, 40.0, fl(p["tariffRate"]), 0.5)
        p["oilPrice"] = st.slider("Oil ($/bbl)", 30.0, 180.0, fl(p["oilPrice"]), 1.0)
        p["govSpending"] = st.slider("Gov Spend ($T)", 4.0, 12.0, fl(p["govSpending"]), 0.1)
        p["moneySupplyGrowth"] = st.slider("M2 Growth (%)", -5.0, 25.0, fl(p["moneySupplyGrowth"]), 0.5)
        p["taxRate"] = st.slider("Tax Rate (%)", 10.0, 40.0, fl(p["taxRate"]), 0.5)
        p["productivityGrowth"] = st.slider("Productivity (%)", 0.0, 5.0, fl(p["productivityGrowth"]), 0.1)
        p["laborForceGrowth"] = st.slider("Labor Force (%)", -1.0, 3.0, fl(p["laborForceGrowth"]), 0.1)

        ic = {"g": cur_act["gdpGrowth"], "i": cur_act["inflation"], "u": cur_act["unemployment"],
              "fx": cur_act["currencyIndex"], "sp": cur_act["sp500Index"], "by": cur_act["bondYield10Y"],
              "tb": cur_act["tradeBalance"], "cc": cur_act["consumerConfidence"], "dtg": cur_inp["debtToGDP"],
              "gold": cur_act.get("goldPrice", 2900)}
        # Build baseline as input series (24 constant quarters) so it matches
        # stress test and Monte Carlo code paths exactly — same lags, same noise, same equity trend.
        baseline_series = [{**p, "q": f"Q{q+1}"} for q in range(24)]
        results = simulate(baseline_series[0], baseline_series, ic)

        # ── Earnings Model ──
        st.divider()
        eq_mode = st.radio("S&P 500 Model", ["Macro Only", "Earnings Only", "Compare Both"], index=2,
                           help="Macro: GDP-trend driven. Earnings: EPS consensus + P/E multiples.")

        if eq_mode != "Macro Only":
            with st.expander("📊 Earnings Assumptions", expanded=False):
                auto_consensus = consensus_from_macro(
                    cur_act["gdpGrowth"], cur_act["inflation"],
                    p["fedRate"], p["productivityGrowth"], cur_act["sp500Index"])
                st.caption(f"Auto-derived from macro state. Override below:")
                e_trailing = st.number_input("Trailing EPS ($)", 100.0, 400.0, fl(auto_consensus.trailing_eps), 5.0)
                e_y1 = st.slider("EPS Growth Y1 (%)", -10.0, 25.0, fl(auto_consensus.eps_growth_y1), 0.5)
                e_y2 = st.slider("EPS Growth Y2 (%)", -5.0, 20.0, fl(auto_consensus.eps_growth_y2), 0.5)
                e_y3 = st.slider("EPS Growth Y3 (%)", -5.0, 15.0, fl(auto_consensus.eps_growth_y3), 0.5)
                e_lt = st.slider("EPS Growth Y4-6 (%)", 0.0, 12.0, fl(auto_consensus.eps_growth_y4_6), 0.5)
                e_bb = st.slider("Buyback Yield (%)", 0.0, 5.0, fl(auto_consensus.buyback_yield), 0.25)
                e_erp = st.slider("Equity Risk Premium (%)", 2.0, 8.0, fl(auto_consensus.equity_risk_premium), 0.25)

                consensus = EarningsConsensus(
                    trailing_eps=e_trailing, eps_growth_y1=e_y1, eps_growth_y2=e_y2,
                    eps_growth_y3=e_y3, eps_growth_y4_6=e_lt,
                    buyback_yield=e_bb, equity_risk_premium=e_erp)
            sp_earnings = earnings_equity_model(consensus, results, cur_act["sp500Index"])
            for i, val in enumerate(sp_earnings):
                results[i].sp500Earnings = val

        # ── Stress Test ──
        st.divider()
        shock_profiles = get_shock_profiles()
        enable_stress = st.checkbox("🔥 Enable Stress Test", value=False,
                                     help="Inject a crisis shock into the forward projection")
        stress_results = None
        stress_df = None
        stress_nopol_results = None
        stress_nopol_df = None
        if enable_stress and shock_profiles:
            shock_id = st.selectbox("Crisis Type",
                list(shock_profiles.keys()),
                format_func=lambda k: shock_profiles[k].name)
            shock = shock_profiles[shock_id]
            st.caption(shock.description)
            onset_q = st.slider("Shock Onset (quarter)", 1, 18, 4,
                                help="Which quarter the crisis begins")
            severity = st.slider("Severity", 0.25, 2.0, 1.0, 0.25,
                                 help="1.0 = historical magnitude. 2.0 = twice as severe.")

            st.markdown("##### Policy Response Controls")
            fed_aggr = st.slider("Fed Aggressiveness", 0.0, 2.0, 1.0, 0.25,
                key="fed_aggr",
                help="0 = Fed does nothing. 1 = standard Taylor. 2 = double-speed response.")
            fiscal_str = st.slider("Fiscal Stimulus", 0.0, 2.0, 1.0, 0.25,
                key="fisc_str",
                help="0 = no auto-stabilizers. 1 = normal. 2 = aggressive counter-cyclical spending.")

            stressed_series = apply_shock_to_params(p, shock, onset_q - 1, severity, 24)

            # Run 1: Stressed with NO policy response
            stress_nopol_results = simulate(stressed_series[0], stressed_series, ic,
                                            taylor_enabled=False, fiscal_response=0.0)
            stress_nopol_df = results_to_df(stress_nopol_results)

            # Run 2: Stressed WITH adjustable policy response
            stress_results = simulate(stressed_series[0], stressed_series, ic,
                                       taylor_enabled=(fed_aggr > 0),
                                       fiscal_response=fiscal_str)
            # Override Taylor aggressiveness via scaling the adjustment
            if fed_aggr != 1.0 and fed_aggr > 0:
                for i, r in enumerate(stress_results):
                    base_taylor = stress_results[i].taylorAdj
                    scaled = base_taylor * fed_aggr
                    stress_results[i].taylorAdj = round(scaled, 3)
                    stress_results[i].effRate = round(
                        max(0, stressed_series[i]["fedRate"] + scaled), 3)
            stress_df = results_to_df(stress_results)

        # ── Monte Carlo ──
        st.divider()
        enable_mc = st.checkbox("📊 Probability Cones (Monte Carlo)", value=False,
                                help="Run 200 simulations with random input noise to show 10/50/90 percentile bands")
        mc_pctiles = None
        if enable_mc:
            n_sims = st.select_slider("Simulations", [50, 100, 200, 500], value=200)
            with st.spinner(f"Running {n_sims} Monte Carlo simulations..."):
                mc_raw = monte_carlo_simulate(p, ic, n_sims=n_sims, n_quarters=24)
                mc_pctiles = compute_percentiles(mc_raw)

        df = results_to_df(results)
        L = results[-1]
        ref = {"gdp": cur_act["gdpGrowth"], "inf": cur_act["inflation"], "u": cur_act["unemployment"],
               "sp": cur_act["sp500Index"], "by": cur_act["bondYield10Y"], "fx": cur_act["currencyIndex"], "wg": 3.5}
        # Store for Data tab
        data_source = src
        policy_inputs = p.copy()
        initial_state = cur_act.copy()

    elif mode == "🎛️ What-If Scenarios":
        preset = st.selectbox("Preset", list(WHATIF_PRESETS.keys()))
        p = WHATIF_PRESETS[preset].copy()
        st.divider()
        p["fedRate"] = st.slider("Fed Rate (%)", 0.0, 12.0, fl(p["fedRate"]), 0.25)
        p["moneySupplyGrowth"] = st.slider("M2 Growth (%)", -2.0, 20.0, fl(p["moneySupplyGrowth"]), 0.5)
        p["govSpending"] = st.slider("Gov Spend ($T)", 4.0, 12.0, fl(p["govSpending"]), 0.1)
        p["taxRate"] = st.slider("Tax Rate (%)", 10.0, 40.0, fl(p["taxRate"]), 0.5)
        p["debtToGDP"] = st.slider("Debt/GDP (%)", 60.0, 200.0, fl(p["debtToGDP"]), 1.0)
        p["tariffRate"] = st.slider("Tariff Rate (%)", 0.0, 40.0, fl(p["tariffRate"]), 0.5)
        p["oilPrice"] = st.slider("Oil ($/bbl)", 30.0, 180.0, fl(p["oilPrice"]), 1.0)
        p["productivityGrowth"] = st.slider("Productivity (%)", 0.0, 5.0, fl(p["productivityGrowth"]), 0.1)
        p["laborForceGrowth"] = st.slider("Labor Force (%)", -1.0, 3.0, fl(p["laborForceGrowth"]), 0.1)
        baseline_series = [{**p, "q": f"Q{q+1}"} for q in range(24)]
        wi_ic = {"g": 2.5, "i": 3.2, "u": 4.0, "fx": 100, "sp": 5200, "by": 4.3,
                 "tb": -780, "cc": 98, "dtg": p["debtToGDP"], "gold": 2900}
        results = simulate(baseline_series[0], baseline_series, wi_ic)

        # ── Earnings Model ──
        st.divider()
        eq_mode = st.radio("S&P 500 Model", ["Macro Only", "Earnings Only", "Compare Both"], index=0,
                           key="whatif_eq", help="Macro: GDP-trend. Earnings: EPS consensus + P/E.")
        if eq_mode != "Macro Only":
            with st.expander("📊 Earnings Assumptions", expanded=False):
                auto_c = consensus_from_macro(2.5, 3.2, p["fedRate"], p["productivityGrowth"], 5200)
                e_y1 = st.slider("EPS Growth Y1 (%)", -10.0, 25.0, fl(auto_c.eps_growth_y1), 0.5, key="wi_y1")
                e_y2 = st.slider("EPS Growth Y2 (%)", -5.0, 20.0, fl(auto_c.eps_growth_y2), 0.5, key="wi_y2")
                e_lt = st.slider("EPS Growth Y4-6 (%)", 0.0, 12.0, fl(auto_c.eps_growth_y4_6), 0.5, key="wi_lt")
                consensus = EarningsConsensus(trailing_eps=auto_c.trailing_eps,
                    eps_growth_y1=e_y1, eps_growth_y2=e_y2, eps_growth_y3=(e_y1+e_lt)/2,
                    eps_growth_y4_6=e_lt, equity_risk_premium=auto_c.equity_risk_premium)
            sp_earn = earnings_equity_model(consensus, results, 5200)
            for i, val in enumerate(sp_earn):
                results[i].sp500Earnings = val

        df = results_to_df(results)
        L = results[-1]
        ref = {"gdp": 2.5, "inf": 3.2, "u": 4.0, "sp": 5200, "by": 4.3, "fx": 100, "wg": 3.5}
        # Store for Data tab
        data_source = f"What-If Preset: {preset}"
        policy_inputs = p.copy()
        initial_state = {"gdpGrowth": 2.5, "inflation": 3.2, "unemployment": 4.0,
                         "currencyIndex": 100, "sp500Index": 5200, "goldPrice": 2900,
                         "bondYield10Y": 4.3, "tradeBalance": -780, "consumerConfidence": 98}
        stress_df = None
        stress_results = None
        stress_nopol_df = None
        stress_nopol_results = None
        mc_pctiles = None
    else:
        results = df = L = ref = None
        stress_df = None
        stress_results = None
        stress_nopol_df = None
        stress_nopol_results = None
        mc_pctiles = None


# ═══════════════════════════════════════════════════════════════
# FORECAST / WHAT-IF CONTENT
# ═══════════════════════════════════════════════════════════════
if mode in ["🔮 Forecast from Now", "🎛️ What-If Scenarios"]:
    tab_list = ["◉ Dashboard", "📈 Growth", "🔥 Prices", "💹 Markets", "📋 Data", "❓ Help"]
    if stress_df is not None:
        tab_list.insert(4, "🔥 Stress Test")
    tabs = st.tabs(tab_list)
    if stress_df is not None:
        tab_d, tab_g, tab_p, tab_m, tab_stress, tab_data, tab_h = tabs
    else:
        tab_d, tab_g, tab_p, tab_m, tab_data, tab_h = tabs
        tab_stress = None

    with tab_d:
        if "Forecast" in mode:
            st.info(f"Projecting 24 quarters from **{src}** conditions. Adjust sliders to test policy changes.", icon="🔮")
        show_kpis(df, L, ref)
        # Monte Carlo cones on dashboard
        if mc_pctiles is not None:
            st.markdown("##### Probability Cones (Monte Carlo)")
            for col, title, color in [
                ("gdpGrowth", "GDP Growth (%)", CL["cy"]),
                ("sp500Index", "S&P 500", CL["gn"]),
                ("inflation", "Inflation (%)", CL["rd"]),
                ("unemployment", "Unemployment (%)", CL["am"]),
            ]:
                st.plotly_chart(plot_cone(df, col, title, color, mc_pctiles), use_container_width=True)

    with tab_g:
        if mc_pctiles is not None:
            for c, t, clr in [("gdpGrowth", "GDP Growth", CL["cy"]),
                               ("unemployment", "Unemployment", CL["am"]),
                               ("wageGrowth", "Wage Growth", CL["pk"]),
                               ("consumerConfidence", "Consumer Confidence", CL["gn"])]:
                st.plotly_chart(plot_cone(df, c, t, clr, mc_pctiles), use_container_width=True)
        else:
            for c, t, clr in [("gdpGrowth", "GDP Growth (%)", CL["cy"]),
                               ("unemployment", "Unemployment (%)", CL["am"])]:
                st.plotly_chart(plot_pct(df, c, t, clr, t), use_container_width=True)
            st.plotly_chart(plot_pct(df, "wageGrowth", "Wage Growth (%)", CL["pk"], "Wage Growth (%)"), use_container_width=True)
            st.plotly_chart(plot_market(df, "consumerConfidence", "Consumer Confidence", CL["gn"], "Consumer Confidence"), use_container_width=True)

    with tab_p:
        st.plotly_chart(plot_lines(df, ["inflation", "inflExpectations", "wageGrowth"],
            ["CPI", "Expectations", "Wages"], [CL["rd"], CL["pu"], CL["pk"]],
            "Inflation · Expectations · Wages"), use_container_width=True)

        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(x=df["label"], y=df["bondYield10Y"], name="10Y Yield",
                                  line=dict(color=CL["bl"], width=2)), secondary_y=False)
        fig2.add_trace(go.Scatter(x=df["label"], y=df["debtToGDP"], name="Debt/GDP",
                                  line=dict(color=CL["or"], width=2)), secondary_y=True)
        fig2.update_layout(**DARK_LAYOUT, height=350, title="Bonds · Debt")
        st.plotly_chart(fig2, use_container_width=True)

    with tab_m:
        # S&P 500 — dual model comparison
        has_earnings = df["sp500Earnings"].max() > 0
        if has_earnings:
            fig_sp = go.Figure()
            fig_sp.add_trace(go.Scatter(x=df["label"], y=df["sp500Index"], mode="lines",
                name="Macro Model (GDP-trend)", line=dict(color=CL["gn"], width=2.5)))
            fig_sp.add_trace(go.Scatter(x=df["label"], y=df["sp500Earnings"], mode="lines",
                name="Earnings Model (EPS×P/E)", line=dict(color=CL["am"], width=2.5, dash="dot")))
            fig_sp.update_layout(**DARK_LAYOUT, height=350, title="S&P 500 — Macro vs Earnings Model")
            st.plotly_chart(fig_sp, use_container_width=True)

            # Summary comparison
            c1, c2 = st.columns(2)
            macro_end = df["sp500Index"].iloc[-1]
            earn_end = df["sp500Earnings"].iloc[-1]
            macro_start = df["sp500Index"].iloc[0]
            earn_start = df["sp500Earnings"].iloc[0]
            c1.metric("Macro Model (6Y)", f"{macro_end:,.0f}",
                      f"{(macro_end/macro_start - 1)*100:+.1f}% total")
            c2.metric("Earnings Model (6Y)", f"{earn_end:,.0f}",
                      f"{(earn_end/earn_start - 1)*100:+.1f}% total")
            spread = earn_end - macro_end
            st.caption(f"Spread: **{spread:+,.0f}** points ({spread/macro_end*100:+.1f}%). "
                       f"{'Earnings model more bullish' if spread > 0 else 'Macro model more bullish'}. "
                       f"Divergence reflects the gap between fundamentals-driven and sentiment/earnings-driven valuations.")
        else:
            st.plotly_chart(plot_market(df, "sp500Index", "S&P 500", CL["gn"], "S&P 500 (Macro Model)"), use_container_width=True)

        # DXY
        if mc_pctiles is not None:
            st.plotly_chart(plot_cone(df, "currencyIndex", "DXY", CL["cy"], mc_pctiles), use_container_width=True)
        else:
            st.plotly_chart(plot_market(df, "currencyIndex", "DXY", CL["cy"], "USD Index (DXY)"), use_container_width=True)
        # Gold
        if mc_pctiles is not None:
            st.plotly_chart(plot_cone(df, "goldPrice", "Gold", CL["am"], mc_pctiles), use_container_width=True)
        else:
            st.plotly_chart(plot_market(df, "goldPrice", "Gold", CL["am"], "Gold ($/oz)"), use_container_width=True)
        # Trade Balance
        if mc_pctiles is not None:
            st.plotly_chart(plot_cone(df, "tradeBalance", "Trade Balance", CL["pk"], mc_pctiles), use_container_width=True)
        else:
            st.plotly_chart(plot_market(df, "tradeBalance", "Trade Balance ($B)", CL["pk"], "Trade Balance ($B)"), use_container_width=True)

    # ── STRESS TEST TAB ──
    if tab_stress is not None and stress_df is not None:
        with tab_stress:
            st.markdown(f"### 🔥 Stress Test: {shock_profiles[shock_id].name}")
            st.caption(f"Shock onset: Q{onset_q} | Severity: {severity}x | "
                       f"Fed response: {fed_aggr}x | Fiscal: {fiscal_str}x | "
                       f"Duration: {shock.n_quarters}Q + recovery")

            # Three-curve charts
            for var_key, var_label, color_base in [
                ("gdpGrowth", "GDP Growth (%)", CL["cy"]),
                ("unemployment", "Unemployment (%)", CL["am"]),
                ("inflation", "Inflation (%)", CL["pk"]),
                ("sp500Index", "S&P 500", CL["gn"]),
                ("goldPrice", "Gold ($/oz)", CL["am"]),
                ("bondYield10Y", "10Y Yield (%)", CL["bl"]),
                ("currencyIndex", "DXY", CL["cy"]),
            ]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["label"], y=df[var_key], mode="lines",
                    name="Baseline", line=dict(color=color_base, width=2)))
                fig.add_trace(go.Scatter(x=stress_nopol_df["label"], y=stress_nopol_df[var_key],
                    mode="lines", name="Stressed (No Response)",
                    line=dict(color=CL["rd"], width=2, dash="dot")))
                fig.add_trace(go.Scatter(x=stress_df["label"], y=stress_df[var_key],
                    mode="lines", name="Stressed (With Policy)",
                    line=dict(color=CL["am"], width=2.5)))
                fig.add_vrect(x0=f"Q{onset_q}", x1=f"Q{min(onset_q + shock.n_quarters, 24)}",
                              fillcolor="rgba(255,82,82,0.08)", line_width=0)
                fig.update_layout(**DARK_LAYOUT, height=280, title=var_label)
                st.plotly_chart(fig, use_container_width=True)

            # Effective Fed Rate chart
            fig_fed = go.Figure()
            fig_fed.add_trace(go.Scatter(x=df["label"], y=df["effRate"], mode="lines",
                name="Baseline", line=dict(color=CL["bl"], width=2)))
            fig_fed.add_trace(go.Scatter(x=stress_nopol_df["label"], y=stress_nopol_df["effRate"],
                mode="lines", name="No Response (flat)",
                line=dict(color=CL["rd"], width=2, dash="dot")))
            fig_fed.add_trace(go.Scatter(x=stress_df["label"], y=stress_df["effRate"],
                mode="lines", name=f"Taylor Response ({fed_aggr}x)",
                line=dict(color=CL["am"], width=2.5)))
            fig_fed.add_vrect(x0=f"Q{onset_q}", x1=f"Q{min(onset_q + shock.n_quarters, 24)}",
                              fillcolor="rgba(255,82,82,0.08)", line_width=0)
            fig_fed.update_layout(**DARK_LAYOUT, height=280,
                title="Effective Fed Rate — Policy Response Timing")
            st.plotly_chart(fig_fed, use_container_width=True)

            # Impact summary at trough
            st.markdown("##### Impact at Trough (worst GDP quarter)")
            stress_nopol_gdp = [r.gdpGrowth for r in stress_nopol_results]
            worst_q = int(np.argmin(stress_nopol_gdp))

            def _v(r_list, q, k):
                v = getattr(r_list[q], k)
                if isinstance(v, float) and abs(v) >= 100:
                    return f"{v:,.0f}"
                elif isinstance(v, float):
                    return f"{v:.1f}"
                return str(v)

            vars_to_show = [
                ("GDP Growth", "gdpGrowth", "%"), ("Unemployment", "unemployment", "%"),
                ("Inflation", "inflation", "%"), ("S&P 500", "sp500Index", ""),
                ("Gold", "goldPrice", ""), ("10Y Yield", "bondYield10Y", "%"),
                ("Fed Rate (eff)", "effRate", "%"),
            ]
            impact_rows = []
            for label, key, unit in vars_to_show:
                b = getattr(results[worst_q], key)
                np_ = getattr(stress_nopol_results[worst_q], key)
                wp = getattr(stress_results[worst_q], key)
                fmt = lambda v: f"{v:,.0f}" if abs(v) >= 100 else f"{v:.2f}"
                impact_rows.append({
                    "Variable": label,
                    "Baseline": f"{fmt(b)}{unit}",
                    "No Response": f"{fmt(np_)}{unit}",
                    "With Policy": f"{fmt(wp)}{unit}",
                    "Policy Effect": f"{wp - np_:+.2f}" if abs(wp) < 100 else f"{wp - np_:+,.0f}",
                })
            st.dataframe(pd.DataFrame(impact_rows), use_container_width=True, hide_index=True)
            st.caption(f"Trough: Q{worst_q + 1} | "
                       f"No-response regime: {stress_nopol_results[worst_q].regime} (FCI {stress_nopol_results[worst_q].fci:.2f}) | "
                       f"With-policy regime: {stress_results[worst_q].regime} (FCI {stress_results[worst_q].fci:.2f})")

    with tab_data:
        st.markdown("### 📋 Input Data Used for This Forecast")
        st.caption(f"Source: **{data_source}**")

        col_pol, col_state = st.columns(2)

        with col_pol:
            st.markdown("##### Policy Inputs (slider values)")
            pol_display = {
                "Fed Funds Rate (%)": policy_inputs.get("fedRate"),
                "Gov Spending ($T)": policy_inputs.get("govSpending"),
                "Tax Rate (%)": policy_inputs.get("taxRate"),
                "M2 Growth (%)": policy_inputs.get("moneySupplyGrowth"),
                "Tariff Rate (%)": policy_inputs.get("tariffRate"),
                "Oil Price ($/bbl)": policy_inputs.get("oilPrice"),
                "Labor Force Growth (%)": policy_inputs.get("laborForceGrowth"),
                "Productivity Growth (%)": policy_inputs.get("productivityGrowth"),
                "Debt/GDP (%)": policy_inputs.get("debtToGDP"),
            }
            st.dataframe(
                pd.DataFrame({"Input": pol_display.keys(), "Value": pol_display.values()}),
                use_container_width=True, hide_index=True,
            )

        with col_state:
            st.markdown("##### Starting Economic State")
            state_display = {
                "GDP Growth (%)": initial_state.get("gdpGrowth"),
                "Inflation / CPI (%)": initial_state.get("inflation"),
                "Unemployment (%)": initial_state.get("unemployment"),
                "S&P 500": initial_state.get("sp500Index"),
                "Gold ($/oz)": initial_state.get("goldPrice"),
                "10Y Yield (%)": initial_state.get("bondYield10Y"),
                "DXY (USD Index)": initial_state.get("currencyIndex"),
                "Trade Balance ($B)": initial_state.get("tradeBalance"),
                "Consumer Confidence": initial_state.get("consumerConfidence"),
            }
            st.dataframe(
                pd.DataFrame({"Variable": state_display.keys(), "Value": state_display.values()}),
                use_container_width=True, hide_index=True,
            )

        # Engine-derived adaptive parameters
        st.markdown("##### Engine-Derived Parameters (auto from starting state)")
        r0 = results[0]
        ada_display = {
            "Adaptive NAIRU": f"{max(3.5, min(6, initial_state.get('unemployment', 4) * 0.85)):.1f}%",
            "Inflation Anchor": f"{max(1.5, min(8, initial_state.get('inflation', 3) * 0.9)):.1f}%",
            "Starting Regime": r0.regime,
            "Starting FCI": f"{r0.fci:.3f}",
            "Fiscal Multiplier": f"{r0.fiscalMultiplier:.2f}x",
        }
        st.dataframe(
            pd.DataFrame({"Parameter": ada_display.keys(), "Value": ada_display.values()}),
            use_container_width=True, hide_index=True,
        )

        # Full projection table (downloadable)
        st.markdown("##### Full Projection (24 quarters)")
        proj_cols = ["label", "gdpGrowth", "inflation", "unemployment", "sp500Index",
                     "goldPrice", "bondYield10Y", "currencyIndex", "tradeBalance",
                     "wageGrowth", "consumerConfidence", "fci", "regime"]
        proj_df = df[proj_cols].copy()
        proj_df.columns = ["Quarter", "GDP%", "CPI%", "U%", "S&P", "Gold", "10Y%",
                           "DXY", "Trade$B", "Wages%", "Conf", "FCI", "Regime"]
        st.dataframe(proj_df, use_container_width=True, hide_index=True)

        # CSV download
        csv = proj_df.to_csv(index=False)
        st.download_button("📥 Download Projection CSV", csv, "macroscope_forecast.csv", "text/csv")

    with tab_h:
        st.markdown("""## MacroScope V6 — Technical Manual

---

### 1. Platform Overview

MacroScope is a macroeconomic policy simulation platform that models how changes in monetary policy,
fiscal policy, trade policy, and external shocks propagate through the US economy. It projects
9 macro indicators over 24 quarters (6 years) using a 21-channel transmission engine with
90+ calibrated coefficients, validated against 45 years of US economic history (1979–2024).

**Three operating modes:**
- **🔮 Forecast from Now** — Loads real-time US data, projects forward with adjustable policy overrides
- **🎛️ What-If Scenarios** — Hypothetical policy experiments from preset starting points
- **🔬 Historical Backtest** — Validates model against 5 real crisis periods

---

### 2. Data Sources

#### 2.1 Current Economic State (Forecast Mode)
Three-tier fallback strategy:
1. **FRED Live** — Public CSV endpoint `fred.stlouisfed.org/graph/fredgraph.csv` (no API key)
   - Series: FEDFUNDS, CPIAUCSL (→ YoY%), UNRATE, DGS10, M2SL (→ YoY%), GFDEGDQ188S
2. **Local Cache** — `~/.macroscope_cache.json`, auto-saved on successful FRED fetch, expires 90 days
3. **Hardcoded Defaults** — Q1 2025 snapshot in `current_state.py`, manually curated from FRED/BEA/BLS

**Variables NOT from FRED** (require manual update in `current_state.py`):
- Gov Spending ($T): BEA national accounts
- Tax Rate (%): CBO effective federal revenue/GDP
- Tariff Rate (%): USITC weighted average
- Oil Price ($/bbl): EIA WTI spot
- Labor Force Growth (%): BLS civilian labor force YoY
- Productivity Growth (%): BLS nonfarm business productivity
- S&P 500, Gold, DXY, Trade Balance, Consumer Confidence: manual/Yahoo/ICE/BEA/Conference Board

#### 2.2 Historical Data (Backtest Mode)
112 quarters of manually curated US macro data across 5 crisis periods, stored in `data.py`:
- **Volcker Crisis 1979–83** (20 quarters): Fed rate 10→18.5→8.5%, CPI 9.9→3.7%
- **Gulf War 1989–93** (20 quarters): Oil shock, mild recession, rate cuts 9.75→3%
- **Dot-Com 2000–03** (16 quarters): Tech bust, 9/11, rate cuts 5.75→1%
- **GFC 2007–10** (16 quarters): Subprime, Lehman, GDP -8.5%, unemployment to 10.7%
- **Modern 2015–24** (40 quarters): ZLB exit, trade war, COVID, inflation surge

Input data sourced from FRED (FEDFUNDS, CPIAUCSL, UNRATE, DGS10, M2SL), BEA (GDP, trade),
BLS (labor, productivity), EIA (oil), ICE (DXY), S&P Dow Jones (S&P 500 index).

---

### 3. Simulation Engine (engine.py)

#### 3.1 Architecture
The engine runs a quarterly time-step simulation. Each quarter:
1. Reads policy inputs (rates, spending, tariffs, oil, etc.)
2. Computes quarter-over-quarter deltas (rate *changes*, not just levels)
3. Runs 21 transmission channels sequentially
4. Outputs 9 macro indicators + internal state (FCI, regime, expectations)

#### 3.2 Lag Structure
Policy changes take effect gradually:
- Monetary lag (lM): starts at 0.70, ramps to 0.92 over ~12 quarters
- Fiscal lag (lF): starts at 0.65, ramps to 0.95
- Structural lag (lS): starts at 0.40, ramps to 0.85

These are faster than the original Friedman "long and variable lags" because
modern financial markets transmit monetary policy faster via forward guidance and QE.

#### 3.3 Era-Adaptive Design (no hardcoded dates)
The engine derives key structural parameters from starting conditions:
- **NAIRU** = startingUnemployment × 0.85, clamped [3.5, 6.0]
- **Inflation Anchor** = startingCPI × 0.9, clamped [1.5, 8.0]
- **Regime** classified endogenously from GDP/unemployment via smoothstep functions

This means the same engine works for Volcker (NAIRU≈4.9, anchor≈8.0) and
Modern era (NAIRU≈3.6, anchor≈2.5) without any era-specific code.

#### 3.4 GDP Channel
```
GDP = potential_growth + fiscal_impulse + rate_impulse + rate_level_drag
      + oil_impulse + oil_level_drag - FCI_drag + money_supply
      + wealth_effect + inventory_cycle - global_spillover
      + fiscal_auto_stabilizer
```
**Key innovation:** Uses quarter-over-quarter rate CHANGES for impulse (captures the
shock of rate hikes) plus sustained rate LEVEL drag for rates above 6%. Previous versions
only used level deltas from Q1 baseline, which produced flat GDP regardless of massive
rate swings during Volcker.

- **Fiscal multiplier**: Regime-dependent (1.0 normal, 1.8 recession, 0.4 overheating)
- **ZLB constraint**: Below 0.5% rate, monetary transmission drops to 15% effectiveness
- **Asymmetric response**: Negative GDP shocks amplified by 1.6x (recession asymmetry)

#### 3.5 Inflation Channel
```
Inflation = anchor + Phillips_curve + demand_pull + oil_passthrough
            + tariff_passthrough + money_supply + expectations_feedback
            + FX_passthrough + wage_pressure + rate_disinflation
```
- **Non-linear Phillips Curve**: Convex below NAIRU (unemployment at 3% generates
  much more inflation than unemployment at 4%). Linear above NAIRU.
- **Adaptive expectations**: Blend speed 25% normal, 40% when inflation > anchor+3%
- **De-anchoring**: When CPI > 6% for 4+ quarters, expectations accelerate away from anchor
- **Rate → CPI channel** (`inf_rate=0.093`): Direct monetary restraint → disinflation.
  Critical for modeling the Volcker disinflation (14% → 4% CPI).

#### 3.6 Unemployment Channel (7-channel model)
```
Unemployment = startU + NAIRU_shift
               + Okun's_law(output_gap)           # Ch1: traditional
               + GDP_level_tracking(GDP)            # Ch2: when GDP<1% → sharp rise
               + FCI_stress(fci)                    # Ch3: credit tightening → layoffs
               + rate_drag(fedRate)                 # Ch4: high rates → housing/auto losses
               + labor_collapse(laborForceGrowth)   # Ch5: COVID-type labor shock
               + tariff_drag + labor_supply         # Ch6-7: smaller effects
```
**Key innovation:** Channel 2 tracks the GDP *level* directly, not just the dampened
output gap. When GDP drops below 1%, unemployment rises 0.4pp per point. Below 0%,
rises 0.6pp per point. This is why GFC unemployment now tracks from 4.5% → 7.0%
(actual peaked at 10.7%) instead of staying stuck at 5%.

- **Hysteresis**: After unemployment > 6% for 4+ quarters, NAIRU drifts upward (scarring)
- **Crisis Okun's**: Amplifies from 0.12 (normal) to 0.27 (crisis) based on FCI

#### 3.7 Currency (DXY) Channel
```
DXY = baseline + rate_differential × rate_sensitivity + FCI_effect
      - money_dilution + current_account + GDP_cycle_feedback
      + PPP_drift + momentum
```
- **Rate sensitivity** (`fx_rb=1.45`): Strong positive relationship with interest rates
- **PPP drift**: High inflation gradually weakens currency over time
- **GDP cycle**: Strong growth attracts capital → stronger dollar

#### 3.8 Equity (S&P 500) Channel — Forward-Looking
```
S&P_change = GDP_effect + productivity_effect - rate_trajectory_effect
             + money_supply + FCI_crash - panic_multiplier
             + momentum + mean_reversion_to_actual + trend_growth
```
**Key innovation:** Uses rate *trajectory* (acceleration/deceleration) not rate level.
When rate hikes decelerate, markets rally in anticipation — this is why the Volcker
S&P model now recovers from 50 to 160 in 1983 (actual: 165) as markets anticipated
the end of the tightening cycle.

- **Panic multiplier**: When GDP < -1% AND FCI > 0.3, equity losses amplified 384x
- **Mean-reversion** (`eq_mr=0.15`): In backtest mode, pulls toward actual S&P data
- **Trend growth**: In forward mode, ≈ (GDP + CPI) × 0.25 per quarter (~5% annualized real)

#### 3.9 Bond (10Y Yield) Channel — Forward-Looking Term Structure
```
10Y = expected_average_future_rate + term_premium + flight_to_quality
```
**Key innovation:** Markets expect the fed rate to normalize toward neutral
(inflation_anchor + 2%). When the fed rate is 18% (Volcker), the 10Y doesn't
go to 18% — it prices in normalization toward ~10%, producing ~13% (matching reality).

- **Neutral rate** = inflation anchor + 2% real
- **Normalization speed**: 15-65% per year depending on rate gap
- **Term premium**: Base 0.76 + inflation volatility + fiscal risk + debt premium
- **Flight to quality**: During high FCI, 10Y drops (safe haven buying)
- **Smoothing**: 60% previous / 40% target per quarter (yields don't jump)

#### 3.10 Gold Channel
```
Gold = previous + real_rate_effect + inflation_hedge + safe_haven
       + dollar_inverse + debt_debasement + central_bank_trend
```
- **Real rates**: -25 × (10Y - inflExpectations - 1.0). Gold rises when real rates < 1%
- **Inflation hedge**: +8 × max(0, expectations - 2.5%)
- **Safe haven**: +40 × max(0, FCI). Convex crisis demand
- **Dollar inverse**: -6 × (DXY change from baseline)
- **Debt debasement**: +3 × max(0, debt/GDP - 120%)
- **Central bank trend**: +0.5%/quarter (~2%/yr) in forward mode. Reflects structural
  central bank gold accumulation (China, India, Russia diversifying reserves)

#### 3.11 Taylor Rule (Endogenous Fed Response)
Active during stress tests. The Fed adjusts rates based on:
```
taylor_adj = 0.25 × [0.5 × (inflation - anchor) + 0.5 × output_gap]
effective_rate = fed_rate + taylor_adj
```
- 25% response speed per quarter (gradual, realistic)
- Clamped to ±1.0pp maximum quarterly adjustment
- User-adjustable aggressiveness slider (0x to 2x)

#### 3.12 Fiscal Auto-Stabilizer
Active during stress tests:
```
if GDP < 1.5%: fiscal_boost = fiscal_response × 0.12 × (1.5 - GDP)
if GDP > 3.5%: fiscal_drag = -fiscal_response × 0.04 × (GDP - 3.5)
```
Uses previous quarter's GDP (lagged response, realistic). User-adjustable strength (0x to 2x).

#### 3.13 FCI (Financial Conditions Index)
Composite stress indicator combining:
- Equity market decline → tighter
- High debt/GDP → tighter
- Low GDP growth → tighter
- Yield curve inversion (fed rate > 10Y + 0.5) → tighter
- Equity crash detection (>10% drawdown in 2 quarters) → tighter

Dynamic range: [-0.5, +2.5]. Values above 0.3 activate crisis amplifiers in GDP and unemployment.

#### 3.14 Other Channels
- **Wages**: Adaptive, driven by NAIRU gap + productivity + inflation persistence
- **Consumer Confidence**: Unemployment, wealth effect, inflation, FCI, GDP
- **Housing**: Mortgage rate (10Y + 1.7%), consumer confidence
- **Trade Balance**: FX level, tariffs (with J-curve), fiscal spending, oil imports, GDP cycle
- **Debt/GDP**: Domar dynamics with tipping points at 150% and 180%

---

### 4. Earnings Model (earnings.py)

#### 4.1 Purpose
Alternative S&P 500 valuation model. While the macro model drives equities from GDP trends,
the earnings model uses a bottom-up EPS × P/E framework.

#### 4.2 Data Sources — IMPORTANT LIMITATION
**The EPS estimates are NOT sourced from real analyst consensus data** (Bloomberg, IBES, FactSet).
They are auto-derived from macro conditions using this formula:

```
Y1_EPS_growth ≈ nominal_GDP + productivity_premium + buyback_boost
Y2_growth ≈ 0.9 × nominal_GDP + buyback_boost
Y3_growth ≈ 0.85 × nominal_GDP + 0.8 × buyback_boost
Y4-6_growth ≈ 0.8 × nominal_GDP + 0.5 × buyback_boost
```

Where:
- nominal_GDP = real GDP growth + inflation (e.g., 2.4% + 2.8% = 5.2%)
- productivity_premium = max(0, productivity - 1.0) × 2
- buyback_boost = 2.0% (S&P 500 average share count reduction)

**To use real consensus data**: Override the auto-derived values in the "Earnings Assumptions"
expander with actual Y1/Y2/Y3 EPS growth estimates from your Bloomberg/FactSet terminal.

#### 4.3 P/E Multiple
Derived via Gordon Growth Model:
```
base_P/E = 1 / (required_return - eps_growth)
required_return = 10Y_yield + equity_risk_premium (default 4.5%)
```
Then adjusted for macro conditions:
- FCI stress: -2 points per unit of FCI
- GDP above trend: +0.5 per percentage point
- 10Y above 4.5%: -0.8 per percentage point
- Clamped to [12, 30] P/E range

#### 4.4 Price Path
```
fair_value = EPS × adjusted_P/E
buyback_lift = S&P × buyback_yield / 4 (quarterly)
S&P_new = S&P + 0.08 × (fair_value - S&P) + 0.15 × momentum + buyback_lift
```
8% quarterly mean-reversion to fair value, 15% momentum carry from previous change.

---

### 5. Stress Test System (stress.py)

#### 5.1 Shock Profiles
**5 historically calibrated** (extracted from `data.py`):
- Only EXOGENOUS variables extracted: oil price, labor force, productivity, tariffs
- POLICY variables (fed rate, spending, tax, M2) excluded — engine generates response via Taylor Rule
- Source quarters documented in code for traceability

**2 hypothetical** (hand-crafted, no direct historical parallel):
- Trade War Escalation: tariffs +25pp over 8 quarters
- Stagflation: oil +$50 + productivity collapse + tariff escalation

#### 5.2 Three-Curve Comparison
For each stress test, three simulations run:
1. **Baseline** — no crisis
2. **Stressed, No Response** — crisis applied, Taylor Rule OFF, fiscal OFF
3. **Stressed, With Policy** — crisis applied, Taylor + fiscal active (adjustable)

Difference between curves 2 and 3 shows exactly how much policy intervention helps.

#### 5.3 Shock Injection Mechanics
```
Before onset: constant baseline policy
During shock: baseline + exogenous_deltas × severity
After shock: deltas decay exponentially (80%/quarter toward baseline)
```

---

### 6. Monte Carlo Simulation

#### 6.1 Noise Model
Correlated random walk on all 8 policy inputs:
```
noise_q = 0.7 × (previous_noise) + 0.3 × N(0, σ) + N(0, 0.3σ)
```
Standard deviations calibrated to historical quarterly variation:
- Fed rate: ±15bp, M2: ±1pp, Oil: ±$5, Tariffs: ±0.3pp
- Gov spending: ±$50B, Tax: ±0.2pp, Labor: ±0.15pp, Productivity: ±0.3pp

#### 6.2 Output
Runs 50-500 simulations, computes 10th/50th/90th percentile cones.
The 10-90% band represents the range of plausible outcomes given random input variation.
Baseline (white line) should sit near the median (dashed).

---

### 7. Scoring System (scoring.py)

#### 7.1 Per-Variable Score
```
score = 0.6 × MAE_score + 0.4 × directional_score
MAE_score = 100 × (1 - MAE / adaptive_scale)
directional_score = 100 × (fraction of quarters where model direction matches actual)
```

The 60/40 blend penalizes models that minimize absolute error but get the direction wrong
(e.g., predicting S&P decline when actual rallied).

#### 7.2 Era-Adaptive Scale
Each variable's scoring scale adapts to the era's volatility:
```
scale = max(base_scale, std(actual_data) × 2.5)
```
So a GDP error of 4pp in the Volcker era (where GDP swung 16pp) is scored proportionally
to a 1.5pp error in the modern era (where GDP swung 6pp).

#### 7.3 Score Interpretation
- **80-100**: Excellent tracking (rare for any macro model)
- **65-80**: Good — captures direction and approximate magnitude
- **50-65**: Fair — gets the general trend but misses amplitude or timing
- **30-50**: Poor — significant structural misfit
- **0-30**: Model fails for this variable/era combination

---

### 8. Coefficient Calibration (optimizer.py)

25 key coefficients optimized via scipy `differential_evolution` across all 5 crisis eras simultaneously.
Objective: minimize weighted-average error (GFC and Modern weighted 1.5x, others 0.8-1.0x).

Run offline:
```bash
python optimizer.py --quick    # 12 params, ~15 seconds
python optimizer.py            # 25 params, ~2.5 minutes
python optimizer.py --full     # 50 params, ~15-30 minutes
```
Output: `optimized_K.json` — copy values into `engine.py`'s K dict.

---

### 9. Known Limitations

1. **GDP amplitude dampened** — Model GDP swings are smaller than reality, especially in pre-2000 eras
2. **Gulf War unemployment** — Model doesn't capture the slow unemployment rise because GDP stays too high
3. **COVID shutdown** — Administrative lockdowns cannot be modeled; COVID GDP drop (-28%) is an outlier
4. **Sentiment-driven equity moves** — AI hype, meme stocks, speculative bubbles are outside model scope
5. **Adaptive expectations only** — No rational expectations or forward-looking consumer behavior
6. **EPS estimates synthetic** — Not from real analyst consensus; override with Bloomberg data if available
7. **DXY in Volcker** — Model undershoots the massive dollar rally (87→124) driven by capital inflows
8. **Single-country model** — No global trade partner modeling, no EM spillover effects

---

### 10. File Architecture

```
app.py             ← Streamlit UI (753 lines) — 3 modes, 6 tabs, Plotly charts
engine.py          ← Simulation engine (522 lines) — 21 channels, 90+ coefficients
earnings.py        ← Earnings-based equity model (208 lines) — EPS × P/E + buybacks
scoring.py         ← Backtest scoring (122 lines) — MAE + directional blend
data.py            ← Historical data (197 lines) — 112 quarters across 5 eras
current_state.py   ← Current state loader (203 lines) — FRED CSV → cache → offline
stress.py          ← Stress test + Monte Carlo (279 lines) — 7 crisis profiles + MC cones
optimizer.py       ← Coefficient optimizer (321 lines) — differential evolution
```

### 11. Theoretical Foundations

The engine draws on established macroeconomic theory:
- **IS-LM/AD-AS framework** for GDP-rate-inflation transmission
- **Phillips Curve** (non-linear, expectations-augmented) for inflation-unemployment tradeoff
- **Okun's Law** (state-dependent) for GDP-unemployment linkage
- **Domar debt dynamics** for fiscal sustainability
- **Gordon Growth Model** for equity valuation
- **Expectations Hypothesis** for bond term structure
- **Taylor Rule** for endogenous monetary policy response
- **Purchasing Power Parity** for long-run FX drift
- **Financial Accelerator** (simplified via FCI) for credit-cycle amplification

No DSGE structure — this is a reduced-form model optimized for practical scenario analysis
rather than theoretical consistency. Trade-off: faster computation, more intuitive controls,
but cannot claim micro-founded welfare analysis.
""")



# ═══════════════════════════════════════════════════════════════
# BACKTEST CONTENT
# ═══════════════════════════════════════════════════════════════
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
        with st.expander("Quarter-by-Quarter Detail"):
            st.dataframe(pdf.round(2), use_container_width=True, hide_index=True)

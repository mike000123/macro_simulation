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
from stress import get_shock_profiles, apply_shock_to_params, SHOCK_CONFIGS

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
        results = simulate(p, initial_conditions=ic)

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
                                     help="Inject a historical crisis shock into the forward projection")
        stress_results = None
        stress_df = None
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
            shocked_series = apply_shock_to_params(p, shock, onset_q - 1, severity, 24)
            stress_results = simulate(shocked_series[0], shocked_series, ic)
            stress_df = results_to_df(stress_results)

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
        results = simulate(p)

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
    else:
        results = df = L = ref = None
        stress_df = None
        stress_results = None


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

    with tab_g:
        # GDP and Unemployment: fill-to-zero is fine for percentages
        for c, t, clr in [("gdpGrowth", "GDP Growth (%)", CL["cy"]),
                           ("unemployment", "Unemployment (%)", CL["am"])]:
            st.plotly_chart(plot_pct(df, c, t, clr, t), use_container_width=True)
        # Wages: also a percentage
        st.plotly_chart(plot_pct(df, "wageGrowth", "Wage Growth (%)", CL["pk"], "Wage Growth (%)"), use_container_width=True)
        # Confidence: level variable (60-140 range), use market chart
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
        st.plotly_chart(plot_market(df, "currencyIndex", "DXY", CL["cy"], "USD Index (DXY)"), use_container_width=True)
        # Gold
        st.plotly_chart(plot_market(df, "goldPrice", "Gold", CL["am"], "Gold ($/oz)"), use_container_width=True)
        # Trade Balance
        st.plotly_chart(plot_market(df, "tradeBalance", "Trade Balance ($B)", CL["pk"], "Trade Balance ($B)"), use_container_width=True)

    # ── STRESS TEST TAB ──
    if tab_stress is not None and stress_df is not None:
        with tab_stress:
            st.markdown(f"### 🔥 Stress Test: {shock_profiles[shock_id].name}")
            st.caption(f"Shock onset: Q{onset_q} | Severity: {severity}x | Duration: {shock.n_quarters} quarters + recovery")

            # Comparison charts: baseline vs stressed
            for var_key, var_label, color_base, color_stress in [
                ("gdpGrowth", "GDP Growth (%)", CL["cy"], CL["rd"]),
                ("unemployment", "Unemployment (%)", CL["am"], CL["rd"]),
                ("inflation", "Inflation (%)", CL["pk"], CL["rd"]),
                ("sp500Index", "S&P 500", CL["gn"], CL["rd"]),
                ("goldPrice", "Gold ($/oz)", CL["am"], CL["rd"]),
                ("bondYield10Y", "10Y Yield (%)", CL["bl"], CL["rd"]),
                ("currencyIndex", "DXY", CL["cy"], CL["rd"]),
            ]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["label"], y=df[var_key], mode="lines",
                    name="Baseline", line=dict(color=color_base, width=2)))
                fig.add_trace(go.Scatter(x=stress_df["label"], y=stress_df[var_key], mode="lines",
                    name="Stressed", line=dict(color=color_stress, width=2.5, dash="dot")))
                # Shade the shock period
                fig.add_vrect(x0=f"Q{onset_q}", x1=f"Q{min(onset_q + shock.n_quarters, 24)}",
                              fillcolor="rgba(255,82,82,0.08)", line_width=0)
                fig.update_layout(**DARK_LAYOUT, height=280, title=var_label)
                st.plotly_chart(fig, use_container_width=True)

            # Impact summary
            st.markdown("##### Impact Summary (stressed vs baseline at trough)")
            # Find worst quarter for GDP in stress
            stress_gdp = [r.gdpGrowth for r in stress_results]
            base_gdp = [r.gdpGrowth for r in results]
            worst_q = int(np.argmin(stress_gdp))

            impact_data = {
                "Variable": ["GDP Growth", "Unemployment", "Inflation", "S&P 500", "Gold", "10Y Yield", "DXY"],
                "Baseline": [
                    f"{results[worst_q].gdpGrowth:.1f}%",
                    f"{results[worst_q].unemployment:.1f}%",
                    f"{results[worst_q].inflation:.1f}%",
                    f"{results[worst_q].sp500Index:,.0f}",
                    f"{results[worst_q].goldPrice:,.0f}",
                    f"{results[worst_q].bondYield10Y:.2f}%",
                    f"{results[worst_q].currencyIndex:.1f}",
                ],
                "Stressed": [
                    f"{stress_results[worst_q].gdpGrowth:.1f}%",
                    f"{stress_results[worst_q].unemployment:.1f}%",
                    f"{stress_results[worst_q].inflation:.1f}%",
                    f"{stress_results[worst_q].sp500Index:,.0f}",
                    f"{stress_results[worst_q].goldPrice:,.0f}",
                    f"{stress_results[worst_q].bondYield10Y:.2f}%",
                    f"{stress_results[worst_q].currencyIndex:.1f}",
                ],
                "Impact": [
                    f"{stress_results[worst_q].gdpGrowth - results[worst_q].gdpGrowth:+.1f}pp",
                    f"{stress_results[worst_q].unemployment - results[worst_q].unemployment:+.1f}pp",
                    f"{stress_results[worst_q].inflation - results[worst_q].inflation:+.1f}pp",
                    f"{stress_results[worst_q].sp500Index - results[worst_q].sp500Index:+,.0f}",
                    f"{stress_results[worst_q].goldPrice - results[worst_q].goldPrice:+,.0f}",
                    f"{stress_results[worst_q].bondYield10Y - results[worst_q].bondYield10Y:+.2f}pp",
                    f"{stress_results[worst_q].currencyIndex - results[worst_q].currencyIndex:+.1f}",
                ],
            }
            st.dataframe(pd.DataFrame(impact_data), use_container_width=True, hide_index=True)
            st.caption(f"Trough quarter: Q{worst_q + 1} | Regime: {stress_results[worst_q].regime} | FCI: {stress_results[worst_q].fci:.2f}")

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
        st.markdown("""## How to Use MacroScope

### 🔮 Forecast from Now (default)
Loads latest US macro conditions and projects 24 quarters forward.
Adjust sidebar sliders to test policy deviations. Uses FRED public CSV (no API key needed),
falls back to built-in Q1 2025 defaults if unreachable.

### 🎛️ What-If Scenarios
Hypothetical mode. Pick a preset, adjust all inputs freely.

### 🔬 Historical Backtest
Validates against 5 crisis periods (1979–2024). Era-adaptive scoring. 55–65 = institutional-grade.

---

### 📊 S&P 500: Dual Model System

Use the **S&P 500 Model** radio in the sidebar to choose **Macro Only**, **Earnings Only**, or **Compare Both**.

**Macro Model (GDP-Trend):** Equities grow with nominal GDP (real growth + inflation).
At 2.4% GDP + 2.8% CPI ≈ 5.2% annualized. This is the sustainable fundamentals floor.
Responds to policy changes but cannot capture earnings surprises or sentiment.

**Earnings Model (EPS × P/E):**
1. **EPS Path** — Quarterly trajectory from annual growth estimates (Y1/Y2/Y3/Y4-6), auto-derived from macro or manually overridden
2. **P/E Multiple** — Gordon Growth Model (1 / (required return − growth)), adjusted for:
   - FCI stress → compresses up to 2 points
   - Above-trend GDP → expands ~0.5 per point
   - 10Y above 4.5% → compresses ~0.8 per point
3. **Fair Value** = EPS × Adjusted P/E
4. **Buyback Lift** — ~2%/yr mechanical support from share reduction
5. **Price Path** — 8%/quarter mean-reversion to fair value + 15% momentum carry

**Editable inputs** (Earnings Assumptions expander): trailing EPS, forward growth rates,
buyback yield, equity risk premium. All auto-populated — override any to test your thesis.

**When models diverge:** Large spread signals either aggressive earnings estimates
(bullish consensus) or a structural shift the macro model misses (e.g. AI productivity).

---

### 🥇 Gold Model
Gold responds to five channels:
- **Real rates** (inverse) — rises when 10Y minus inflation expectations falls below 1%
- **Inflation expectations** — hedge demand above 2.5%
- **FCI (safe haven)** — crisis buying
- **Dollar weakness** — inverse DXY relationship
- **Debt/GDP concerns** — debasement fears above 120%
- **Central bank trend** — ~2%/yr structural demand from reserve diversification

---

### Charts
- **Dashboard**: GDP, CPI, unemployment, wages overlay
- **Growth**: Individual area charts for GDP, unemployment, wages, confidence
- **Prices**: Inflation vs expectations; bonds vs debt trajectory
- **Markets**: S&P (macro vs earnings), DXY, Gold, Trade Balance

### Architecture
```
app.py             ← Streamlit UI
engine.py          ← 21-channel simulation + coefficients
earnings.py        ← EPS consensus + P/E fair value model
scoring.py         ← Era-adaptive backtest scoring
data.py            ← 5 crisis datasets (112 quarters)
current_state.py   ← Current macro state (offline + FRED CSV)
```""")


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

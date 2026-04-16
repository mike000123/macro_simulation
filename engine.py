"""
M2+M3: Coefficients & Simulation Engine — engine.py
V6: Era-adaptive, 21 channels, 7 crisis-regime fixes
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict

# ╔═══════════════════════════════════════════════════════════════╗
# ║  COEFFICIENTS (Nelder-Mead calibrated + 7 crisis fixes)     ║
# ╚═══════════════════════════════════════════════════════════════╝

K = {
    # GDP — optimized
    "gdp_rate": 0.116, "gdp_rate_high": 0.015, "gdp_rate_th": 5,
    "gdp_fs": 0.8, "gdp_ft": 0.235, "gdp_fc": 3.5,
    "gdp_m": 0.041, "gdp_ta": 0.18, "gdp_o": 0.00593,
    "gdp_fci_base": 0.215, "gdp_fci_crisis": 0.291, "gdp_fci_th": 0.3,
    "gdp_p": 0.6, "gdp_l": 0.4, "gdp_pb": 2.0,
    "gdp_ic": 0.3, "gdp_w": 0.000485, "gdp_g": 0.1,
    # Inflation — optimized
    "inf_pl": 0.35, "inf_pc": 0.6, "inf_ps": 0.168,
    "inf_ta": 0.139, "inf_o": 0.05, "inf_m": 0.03,
    "inf_dc": 0.25, "inf_dl": 0.0535,
    "inf_ew": 0.259, "inf_an": 0.25, "inf_ah": 0.4,
    "inf_dt": 6, "inf_da": 0.1, "inf_fx": 0.016, "inf_wp": 0.15,
    "inf_rate": 0.0932,
    # Unemployment — optimized
    "uo_b": 0.117, "uo_c": 0.270, "uo_t": 0.4,
    "uo_rate_th": 8, "uo_rate_amp": 0.00813,
    "ul": 0.2, "ut": 0.08, "ul_crisis": 1.472,
    "uh_t": 6, "uh_d": 4, "uh_r": 0.05,
    # Currency — optimized
    "fx_rb": 1.448, "fx_rl": 0.1, "fx_f": 1.5, "fx_m": 0.472,
    "fx_ct": 170, "fx_ca": 0.05, "fx_to": 0.01,
    # Equities — optimized
    "eq_e": 2.848, "eq_p": 30, "eq_r": 225.6, "eq_m": 40,
    "eq_f": 329, "eq_pg": -1, "eq_pf": 0.3, "eq_pm": 384,
    "eq_wf": 0.005, "eq_mo": 0.5, "eq_mr": 0.154,
    # Bonds — optimized
    "bf": 0.45, "bfl": 0.14, "bi": 0.35, "bt": 0.764,
    "bv": 0.340, "bs": 0.0176, "bif": 0.384,
    # Debt
    "ds": 1.5, "dtt": 0.3, "d1": 150, "d2": 180,
    "dq": 0.3, "drs": 0.5, "da": 0.4,
    # Trade
    "tf": 2, "tt": 8, "ts": 15, "tj": 4, "tr": 0.3,
    # FCI — optimized
    "fe": 0.0797, "fd_t": 135, "fd": 0.35,
    "fg_t": 0.5, "fg": 0.394, "fa_t": 0.28, "fa": 1.6, "fc": 0.22,
    "fy": 0.361, "fec": 0.673,
    # Consumer / Housing
    "ccl": 5, "cci": 3, "ccf": 8, "ccm": 0.2, "ccg": 3,
    "hr": 8, "hc": 1.3, "hd": 0.15,
    # Regime
    "rr": (1, -1), "ru": (4.5, 7), "ro": (3.5, 5.5), "rou": (4, 2.5),
    "rfr": 1.8, "rfn": 1.0, "rfo": 0.4, "ra": 1.6,
    # Lags (no longer used for main lags but kept for compatibility)
    "lm": 0.519, "lf": 0.435, "ls": 0.2, "lr": 0.031,
    # Shock
    "st": -2, "sa": 4, "sr": 0.5,
}


# ╔═══════════════════════════════════════════════════════════════╗
# ║  MATH HELPERS                                                ║
# ╚═══════════════════════════════════════════════════════════════╝

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def lerp(a, b, t):
    return a + (b - a) * clamp(t, 0, 1)

def smoothstep(e0, e1, x):
    t = clamp((x - e0) / (e1 - e0) if e1 != e0 else 0, 0, 1)
    return t * t * (3 - 2 * t)

def logistic(x, cap, s=1):
    return cap * (2 / (1 + np.exp(-s * x / cap)) - 1) if cap != 0 else 0


# ╔═══════════════════════════════════════════════════════════════╗
# ║  DEBT DYNAMICS                                               ║
# ╚═══════════════════════════════════════════════════════════════╝

def debt_calc(dr, nr, g, pd):
    r, gg = nr / 100, g / 100
    denom = 1 + gg if gg != -1 else 1
    snowball = dr * (r - gg) / denom
    nd = dr + snowball * 0.25 + pd * 0.25
    rp = 0
    if nd > K["d1"]:
        rp = K["dq"] * ((nd - K["d1"]) / 50) ** 2
    if nd > K["d2"]:
        rp += K["drs"]
    return clamp(nd, 40, 250), rp


# ╔═══════════════════════════════════════════════════════════════╗
# ║  SIMULATION ENGINE                                           ║
# ╚═══════════════════════════════════════════════════════════════╝

@dataclass
class SimResult:
    """One quarter of simulation output."""
    label: str = ""
    gdpGrowth: float = 0
    inflation: float = 0
    unemployment: float = 0
    currencyIndex: float = 100
    sp500Index: float = 5200
    sp500Earnings: float = 0       # Earnings-model S&P (0 = not computed)
    goldPrice: float = 2900
    bondYield10Y: float = 4.3
    tradeBalance: float = -780
    debtToGDP: float = 123
    consumerConfidence: float = 98
    housingIndex: float = 200
    inflExpectations: float = 3.0
    fci: float = 0
    outputGap: float = 0
    wageGrowth: float = 3.5
    regime: str = "normal"
    regime_r: float = 0
    regime_n: float = 1
    regime_o: float = 0
    fiscalMultiplier: float = 1.0
    nairuShift: float = 0
    taylorAdj: float = 0         # Taylor Rule rate adjustment
    effRate: float = 0           # Effective fed rate after Taylor
    fiscalAdj: float = 0         # Counter-cyclical fiscal adjustment ($T)


def simulate(params: dict,
             input_series: Optional[list] = None,
             initial_conditions: Optional[dict] = None,
             taylor_enabled: bool = True,
             fiscal_response: float = 0.0) -> List[SimResult]:
    """
    Run the macro simulation engine.

    Args:
        params: Static policy parameters (for forward projection)
        input_series: List of per-quarter input dicts (for backtesting)
        initial_conditions: Starting state {g, i, u, fx, sp, by, tb, cc, dtg, hsp}
        taylor_enabled: If True, Fed adjusts rates endogenously via Taylor Rule
        fiscal_response: Auto-stabilizer strength (0=none, 1=normal, 2=aggressive).
                         Adds counter-cyclical spending when GDP falls below potential.

    Returns:
        List of SimResult for each quarter
    """
    bt = input_series is not None
    N = len(input_series) if bt else 24
    results = []

    # Initial state
    ic = initial_conditions or {}
    sU = ic.get("u", 4.0)
    sI = ic.get("i", 3.2)
    sFX = ic.get("fx", 100.0)
    sSP = ic.get("sp", 5200.0)
    sBY = ic.get("by", 4.3)
    sTB = ic.get("tb", -780.0)
    sCC = ic.get("cc", 98.0)
    sDTG = ic.get("dtg", params.get("debtToGDP", 123.0))
    HSP = ic.get("hsp", None)

    # Era-adaptive anchors
    NAIRU = clamp(sU * 0.85, 3.5, 6.0)
    ANCHOR = clamp(sI * 0.9, 1.5, 8.0)

    # State variables
    g = ic.get("g", 2.5)
    inf = sI
    u = sU
    fx = sFX
    sp = sSP
    by = sBY
    tb = sTB
    dtg = sDTG
    cc = sCC
    hi = 200.0
    ie = sI
    fci = 0.0
    gap = 0.0
    pot = 100.0
    hud = 0
    ns = 0.0
    wg = clamp(sI + 1, 2, 8)
    pS = sp
    ppS = sp
    gold = ic.get("gold", 2900.0)   # $/oz
    prev_fx = sFX
    prev_tb = sTB
    prev_gold = gold

    ref = input_series[0] if bt else params

    def get_input(q_idx):
        if bt:
            row = input_series[q_idx]
            if isinstance(row, dict):
                return row
            # DataFrame row
            return row.to_dict() if hasattr(row, 'to_dict') else row
        return params

    ref_d = get_input(0) if bt else params

    for q in range(N):
        p = get_input(q)
        # Lag structure — starts high so policy changes hit GDP within 1-2 quarters
        lM = min(0.92, 0.7 + q * 0.02) if bt else 1 - np.exp(-1.5 * q / N)
        lF = min(0.95, 0.65 + q * 0.025) if bt else 1 - np.exp(-2.5 * q / N)
        lS = min(0.85, 0.4 + q * 0.02) if bt else 1 - np.exp(-q / N)
        n = np.sin(q * 3.7) * 0.12 + np.cos(q * 5.3) * 0.08

        # Use PREVIOUS quarter as reference for deltas (not always Q1)
        # This captures the quarter-over-quarter change that drives GDP volatility
        prev_d = get_input(max(0, q - 1)) if bt else ref_d
        # Also keep absolute reference for level-based calculations
        abs_ref = ref_d

        # Regime classification
        rW = smoothstep(K["rr"][0], K["rr"][1], g) * smoothstep(K["ru"][0], K["ru"][1], u)
        oW = smoothstep(K["ro"][0], K["ro"][1], g) * smoothstep(K["rou"][0], K["rou"][1], u)
        nW = clamp(1 - rW - oW, 0, 1)
        fm = lerp(lerp(K["rfn"], K["rfr"], rW), K["rfo"], oW)
        mp = lerp(lerp(1, 0.3, smoothstep(1, 0, p["fedRate"])), 1.4, oW)
        asym = lerp(1, K["ra"], rW)
        zlb = 0.15 if p["fedRate"] <= 0.5 else (
            smoothstep(0.5, 2, p["fedRate"]) * 0.7 + 0.15 if p["fedRate"] <= 2 else 1.0)

        # Deltas: quarter-over-quarter for GDP impulse (captures rate CHANGES)
        dR_qoq = p["fedRate"] - prev_d["fedRate"]
        dS_qoq = p["govSpending"] - prev_d["govSpending"]
        dT_qoq = p["taxRate"] - prev_d["taxRate"]
        dO_qoq = p["oilPrice"] - prev_d["oilPrice"]
        # Absolute deltas from baseline for level-dependent calcs (FX, trade, etc.)
        dR = p["fedRate"] - abs_ref["fedRate"]
        dS = p["govSpending"] - abs_ref["govSpending"]
        dT = p["taxRate"] - abs_ref["taxRate"]
        dM = p["moneySupplyGrowth"] - abs_ref["moneySupplyGrowth"]
        dTa = p["tariffRate"] - abs_ref["tariffRate"]
        dO = p["oilPrice"] - abs_ref["oilPrice"]
        dL = p["laborForceGrowth"] - 0.5
        dP = p["productivityGrowth"] - 1.5

        # Shock detection
        sf = 1.0
        if p["laborForceGrowth"] < K["st"]:
            sf = K["sa"] * abs(p["laborForceGrowth"] / K["st"])
        prevL = (get_input(q - 1)["laborForceGrowth"] if q > 0 and bt
                 else p["laborForceGrowth"])
        isReb = prevL < K["st"] and p["laborForceGrowth"] > 0

        # Adaptive Okun's
        rateStress = max(0, p["fedRate"] - K["uo_rate_th"]) * K["uo_rate_amp"]
        eOkun = clamp(lerp(K["uo_b"], K["uo_c"], smoothstep(K["uo_t"], 1, fci)) + rateStress, K["uo_b"], 0.7)

        # ── GDP ──
        # Pre-compute fiscal auto-stabilizer from PREVIOUS quarter's gap (lagged response)
        fiscal_adj = 0.0
        if fiscal_response > 0 and bt:
            # Trigger when GDP growth drops below 1.5% (not just negative gap)
            gdp_shortfall = max(0, 1.5 - g)  # how far below trend
            if gdp_shortfall > 0:
                fiscal_adj = fiscal_response * 0.12 * gdp_shortfall  # spending boost
            elif g > 3.5:
                fiscal_adj = -fiscal_response * 0.04 * (g - 3.5)  # modest austerity

        potG = K["gdp_pb"] + dP * K["gdp_p"] * lS + dL * K["gdp_l"] * lS - ns * 0.15
        pot *= (1 + potG / 400)
        fi = logistic(dS_qoq * K["gdp_fs"] - dT_qoq * K["gdp_ft"], K["gdp_fc"], 1.2) * fm
        effRateSens = K["gdp_rate"] + K["gdp_rate_high"] * max(0, p["fedRate"] - K["gdp_rate_th"])
        effFciDrag = K["gdp_fci_base"] + K["gdp_fci_crisis"] * max(0, fci - K["gdp_fci_th"])
        
        # GDP impulse: combine QoQ rate change (direction) + rate level effect (sustained drag)
        rate_impulse = -effRateSens * dR_qoq * zlb * mp * lM  # rate CHANGE effect
        rate_level = -0.08 * max(0, p["fedRate"] - 6) * lM     # sustained high-rate drag above 6%
        oil_impulse = -K["gdp_o"] * dO_qoq * lM * 2            # oil QoQ changes hit hard
        oil_level = -0.003 * max(0, p["oilPrice"] - 80) * lM   # sustained high oil drag
        
        rI = (fi + rate_impulse + rate_level + oil_impulse + oil_level
              - logistic(dTa * K["gdp_ta"], 2) * lF
              - effFciDrag * fci * lM
              + K["gdp_m"] * dM * lM + K["gdp_w"] * (sp - pS)
              + K["gdp_ic"] * np.sin(q * np.pi / 3) * np.exp(-0.08 * q)
              - K["gdp_g"] * max(0, dTa) * lF + K["da"] * max(0, u - 5) * 0.1)
        if sf > 1:
            rI += dL * sf
        if isReb:
            rI += p["laborForceGrowth"] * K["sr"] * 3
        # Fiscal auto-stabilizer boost (from fiscal_response parameter)
        rI += fiscal_adj * K["gdp_fs"] * fm
        rG = potG + rI * lF
        sh = rG - potG
        g = clamp(potG + (sh * asym if sh < 0 else sh) + n, -35, 40)
        gap = g - potG

        # ── Wages ──
        wg = clamp(lerp(wg, potG + clamp(NAIRU - u, -3, 3) * 0.8 + 2, 0.3) + wg * K["inf_wp"] * 0.1, -1, 12)

        # ── Inflation ──
        uG = NAIRU - u
        ph = K["inf_pl"] * uG + K["inf_pc"] * uG * uG if uG > 0 else K["inf_ps"] * uG
        demPull = K["inf_dc"] * gap * gap if gap > 0 else K["inf_dl"] * gap
        # Direct rate→inflation: high rates crush demand and reduce inflation (Volcker mechanism)
        rate_disinflation = -K["inf_rate"] * max(0, dR) * lM  # only when rates RISE above baseline
        inf = clamp(ANCHOR + ph
                     + (0.15 * (ie - ANCHOR - 2) if ie > ANCHOR + 2 else 0)
                     + K["inf_ta"] * dTa * lF + K["inf_o"] * dO * lM
                     + K["inf_m"] * dM * lM + demPull + rate_disinflation
                     + K["inf_ew"] * (ie - ANCHOR)
                     + K["inf_fx"] * max(0, 100 - fx) * lM
                     + 0.1 * max(0, wg - potG - 1) + n * 0.4, -3, 20)
        aw = K["inf_ah"] if inf > ANCHOR + 3 else K["inf_an"]
        ie = lerp(ie, inf, aw) * 0.7 + ANCHOR * 0.3
        if inf > K["inf_dt"] and q > 4:
            ie += K["inf_da"] * (inf - K["inf_dt"])

        # ── Unemployment (multi-channel with cumulative dynamics) ──
        if u > K["uh_t"]:
            hud += 1
        else:
            hud = max(0, hud - 1)
        ns = (K["uh_r"] * (hud - K["uh_d"]) * (u - K["uh_t"])
              if u > K["uh_t"] and hud > K["uh_d"] else 0)

        # Channel 1: Okun's law (output gap → unemployment)
        uI = -eOkun * gap

        # Channel 2: Direct GDP level tracking — when GDP < 1%, unemployment rises sharply
        # This is the key fix: actual GDP swings drive unemployment, not just the dampened output gap
        gdp_unemp = 0.4 * max(0, 1.0 - g)   # rises fast below 1% GDP
        if g < 0:
            gdp_unemp += 0.6 * abs(g)         # much sharper below zero (recession)

        # Channel 3: Direct FCI stress (credit tightening → layoffs)
        fci_unemp = 0.4 * max(0, fci - 0.1) ** 1.3

        # Channel 4: Rate drag (high rates kill housing/auto/capex employment)
        rate_unemp = K["uo_rate_amp"] * max(0, p["fedRate"] - K["uo_rate_th"])

        # Channel 5: Labor force collapse (direct)
        labor_unemp = 0.0
        if p["laborForceGrowth"] < -1:
            labor_unemp = K["ul_crisis"] * abs(p["laborForceGrowth"] + 1)
        if sf > 1:
            labor_unemp += abs(dL) * 1.5

        # Channel 6: Tariff + labor supply
        other_unemp = K["ut"] * dTa * lF - K["ul"] * dL * lS

        uI += gdp_unemp + fci_unemp + rate_unemp + labor_unemp + other_unemp
        u = clamp((sU + ns) + uI + n * 0.25, 1.5, 18)

        # ── Taylor Rule (endogenous Fed response) ──
        taylor_adj = 0.0
        if taylor_enabled and bt:
            r_neutral = ref_d["fedRate"]
            taylor_target = 0.5 * (inf - ANCHOR) + 0.5 * gap
            taylor_adj = clamp(taylor_target * 0.25, -1.0, 1.0)
        eff_rate = clamp(p["fedRate"] + taylor_adj, 0.0, 20.0)
        dR_eff = eff_rate - ref_d["fedRate"]

        # ── Currency (with cyclical dynamics) ──
        fxRS = K["fx_rb"] + K["fx_rl"] * abs(eff_rate)
        # Core FX: rate differential + FCI + money + trade + oil
        fx_core = sFX + fxRS * dR_eff * lM * zlb - K["fx_f"] * fci \
                  - K["fx_m"] * dM * lM + K["fx_ca"] * (tb - sTB) * 0.01 \
                  - K["fx_to"] * dO
        # Add GDP-cycle feedback: strong growth attracts capital → stronger dollar
        fx_gdp = 0.3 * (g - 2.0)
        # Add gradual drift from inflation differential (PPP channel)
        fx_ppp = -0.15 * max(0, inf - 2.5) * (q / max(N, 1))
        # Momentum (small carry-over from previous move)
        fx_mom = 0.2 * (fx - prev_fx)
        prev_fx = fx
        fx = clamp(fx_core + fx_gdp + fx_ppp + fx_mom + n * 0.5, 55, 150)

        # ── Equities (forward-looking) ──
        ppS = pS
        pS = sp
        spT = HSP[q] if bt and HSP and q < len(HSP) else sp
        eq_trend = sp * (max(0, g + inf) / 100) * 0.25 if (not bt or not HSP) else 0

        # Forward-looking rate effect: markets price in expected rate TRAJECTORY
        # If rates are very high but falling, equities anticipate relief (Volcker 1982 rally)
        # If rates are low but rising, equities anticipate tightening
        rate_trajectory = dR_qoq  # positive = rates rising, negative = rates falling
        if bt and q > 0:
            # 2-quarter rate momentum: are rates accelerating or decelerating?
            prev_prev_rate = get_input(max(0, q-2))["fedRate"]
            rate_accel = dR_qoq - (prev_d["fedRate"] - prev_prev_rate)
            # Markets rally on decelerating hikes (even before cuts)
            rate_fwd = rate_trajectory + 0.5 * rate_accel
        else:
            rate_fwd = rate_trajectory

        spC = ((g * K["eq_e"] + dP * K["eq_p"] * lS
                - K["eq_r"] * rate_fwd * lM   # forward rate effect, not level
                + K["eq_m"] * dM * lM - K["eq_f"] * fci
                - (K["eq_pm"] * abs(g - K["eq_pg"]) * fci
                   if g < K["eq_pg"] and fci > K["eq_pf"] else 0))
               * lF / N
               + K["eq_mo"] * (sp - pS) * 0.3
               + K["eq_mr"] * (spT - sp)
               + eq_trend)
        sp = clamp(sp + spC + n * 30, 50, 14000)

        # ── Bonds (forward-looking term structure) ──
        # 10Y yield = expected average short rate over next 10Y + term premium
        # Key insight: when fed rate is 18%, market expects it to return to neutral (~5-6%)
        # so 10Y prices in the full path, not just the spot rate.
        _, rp = debt_calc(dtg, by, g, dS * K["ds"] * 0.2 - dT * K["dtt"] * 0.5)
        
        # Expected rate path: blend of current rate toward long-run neutral
        # Neutral rate ≈ inflation anchor + real neutral (~2%)
        r_neutral = ANCHOR + 2.0
        # How fast the market expects rates to normalize (faster when rates are extreme)
        rate_gap = abs(eff_rate - r_neutral)
        norm_speed = 0.15 + 0.05 * min(rate_gap, 10)  # 15-65% per "year" toward neutral
        # Expected average rate over 10Y horizon (geometric mean reversion)
        exp_avg_rate = eff_rate * (1 - norm_speed) + r_neutral * norm_speed
        # For very high rates (Volcker), heavier weight on normalization
        if eff_rate > 10:
            exp_avg_rate = exp_avg_rate * 0.7 + r_neutral * 0.3
        
        # Term premium: rises with inflation volatility, debt, and uncertainty
        term_prem = (K["bt"]
                     + K["bv"] * abs(inf - ANCHOR) * 0.3     # inflation risk
                     + rp                                       # fiscal risk premium
                     + K["bs"] * max(0, dtg - 100)             # debt premium
                     + (K["bif"] * (inf - ANCHOR - 1) if inf > ANCHOR + 1 else 0))
        
        # Flight to quality: during crises, 10Y drops (safe haven)
        flight = -0.5 * max(0, fci - 0.3) if fci > 0.3 else 0
        
        by_target = exp_avg_rate + term_prem + flight
        # Smooth toward target (yields don't jump instantly)
        by = clamp(by * 0.6 + by_target * 0.4 + n * 0.08, 0.1, 18)

        # ── Debt ──
        dtg, _ = debt_calc(dtg, by, g, dS * K["ds"] - dT * K["dtt"] + K["da"] * max(0, u - 5))

        # ── Trade (with cyclical dynamics) ──
        jc = -1.2 if q < K["tj"] else 0.6
        tb_core = sTB - K["tf"] * (fx - sFX) + dTa * K["tt"] * jc * lF \
                  - dS * K["ts"] * lF - K["tr"] * max(0, dTa) * 10 * lF
        # GDP cycle: strong growth sucks in imports → trade worsens
        tb_gdp = -3.0 * max(0, g - 2.0)
        # Oil import cost effect
        tb_oil = -0.08 * dO
        # Structural deficit trend (US trade deficit slowly worsens over time)
        tb_trend = -1.0 * q / max(N, 1) if not bt else 0
        # Momentum
        tb_mom = 0.15 * (tb - prev_tb)
        prev_tb = tb
        tb = clamp(tb_core + tb_gdp + tb_oil + tb_trend + tb_mom + n * 3, -1800, 300)

        # ── FCI ──
        fR = -K["fe"] * spC
        if dtg > K["fd_t"]:
            fR += K["fd"] * smoothstep(K["fd_t"], 180, dtg)
        if g < K["fg_t"]:
            fR += K["fg"] * smoothstep(K["fg_t"], -2, g)
        fR += K["fc"] * smoothstep(2, -1, g)
        if p["fedRate"] > by + 0.5:
            fR += K["fy"] * (p["fedRate"] - by - 0.5)
        if ppS > 0 and (sp - ppS) / ppS < -0.1:
            fR += K["fec"]
        if fR > K["fa_t"]:
            fR = K["fa_t"] + (fR - K["fa_t"]) * K["fa"]
        fci = clamp(fR, -0.5, 2.5)

        # ── Consumer Confidence ──
        ccF = (sCC - K["ccl"] * (u - sU) + K["eq_wf"] * (sp - sSP)
               - (K["cci"] * (inf - ANCHOR - 2) if inf > ANCHOR + 2 else 0.5 * (inf - ANCHOR))
               - K["ccf"] * fci + K["ccg"] * (g - 2.5))
        cc = clamp(cc * K["ccm"] + ccF * (1 - K["ccm"]) + n * 2, 20, 150)

        # ── Housing ──
        mr = by + 1.7
        hi = clamp(hi + ((15 * (5 - mr) if mr < 5 else -K["hr"] * (mr - 5) ** K["hc"])
                         + 2 * (cc - sCC) * K["hd"] + n * 2) * 0.15, 100, 600)

        # ── Gold (multi-channel with structural demand) ──
        # Gold responds to BOTH cyclical factors (real rates, FCI) AND
        # structural factors (central bank buying, de-dollarization, fiscal concerns)
        # that dominated post-2022.
        real_rate = by - ie  # 10Y yield minus inflation expectations

        # CYCLICAL channels (traditional)
        # Real rate sensitivity reduced — gold's real-rate correlation has weakened
        gold_real_rate = -12.0 * (real_rate - 1.0)  # was -25, now less sensitive
        gold_inflation = 6.0 * max(0, ie - 2.5)     # hedge above anchor
        gold_safe_haven = 35.0 * max(0, fci)         # flight to safety
        gold_dollar = -5.0 * (fx - sFX)              # inverse dollar

        # STRUCTURAL channels (post-2022 regime)
        # Central bank buying: accelerated sharply after Russia sanctions (2022)
        # WGC data: CB gold purchases averaged ~500t/yr pre-2022, ~1000t/yr post-2022
        # Modeled as accelerating trend, scaling with current gold price
        cb_base = 0.012  # ~5% annual baseline CB accumulation
        # De-dollarization momentum: stronger when debt/GDP is high and dollar reserves being diversified
        dedollar_boost = 0.004 * min(1.0, max(0, dtg - 100) / 30)  # activates above 100% debt/GDP
        # Geopolitical risk premium: stays elevated even when FCI is normal
        # Modeled as a persistent component that doesn't reset
        geo_premium = 0.005  # ~2% annual geopolitical/structural premium
        # Fiscal sustainability concern: US deficit-driven
        # High debt + high rates = compounding interest costs = gold demand
        fiscal_premium = 0.003 * max(0, dtg - 110) * max(0, by - 3.5) / 50

        # Combined structural trend (percentage-based, compounds with gold price)
        structural_pct = cb_base + dedollar_boost + geo_premium + fiscal_premium
        gold_structural = gold * structural_pct if (not bt or not HSP) else gold * structural_pct * 0.5

        # Momentum channel: once gold breaks trend, CTAs amplify the move
        gold_momentum = 0.08 * (gold - prev_gold)
        prev_gold = gold

        gold = clamp(gold + gold_real_rate + gold_inflation + gold_safe_haven
                     + gold_dollar + gold_structural + gold_momentum + n * 15, 800, 8000)

        rl = "recession" if rW > 0.5 else ("overheating" if oW > 0.5 else "normal")
        label = p.get("q", f"Q{q + 1}") if isinstance(p, dict) else f"Q{q + 1}"

        results.append(SimResult(
            label=label,
            gdpGrowth=round(g, 2), inflation=round(inf, 2),
            unemployment=round(u, 2), currencyIndex=round(fx, 1),
            sp500Index=round(sp), goldPrice=round(gold),
            bondYield10Y=round(by, 2),
            tradeBalance=round(tb), debtToGDP=round(dtg, 1),
            consumerConfidence=round(cc, 1), housingIndex=round(hi),
            inflExpectations=round(ie, 2), fci=round(fci, 3),
            outputGap=round(gap, 2), wageGrowth=round(wg, 2),
            regime=rl, regime_r=round(rW, 2), regime_n=round(nW, 2),
            regime_o=round(oW, 2), fiscalMultiplier=round(fm, 2),
            nairuShift=round(ns, 3), taylorAdj=round(taylor_adj, 3),
            effRate=round(eff_rate, 3), fiscalAdj=round(fiscal_adj, 3),
        ))

    return results

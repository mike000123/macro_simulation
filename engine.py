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
    # GDP
    "gdp_rate": 0.1, "gdp_rate_high": 0.015, "gdp_rate_th": 5,
    "gdp_fs": 0.8, "gdp_ft": 0.235, "gdp_fc": 3.5,
    "gdp_m": 0.041, "gdp_ta": 0.18, "gdp_o": 0.007,
    "gdp_fci_base": 0.2, "gdp_fci_crisis": 0.6, "gdp_fci_th": 0.3,
    "gdp_p": 0.6, "gdp_l": 0.4, "gdp_pb": 2.0,
    "gdp_ic": 0.3, "gdp_w": 0.000485, "gdp_g": 0.1,
    # Inflation — added inf_rate for Volcker disinflation
    "inf_pl": 0.35, "inf_pc": 0.6, "inf_ps": 0.28,
    "inf_ta": 0.139, "inf_o": 0.05, "inf_m": 0.03,
    "inf_dc": 0.25, "inf_dl": 0.12,
    "inf_ew": 0.137, "inf_an": 0.25, "inf_ah": 0.4,
    "inf_dt": 6, "inf_da": 0.1, "inf_fx": 0.016, "inf_wp": 0.15,
    "inf_rate": 0.04,                                               # NEW: direct rate → CPI (moderate)
    # Unemployment
    "uo_b": 0.2, "uo_c": 0.45, "uo_t": 0.4,
    "uo_rate_th": 8, "uo_rate_amp": 0.015,
    "ul": 0.2, "ut": 0.08, "ul_crisis": 1.8,
    "uh_t": 6, "uh_d": 4, "uh_r": 0.05,
    # Currency
    "fx_rb": 1.0, "fx_rl": 0.1, "fx_f": 1.5, "fx_m": 0.609,
    "fx_ct": 170, "fx_ca": 0.05, "fx_to": 0.01,
    # Equities — stronger crash + mean-reversion
    "eq_e": 2.848, "eq_p": 30, "eq_r": 225.6, "eq_m": 40,
    "eq_f": 300, "eq_pg": -1, "eq_pf": 0.3, "eq_pm": 450,        # ↑ FCI crash
    "eq_wf": 0.005, "eq_mo": 0.5, "eq_mr": 0.22,                  # ↑ mean-reversion
    # Bonds — higher fed passthrough
    "bf": 0.45, "bfl": 0.14, "bi": 0.35, "bt": 0.25,
    "bv": 0.15, "bs": 0.02, "bif": 0.25,
    # Debt
    "ds": 1.5, "dtt": 0.3, "d1": 150, "d2": 180,
    "dq": 0.3, "drs": 0.5, "da": 0.4,
    # Trade
    "tf": 2, "tt": 8, "ts": 15, "tj": 4, "tr": 0.3,
    # FCI — lower thresholds for earlier activation
    "fe": 0.12, "fd_t": 135, "fd": 0.35,
    "fg_t": 0.5, "fg": 0.55, "fa_t": 0.28, "fa": 1.6, "fc": 0.22,
    "fy": 0.35, "fec": 0.45,
    # Consumer / Housing
    "ccl": 5, "cci": 3, "ccf": 8, "ccm": 0.2, "ccg": 3,
    "hr": 8, "hc": 1.3, "hd": 0.15,
    # Regime
    "rr": (1, -1), "ru": (4.5, 7), "ro": (3.5, 5.5), "rou": (4, 2.5),
    "rfr": 1.8, "rfn": 1.0, "rfo": 0.4, "ra": 1.6,
    # Lags
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
        # Lag structure
        lM = min(0.85, K["lm"] + q * K["lr"]) if bt else 1 - np.exp(-1.5 * q / N)
        lF = min(0.9, K["lf"] + q * K["lr"]) if bt else 1 - np.exp(-2.5 * q / N)
        lS = min(0.8, K["ls"] + q * K["lr"] * 0.75) if bt else 1 - np.exp(-q / N)
        n = np.sin(q * 3.7) * 0.12 + np.cos(q * 5.3) * 0.08

        # Regime classification
        rW = smoothstep(K["rr"][0], K["rr"][1], g) * smoothstep(K["ru"][0], K["ru"][1], u)
        oW = smoothstep(K["ro"][0], K["ro"][1], g) * smoothstep(K["rou"][0], K["rou"][1], u)
        nW = clamp(1 - rW - oW, 0, 1)
        fm = lerp(lerp(K["rfn"], K["rfr"], rW), K["rfo"], oW)
        mp = lerp(lerp(1, 0.3, smoothstep(1, 0, p["fedRate"])), 1.4, oW)
        asym = lerp(1, K["ra"], rW)
        zlb = 0.15 if p["fedRate"] <= 0.5 else (
            smoothstep(0.5, 2, p["fedRate"]) * 0.7 + 0.15 if p["fedRate"] <= 2 else 1.0)

        # Deltas from reference
        dR = p["fedRate"] - ref_d["fedRate"]
        dS = p["govSpending"] - ref_d["govSpending"]
        dT = p["taxRate"] - ref_d["taxRate"]
        dM = p["moneySupplyGrowth"] - ref_d["moneySupplyGrowth"]
        dTa = p["tariffRate"] - ref_d["tariffRate"]
        dO = p["oilPrice"] - ref_d["oilPrice"]
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
        fi = logistic(dS * K["gdp_fs"] - dT * K["gdp_ft"], K["gdp_fc"], 1.2) * fm
        effRateSens = K["gdp_rate"] + K["gdp_rate_high"] * max(0, p["fedRate"] - K["gdp_rate_th"])
        effFciDrag = K["gdp_fci_base"] + K["gdp_fci_crisis"] * max(0, fci - K["gdp_fci_th"])
        rI = (fi - effRateSens * dR * zlb * mp * lM
              - logistic(dTa * K["gdp_ta"], 2) * lF
              - K["gdp_o"] * dO * lM - effFciDrag * fci * lM
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

        # ── Unemployment ──
        if u > K["uh_t"]:
            hud += 1
        else:
            hud = max(0, hud - 1)
        ns = (K["uh_r"] * (hud - K["uh_d"]) * (u - K["uh_t"])
              if u > K["uh_t"] and hud > K["uh_d"] else 0)
        uI = -eOkun * gap - K["ul"] * dL * lS + K["ut"] * dTa * lF
        if fci > 0.3:
            uI += 0.3 * fci
        if p["laborForceGrowth"] < -1:
            uI += K["ul_crisis"] * abs(p["laborForceGrowth"] + 1)
        if sf > 1:
            uI += abs(dL) * 1.5
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

        # ── Equities ──
        ppS = pS
        pS = sp
        spT = HSP[q] if bt and HSP and q < len(HSP) else sp
        # Earnings-driven trend: GDP growth drives long-run equity appreciation
        # ~2% quarterly earnings growth at trend GDP (≈8% annualized nominal return)
        eq_trend = sp * (max(0, g + inf) / 100) * 0.25 if (not bt or not HSP) else 0  # quarterly nominal return
        spC = ((g * K["eq_e"] + dP * K["eq_p"] * lS - K["eq_r"] * dR_eff * lM
                + K["eq_m"] * dM * lM - K["eq_f"] * fci
                - (K["eq_pm"] * abs(g - K["eq_pg"]) * fci
                   if g < K["eq_pg"] and fci > K["eq_pf"] else 0))
               * lF / N
               + K["eq_mo"] * (sp - pS) * 0.3
               + K["eq_mr"] * (spT - sp)
               + eq_trend)
        sp = clamp(sp + spC + n * 30, 50, 14000)

        # ── Bonds (sqrt scaling) ──
        _, rp = debt_calc(dtg, by, g, dS * K["ds"] * 0.2 - dT * K["dtt"] * 0.5)
        bfp = K["bf"] + K["bfl"] * np.sqrt(max(0, eff_rate))
        itp = K["bif"] * (inf - ANCHOR - 1) if inf > ANCHOR + 1 else 0
        by = clamp(bfp * eff_rate + K["bi"] * ie + K["bt"] + rp
                    + K["bv"] * abs(g - 2.5) * 0.2
                    + K["bs"] * max(0, dtg - 100) + itp + n * 0.08, 0.1, 16)

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
        fci = clamp(fR, -1, 2.5)

        # ── Consumer Confidence ──
        ccF = (sCC - K["ccl"] * (u - sU) + K["eq_wf"] * (sp - sSP)
               - (K["cci"] * (inf - ANCHOR - 2) if inf > ANCHOR + 2 else 0.5 * (inf - ANCHOR))
               - K["ccf"] * fci + K["ccg"] * (g - 2.5))
        cc = clamp(cc * K["ccm"] + ccF * (1 - K["ccm"]) + n * 2, 20, 150)

        # ── Housing ──
        mr = by + 1.7
        hi = clamp(hi + ((15 * (5 - mr) if mr < 5 else -K["hr"] * (mr - 5) ** K["hc"])
                         + 2 * (cc - sCC) * K["hd"] + n * 2) * 0.15, 100, 600)

        # ── Gold ──
        # Gold responds to: real rates (inverse), inflation expectations, FCI (safe haven),
        # dollar weakness, debt concerns, and has long-run inflation hedge trend
        real_rate = by - ie  # 10Y yield minus inflation expectations
        gold_real_rate = -25.0 * (real_rate - 1.0)  # gold rises when real rates fall below 1%
        gold_inflation = 8.0 * max(0, ie - 2.5)     # inflation hedge above anchor
        gold_safe_haven = 40.0 * max(0, fci)         # flight to safety
        gold_dollar = -6.0 * (fx - sFX)              # inverse dollar relationship
        gold_debt = 3.0 * max(0, dtg - 120)          # debasement fears
        gold_trend = gold * 0.005 if (not bt or not HSP) else 0   # ~2% annual trend
        gold = clamp(gold + gold_real_rate + gold_inflation + gold_safe_haven
                     + gold_dollar + gold_debt + gold_trend + n * 15, 800, 8000)

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

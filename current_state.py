"""
Current Macro State — current_state.py
Offline defaults from Q1 2025 + keyless FRED CSV auto-fetch.

Uses the public FRED fredgraph CSV endpoint (no API key required),
matching the Macro Cockpit style of data loading.
"""

from datetime import datetime

import pandas as pd

# ═══════════════════════════════════════════════════════════════
# OFFLINE DEFAULTS (Q1 2025 — update periodically)
# ═══════════════════════════════════════════════════════════════

OFFLINE_INPUTS = {
    "fedRate": 4.375,           # FRED: FEDFUNDS
    "govSpending": 5.3,         # BEA / manual
    "taxRate": 17.5,            # CBO / manual
    "moneySupplyGrowth": 3.8,   # FRED: M2SL YoY%
    "tariffRate": 12.0,         # USITC / manual
    "oilPrice": 65.0,           # EIA / manual
    "laborForceGrowth": 0.4,    # BLS / manual
    "productivityGrowth": 1.8,  # BLS / manual
    "debtToGDP": 124.0,         # FRED: GFDEGDQ188S
}

OFFLINE_ACTUALS = {
    "gdpGrowth": 2.4,           # BEA / manual
    "inflation": 2.8,           # CPIAUCSL YoY
    "unemployment": 4.2,        # UNRATE
    "currencyIndex": 104.0,     # ICE DXY / manual
    "sp500Index": 5600,         # manual
    "bondYield10Y": 4.35,       # DGS10
    "tradeBalance": -850,       # manual
    "consumerConfidence": 98.0, # manual
}

OFFLINE_DATE = "Q1 2025"


def _fred_csv(series_id: str) -> pd.Series:
    """Fetch a FRED series through the public CSV endpoint. No API key required."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    if df.empty or len(df.columns) < 2:
        raise ValueError(f"Empty or malformed FRED CSV for {series_id}")

    value_col = df.columns[1]
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    series = df[value_col].dropna()
    if series.empty:
        raise ValueError(f"No numeric data returned for {series_id}")
    return series


def _latest_value(series_id: str, fallback: float) -> float:
    try:
        s = _fred_csv(series_id)
        return float(s.iloc[-1])
    except Exception:
        return float(fallback)


def _yoy_from_level(series_id: str, fallback: float) -> float:
    """Convert a level series to year-over-year percent."""
    try:
        s = _fred_csv(series_id)
        if len(s) < 13:
            return float(fallback)
        now = float(s.iloc[-1])
        prev = float(s.iloc[-13])
        if prev == 0:
            return float(fallback)
        return ((now / prev) - 1.0) * 100.0
    except Exception:
        return float(fallback)


def _try_fred_fetch():
    """Try to fetch latest data from FRED CSV. Returns (inputs, actuals, date) or None."""
    try:
        fed_rate = _latest_value("FEDFUNDS", OFFLINE_INPUTS["fedRate"])
        cpi_yoy = _yoy_from_level("CPIAUCSL", OFFLINE_ACTUALS["inflation"])
        unemp = _latest_value("UNRATE", OFFLINE_ACTUALS["unemployment"])
        dgs10 = _latest_value("DGS10", OFFLINE_ACTUALS["bondYield10Y"])
        m2_growth = _yoy_from_level("M2SL", OFFLINE_INPUTS["moneySupplyGrowth"])
        debt_gdp = _latest_value("GFDEGDQ188S", OFFLINE_INPUTS["debtToGDP"])

        inputs = {
            **OFFLINE_INPUTS,
            "fedRate": round(fed_rate, 3),
            "moneySupplyGrowth": round(m2_growth, 1),
            "debtToGDP": round(debt_gdp, 1),
        }
        actuals = {
            **OFFLINE_ACTUALS,
            "inflation": round(cpi_yoy, 1),
            "unemployment": round(unemp, 1),
            "bondYield10Y": round(dgs10, 2),
        }
        date = f"FRED CSV ({datetime.now().strftime('%b %Y')})"
        return inputs, actuals, date

    except Exception:
        return None


def get_current_state():
    """Returns (inputs, actuals, source_label). Tries FRED CSV first, falls back to offline defaults."""
    result = _try_fred_fetch()
    if result:
        return result
    return OFFLINE_INPUTS.copy(), OFFLINE_ACTUALS.copy(), f"Offline ({OFFLINE_DATE})"

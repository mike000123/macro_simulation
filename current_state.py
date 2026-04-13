"""
Current Macro State — current_state.py
Offline defaults from Q1 2025 (manually updated) + FRED auto-fetch if available.

To update offline defaults: edit OFFLINE_STATE below with latest FRED/BEA/BLS data.
To enable live FRED: pip install fredapi, set FRED_API_KEY env variable.
"""

import os
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# OFFLINE DEFAULTS (Q1 2025 — update periodically)
# ═══════════════════════════════════════════════════════════════

OFFLINE_INPUTS = {
    "fedRate": 4.375,           # FRED: FEDFUNDS (Apr 2025 — Fed held after 3 cuts in late 2024)
    "govSpending": 5.3,         # BEA: Federal spending ~$5.3T annual rate
    "taxRate": 17.5,            # CBO: Federal revenue ~17.5% of GDP
    "moneySupplyGrowth": 3.8,   # FRED: M2SL YoY% (recovering from 2023 contraction)
    "tariffRate": 12.0,         # USITC: Effective avg tariff rate (post Trump 2025 tariffs)
    "oilPrice": 65.0,           # EIA: WTI average Q1 2025
    "laborForceGrowth": 0.4,    # BLS: Civilian labor force YoY%
    "productivityGrowth": 1.8,  # BLS: Nonfarm productivity YoY%
    "debtToGDP": 124.0,         # FRED: GFDEGDQ188S
}

OFFLINE_ACTUALS = {
    "gdpGrowth": 2.4,           # BEA: Q4 2024 advance estimate
    "inflation": 2.8,           # BLS: CPI-U YoY% (Mar 2025)
    "unemployment": 4.2,        # BLS: U-3 (Mar 2025)
    "currencyIndex": 104.0,     # ICE: DXY (Q1 2025 avg)
    "sp500Index": 5600,         # Yahoo: S&P 500 (Q1 2025 avg)
    "bondYield10Y": 4.35,       # FRED: DGS10 (Q1 2025 avg)
    "tradeBalance": -850,       # BEA: Goods+services balance annualized
    "consumerConfidence": 98.0, # Conference Board: CCI (Mar 2025)
}

OFFLINE_DATE = "Q1 2025"


def _try_fred_fetch():
    """Try to fetch latest data from FRED. Returns (inputs, actuals, date) or None."""
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        return None

    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)

        def latest(series_id, fallback=None):
            try:
                s = fred.get_series(series_id)
                val = s.dropna().iloc[-1]
                return float(val)
            except Exception:
                return fallback

        fed_rate = latest("FEDFUNDS", OFFLINE_INPUTS["fedRate"])
        cpi_yoy = latest("CPIAUCSL", OFFLINE_ACTUALS["inflation"])  # level, need YoY
        unemp = latest("UNRATE", OFFLINE_ACTUALS["unemployment"])
        dgs10 = latest("DGS10", OFFLINE_ACTUALS["bondYield10Y"])
        m2_level = latest("M2SL")  # level, need YoY
        debt_gdp = latest("GFDEGDQ188S", OFFLINE_INPUTS["debtToGDP"])

        # CPI: convert level to YoY% (need 12-month lag)
        try:
            cpi_series = fred.get_series("CPIAUCSL")
            cpi_now = cpi_series.dropna().iloc[-1]
            cpi_12m = cpi_series.dropna().iloc[-13] if len(cpi_series.dropna()) > 13 else cpi_now
            cpi_yoy = ((cpi_now / cpi_12m) - 1) * 100
        except Exception:
            cpi_yoy = OFFLINE_ACTUALS["inflation"]

        # M2: convert level to YoY%
        m2_growth = OFFLINE_INPUTS["moneySupplyGrowth"]
        try:
            m2_series = fred.get_series("M2SL")
            m2_now = m2_series.dropna().iloc[-1]
            m2_12m = m2_series.dropna().iloc[-13] if len(m2_series.dropna()) > 13 else m2_now
            m2_growth = ((m2_now / m2_12m) - 1) * 100
        except Exception:
            pass

        inputs = {
            **OFFLINE_INPUTS,
            "fedRate": fed_rate,
            "moneySupplyGrowth": round(m2_growth, 1),
            "debtToGDP": debt_gdp if debt_gdp else OFFLINE_INPUTS["debtToGDP"],
        }
        actuals = {
            **OFFLINE_ACTUALS,
            "inflation": round(cpi_yoy, 1),
            "unemployment": unemp,
            "bondYield10Y": dgs10,
        }
        date = f"FRED Live ({datetime.now().strftime('%b %Y')})"
        return inputs, actuals, date

    except ImportError:
        return None
    except Exception:
        return None


def get_current_state():
    """
    Returns (inputs, actuals, source_label).
    Tries FRED first, falls back to offline defaults.
    """
    result = _try_fred_fetch()
    if result:
        return result
    return OFFLINE_INPUTS.copy(), OFFLINE_ACTUALS.copy(), f"Offline ({OFFLINE_DATE})"

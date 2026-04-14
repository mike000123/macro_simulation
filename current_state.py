"""
Current Macro State — current_state.py

3-tier data strategy:
  1. FRED live (public CSV, no API key) → fetches + saves to cache
  2. Local cache (~/.macroscope_cache.json) → used when offline
  3. Hardcoded defaults (Q1 2025) → ultimate fallback

Cache auto-refreshes when online. When offline, uses last cached data
with age indicator so the analyst knows how stale it is.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

# ═══════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════

CACHE_FILE = Path.home() / ".macroscope_cache.json"

# ═══════════════════════════════════════════════════════════════
# TIER 3: HARDCODED DEFAULTS (Q1 2025)
# ═══════════════════════════════════════════════════════════════

OFFLINE_INPUTS = {
    "fedRate": 4.375,
    "govSpending": 5.3,
    "taxRate": 17.5,
    "moneySupplyGrowth": 3.8,
    "tariffRate": 12.0,
    "oilPrice": 65.0,
    "laborForceGrowth": 0.4,
    "productivityGrowth": 1.8,
    "debtToGDP": 124.0,
}

OFFLINE_ACTUALS = {
    "gdpGrowth": 2.4,
    "inflation": 2.8,
    "unemployment": 4.2,
    "currencyIndex": 104.0,
    "sp500Index": 5600,
    "goldPrice": 2900,
    "bondYield10Y": 4.35,
    "tradeBalance": -850,
    "consumerConfidence": 98.0,
}

OFFLINE_DATE = "Q1 2025"


# ═══════════════════════════════════════════════════════════════
# TIER 2: CACHE READ/WRITE
# ═══════════════════════════════════════════════════════════════

def _save_cache(inputs: dict, actuals: dict):
    """Save fetched data to local JSON cache."""
    try:
        cache = {
            "inputs": inputs,
            "actuals": actuals,
            "timestamp": datetime.now().isoformat(),
        }
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass  # cache write failure is non-critical


def _load_cache():
    """
    Load cached data. Returns (inputs, actuals, source_label) or None.
    """
    try:
        if not CACHE_FILE.exists():
            return None
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)

        ts = datetime.fromisoformat(cache["timestamp"])
        age = datetime.now() - ts

        if age.days > 90:
            # Cache older than 90 days — too stale, ignore
            return None

        if age.days == 0:
            age_str = f"{age.seconds // 3600}h ago"
        elif age.days == 1:
            age_str = "1 day ago"
        else:
            age_str = f"{age.days} days ago"

        label = f"Cached ({ts.strftime('%d %b %Y')}, {age_str})"
        return cache["inputs"], cache["actuals"], label

    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
# TIER 1: FRED LIVE FETCH
# ═══════════════════════════════════════════════════════════════

def _fred_csv(series_id: str) -> pd.Series:
    """Fetch a FRED series via public CSV endpoint. No API key required."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url, timeout=10)
    if df.empty or len(df.columns) < 2:
        raise ValueError(f"Empty FRED CSV for {series_id}")
    value_col = df.columns[1]
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    series = df[value_col].dropna()
    if series.empty:
        raise ValueError(f"No data for {series_id}")
    return series


def _latest(series_id: str, fallback: float) -> float:
    try:
        return float(_fred_csv(series_id).iloc[-1])
    except Exception:
        return float(fallback)


def _yoy(series_id: str, fallback: float) -> float:
    """Level series → year-over-year %."""
    try:
        s = _fred_csv(series_id)
        if len(s) < 13:
            return float(fallback)
        now, prev = float(s.iloc[-1]), float(s.iloc[-13])
        return ((now / prev) - 1.0) * 100.0 if prev != 0 else float(fallback)
    except Exception:
        return float(fallback)


def _try_fred_fetch():
    """
    Fetch latest data from FRED. On success, saves to cache.
    Returns (inputs, actuals, label) or None.
    """
    try:
        fed_rate = _latest("FEDFUNDS", OFFLINE_INPUTS["fedRate"])
        cpi_yoy = _yoy("CPIAUCSL", OFFLINE_ACTUALS["inflation"])
        unemp = _latest("UNRATE", OFFLINE_ACTUALS["unemployment"])
        dgs10 = _latest("DGS10", OFFLINE_ACTUALS["bondYield10Y"])
        m2_growth = _yoy("M2SL", OFFLINE_INPUTS["moneySupplyGrowth"])
        debt_gdp = _latest("GFDEGDQ188S", OFFLINE_INPUTS["debtToGDP"])

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

        # Save to cache for offline use
        _save_cache(inputs, actuals)

        label = f"FRED Live ({datetime.now().strftime('%d %b %Y')})"
        return inputs, actuals, label

    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════

def get_current_state():
    """
    Returns (inputs, actuals, source_label).

    Resolution order:
      1. FRED live fetch → saves cache on success
      2. Local cache file (~/.macroscope_cache.json)
      3. Hardcoded Q1 2025 defaults
    """
    # Tier 1: Try live FRED
    result = _try_fred_fetch()
    if result:
        return result

    # Tier 2: Try local cache
    cached = _load_cache()
    if cached:
        return cached

    # Tier 3: Hardcoded defaults
    return OFFLINE_INPUTS.copy(), OFFLINE_ACTUALS.copy(), f"Offline ({OFFLINE_DATE})"

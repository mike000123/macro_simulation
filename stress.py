"""
Stress Test Module — stress.py

Extracts crisis "shock profiles" from historical data and injects them
into forward simulations at a user-selected quarter.

A shock profile is the set of policy/macro deltas relative to pre-crisis
baseline — not raw values, but changes. This makes them era-independent
and reusable across different starting conditions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from data import DATASETS
import numpy as np


@dataclass
class ShockProfile:
    """
    A crisis signature expressed as quarterly deltas from pre-crisis baseline.
    Each field is a list of per-quarter changes to apply.
    """
    name: str
    description: str
    n_quarters: int                   # how many quarters the shock lasts
    fedRate_deltas: List[float] = field(default_factory=list)
    govSpending_deltas: List[float] = field(default_factory=list)
    taxRate_deltas: List[float] = field(default_factory=list)
    moneySupplyGrowth_deltas: List[float] = field(default_factory=list)
    tariffRate_deltas: List[float] = field(default_factory=list)
    oilPrice_deltas: List[float] = field(default_factory=list)
    laborForceGrowth_deltas: List[float] = field(default_factory=list)
    productivityGrowth_deltas: List[float] = field(default_factory=list)


def extract_shock_profile(crisis_key: str, shock_start_q: int = 0, shock_end_q: Optional[int] = None) -> ShockProfile:
    """
    Extract a shock profile from a historical crisis dataset.

    The profile captures the delta (change) of each input variable relative
    to the first quarter of the crisis period (pre-shock baseline).

    Args:
        crisis_key: Key into DATASETS (e.g. 'gfc', 'volcker')
        shock_start_q: Which quarter of the dataset is the pre-shock baseline (0 = first)
        shock_end_q: Last quarter to include (None = all)

    Returns:
        ShockProfile with per-quarter deltas
    """
    ds = DATASETS[crisis_key]
    inputs = ds.inputs.to_dict("records")
    n = len(inputs) if shock_end_q is None else min(shock_end_q + 1, len(inputs))

    baseline = inputs[shock_start_q]
    fields_map = {
        "fedRate": "fedRate_deltas",
        "govSpending": "govSpending_deltas",
        "taxRate": "taxRate_deltas",
        "moneySupplyGrowth": "moneySupplyGrowth_deltas",
        "tariffRate": "tariffRate_deltas",
        "oilPrice": "oilPrice_deltas",
        "laborForceGrowth": "laborForceGrowth_deltas",
        "productivityGrowth": "productivityGrowth_deltas",
    }

    profile = ShockProfile(
        name=ds.label,
        description=ds.desc,
        n_quarters=n - shock_start_q,
    )

    for inp_key, prof_key in fields_map.items():
        deltas = []
        for q in range(shock_start_q, n):
            deltas.append(inputs[q][inp_key] - baseline[inp_key])
        setattr(profile, prof_key, deltas)

    return profile


# ═══════════════════════════════════════════════════════════════
# PRE-BUILT SHOCK PROFILES
# ═══════════════════════════════════════════════════════════════

# Extract from the "crisis phase" of each dataset (not the full period)
# GFC: Q4 2007 (cracks) through Q4 2009 (trough) = indices 3-11
# Volcker: Q4 1979 (Volcker shock) through Q4 1982 = indices 3-15
# Gulf War: Q3 1990 (invasion) through Q2 1991 = indices 6-9
# Dot-Com: Q1 2001 (bust) through Q4 2002 = indices 4-11
# Modern: Q1 2020 (COVID) through Q4 2020 = indices 20-23

SHOCK_CONFIGS = {
    "gfc": {"key": "gfc", "start": 3, "end": 11,
            "label": "🏠 GFC / Credit Crisis",
            "desc": "Lehman-style credit freeze: rates slashed 475bp, massive fiscal injection, labor collapse, oil crash. 8-quarter shock."},
    "volcker": {"key": "volcker", "start": 3, "end": 15,
                "label": "📈 Volcker Rate Shock",
                "desc": "Aggressive tightening: rates surge to 18%+, money supply crushed. Tests what happens with extreme hawkish policy. 12-quarter shock."},
    "gulfwar": {"key": "gulfwar", "start": 6, "end": 11,
                "label": "🛢️ Oil Shock / Gulf War",
                "desc": "Sudden oil price doubling ($18→$35), mild recession. Tests energy supply disruption. 5-quarter shock."},
    "dotcom": {"key": "dotcom", "start": 4, "end": 11,
               "label": "💻 Tech Bust / 9-11",
               "desc": "Rate slashing 650bp, equity crash, mild labor decline. Tests bursting asset bubble. 8-quarter shock."},
    "covid": {"key": "modern", "start": 20, "end": 23,
              "label": "🦠 COVID Pandemic",
              "desc": "Extreme labor collapse (-6.5%), massive fiscal/monetary injection, oil crash. 4-quarter acute shock."},
}


def get_shock_profiles() -> Dict[str, ShockProfile]:
    """Build all pre-defined shock profiles."""
    profiles = {}
    for shock_id, cfg in SHOCK_CONFIGS.items():
        try:
            p = extract_shock_profile(cfg["key"], cfg["start"], cfg["end"])
            p.name = cfg["label"]
            p.description = cfg["desc"]
            profiles[shock_id] = p
        except Exception:
            pass
    return profiles


def apply_shock_to_params(
    base_params: dict,
    shock: ShockProfile,
    onset_quarter: int,
    severity: float = 1.0,
    n_total_quarters: int = 24,
) -> List[dict]:
    """
    Generate a per-quarter input series by injecting a shock profile
    into constant base parameters at a specified quarter.

    Args:
        base_params: The constant policy inputs (from sidebar sliders)
        shock: The crisis shock profile to inject
        onset_quarter: Which quarter (0-indexed) the shock begins
        severity: Multiplier on shock deltas (1.0 = historical, 0.5 = half, 2.0 = double)
        n_total_quarters: Total quarters in the simulation

    Returns:
        List of per-quarter input dicts
    """
    series = []
    shock_fields = [
        ("fedRate", "fedRate_deltas"),
        ("govSpending", "govSpending_deltas"),
        ("taxRate", "taxRate_deltas"),
        ("moneySupplyGrowth", "moneySupplyGrowth_deltas"),
        ("tariffRate", "tariffRate_deltas"),
        ("oilPrice", "oilPrice_deltas"),
        ("laborForceGrowth", "laborForceGrowth_deltas"),
        ("productivityGrowth", "productivityGrowth_deltas"),
    ]

    for q in range(n_total_quarters):
        params_q = base_params.copy()
        params_q["q"] = f"Q{q + 1}"

        shock_q = q - onset_quarter  # how far into the shock we are

        if 0 <= shock_q < shock.n_quarters:
            for param_key, delta_key in shock_fields:
                deltas = getattr(shock, delta_key)
                if shock_q < len(deltas):
                    delta = deltas[shock_q] * severity

                    # Apply delta with bounds
                    new_val = params_q[param_key] + delta

                    # Respect natural bounds
                    if param_key == "fedRate":
                        new_val = max(0.0, new_val)  # can't go negative
                    elif param_key == "oilPrice":
                        new_val = max(10.0, new_val)  # floor
                    elif param_key in ("tariffRate", "govSpending"):
                        new_val = max(0.0, new_val)

                    params_q[param_key] = round(new_val, 3)

        elif shock_q >= shock.n_quarters:
            # Post-shock: gradual recovery (80% reversion per quarter toward baseline)
            recovery_q = shock_q - shock.n_quarters
            recovery_factor = 0.8 ** (recovery_q + 1)  # exponential decay

            for param_key, delta_key in shock_fields:
                deltas = getattr(shock, delta_key)
                if deltas:
                    final_delta = deltas[-1] * severity * recovery_factor
                    new_val = base_params[param_key] + final_delta
                    if param_key == "fedRate":
                        new_val = max(0.0, new_val)
                    elif param_key == "oilPrice":
                        new_val = max(10.0, new_val)
                    params_q[param_key] = round(new_val, 3)

        # Ensure debtToGDP is always present
        if "debtToGDP" not in params_q:
            params_q["debtToGDP"] = base_params.get("debtToGDP", 124)

        series.append(params_q)

    return series

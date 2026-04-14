"""
Stress Test + Monte Carlo — stress.py

Two features:
1. Crisis shock injection: Hand-crafted exogenous shock profiles (NOT extracted
   from historical data with policy responses mixed in). Only the crisis trigger
   variables change — the engine generates recessions, rate cuts, etc. endogenously.

2. Monte Carlo cone: Adds random quarterly noise to policy inputs to generate
   probability distributions (10th/50th/90th percentile cones).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from engine import simulate, SimResult


# ═══════════════════════════════════════════════════════════════
# SHOCK PROFILES — hand-crafted exogenous shocks
# Only crisis TRIGGER variables change. Policy responses are endogenous.
# ═══════════════════════════════════════════════════════════════

@dataclass
class ShockProfile:
    name: str
    description: str
    n_quarters: int
    # Per-quarter deltas for each input variable
    fedRate_deltas: List[float] = field(default_factory=list)
    govSpending_deltas: List[float] = field(default_factory=list)
    taxRate_deltas: List[float] = field(default_factory=list)
    moneySupplyGrowth_deltas: List[float] = field(default_factory=list)
    tariffRate_deltas: List[float] = field(default_factory=list)
    oilPrice_deltas: List[float] = field(default_factory=list)
    laborForceGrowth_deltas: List[float] = field(default_factory=list)
    productivityGrowth_deltas: List[float] = field(default_factory=list)


def _pad(deltas, n):
    """Pad or truncate delta list to n quarters."""
    if len(deltas) >= n:
        return deltas[:n]
    return deltas + [deltas[-1]] * (n - len(deltas))


SHOCK_PROFILES: Dict[str, ShockProfile] = {
    "oil_shock": ShockProfile(
        name="🛢️ Oil Supply Shock",
        description="Oil price doubles over 2 quarters (like 1990 Kuwait invasion or 2022 Russia/Ukraine). "
                    "Triggers stagflation: GDP contracts via cost-push, inflation surges, consumer confidence drops.",
        n_quarters=6,
        oilPrice_deltas=[15, 35, 40, 30, 15, 5],       # +$40/bbl peak at Q3
        laborForceGrowth_deltas=[0, -0.2, -0.4, -0.3, -0.1, 0],  # mild labor hit
        productivityGrowth_deltas=[0, -0.3, -0.5, -0.3, -0.1, 0],
    ),
    "credit_crisis": ShockProfile(
        name="🏠 Credit / Banking Crisis",
        description="Lehman-style credit freeze. Interbank lending seizes, labor force contracts, "
                    "productivity collapses. The engine's FCI amplifier and panic equity channel activate.",
        n_quarters=8,
        laborForceGrowth_deltas=[0, -0.3, -0.8, -1.5, -2.0, -1.5, -0.8, -0.3],
        productivityGrowth_deltas=[0, -0.3, -0.8, -1.2, -1.0, -0.5, 0, 0.3],
        moneySupplyGrowth_deltas=[0, 1, 3, 6, 8, 6, 4, 2],     # QE response
        govSpending_deltas=[0, 0, 0.2, 0.5, 0.8, 0.6, 0.4, 0.2],  # fiscal stimulus
        oilPrice_deltas=[0, 5, 10, -20, -30, -20, -10, 0],       # oil crashes in crisis
    ),
    "rate_shock": ShockProfile(
        name="📈 Aggressive Rate Hike",
        description="Volcker-style emergency tightening: +300-500bp over 4 quarters to fight inflation. "
                    "Tests the economy's resilience to rapid monetary tightening.",
        n_quarters=8,
        fedRate_deltas=[0.5, 1.5, 2.5, 3.5, 4.0, 3.5, 3.0, 2.5],  # +400bp peak
        moneySupplyGrowth_deltas=[0, -1, -3, -5, -6, -5, -4, -3],   # M2 contracts
    ),
    "trade_war": ShockProfile(
        name="🛃 Trade War Escalation",
        description="Tariffs surge 20pp+ (like 2018-19 US-China but more severe). "
                    "Supply chains disrupted, productivity hit, retaliatory tariffs.",
        n_quarters=8,
        tariffRate_deltas=[5, 12, 20, 25, 25, 20, 15, 10],
        oilPrice_deltas=[0, 5, 10, 15, 10, 5, 0, 0],
        productivityGrowth_deltas=[0, -0.2, -0.5, -0.8, -0.6, -0.4, -0.2, 0],
    ),
    "pandemic": ShockProfile(
        name="🦠 Pandemic Shock",
        description="COVID-style lockdown: extreme labor force collapse, massive fiscal/monetary response, "
                    "oil crash. 4 acute quarters followed by sharp rebound.",
        n_quarters=6,
        laborForceGrowth_deltas=[-1, -6, 3, 1, 0.5, 0.2],
        productivityGrowth_deltas=[1, 3, 1.5, 0.5, 0, 0],      # remote work boost
        govSpending_deltas=[0.5, 2.5, 1.5, 0.8, 0.3, 0],       # stimulus
        taxRate_deltas=[0, -2, -1, -0.5, 0, 0],                 # tax cuts
        moneySupplyGrowth_deltas=[3, 15, 12, 8, 4, 1],          # money printing
        oilPrice_deltas=[-10, -30, -15, -5, 0, 5],              # oil crash
    ),
    "stagflation": ShockProfile(
        name="📉 Stagflation (Supply Shock + Weak Demand)",
        description="Combined oil shock + productivity collapse + tariff escalation. "
                    "Growth stalls while inflation rises — the worst macro combination.",
        n_quarters=8,
        oilPrice_deltas=[10, 25, 40, 50, 45, 35, 25, 15],
        tariffRate_deltas=[3, 8, 12, 15, 15, 12, 8, 5],
        productivityGrowth_deltas=[0, -0.5, -1.0, -1.2, -1.0, -0.8, -0.5, -0.2],
        laborForceGrowth_deltas=[0, -0.2, -0.5, -0.5, -0.3, -0.1, 0, 0],
    ),
}


def get_shock_profiles() -> Dict[str, ShockProfile]:
    return SHOCK_PROFILES


# ═══════════════════════════════════════════════════════════════
# SHOCK INJECTION
# ═══════════════════════════════════════════════════════════════

def apply_shock_to_params(
    base_params: dict,
    shock: ShockProfile,
    onset_quarter: int,
    severity: float = 1.0,
    n_total_quarters: int = 24,
) -> List[dict]:
    """
    Generate per-quarter input series by injecting a shock profile.
    Before onset: constant baseline. During shock: deltas applied.
    After shock: exponential recovery (80%/quarter decay).
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
        pq = base_params.copy()
        pq["q"] = f"Q{q + 1}"
        shock_q = q - onset_quarter

        if 0 <= shock_q < shock.n_quarters:
            for param_key, delta_key in shock_fields:
                deltas = getattr(shock, delta_key)
                if shock_q < len(deltas):
                    delta = deltas[shock_q] * severity
                    pq[param_key] = max(0 if param_key != "oilPrice" else 10,
                                        pq[param_key] + delta)
        elif shock_q >= shock.n_quarters:
            recovery_factor = 0.8 ** (shock_q - shock.n_quarters + 1)
            for param_key, delta_key in shock_fields:
                deltas = getattr(shock, delta_key)
                if deltas:
                    final_delta = deltas[-1] * severity * recovery_factor
                    pq[param_key] = max(0 if param_key != "oilPrice" else 10,
                                        base_params[param_key] + final_delta)

        if "debtToGDP" not in pq:
            pq["debtToGDP"] = base_params.get("debtToGDP", 124)
        series.append(pq)

    return series


# ═══════════════════════════════════════════════════════════════
# MONTE CARLO SIMULATION
# ═══════════════════════════════════════════════════════════════

# Standard deviations for quarterly noise on each input variable
# Based on historical quarterly variation in US data
MC_NOISE_STD = {
    "fedRate": 0.15,             # ±15bp typical quarter
    "govSpending": 0.05,         # ±$50B
    "taxRate": 0.2,              # ±0.2pp
    "moneySupplyGrowth": 1.0,    # ±1pp (M2 volatile)
    "tariffRate": 0.3,           # ±0.3pp
    "oilPrice": 5.0,             # ±$5/bbl
    "laborForceGrowth": 0.15,    # ±0.15pp
    "productivityGrowth": 0.3,   # ±0.3pp
}


def monte_carlo_simulate(
    base_params: dict,
    initial_conditions: dict,
    n_sims: int = 200,
    n_quarters: int = 24,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Run Monte Carlo simulation with random quarterly noise on inputs.

    Returns dict of {variable_name: (n_quarters, n_sims) array}
    for computing percentile cones.
    """
    rng = np.random.RandomState(seed)
    output_keys = ["gdpGrowth", "inflation", "unemployment", "sp500Index",
                   "goldPrice", "bondYield10Y", "currencyIndex", "tradeBalance",
                   "wageGrowth", "consumerConfidence"]

    # Pre-allocate output arrays
    results_array = {k: np.zeros((n_quarters, n_sims)) for k in output_keys}

    for sim_i in range(n_sims):
        # Generate noisy input series
        noisy_series = []
        for q in range(n_quarters):
            pq = base_params.copy()
            pq["q"] = f"Q{q + 1}"
            for param_key, std in MC_NOISE_STD.items():
                if param_key in pq:
                    noise = rng.normal(0, std)
                    # Cumulative random walk (correlated across quarters)
                    if q > 0 and noisy_series:
                        prev_val = noisy_series[-1].get(param_key, pq[param_key])
                        noise = 0.7 * (prev_val - base_params[param_key]) + 0.3 * noise + rng.normal(0, std * 0.3)
                    new_val = pq[param_key] + noise
                    # Bounds
                    if param_key == "fedRate":
                        new_val = max(0, new_val)
                    elif param_key == "oilPrice":
                        new_val = max(20, new_val)
                    elif param_key in ("tariffRate", "govSpending"):
                        new_val = max(0, new_val)
                    pq[param_key] = new_val
            if "debtToGDP" not in pq:
                pq["debtToGDP"] = base_params.get("debtToGDP", 124)
            noisy_series.append(pq)

        # Run simulation
        sim_results = simulate(noisy_series[0], noisy_series, initial_conditions)

        # Extract outputs
        for q_idx, r in enumerate(sim_results):
            for k in output_keys:
                results_array[k][q_idx, sim_i] = getattr(r, k)

    return results_array


def compute_percentiles(mc_results: Dict[str, np.ndarray],
                        pctiles: List[int] = [10, 50, 90]) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Compute percentile paths from Monte Carlo results.

    Returns: {variable: {10: array, 50: array, 90: array}}
    """
    out = {}
    for k, arr in mc_results.items():
        out[k] = {}
        for p in pctiles:
            out[k][p] = np.percentile(arr, p, axis=1)
    return out

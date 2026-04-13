"""
M4: Scoring Engine — scoring.py
Era-adaptive: scales derived from actual data volatility
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

SCORE_VARS = [
    {"k": "gdpGrowth",    "l": "GDP",   "w": 1.0, "base_scale": 5},
    {"k": "inflation",     "l": "CPI",   "w": 1.0, "base_scale": 5},
    {"k": "unemployment",  "l": "Unemp", "w": 1.0, "base_scale": 3},
    {"k": "bondYield10Y",  "l": "10Y",   "w": 0.8, "base_scale": 2},
    {"k": "currencyIndex", "l": "DXY",   "w": 0.5, "base_scale": 15},
    {"k": "sp500Index",    "l": "S&P",   "w": 0.4, "base_scale": 800},
]


@dataclass
class VarScore:
    label: str
    mae: float
    rmse: float
    bias: float
    directional: float  # 0-1
    score: float        # 0-100
    scale: float        # era-adaptive scale used
    weight: float
    pairs: list         # [{q, pred, actual, error}]


@dataclass
class BacktestScore:
    overall: float
    variables: Dict[str, VarScore]


def _adaptive_scale(actuals: list, key: str, base_scale: float) -> float:
    """Compute era-adaptive scoring scale from data volatility."""
    vals = [a[key] for a in actuals if key in a and a[key] is not None]
    if len(vals) < 3:
        return base_scale
    std = np.std(vals)
    return max(base_scale, std * 2.5)


def score(predictions: list, actuals: list) -> BacktestScore:
    """
    Score predictions against actuals with era-adaptive scales.

    Args:
        predictions: List of SimResult or dicts with predicted values
        actuals: List of dicts with actual values

    Returns:
        BacktestScore with overall and per-variable scores
    """
    variables = {}
    total_weight = 0
    total_score = 0

    n = min(len(predictions), len(actuals))

    for sv in SCORE_VARS:
        k = sv["k"]
        errors = []
        pairs = []

        for i in range(n):
            # Handle both SimResult objects and dicts
            pred = predictions[i]
            p_val = getattr(pred, k, None) if hasattr(pred, k) else pred.get(k)
            a_val = actuals[i].get(k) if isinstance(actuals[i], dict) else getattr(actuals[i], k, None)

            if p_val is not None and a_val is not None:
                err = p_val - a_val
                errors.append(err)
                q_label = actuals[i].get("q", f"Q{i+1}") if isinstance(actuals[i], dict) else f"Q{i+1}"
                pairs.append({"q": q_label, "pred": p_val, "actual": a_val, "error": err})

        if not errors:
            continue

        errors = np.array(errors)
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        bias = np.mean(errors)

        # Directional accuracy
        dir_correct = 0
        dir_total = 0
        for i in range(1, len(pairs)):
            a_delta = pairs[i]["actual"] - pairs[i-1]["actual"]
            p_delta = pairs[i]["pred"] - pairs[i-1]["pred"]
            if a_delta != 0:
                dir_total += 1
                if a_delta * p_delta > 0:
                    dir_correct += 1

        directional = dir_correct / dir_total if dir_total > 0 else 0

        # Era-adaptive scale
        actual_dicts = [a if isinstance(a, dict) else {} for a in actuals]
        scale = _adaptive_scale(actual_dicts, k, sv["base_scale"])
        sc = max(0, min(100, 100 * (1 - mae / scale)))

        variables[k] = VarScore(
            label=sv["l"], mae=mae, rmse=rmse, bias=bias,
            directional=directional, score=sc, scale=scale,
            weight=sv["w"], pairs=pairs,
        )
        total_weight += sv["w"]
        total_score += sc * sv["w"]

    overall = total_score / total_weight if total_weight > 0 else 0
    return BacktestScore(overall=overall, variables=variables)

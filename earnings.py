"""
Earnings-Based Equity Model — earnings.py

Two approaches to S&P 500 valuation:
1. Bottom-up: EPS estimates → P/E multiple → fair value
2. Top-down: nominal GDP growth → corporate earnings proxy → valuation

Users can input custom EPS estimates or use the macro-derived defaults.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List


# ═══════════════════════════════════════════════════════════════
# DEFAULT CONSENSUS ESTIMATES (editable by analyst)
# ═══════════════════════════════════════════════════════════════

@dataclass
class EarningsConsensus:
    """
    Analyst consensus inputs for the earnings-based equity model.
    All values are annualized. The model interpolates quarterly.
    """
    # S&P 500 trailing EPS (last 4 quarters)
    trailing_eps: float = 240.0       # ~$240 as of Q1 2025

    # Forward EPS growth estimates (annualized %)
    eps_growth_y1: float = 12.0       # Year 1: consensus ~12% (AI/tech driven)
    eps_growth_y2: float = 10.0       # Year 2: slightly lower
    eps_growth_y3: float = 8.0        # Year 3: normalizing
    eps_growth_y4_6: float = 6.0      # Years 4-6: long-run trend

    # Buyback yield (% of market cap returned via buybacks annually)
    buyback_yield: float = 2.0        # ~2% for S&P 500

    # Equity risk premium (ERP) — excess return demanded over risk-free
    equity_risk_premium: float = 4.5  # historical avg ~4.5%

    # Terminal P/E assumption (what the market pays at end of projection)
    terminal_pe: Optional[float] = None  # None = model-derived from rates


def build_eps_path(consensus: EarningsConsensus, n_quarters: int = 24) -> np.ndarray:
    """
    Build quarterly EPS trajectory from annual growth estimates.

    Returns array of forward EPS for each quarter.
    """
    eps = np.zeros(n_quarters)
    current_eps = consensus.trailing_eps

    for q in range(n_quarters):
        year = q / 4  # 0-6
        if year < 1:
            annual_growth = consensus.eps_growth_y1
        elif year < 2:
            annual_growth = consensus.eps_growth_y2
        elif year < 3:
            annual_growth = consensus.eps_growth_y3
        else:
            annual_growth = consensus.eps_growth_y4_6

        quarterly_growth = (1 + annual_growth / 100) ** 0.25 - 1
        current_eps *= (1 + quarterly_growth)
        eps[q] = current_eps

    return eps


def compute_fair_pe(risk_free_rate: float, erp: float, eps_growth: float) -> float:
    """
    Gordon Growth Model-style P/E ratio.
    P/E = 1 / (required_return - growth)
    with guardrails to avoid division by zero or negative P/E.
    """
    required_return = (risk_free_rate + erp) / 100
    growth = eps_growth / 100
    denominator = required_return - growth
    if denominator <= 0.005:
        return 30.0  # cap at 30x when growth approaches required return
    pe = 1.0 / denominator
    return max(10, min(35, pe))  # realistic P/E range


def earnings_equity_model(
    consensus: EarningsConsensus,
    macro_results: list,
    starting_sp: float = 5600.0,
) -> List[float]:
    """
    Compute S&P 500 trajectory from earnings consensus + macro conditions.

    The model works as follows:
    1. EPS path: interpolates from analyst growth estimates per year
    2. P/E multiple: derived from 10Y yield (via Gordon Growth) + macro adjustment
       - Higher rates → lower P/E (discount rate up)
       - Higher FCI → lower P/E (risk premium up)
       - Higher GDP → slight P/E expansion (confidence)
    3. Fair value = EPS × P/E
    4. Actual path: blends toward fair value with momentum + mean-reversion

    Args:
        consensus: Analyst earnings estimates
        macro_results: List of SimResult from engine (for rates, FCI, GDP)
        starting_sp: Current S&P 500 level

    Returns:
        List of S&P 500 values for each quarter
    """
    n = len(macro_results)
    eps_path = build_eps_path(consensus, n)
    sp_path = []
    sp = starting_sp
    prev_sp = sp

    for q in range(n):
        mr = macro_results[q]
        eps = eps_path[q]

        # Year for growth rate lookup
        year = q / 4
        if year < 1:
            fwd_growth = consensus.eps_growth_y1
        elif year < 2:
            fwd_growth = consensus.eps_growth_y2
        elif year < 3:
            fwd_growth = consensus.eps_growth_y3
        else:
            fwd_growth = consensus.eps_growth_y4_6

        # P/E from fundamentals
        if consensus.terminal_pe is not None and year > 3:
            base_pe = consensus.terminal_pe
        else:
            base_pe = compute_fair_pe(mr.bondYield10Y, consensus.equity_risk_premium, fwd_growth)

        # Macro adjustments to P/E
        fci_adj = -2.0 * max(0, mr.fci)        # tight FCI compresses multiples
        gdp_adj = 0.5 * (mr.gdpGrowth - 2.0)   # above-trend GDP supports multiples
        rate_adj = -0.8 * max(0, mr.bondYield10Y - 4.5)  # rates above 4.5% compress

        adjusted_pe = max(12, min(30, base_pe + fci_adj + gdp_adj + rate_adj))

        # Fair value
        fair_value = eps * adjusted_pe

        # Buyback contribution (reduces shares → mechanically lifts price)
        buyback_lift = sp * (consensus.buyback_yield / 100) * 0.25  # quarterly

        # Blend: 80% mean-reversion toward fair value, 20% momentum
        mean_rev_speed = 0.08  # per quarter
        momentum = 0.15 * (sp - prev_sp) if q > 0 else 0

        prev_sp = sp
        sp = sp + mean_rev_speed * (fair_value - sp) + momentum + buyback_lift

        sp_path.append(round(max(1000, sp)))

    return sp_path


# ═══════════════════════════════════════════════════════════════
# MACRO-DERIVED CONSENSUS (auto-generate from engine state)
# ═══════════════════════════════════════════════════════════════

def consensus_from_macro(
    gdp_growth: float = 2.4,
    inflation: float = 2.8,
    fed_rate: float = 4.375,
    productivity_growth: float = 1.8,
    current_sp: float = 5600,
) -> EarningsConsensus:
    """
    Auto-derive reasonable earnings estimates from macro conditions.
    This is the 'no analyst input' fallback.

    Logic:
    - EPS growth Y1 ≈ nominal GDP growth + productivity premium + buyback effect
    - EPS growth Y2-Y3 mean-reverts toward nominal GDP growth
    - Long-run growth ≈ nominal GDP growth
    """
    nominal_gdp = gdp_growth + inflation
    prod_premium = max(0, productivity_growth - 1.0) * 2  # extra growth from productivity
    buyback_eps_boost = 2.0  # ~2% fewer shares per year → EPS boost

    y1_growth = nominal_gdp + prod_premium + buyback_eps_boost
    y2_growth = nominal_gdp * 0.9 + buyback_eps_boost  # slight fade
    y3_growth = nominal_gdp * 0.85 + buyback_eps_boost * 0.8
    lt_growth = nominal_gdp * 0.8 + buyback_eps_boost * 0.5

    # ERP adjusts with rate level (higher rates → higher ERP demanded)
    erp = 4.0 + max(0, fed_rate - 4) * 0.3

    # Trailing EPS estimate from current price and reasonable P/E
    implied_pe = current_sp / max(180, current_sp / 25)  # rough
    trailing_eps = current_sp / max(18, min(25, implied_pe))

    return EarningsConsensus(
        trailing_eps=round(trailing_eps, 1),
        eps_growth_y1=round(min(20, max(2, y1_growth)), 1),
        eps_growth_y2=round(min(15, max(2, y2_growth)), 1),
        eps_growth_y3=round(min(12, max(2, y3_growth)), 1),
        eps_growth_y4_6=round(min(10, max(2, lt_growth)), 1),
        buyback_yield=2.0,
        equity_risk_premium=round(erp, 1),
    )

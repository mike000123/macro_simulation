# MacroScope V6 — Macroeconomic Policy Simulator

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501

## Three Modes

### 🔮 Forecast from Now (default)
Loads current US macro conditions (Q1 2025 defaults or live FRED data) and projects
24 quarters forward. Adjust sidebar sliders to test policy changes from reality.

### 🎛️ What-If Scenarios
Hypothetical mode with presets (Hawkish, Stimulus, Trade War, Stagflation, Tech Boom).
Full control over all 9 policy inputs. Starts from neutral baseline.

### 🔬 Historical Backtest
Validates against 5 crisis periods (Volcker 1979, Gulf War 1989, Dot-Com 2000, GFC 2007, Modern 2015).
Era-adaptive scoring. 55-65 = institutional-grade.

## Live FRED Data (optional)

```bash
pip install fredapi
export FRED_API_KEY=your_key_here  # free from https://fred.stlouisfed.org/docs/api/api_key.html
streamlit run app.py
```

Auto-fetches: Fed Funds, CPI, Unemployment, 10Y Yield, M2, Debt/GDP.
Falls back to built-in Q1 2025 defaults if FRED unavailable.

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `app.py` | ~260 | Streamlit UI — 3 modes, 5 tabs, Plotly charts |
| `engine.py` | ~370 | 21-channel simulation engine + 90 coefficients |
| `scoring.py` | ~120 | Era-adaptive backtest scoring |
| `data.py` | ~200 | 5 crisis datasets (112 quarters, 1979-2024) |
| `current_state.py` | ~100 | Current macro state (offline + FRED auto-fetch) |
| `requirements.txt` | 5 | Python dependencies |

## Updating Current State

Edit `OFFLINE_INPUTS` and `OFFLINE_ACTUALS` in `current_state.py` with latest
FRED/BEA/BLS data. Or install `fredapi` for automatic updates.

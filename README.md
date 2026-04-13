# MacroScope V6 — Python Edition

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI — interactive dashboard, backtests, help |
| `engine.py` | Simulation engine (21 channels, 90 coefficients) + coefficient dict |
| `scoring.py` | Backtest scoring with era-adaptive scales |
| `data.py` | Historical data for 5 crisis periods (112 quarters) |
| `requirements.txt` | Python dependencies |

## Usage

### Forward Simulation
Adjust policy sliders in the sidebar. Charts update instantly across 4 tabs (Dashboard, Growth, Prices, Markets).

### Backtest
Select a crisis period in the 🔬 Backtest tab. The model receives actual historical policy inputs and is scored against what really happened.

### Extending
- Add new crisis periods: add entries to `DATASETS` dict in `data.py`
- Tune coefficients: edit `K` dict in `engine.py`
- Add variables to scoring: append to `SCORE_VARS` in `scoring.py`
- Connect to FRED API: replace static data with `fredapi` live pulls

## Engine Architecture
21 transmission channels, era-adaptive (no hardcoded dates):
- Monetary (ZLB, regime-dependent), Fiscal (logistic saturation), Phillips Curve (non-linear)
- Debt dynamics (Domar with tipping points), Inflation expectations (adaptive + de-anchoring)
- Financial conditions (endogenous FCI, yield curve, equity crash detection)
- Trade (J-curve, retaliation), Unemployment (hysteresis, crisis-amplified Okun's)

V6 crisis fixes: sqrt bond passthrough, FCI crisis amplifier, high-rate GDP sensitivity,
adaptive Okun's, labor collapse → unemployment channel, symmetric deflation, anchor cap 8%.

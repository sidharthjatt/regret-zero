# RegretZero

A decision-aware demand & inventory engine. Forecasts weekly product demand
at multiple quantiles and (planned) turns those forecasts into inventory
decisions that minimize decision regret.

## Pipeline

| Stage | Script | Output |
|---|---|---|
| Data prep | `src/01_data_prep.py` | `data/demand.csv` (stock_code, week_start_date, demand) |
| EDA | `src/02_demand_eda.py` | `outputs/weekly_demand_distribution.png` |
| Forecast | `src/02_forecast.py` | `outputs/forecasts.csv` (P50 / P90 per product-week) |
| Optimize | `src/03_optimize.py` *(planned)* | newsvendor + critical ratio + decision-regret |

Run scripts from the project root, e.g. `python src/01_data_prep.py`.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Setup notes

- **macOS — LightGBM needs `libomp`.** LightGBM's compiled binary depends on
  the OpenMP runtime, which is *not* bundled with the pip wheel. Without it,
  `import lightgbm` fails with `Library not loaded: @rpath/libomp.dylib`.
  Install it via Homebrew (it cannot come from pip):

  ```bash
  brew install libomp
  ```

  `libomp` is keg-only, but LightGBM looks for it at
  `/opt/homebrew/opt/libomp/lib/libomp.dylib`, where Homebrew places it — no
  extra env vars are needed just to run the forecaster.

## Modelling notes

- Forecasting uses **LightGBM quantile regression (pinball loss)**, not a
  Gaussian point forecast, because weekly demand is extremely right-skewed.
- Extreme weekly-demand values are **kept, not winsorized** — see
  `reports/01_outlier_decision.md`.
- Latest forecast calibration: P90 coverage ≈ 91.7% on the held-out test
  weeks (target 90%); see `src/02_forecast.py`.

## Future polish

- Break forecast metrics (pinball / coverage) out by demand-volume tier — the
  aggregate hides per-SKU behaviour, especially for intermittent-demand items.

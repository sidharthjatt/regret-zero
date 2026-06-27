"""
02_forecast.py — RegretZero forecasting layer

Trains LightGBM quantile-regression models on data/demand.csv to forecast
weekly demand per product at multiple quantiles (P50, P90). Quantile models
(pinball loss) are used instead of a Gaussian point forecast because weekly
demand is extremely right-skewed and heavy-tailed; see
reports/01_outlier_decision.md. Extreme weeks are kept, not winsorized.

LEAKAGE IS THE PRIMARY CONCERN. This is time-series data, so:
  * the split is strictly time-based (train = earlier weeks, test = most
    recent weeks); nothing is shuffled,
  * every feature is a backward-looking shift computed PER product, so a row
    only ever sees its own product's past, never the present or future,
  * no global statistic (scaler / target encoding) is fit on the full data.

Input : data/demand.csv            (stock_code, week_start_date, demand)
Output: outputs/forecasts.csv      (stock_code, week_start_date, actual, p50, p90)

Run from the project root:
    python src/02_forecast.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

# --------------------------------------------------------------------------
# Paths (resolved from project root, consistent with 01_data_prep.py).
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMAND_PATH = PROJECT_ROOT / "data" / "demand.csv"
FORECAST_PATH = PROJECT_ROOT / "outputs" / "forecasts.csv"

# --------------------------------------------------------------------------
# Config — everything tunable lives up top.
# --------------------------------------------------------------------------
# Quantiles to forecast. These are chosen to match the optimizer's per-tier
# critical ratios (P33/P67/P82) so 03_optimize can order the true F^{-1}(CR);
# P50 and P90 are kept too for calibration diagnostics. Add more here to
# extend; the rest of the script adapts automatically.
QUANTILES = [0.333, 0.5, 0.667, 0.818, 0.9]

TEST_WEEKS = 12   # most-recent weeks held out for the test set
VALID_WEEKS = 8   # weeks just before the test set, used for early stopping

LAGS = [1, 2, 3, 4]      # demand k weeks ago
ROLL_WINDOW = 4          # window for rolling mean/std of recent demand

# Reindex each product to a continuous weekly grid and fill missing weeks
# with demand=0? A week with no sales is genuinely zero demand, and without
# this the lag features would be wrong (gaps would make "1 week ago" mean
# "previous week that happened to have a sale"). Strongly recommended True.
FILL_MISSING_WEEKS = True

RANDOM_STATE = 42


def load_demand() -> pd.DataFrame:
    """Load the cleaned weekly demand table with parsed dates, sorted."""
    df = pd.read_csv(DEMAND_PATH, parse_dates=["week_start_date"])
    return df.sort_values(["stock_code", "week_start_date"]).reset_index(drop=True)


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to a complete product x week grid, filling gaps with 0 demand.

    The canonical calendar is every Monday from the global first to the global
    last week (7-day steps from the first Monday, so all points are Mondays).
    Each product is reindexed from ITS first observed week to the global last
    week — never earlier, which would invent demand before the product existed.
    """
    if not FILL_MISSING_WEEKS:
        return df

    full_weeks = pd.date_range(
        df["week_start_date"].min(), df["week_start_date"].max(), freq="7D"
    )

    # Cross product of (product, full calendar), then keep only weeks at or
    # after each product's first appearance.
    first_week = (
        df.groupby("stock_code")["week_start_date"].min().rename("first_week").reset_index()
    )
    grid = first_week.merge(
        pd.DataFrame({"week_start_date": full_weeks}), how="cross"
    )
    grid = grid[grid["week_start_date"] >= grid["first_week"]].drop(columns="first_week")

    panel = grid.merge(df, on=["stock_code", "week_start_date"], how="left")
    panel["demand"] = panel["demand"].fillna(0).astype("int64")
    return panel.sort_values(["stock_code", "week_start_date"]).reset_index(drop=True)


def add_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add causal (past-only) features, computed per product.

    Lags use groupby().shift(k) so they cannot cross product boundaries.
    Rolling stats are computed on shift(1) first, so the window covers weeks
    t-1 .. t-ROLL_WINDOW and NEVER includes the current week's demand — the
    classic rolling-leakage trap.
    """
    g = panel.groupby("stock_code")["demand"]

    for k in LAGS:
        panel[f"lag_{k}"] = g.shift(k)

    # shift(1) then rolling -> window is strictly in the past, current excluded.
    panel["roll_mean_4"] = g.transform(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=ROLL_WINDOW).mean()
    )
    panel["roll_std_4"] = g.transform(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=ROLL_WINDOW).std()
    )

    # Calendar / seasonality features — known ahead of time, so causal.
    iso = panel["week_start_date"].dt.isocalendar()
    panel["woy"] = iso["week"].astype("int32")
    panel["month"] = panel["week_start_date"].dt.month.astype("int32")

    # Product identity as a categorical feature (lets the model learn
    # per-product demand levels). Identity mapping -> no leakage.
    panel["stock_code_cat"] = panel["stock_code"].astype("category")

    feature_cols = (
        [f"lag_{k}" for k in LAGS]
        + ["roll_mean_4", "roll_std_4", "woy", "month", "stock_code_cat"]
    )

    # Drop warm-up rows where the longest lag / rolling window isn't available
    # yet (each product's first ROLL_WINDOW weeks).
    before = len(panel)
    panel = panel.dropna(subset=[f"lag_{max(LAGS)}", "roll_mean_4", "roll_std_4"])
    print(f"dropped {before - len(panel):,} warm-up rows (insufficient history)")
    return panel.reset_index(drop=True), feature_cols


def time_split(panel: pd.DataFrame):
    """Strict chronological split into train / valid / test by week."""
    weeks = np.sort(panel["week_start_date"].unique())
    test_start = weeks[-TEST_WEEKS]
    valid_start = weeks[-(TEST_WEEKS + VALID_WEEKS)]

    train = panel[panel["week_start_date"] < valid_start]
    valid = panel[
        (panel["week_start_date"] >= valid_start)
        & (panel["week_start_date"] < test_start)
    ]
    test = panel[panel["week_start_date"] >= test_start]

    print("\n--- Time-based split (no shuffling) ---")
    for name, part in [("train", train), ("valid", valid), ("test", test)]:
        print(
            f"{name:>5}: {len(part):>8,} rows  "
            f"{part['week_start_date'].min().date()} -> "
            f"{part['week_start_date'].max().date()}"
        )
    return train, valid, test


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Mean pinball (quantile) loss — the objective LightGBM optimizes."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def train_quantile_model(q, X_fit, y_fit, X_valid, y_valid, cat_features):
    """Train one LightGBM quantile model with early stopping on the
    validation set (which lies strictly before the test period)."""
    model = LGBMRegressor(
        objective="quantile",
        alpha=q,
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_valid, y_valid)],
        eval_metric="quantile",
        categorical_feature=cat_features,
        callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
    )
    return model


def main() -> None:
    if not DEMAND_PATH.exists():
        raise FileNotFoundError(
            f"{DEMAND_PATH} not found — run src/01_data_prep.py first."
        )

    print("Loading demand ...")
    df = load_demand()
    print(f"loaded {len(df):,} observed product-weeks, {df['stock_code'].nunique():,} products")

    panel = build_panel(df)
    if FILL_MISSING_WEEKS:
        print(f"panel after zero-fill: {len(panel):,} product-weeks")

    panel, feature_cols = add_features(panel)
    cat_features = ["stock_code_cat"]

    train, valid, test = time_split(panel)

    X_fit, y_fit = train[feature_cols], train["demand"]
    X_valid, y_valid = valid[feature_cols], valid["demand"]
    X_test, y_test = test[feature_cols], test["demand"]

    # Train one model per quantile and collect test predictions.
    print("\n--- Training quantile models ---")
    preds = {}
    for q in QUANTILES:
        print(f"training alpha={q} ...")
        model = train_quantile_model(q, X_fit, y_fit, X_valid, y_valid, cat_features)
        preds[q] = model.predict(X_test)

    # ---- Monotonic rearrangement (fix quantile crossings) -----------------
    # The quantile models are trained independently, so their predictions can
    # cross (e.g. P50 > P67 on some rows). We enforce coherence by sorting each
    # row's predictions ascending and re-assigning them to the ascending
    # quantiles. This is the standard monotone-rearrangement fix and cannot
    # make calibration worse (it only reorders, never invents, values).
    qs_sorted = sorted(QUANTILES)
    stacked = np.column_stack([preds[q] for q in qs_sorted])
    stacked = np.sort(stacked, axis=1)
    for i, q in enumerate(qs_sorted):
        preds[q] = stacked[:, i]

    # ---- Evaluation -------------------------------------------------------
    print("\n--- Test-set evaluation ---")
    y_true = y_test.to_numpy()
    for q in QUANTILES:
        pl = pinball_loss(y_true, preds[q], q)
        coverage = float(np.mean(y_true <= preds[q]))
        print(
            f"P{round(q*100):<2}  pinball={pl:8.4f}   "
            f"coverage(actual<=pred)={coverage*100:5.1f}%  (target {q*100:.1f}%)"
        )

    # Quantile-crossing diagnostic across ALL adjacent pairs (should be ~0
    # after the monotonic rearrangement above).
    print("\n--- Quantile crossings (adjacent pairs) ---")
    any_cross = np.zeros(len(y_true), dtype=bool)
    for lo, hi in zip(qs_sorted[:-1], qs_sorted[1:]):
        cross = preds[hi] < preds[lo] - 1e-9
        any_cross |= cross
        print(f"P{round(lo*100)} > P{round(hi*100)}: {cross.mean()*100:.3f}%")
    print(f"any adjacent crossing: {any_cross.mean()*100:.3f}%")

    # ---- Save forecasts ---------------------------------------------------
    out = pd.DataFrame(
        {
            "stock_code": test["stock_code"].to_numpy(),
            "week_start_date": test["week_start_date"].to_numpy(),
            "actual": y_true,
        }
    )
    for q in QUANTILES:
        out[f"p{round(q*100)}"] = preds[q]
    out = out.sort_values(["stock_code", "week_start_date"]).reset_index(drop=True)
    out.to_csv(FORECAST_PATH, index=False)
    print(f"\nWrote {FORECAST_PATH.relative_to(PROJECT_ROOT)} ({len(out):,} rows)")

    print("\nsample forecasts:")
    print(out.head(8).to_string(index=False))


if __name__ == "__main__":
    main()

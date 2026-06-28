"""
03_optimize.py — RegretZero inventory optimizer & decision-regret layer

This is the core of RegretZero: it turns the quantile demand forecasts from
02_forecast.py into ordering decisions, and proves that a DECISION-AWARE order
beats an ACCURACY-FIRST order in rupee terms — i.e. the best forecast is not
necessarily the best decision.

THE NEWSVENDOR LOGIC
--------------------
Each product-week is a single-period inventory problem: we choose an order
quantity Q before demand D is known.
  * Over-order  -> pay an overage (holding) cost Co per excess unit.
  * Under-order -> pay an underage (stockout) cost Cu per unit short.
Raising Q by one more unit pays off while the expected stockout saving exceeds
the expected holding cost:  P(D > Q)*Cu > P(D <= Q)*Co. The optimum is where
they balance:

    P(D <= Q*) = Cu / (Cu + Co) = CR   =>   Q* = F^{-1}(CR)

So the cost-optimal order is the CR-th QUANTILE of demand — exactly what our
quantile models estimate.

  * accuracy_first : order = P50  (median point forecast, ignores asymmetry)
  * decision_aware : order = CR-th quantile (accounts for the cost asymmetry)

PER-PRODUCT CRITICAL RATIO (price-tier margins)
-----------------------------------------------
We have no real cost data, so costs are transparent assumptions. To make the
critical ratio genuinely VARY across products (otherwise "CR per product" is
pointless), margin is tiered by unit price:

  * Products are split into price tiers (low / mid / premium) by unit-price
    quantiles.
  * Each tier gets its own MARGIN_FRACTION. Premium products carry a higher
    margin, so a stockout forfeits more profit -> higher Cu -> higher CR.
    Cheap commodity products carry a thin margin -> lower CR.
  * HOLDING_FRACTION is kept constant across tiers for simplicity.

  CR = margin / (margin + HOLDING_FRACTION)

Each product orders the NEAREST trained quantile to its critical ratio, so it
approximates the newsvendor optimum F^{-1}(CR). The forecaster trains
P33/P67/P82 to match the three tier CRs, so low->P33, mid->P67, premium->P82.
All thresholds and fractions are top-of-file and easy to change.

Input : outputs/forecasts.csv     (stock_code, week_start_date, actual, p50, p90)
        data/prices.csv           (stock_code, unit_price — from 01_data_prep)
Output: outputs/regret_by_product.csv

Run from the project root:
    python src/03_optimize.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Shared decision logic — single source of truth (src/optimizer.py). Importable
# here because running `python src/03_optimize.py` puts src/ on sys.path.
from optimizer import HOLDING_FRACTION, TIER_LABELS, score

# --------------------------------------------------------------------------
# Paths (resolved from project root, consistent with the other scripts).
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FORECAST_PATH = PROJECT_ROOT / "outputs" / "forecasts.csv"
PRICES_PATH = PROJECT_ROOT / "data" / "prices.csv"
RAW_PATH = PROJECT_ROOT / "data" / "online_retail_II.csv"  # fallback only
REGRET_PATH = PROJECT_ROOT / "outputs" / "regret_by_product.csv"

# --------------------------------------------------------------------------
# Cost model (tier margins, holding fraction, critical-ratio -> quantile,
# asymmetric cost) lives in src/optimizer.py — the single source of truth
# shared with the dashboard. See that module's docstring for the full
# benchmark justification of the cost assumptions. This file owns only the
# I/O and the reporting; the math is imported.
# --------------------------------------------------------------------------


def load_forecasts() -> pd.DataFrame:
    """Load the test-set forecasts. stock_code kept as string (codes can be
    alphanumeric, e.g. '79323P')."""
    return pd.read_csv(
        FORECAST_PATH,
        dtype={"stock_code": "string"},
        parse_dates=["week_start_date"],
    )


def _price_from_raw() -> pd.Series:
    """Fallback: recompute mean unit price per product from the raw file,
    applying the same minimal sales filters as 01_data_prep."""
    raw = pd.read_csv(
        RAW_PATH,
        usecols=["Invoice", "StockCode", "Quantity", "Price"],
        dtype={"Invoice": "string", "StockCode": "string",
               "Quantity": "int64", "Price": "float64"},
    )
    raw = raw[~raw["Invoice"].str.startswith("C", na=False)]
    raw = raw[(raw["Quantity"] > 0) & (raw["Price"] > 0)]
    price = raw.groupby("StockCode")["Price"].mean()
    price.index.name = "stock_code"
    return price.rename("unit_price")


def load_unit_price() -> pd.Series:
    """Load per-product unit price, preferring the small data/prices.csv.

    If prices.csv is missing (e.g. 01_data_prep hasn't been re-run yet), fall
    back to computing it from the raw file ONCE and cache it to prices.csv, so
    subsequent runs are decoupled from the 90 MB raw data.
    """
    if PRICES_PATH.exists():
        s = pd.read_csv(PRICES_PATH, dtype={"stock_code": "string"})
        return s.set_index("stock_code")["unit_price"]

    print("data/prices.csv not found — computing from raw once and caching ...")
    price = _price_from_raw()
    price.reset_index().to_csv(PRICES_PATH, index=False)
    print(f"cached {PRICES_PATH.relative_to(PROJECT_ROOT)} ({len(price):,} products)")
    return price


def main() -> None:
    if not FORECAST_PATH.exists():
        raise FileNotFoundError(
            f"{FORECAST_PATH} not found — run src/02_forecast.py first."
        )

    print("Loading forecasts ...")
    df = load_forecasts()
    print(f"loaded {len(df):,} test product-weeks, {df['stock_code'].nunique():,} products")

    # Per-product price (Series) -> DataFrame [stock_code, unit_price] for the
    # shared scorer. score() returns the per-product-week table with realized
    # costs, plus the per-product economics (used in the tier summary below).
    price = load_unit_price()
    df, econ = score(df, price.reset_index())

    # Actual-demand vector (for the RMSE metric below).
    actual = df["actual"].to_numpy(dtype=float)

    total_acc = df["cost_accuracy"].sum()
    total_dec = df["cost_decision"].sum()
    savings = total_acc - total_dec
    savings_pct = 100 * savings / total_acc if total_acc else 0.0

    # --- RMSE per strategy (accuracy metric, deliberately separate from cost) ---
    rmse_acc = float(np.sqrt(np.mean((df["accuracy_order"] - actual) ** 2)))
    rmse_dec = float(np.sqrt(np.mean((df["decision_order"] - actual) ** 2)))

    # --- Per-product breakdown ---
    by_product = (
        df.groupby("stock_code")
        .agg(
            unit_price=("unit_price", "first"),
            tier=("tier", "first"),
            margin=("margin", "first"),
            cu=("cu", "first"),
            co=("co", "first"),
            cr=("cr", "first"),
            chosen_q=("chosen_q", "first"),
            n_weeks=("actual", "size"),
            cost_accuracy=("cost_accuracy", "sum"),
            cost_decision=("cost_decision", "sum"),
        )
        .reset_index()
    )
    by_product["savings"] = by_product["cost_accuracy"] - by_product["cost_decision"]
    rmse_per = (
        df.assign(
            se_acc=(df["accuracy_order"] - actual) ** 2,
            se_dec=(df["decision_order"] - actual) ** 2,
        )
        .groupby("stock_code")
        .agg(rmse_accuracy=("se_acc", lambda s: np.sqrt(s.mean())),
             rmse_decision=("se_dec", lambda s: np.sqrt(s.mean())))
        .reset_index()
    )
    by_product = by_product.merge(rmse_per, on="stock_code")
    by_product = by_product.sort_values("savings", ascending=False).reset_index(drop=True)
    by_product.to_csv(REGRET_PATH, index=False)

    # --- Summary ---
    print("\n--- Cost model (price-tier margins) ---")
    print(f"HOLDING_FRACTION = {HOLDING_FRACTION}")
    print(f"price tier thresholds: <= {econ.attrs['q_lo']:.2f} (low), "
          f"<= {econ.attrs['q_hi']:.2f} (mid), else (premium)")
    tier_summary = (
        econ.groupby("tier")
        .agg(products=("stock_code", "size"), margin=("margin", "first"),
             cr=("cr", "first"), chosen_q=("chosen_q", "first"))
        .reindex(TIER_LABELS)
    )
    print(tier_summary.to_string())
    print(f"\nCR range across products: {econ['cr'].min():.3f} -> {econ['cr'].max():.3f}")
    q_counts = econ["chosen_q"].value_counts().sort_index()
    print("products by ordered quantile: " + ", ".join(
        f"P{round(q*100)}={n:,}" for q, n in q_counts.items()))

    print("\n--- Decision-regret comparison (test set, total rupee cost) ---")
    print(f"accuracy_first (order P50):    Rs {total_acc:14,.0f}")
    print(f"decision_aware (order CR-qtl): Rs {total_dec:14,.0f}")
    print(f"savings:                       Rs {savings:14,.0f}  ({savings_pct:.1f}%)")

    print("\n--- Accuracy metric (RMSE) — note: best forecast != best decision ---")
    print(f"accuracy_first RMSE: {rmse_acc:.3f}")
    print(f"decision_aware RMSE: {rmse_dec:.3f}")
    if rmse_dec >= rmse_acc and savings > 0:
        print("=> decision_aware has WORSE RMSE but LOWER cost — the headline result.")

    n_better = int((by_product["savings"] > 0).sum())
    n_worse = int((by_product["savings"] < 0).sum())
    print(f"\nproducts where decision_aware is cheaper: {n_better:,} / "
          f"{len(by_product):,}  (worse: {n_worse:,})")

    print(f"\nWrote {REGRET_PATH.relative_to(PROJECT_ROOT)} ({len(by_product):,} products)")
    print("\ntop products by savings:")
    print(by_product.head(8).to_string(index=False))


if __name__ == "__main__":
    main()

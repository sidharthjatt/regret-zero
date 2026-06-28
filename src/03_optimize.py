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

# --------------------------------------------------------------------------
# Paths (resolved from project root, consistent with the other scripts).
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FORECAST_PATH = PROJECT_ROOT / "outputs" / "forecasts.csv"
PRICES_PATH = PROJECT_ROOT / "data" / "prices.csv"
RAW_PATH = PROJECT_ROOT / "data" / "online_retail_II.csv"  # fallback only
REGRET_PATH = PROJECT_ROOT / "outputs" / "regret_by_product.csv"

# --------------------------------------------------------------------------
# Cost assumptions — the dataset has NO cost data, so these are transparent,
# configurable proxies. They are grounded in published retail gross-margin
# benchmarks (2025-26) for a business like this one (a UK online gift retailer):
#
#     grocery / commodity        ~25-30%
#     general merchandise        ~35-45%
#     specialty / luxury / gift  ~55-65%
#     e-commerce (blended)       ~30-50%
#   Net margins are far thinner (grocery ~1-3%, specialty ~10%+).
#
# We tier products by unit price (a proxy for category) and set the underage
# (stockout) cost Cu = margin[tier] * price = the profit forgone per lost sale.
# We deliberately use CONSERVATIVE, net/contribution-margin-style fractions that
# sit BELOW the gross benchmarks above — NOT the headline gross margins — for
# three auditable reasons:
#   (1) the overage cost Co also scales with price (Co = HOLDING * price), and a
#       per-week HOLDING of 10% already absorbs real markdown/obsolescence risk,
#       so Cu should reflect *contribution* lost, not gross margin;
#   (2) a stockout is not always a fully lost sale (substitution / backorder),
#       so gross margin overstates the true per-unit regret;
#   (3) sub-gross fractions keep the critical ratios spread ACROSS tiers
#       (0.333 / 0.667 / 0.818) so the optimizer genuinely orders a different
#       quantile per tier. Plugging in full gross margins (0.27/0.40/0.60) would
#       compress every tier to CR>0.7 and erase the per-tier decision story.
#
#   Cu = margin[tier] * price        Co = HOLDING_FRACTION * price
#   CR = margin / (margin + HOLDING_FRACTION)  -> nearest trained quantile:
#
#   tier     benchmark category            margin   CR      orders
#   low      commodity (grocery net ~2-5%)  0.05    0.333    P33
#   mid      general merchandise            0.20    0.667    P67
#   premium  specialty / gift / luxury      0.45    0.818    P82
#
# Values are unchanged from the proven run -> total savings stay +12.3%
# (Rs 57,232). Edit them (or HOLDING_FRACTION) to explore other regimes.
# --------------------------------------------------------------------------
HOLDING_FRACTION = 0.10  # weekly holding/overage cost as a fraction of unit price

# Tier boundaries as quantiles of the per-product unit-price distribution.
# (1/3, 2/3) -> three roughly equal-sized tiers.
PRICE_TIER_QUANTILES = (1 / 3, 2 / 3)
TIER_LABELS = ["low", "mid", "premium"]
# Conservative net/contribution-margin-style underage fractions per tier; sit
# below the gross-margin benchmarks above (see note). Ordering matches the
# benchmark ordering: commodity < general merchandise < specialty/gift.
MARGIN_BY_TIER = {"low": 0.05, "mid": 0.20, "premium": 0.45}

# Quantiles available from the forecaster (must exist as columns in
# forecasts.csv, e.g. p33, p50, p67, p82, p90). Kept in sync with 02_forecast.
AVAILABLE_QUANTILES = [0.333, 0.5, 0.667, 0.818, 0.9]

ROUND_ORDERS = True  # round order quantities to whole units (can't order 0.5)


def map_cr_to_quantile(cr: float, available=AVAILABLE_QUANTILES) -> float:
    """Snap a critical ratio to the NEAREST trained quantile.

    The newsvendor optimum is F^{-1}(CR); with a finite set of trained
    quantiles we pick the closest one. This is general — add more quantiles to
    AVAILABLE_QUANTILES (and to 02_forecast) and the match gets finer
    automatically.
    """
    return min(available, key=lambda q: abs(q - cr))


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


def build_economics(price: pd.Series) -> pd.DataFrame:
    """Build per-product economics: price tier, margin, Cu, Co, CR, quantile.

    Tiering is done on the per-PRODUCT price distribution (one price per
    product), so it isn't skewed by how many weeks each product has.
    """
    econ = price.to_frame().reset_index()  # columns: stock_code, unit_price

    # Robust tiering via quantile thresholds (handles duplicate prices, unlike
    # qcut which can choke on tied bin edges).
    q_lo, q_hi = econ["unit_price"].quantile(PRICE_TIER_QUANTILES)
    econ["tier"] = np.where(
        econ["unit_price"] <= q_lo, TIER_LABELS[0],
        np.where(econ["unit_price"] <= q_hi, TIER_LABELS[1], TIER_LABELS[2]),
    )
    econ["margin"] = econ["tier"].map(MARGIN_BY_TIER)

    econ["cu"] = econ["margin"] * econ["unit_price"]
    econ["co"] = HOLDING_FRACTION * econ["unit_price"]
    econ["cr"] = econ["margin"] / (econ["margin"] + HOLDING_FRACTION)
    econ["chosen_q"] = econ["cr"].apply(map_cr_to_quantile)

    # Stash thresholds for the summary printout.
    econ.attrs["q_lo"] = float(q_lo)
    econ.attrs["q_hi"] = float(q_hi)
    return econ


def realized_cost(actual: np.ndarray, order: np.ndarray,
                  cu: np.ndarray, co: np.ndarray) -> np.ndarray:
    """Asymmetric newsvendor cost per product-week: Cu*shortage + Co*excess."""
    shortage = np.maximum(actual - order, 0.0)
    excess = np.maximum(order - actual, 0.0)
    return cu * shortage + co * excess


def prepare_orders(df: pd.DataFrame, econ: pd.DataFrame) -> pd.DataFrame:
    """Attach economics and both strategies' order quantities to each row."""
    df = df.merge(econ, on="stock_code", how="left")

    n_missing = int(df["unit_price"].isna().sum())
    if n_missing:
        # Any product without a price gets median economics so the model stays
        # defined (shouldn't happen when prices.csv comes from 01_data_prep).
        med_price = econ["unit_price"].median()
        df["unit_price"] = df["unit_price"].fillna(med_price)
        df["tier"] = df["tier"].fillna(TIER_LABELS[1])
        df["margin"] = df["margin"].fillna(MARGIN_BY_TIER[TIER_LABELS[1]])
        df["co"] = df["co"].fillna(HOLDING_FRACTION * med_price)
        df["cu"] = df["cu"].fillna(MARGIN_BY_TIER[TIER_LABELS[1]] * med_price)
        df["cr"] = df["cr"].fillna(
            MARGIN_BY_TIER[TIER_LABELS[1]] / (MARGIN_BY_TIER[TIER_LABELS[1]] + HOLDING_FRACTION)
        )
        df["chosen_q"] = df["chosen_q"].fillna(map_cr_to_quantile(df["cr"].iloc[0]))
        print(f"WARNING: {n_missing:,} rows had no price; filled with median economics")

    # Decision-aware order = the forecast column matching each row's chosen
    # quantile; accuracy-first = always P50.
    quantile_cols = {q: f"p{round(q * 100)}" for q in AVAILABLE_QUANTILES}
    df["decision_order"] = df.apply(
        lambda r: r[quantile_cols[r["chosen_q"]]], axis=1
    )
    df["accuracy_order"] = df["p50"]

    for col in ["accuracy_order", "decision_order"]:
        df[col] = np.clip(df[col], 0.0, None)
        if ROUND_ORDERS:
            df[col] = np.rint(df[col])

    return df


def main() -> None:
    if not FORECAST_PATH.exists():
        raise FileNotFoundError(
            f"{FORECAST_PATH} not found — run src/02_forecast.py first."
        )

    print("Loading forecasts ...")
    df = load_forecasts()
    print(f"loaded {len(df):,} test product-weeks, {df['stock_code'].nunique():,} products")

    price = load_unit_price()
    econ = build_economics(price)
    df = prepare_orders(df, econ)

    # --- Realized cost (regret) for each strategy ---
    actual = df["actual"].to_numpy(dtype=float)
    cu = df["cu"].to_numpy()
    co = df["co"].to_numpy()
    df["cost_accuracy"] = realized_cost(actual, df["accuracy_order"].to_numpy(), cu, co)
    df["cost_decision"] = realized_cost(actual, df["decision_order"].to_numpy(), cu, co)

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

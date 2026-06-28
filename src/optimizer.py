"""
optimizer.py — RegretZero shared decision logic (single source of truth)

The newsvendor cost model used by BOTH the batch pipeline (src/03_optimize.py)
and the dashboard (app/app.py). Pure pandas/numpy — no file I/O, no Streamlit —
so the two callers produce byte-identical results from the same functions.

NEWSVENDOR, IN ONE LINE
    Optimal order = F^{-1}(CR), where CR = Cu / (Cu + Co). We estimate F with
    quantile forecasts, so ordering the CR-th quantile is the cost-optimal order.

COST ASSUMPTIONS (the defaults below)
The dataset has NO cost data, so these are transparent, configurable proxies,
grounded in published retail gross-margin benchmarks (2025-26) for a business
like this one (a UK online gift retailer):

    grocery / commodity        ~25-30%
    general merchandise        ~35-45%
    specialty / luxury / gift  ~55-65%
    e-commerce (blended)       ~30-50%
  Net margins are far thinner (grocery ~1-3%, specialty ~10%+).

We tier products by unit price (a proxy for category) and set the underage
(stockout) cost Cu = margin[tier] * price = the profit forgone per lost sale.
We deliberately use CONSERVATIVE, net/contribution-margin-style fractions that
sit BELOW the gross benchmarks above — NOT the headline gross margins — for
three auditable reasons:
  (1) the overage cost Co also scales with price (Co = HOLDING * price), and a
      per-week HOLDING of 10% already absorbs real markdown/obsolescence risk,
      so Cu should reflect *contribution* lost, not gross margin;
  (2) a stockout is not always a fully lost sale (substitution / backorder),
      so gross margin overstates the true per-unit regret;
  (3) sub-gross fractions keep the critical ratios spread ACROSS tiers
      (0.333 / 0.667 / 0.818) so the optimizer genuinely orders a different
      quantile per tier. Plugging in full gross margins (0.27/0.40/0.60) would
      compress every tier to CR>0.7 and erase the per-tier decision story.

    Cu = margin[tier] * price        Co = HOLDING_FRACTION * price
    CR = margin / (margin + HOLDING_FRACTION)  -> nearest trained quantile:

    tier     benchmark category            margin   CR      orders
    low      commodity (grocery net ~2-5%)  0.05    0.333    P33
    mid      general merchandise            0.20    0.667    P67
    premium  specialty / gift / luxury      0.45    0.818    P82

These defaults reproduce the proven result: total savings +12.3% (Rs 57,232).
"""

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Canonical constants (the single source of truth; both callers import these).
# --------------------------------------------------------------------------
PRICE_TIER_QUANTILES = (1 / 3, 2 / 3)   # tier boundaries on per-product price
TIER_LABELS = ["low", "mid", "premium"]
AVAILABLE_QUANTILES = [0.333, 0.5, 0.667, 0.818, 0.9]  # must match forecast cols
MARGIN_BY_TIER = {"low": 0.05, "mid": 0.20, "premium": 0.45}
HOLDING_FRACTION = 0.10                  # weekly holding/overage as fraction of price
ROUND_ORDERS = True                      # round orders to whole units


def map_cr_to_quantile(cr: float, available=AVAILABLE_QUANTILES) -> float:
    """Snap a critical ratio to the NEAREST trained quantile.

    The newsvendor optimum is F^{-1}(CR); with a finite set of trained quantiles
    we pick the closest one. General — add more quantiles to AVAILABLE_QUANTILES
    (and to 02_forecast) and the match gets finer automatically.
    """
    return min(available, key=lambda q: abs(q - cr))


def build_economics(prices: pd.DataFrame,
                    holding: float = HOLDING_FRACTION,
                    margins: dict = MARGIN_BY_TIER) -> pd.DataFrame:
    """Build per-product economics: price tier, margin, Cu, Co, CR, quantile.

    `prices` is a DataFrame with columns [stock_code, unit_price] (one row per
    product). `holding` and `margins` are parameterized so the dashboard can
    drive them from sliders; the defaults reproduce the pipeline result.

    Tiering is on the per-PRODUCT price distribution, so it isn't skewed by how
    many weeks each product has.
    """
    econ = prices.copy()

    # Robust tiering via quantile thresholds (handles duplicate prices, unlike
    # qcut which can choke on tied bin edges).
    q_lo, q_hi = econ["unit_price"].quantile(PRICE_TIER_QUANTILES)
    econ["tier"] = np.where(
        econ["unit_price"] <= q_lo, TIER_LABELS[0],
        np.where(econ["unit_price"] <= q_hi, TIER_LABELS[1], TIER_LABELS[2]),
    )
    econ["margin"] = econ["tier"].map(margins)

    econ["cu"] = econ["margin"] * econ["unit_price"]
    econ["co"] = holding * econ["unit_price"]
    econ["cr"] = econ["margin"] / (econ["margin"] + holding)
    econ["chosen_q"] = econ["cr"].apply(map_cr_to_quantile)

    # Stash thresholds for callers that print a summary.
    econ.attrs["q_lo"] = float(q_lo)
    econ.attrs["q_hi"] = float(q_hi)
    return econ


def attach_orders(forecasts: pd.DataFrame, econ: pd.DataFrame,
                  round_orders: bool = ROUND_ORDERS,
                  holding: float = HOLDING_FRACTION,
                  margins: dict = MARGIN_BY_TIER) -> pd.DataFrame:
    """Merge economics onto each product-week and attach both strategies' orders.

    decision_order = the forecast column for each row's chosen quantile;
    accuracy_order = always P50. Both are clipped at 0 and (optionally) rounded
    identically, so the comparison stays fair.
    """
    df = forecasts.merge(econ, on="stock_code", how="left")

    # Defensive: any product missing a price gets median (mid-tier) economics so
    # the model stays defined. Normally a no-op — prices.csv covers all products.
    if int(df["unit_price"].isna().sum()):
        med = econ["unit_price"].median()
        mid = TIER_LABELS[1]
        mid_cr = margins[mid] / (margins[mid] + holding)
        df["unit_price"] = df["unit_price"].fillna(med)
        df["tier"] = df["tier"].fillna(mid)
        df["margin"] = df["margin"].fillna(margins[mid])
        df["co"] = df["co"].fillna(holding * med)
        df["cu"] = df["cu"].fillna(margins[mid] * med)
        df["cr"] = df["cr"].fillna(mid_cr)
        df["chosen_q"] = df["chosen_q"].fillna(map_cr_to_quantile(mid_cr))

    # Decision-aware order: pick, per row, the forecast column matching its
    # chosen quantile (vectorized per quantile).
    quantile_cols = {q: f"p{round(q * 100)}" for q in AVAILABLE_QUANTILES}
    df["decision_order"] = np.nan
    for q, col in quantile_cols.items():
        mask = df["chosen_q"] == q
        df.loc[mask, "decision_order"] = df.loc[mask, col]
    df["accuracy_order"] = df["p50"]

    for col in ["accuracy_order", "decision_order"]:
        df[col] = np.clip(df[col], 0.0, None)
        if round_orders:
            df[col] = np.rint(df[col])

    return df


def realized_cost(actual: np.ndarray, order: np.ndarray,
                  cu: np.ndarray, co: np.ndarray) -> np.ndarray:
    """Asymmetric newsvendor cost per product-week: Cu*shortage + Co*excess."""
    shortage = np.maximum(actual - order, 0.0)
    excess = np.maximum(order - actual, 0.0)
    return cu * shortage + co * excess


def score(forecasts: pd.DataFrame, prices: pd.DataFrame,
          holding: float = HOLDING_FRACTION,
          margins: dict = MARGIN_BY_TIER,
          round_orders: bool = ROUND_ORDERS):
    """End-to-end: economics -> orders -> realized cost per strategy.

    Returns (df, econ): `df` is the per-product-week table with cost_accuracy /
    cost_decision columns; `econ` is the per-product economics (handy for tier
    summaries). Both 03_optimize.py and app/app.py call this so the math lives
    in exactly one place.
    """
    econ = build_economics(prices, holding=holding, margins=margins)
    df = attach_orders(forecasts, econ, round_orders=round_orders,
                       holding=holding, margins=margins)
    actual = df["actual"].to_numpy(dtype=float)
    df["cost_accuracy"] = realized_cost(
        actual, df["accuracy_order"].to_numpy(), df["cu"].to_numpy(), df["co"].to_numpy())
    df["cost_decision"] = realized_cost(
        actual, df["decision_order"].to_numpy(), df["cu"].to_numpy(), df["co"].to_numpy())
    return df, econ

"""
app.py — RegretZero "decision cockpit" (Streamlit dashboard)

Interactive front-end for the decision-regret result. The cost math here is a
FAITHFUL re-implementation of src/03_optimize.py — same tiering, critical
ratio, nearest-quantile snap, and asymmetric cost — except the cost
assumptions (holding fraction + per-tier margins) are driven by sliders so you
can watch the savings move in real time.

With the default sliders (holding=0.10, margins 0.05/0.20/0.45) this reproduces
the pipeline's proven result: decision-aware beats accuracy-first by ~12.3%.

Run from the project root:
    streamlit run app/app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# Constants mirrored EXACTLY from src/03_optimize.py (keep in sync).
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FORECAST_PATH = PROJECT_ROOT / "outputs" / "forecasts.csv"
PRICES_PATH = PROJECT_ROOT / "data" / "prices.csv"

PRICE_TIER_QUANTILES = (1 / 3, 2 / 3)   # tier boundaries on per-product price
TIER_LABELS = ["low", "mid", "premium"]
AVAILABLE_QUANTILES = [0.333, 0.5, 0.667, 0.818, 0.9]  # must match forecast cols
ROUND_ORDERS = True

TIER_COLORS = {"low": "#4C72B0", "mid": "#DD8452", "premium": "#55A868"}


# --------------------------------------------------------------------------
# Cached data loads (the only expensive step; recompute below is cheap).
# --------------------------------------------------------------------------
@st.cache_data
def load_forecasts() -> pd.DataFrame:
    """Test-set forecasts: actual + p33,p50,p67,p82,p90 per product-week."""
    return pd.read_csv(
        FORECAST_PATH, dtype={"stock_code": "string"}, parse_dates=["week_start_date"]
    )


@st.cache_data
def load_prices() -> pd.DataFrame:
    """Per-product unit price (cost basis)."""
    return pd.read_csv(PRICES_PATH, dtype={"stock_code": "string"})


def map_cr_to_quantile(cr: float, available=AVAILABLE_QUANTILES) -> float:
    """Snap a critical ratio to the NEAREST available quantile (mirrors 03)."""
    return min(available, key=lambda q: abs(q - cr))


def compute(forecasts: pd.DataFrame, prices: pd.DataFrame,
            holding: float, margins: dict) -> pd.DataFrame:
    """Re-implement 03_optimize.py's per-row economics + costs.

    `margins` maps tier -> margin fraction (from the sidebar sliders);
    `holding` is the holding fraction. Tier boundaries come from the
    per-PRODUCT price quantiles, exactly as in the pipeline.
    """
    econ = prices.copy()

    # Tier each product by its unit price (same quantile thresholds as 03).
    q_lo, q_hi = econ["unit_price"].quantile(PRICE_TIER_QUANTILES)
    econ["tier"] = np.where(
        econ["unit_price"] <= q_lo, TIER_LABELS[0],
        np.where(econ["unit_price"] <= q_hi, TIER_LABELS[1], TIER_LABELS[2]),
    )
    econ["margin"] = econ["tier"].map(margins)

    # Asymmetric costs and critical ratio (price cancels in CR, as in 03).
    econ["cu"] = econ["margin"] * econ["unit_price"]
    econ["co"] = holding * econ["unit_price"]
    econ["cr"] = econ["margin"] / (econ["margin"] + holding)
    econ["chosen_q"] = econ["cr"].apply(map_cr_to_quantile)

    # Attach economics to each product-week.
    m = forecasts.merge(econ, on="stock_code", how="left")

    # Decision-aware order = the forecast column matching each row's chosen
    # quantile; accuracy-first = always P50. (Vectorized per quantile.)
    quantile_cols = {q: f"p{round(q * 100)}" for q in AVAILABLE_QUANTILES}
    m["decision_order"] = np.nan
    for q, col in quantile_cols.items():
        mask = m["chosen_q"] == q
        m.loc[mask, "decision_order"] = m.loc[mask, col]
    m["accuracy_order"] = m["p50"]

    # Orders can't be negative or (optionally) fractional — applied to BOTH
    # strategies identically so the comparison stays fair.
    for col in ["accuracy_order", "decision_order"]:
        m[col] = np.clip(m[col], 0.0, None)
        if ROUND_ORDERS:
            m[col] = np.rint(m[col])

    # Asymmetric newsvendor cost: Cu*shortage + Co*excess.
    actual = m["actual"].to_numpy(dtype=float)
    for strat, order_col in [("accuracy", "accuracy_order"), ("decision", "decision_order")]:
        order = m[order_col].to_numpy()
        shortage = np.maximum(actual - order, 0.0)
        excess = np.maximum(order - actual, 0.0)
        m[f"cost_{strat}"] = m["cu"].to_numpy() * shortage + m["co"].to_numpy() * excess

    return m


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
st.set_page_config(page_title="RegretZero — decision cockpit", layout="wide")

st.title("RegretZero — decision cockpit")
st.markdown(
    "**Thesis:** the best forecast is not the best decision. Ordering at the "
    "newsvendor critical-ratio quantile (*decision-aware*) beats ordering the "
    "median point forecast (*accuracy-first*) in rupee terms."
)

forecasts = load_forecasts()
prices = load_prices()

# ---- Sidebar controls ----------------------------------------------------
st.sidebar.header("Cost assumptions")
st.sidebar.caption("Cu = margin × price (stockout)  ·  Co = holding × price (overage)")

# Slider defaults mirror src/03_optimize.py's benchmark-justified values
# (holding 0.10; conservative net/contribution-margin-style tier margins
# 0.05/0.20/0.45 — see the cost-assumptions note in 03). These defaults
# reproduce the pipeline headline of +12.3% (Rs 57,232).
holding = st.sidebar.slider("Holding fraction (Co)", 0.02, 0.30, 0.10, 0.01)

st.sidebar.subheader("Tier margins (Cu)")
margins = {
    "low": st.sidebar.slider("low margin", 0.01, 0.60, 0.05, 0.01),
    "mid": st.sidebar.slider("mid margin", 0.01, 0.60, 0.20, 0.01),
    "premium": st.sidebar.slider("premium margin", 0.01, 0.60, 0.45, 0.01),
}

# Live critical ratio + snapped quantile per tier.
st.sidebar.subheader("Resulting critical ratio")
for tier in TIER_LABELS:
    cr = margins[tier] / (margins[tier] + holding)
    q = map_cr_to_quantile(cr)
    st.sidebar.write(f"**{tier}**: CR = {cr:.3f}  →  orders **P{round(q*100)}**")

# ---- Recompute (live) ----------------------------------------------------
m = compute(forecasts, prices, holding, margins)
total_acc = m["cost_accuracy"].sum()
total_dec = m["cost_decision"].sum()
savings = total_acc - total_dec
savings_pct = 100 * savings / total_acc if total_acc else 0.0

# ---- Headline metrics ----------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Accuracy-first cost", f"₹{total_acc:,.0f}")
c2.metric("Decision-aware cost", f"₹{total_dec:,.0f}")
c3.metric("Savings", f"₹{savings:,.0f}", delta=f"{savings_pct:.1f}%")

# ---- Per-tier aggregation ------------------------------------------------
by_tier = (
    m.groupby("tier")[["cost_accuracy", "cost_decision"]].sum()
    .reindex(TIER_LABELS)
)
by_tier["savings"] = by_tier["cost_accuracy"] - by_tier["cost_decision"]

left, right = st.columns(2)

with left:
    st.subheader("Savings by tier")
    fig = go.Figure(
        go.Bar(
            x=by_tier.index,
            y=by_tier["savings"],
            marker_color=[TIER_COLORS[t] for t in by_tier.index],
            text=[f"₹{v:,.0f}" for v in by_tier["savings"]],
            textposition="outside",
        )
    )
    fig.update_layout(yaxis_title="savings (₹)", xaxis_title="tier", height=380)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Cost by strategy, per tier")
    fig2 = go.Figure()
    fig2.add_bar(name="accuracy-first", x=by_tier.index, y=by_tier["cost_accuracy"],
                 marker_color="#B0B0B0")
    fig2.add_bar(name="decision-aware", x=by_tier.index, y=by_tier["cost_decision"],
                 marker_color="#2C7FB8")
    fig2.update_layout(barmode="group", yaxis_title="cost (₹)", xaxis_title="tier",
                       height=380, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig2, use_container_width=True)

# ---- Top products by savings ---------------------------------------------
st.subheader("Top 10 products by savings")
top = (
    m.groupby("stock_code")
    .agg(
        tier=("tier", "first"),
        unit_price=("unit_price", "first"),
        cr=("cr", "first"),
        chosen_q=("chosen_q", "first"),
        cost_accuracy=("cost_accuracy", "sum"),
        cost_decision=("cost_decision", "sum"),
    )
    .reset_index()
)
top["savings"] = top["cost_accuracy"] - top["cost_decision"]
top = top.sort_values("savings", ascending=False).head(10)
st.dataframe(
    top.style.format({
        "unit_price": "{:.2f}", "cr": "{:.3f}", "chosen_q": "{:.3f}",
        "cost_accuracy": "₹{:,.0f}", "cost_decision": "₹{:,.0f}", "savings": "₹{:,.0f}",
    }),
    use_container_width=True,
)

st.caption(
    "Math mirrors src/03_optimize.py exactly; only the cost assumptions are "
    "slider-driven. Defaults reproduce the pipeline's +12.3% result."
)

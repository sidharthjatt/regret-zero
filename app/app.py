"""
app.py: RegretZero "decision cockpit" (Streamlit dashboard)

Interactive front-end for the decision-regret result. The cost math is imported
from src/optimizer.py, the same module the batch pipeline uses, so the
dashboard and the pipeline can never drift. Only the cost assumptions (holding
fraction + per-tier margins) are driven by sliders so you can watch the savings
move in real time.

With the default sliders (holding=0.10, margins 0.05/0.20/0.45) this reproduces
the pipeline's proven result: decision-aware beats accuracy-first by ~12.0%.

Run from the project root:
    streamlit run app/app.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Import the shared cost model (src/optimizer.py), the single source of truth with
# the batch pipeline. app/ is a different folder, so put src/ on sys.path first.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from optimizer import (  # noqa: E402  (import after the sys.path tweak)
    HOLDING_FRACTION,
    MARGIN_BY_TIER,
    TIER_LABELS,
    map_cr_to_quantile,
    score,
)

# --------------------------------------------------------------------------
# Paths + app-only display constants.
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FORECAST_PATH = PROJECT_ROOT / "outputs" / "forecasts.csv"
PRICES_PATH = PROJECT_ROOT / "data" / "prices.csv"
NAMES_PATH = PROJECT_ROOT / "data" / "product_names.csv"

# Same colors as the static charts in assets/ (src/04_make_figures.py).
BASELINE_COLOR = "#8C8C8C"   # accuracy-first: order P50
DECISION_COLOR = "#2F6B9A"   # decision-aware: order the critical-ratio quantile


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


def title_case(name: str) -> str:
    """'WHITE HANGING HEART T-LIGHT HOLDER' -> 'White Hanging Heart T-Light Holder'.
    Capitalizes after spaces and hyphens but not after apostrophes, so "50'S"
    becomes "50's" rather than str.title()'s "50'S"."""
    return re.sub(r"(?<![A-Za-z'])[a-z]", lambda m: m.group(0).upper(), name.lower())


@st.cache_data
def load_names() -> dict:
    """stock_code -> display name. Falls back to an empty map (codes shown
    instead) if data/product_names.csv hasn't been generated."""
    if not NAMES_PATH.exists():
        return {}
    names = pd.read_csv(NAMES_PATH, dtype={"stock_code": "string", "name": "string"})
    return {c: title_case(n) for c, n in zip(names["stock_code"], names["name"])}


def compute(forecasts: pd.DataFrame, prices: pd.DataFrame,
            holding: float, margins: dict) -> pd.DataFrame:
    """Per-product-week economics + realized costs, via the shared scorer
    (src/optimizer.py). Identical math to the batch pipeline; only the cost
    assumptions (holding + tier margins) are slider-driven here."""
    df, _ = score(forecasts, prices, holding=holding, margins=margins)
    return df


# Holding-fraction grid swept by the Sensitivity section.
HOLDING_SWEEP = [round(x, 2) for x in np.arange(0.02, 0.301, 0.01)]


@st.cache_data
def sweep_holding(margins_tuple: tuple) -> pd.DataFrame:
    """Total savings (£ and %) across the holding-fraction grid, with the given
    tier margins held fixed. Reuses compute() exactly, with no separate math.

    Cached on the margins tuple, so moving the *holding* slider (which doesn't
    change this curve) doesn't trigger a re-sweep. The cached data loaders are
    called inside, so the heavy frames aren't re-read or re-hashed per call.
    """
    f, p = load_forecasts(), load_prices()
    margins = dict(zip(TIER_LABELS, margins_tuple))
    rows = []
    for h in HOLDING_SWEEP:
        mm = compute(f, p, h, margins)
        ta = mm["cost_accuracy"].sum()
        td = mm["cost_decision"].sum()
        rows.append({"holding": h, "savings": ta - td,
                     "savings_pct": 100 * (ta - td) / ta if ta else 0.0})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
st.set_page_config(page_title="RegretZero: decision cockpit", layout="wide")

st.title("RegretZero: decision cockpit")
st.markdown(
    "**Thesis:** the best forecast is not the best decision. Ordering at the "
    "newsvendor critical-ratio quantile (*decision-aware*) beats ordering the "
    "median point forecast (*accuracy-first*) in pound (£) terms."
)

forecasts = load_forecasts()
prices = load_prices()
names = load_names()

# ---- Sidebar controls ----------------------------------------------------
st.sidebar.header("Cost assumptions")
st.sidebar.caption("Cu = margin × price (stockout)  ·  Co = holding × price (overage)")

# Slider defaults come straight from the shared cost model's canonical values
# (optimizer.HOLDING_FRACTION and MARGIN_BY_TIER), so the dashboard's default
# load always matches the pipeline (+12.0%, £54,631).
holding = st.sidebar.slider("Holding fraction (Co)", 0.02, 0.30, HOLDING_FRACTION, 0.01)

st.sidebar.subheader("Tier margins (Cu)")
margins = {
    t: st.sidebar.slider(f"{t} margin", 0.01, 0.60, MARGIN_BY_TIER[t], 0.01)
    for t in TIER_LABELS
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
c1.metric("Accuracy-first cost", f"£{total_acc:,.0f}")
c2.metric("Decision-aware cost", f"£{total_dec:,.0f}")
c3.metric("Savings", f"£{savings:,.0f}", delta=f"{savings_pct:.1f}%")

# ---- Per-tier aggregation ------------------------------------------------
by_tier = (
    m.groupby("tier")[["cost_accuracy", "cost_decision"]].sum()
    .reindex(TIER_LABELS)
)
by_tier["savings"] = by_tier["cost_accuracy"] - by_tier["cost_decision"]

left, right = st.columns(2)

with left:
    st.subheader("Savings by tier")
    tier_names = [t.capitalize() for t in by_tier.index]
    fig = go.Figure(
        go.Bar(
            x=tier_names,
            y=by_tier["savings"],
            marker_color=DECISION_COLOR,
            text=[f"£{v:,.0f}" for v in by_tier["savings"]],
            textposition="outside",
        )
    )
    fig.update_layout(yaxis_title="Savings (£)", xaxis_title="Tier", height=380)
    st.plotly_chart(fig, width="stretch")

with right:
    st.subheader("Cost by strategy, per tier")
    fig2 = go.Figure()
    fig2.add_bar(name="Accuracy-first", x=tier_names, y=by_tier["cost_accuracy"],
                 marker_color=BASELINE_COLOR)
    fig2.add_bar(name="Decision-aware", x=tier_names, y=by_tier["cost_decision"],
                 marker_color=DECISION_COLOR)
    fig2.update_layout(barmode="group", yaxis_title="Cost (£)", xaxis_title="Tier",
                       height=380, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig2, width="stretch")

# ---- Top products by savings ---------------------------------------------
st.subheader("Top 10 products by savings")
top = (
    m.groupby("stock_code")
    .agg(
        tier=("tier", "first"),
        unit_price=("unit_price", "first"),
        chosen_q=("chosen_q", "first"),
        cost_accuracy=("cost_accuracy", "sum"),
        cost_decision=("cost_decision", "sum"),
    )
    .reset_index()
)
top["savings"] = top["cost_accuracy"] - top["cost_decision"]
top = top.sort_values("savings", ascending=False).head(10)
top_table = pd.DataFrame({
    "Product": [names.get(c, c) for c in top["stock_code"]],
    "Code": top["stock_code"].to_numpy(),
    "Tier": top["tier"].str.capitalize().to_numpy(),
    "Price (£)": top["unit_price"].to_numpy(),
    "Orders": [f"P{round(q * 100)}" for q in top["chosen_q"]],
    "Cost at P50 (£)": top["cost_accuracy"].to_numpy(),
    "Cost at CR quantile (£)": top["cost_decision"].to_numpy(),
    "Saving (£)": top["savings"].to_numpy(),
})
st.dataframe(
    top_table.style.format({
        "Price (£)": "{:.2f}", "Cost at P50 (£)": "{:,.0f}",
        "Cost at CR quantile (£)": "{:,.0f}", "Saving (£)": "{:,.0f}",
    }),
    width="stretch",
    hide_index=True,
)

# ==========================================================================
# Sensitivity: savings vs holding fraction (robustness check)
# ==========================================================================
st.divider()
st.subheader("Sensitivity: savings vs holding cost")

# Sweep holding across its full range at the current tier margins (cached).
sweep = sweep_holding(tuple(margins[t] for t in TIER_LABELS))

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=sweep["holding"], y=sweep["savings"], mode="lines",
    line=dict(color=DECISION_COLOR, width=2), name="Savings",
))
# Zero line: anything above it means decision-aware still wins.
fig3.add_hline(y=0, line_color=BASELINE_COLOR)
# Mark the current slider value and its savings.
fig3.add_vline(x=holding, line_dash="dash", line_color=BASELINE_COLOR,
               annotation_text=f"current: {holding:.2f}", annotation_position="top")
fig3.add_trace(go.Scatter(
    x=[holding], y=[savings], mode="markers",
    marker=dict(color=DECISION_COLOR, size=11), name="Current setting",
))
fig3.update_layout(xaxis_title="Holding fraction", yaxis_title="Total savings (£)",
                   height=380, legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig3, width="stretch")

# Describe the curve from the swept data, so the text always matches the chart:
# where the current setting lands, and where the saving first stops being
# positive (the breakeven moves as the tier-margin sliders change).
if savings > 0:
    current_msg = (
        f"At a holding fraction of {holding:.2f}, ordering the critical-ratio "
        f"quantile saves £{savings:,.0f} ({savings_pct:.1f}%) against ordering P50."
    )
else:
    current_msg = (
        f"At a holding fraction of {holding:.2f}, ordering the critical-ratio "
        f"quantile costs £{-savings:,.0f} more than ordering P50 ({savings_pct:.1f}%)."
    )

h_lo, h_hi = HOLDING_SWEEP[0], HOLDING_SWEEP[-1]
nonpos = sweep.index[sweep["savings"] <= 0]
if nonpos.empty:
    breakeven_msg = (
        f"At these margins the saving stays positive across the whole range "
        f"shown ({h_lo:.2f} to {h_hi:.2f}), so there is no breakeven in it."
    )
elif nonpos[0] == 0:
    breakeven_msg = (
        f"At these margins the saving is already negative at {h_lo:.2f}, the "
        "lowest holding fraction shown."
    )
else:
    below = sweep.loc[nonpos[0] - 1, "holding"]
    above = sweep.loc[nonpos[0], "holding"]
    breakeven_msg = (
        f"At these margins the breakeven lies between {below:.2f} and {above:.2f}: "
        f"the saving is positive at {below:.2f} and first turns negative at {above:.2f}."
    )
    recovered = sweep.loc[nonpos[0]:][sweep.loc[nonpos[0]:, "savings"] > 0]
    if not recovered.empty:
        breakeven_msg += (
            f" It turns positive again at {recovered['holding'].iloc[0]:.2f}, "
            "because each tier snaps to the nearest trained quantile."
        )
st.markdown(f"{current_msg} {breakeven_msg}")

# ==========================================================================
# Per-product drill-down: how the engine reasons about one product
# ==========================================================================
st.divider()
st.subheader("Per-product drill-down")

# Rank products by savings so the default selection is an instructive one.
per_prod = (
    m.groupby("stock_code")
    .agg(cost_accuracy=("cost_accuracy", "sum"), cost_decision=("cost_decision", "sum"))
    .assign(savings=lambda d: d["cost_accuracy"] - d["cost_decision"])
    .sort_values("savings", ascending=False)
)

with st.expander("Inspect a single product", expanded=False):
    sel = st.selectbox("Product (sorted by savings)", per_prod.index.tolist(), index=0,
                       format_func=lambda c: f"{names.get(c, c)} ({c})")
    sub = m[m["stock_code"] == sel].sort_values("week_start_date")
    r = sub.iloc[0]  # product-level fields are constant across the product's weeks

    # Product economics.
    econ_cols = st.columns(4)
    econ_cols[0].metric("Unit price", f"£{r['unit_price']:.2f}")
    econ_cols[1].metric("Tier", r["tier"])
    econ_cols[2].metric("Critical ratio", f"{r['cr']:.3f}")
    econ_cols[3].metric("Orders quantile", f"P{round(r['chosen_q']*100)}")

    # Cost under each strategy, summed over this product's test weeks.
    ca, cd = sub["cost_accuracy"].sum(), sub["cost_decision"].sum()
    cost_cols = st.columns(3)
    cost_cols[0].metric("Accuracy-first cost", f"£{ca:,.0f}")
    cost_cols[1].metric("Decision-aware cost", f"£{cd:,.0f}")
    cost_cols[2].metric("Savings", f"£{ca - cd:,.0f}",
                        delta=f"{(100 * (ca - cd) / ca) if ca else 0:.1f}%")

    # Actual demand vs the five quantile forecasts over the test weeks.
    fig4 = go.Figure()
    q_palette = {0.333: "#c6dbef", 0.5: "#9ecae1", 0.667: "#6baed6",
                 0.818: "#3182bd", 0.9: "#08519c"}
    for q, color in q_palette.items():
        col = f"p{round(q * 100)}"
        fig4.add_trace(go.Scatter(x=sub["week_start_date"], y=sub[col], mode="lines",
                                  line=dict(color=color, width=1.5), name=col.upper()))
    fig4.add_trace(go.Scatter(x=sub["week_start_date"], y=sub["actual"],
                              mode="lines+markers", line=dict(color="#e6550d", width=2.5),
                              marker=dict(size=6), name="actual"))
    fig4.update_layout(xaxis_title="Week", yaxis_title="Weekly demand (units)",
                       height=400, legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig4, width="stretch")

    # Per-week orders + realized cost under each strategy.
    wk = pd.DataFrame({
        "Week": sub["week_start_date"].dt.date.to_numpy(),
        "Actual": sub["actual"].to_numpy(),
        "Order at P50": sub["accuracy_order"].to_numpy(),
        "Order at CR quantile": sub["decision_order"].to_numpy(),
        "Cost at P50 (£)": sub["cost_accuracy"].to_numpy(),
        "Cost at CR quantile (£)": sub["cost_decision"].to_numpy(),
    })
    st.dataframe(
        wk.style.format({"Order at P50": "{:.0f}", "Order at CR quantile": "{:.0f}",
                         "Cost at P50 (£)": "{:,.2f}", "Cost at CR quantile (£)": "{:,.2f}"}),
        width="stretch",
        hide_index=True,
    )

    # Explain the direction of the order relative to the median, which depends
    # on whether this product's critical ratio sits above or below 0.5.
    q_label = f"P{round(r['chosen_q'] * 100)}"
    basis = (
        f"{q_label} is the trained quantile nearest this product's critical ratio "
        f"({r['cr']:.3f}). The decision-aware strategy orders {q_label}; "
        "accuracy-first orders P50."
    )
    if r["chosen_q"] > 0.5:
        reason = (
            "A missed sale costs more than a leftover unit for this product, so "
            "ordering above the median protects against costly stockouts in "
            "high-demand weeks."
        )
    elif r["chosen_q"] < 0.5:
        reason = (
            "A missed sale is cheap for this product relative to holding a leftover "
            "unit, so it orders below the median to avoid paying for stock that "
            "doesn't sell."
        )
    else:
        reason = "Both strategies order the median for this product, so their costs match."
    st.caption(f"{basis} {reason}")

# --------------------------------------------------------------------------
st.divider()
st.caption(
    "The math comes from src/optimizer.py, the same module the batch pipeline "
    "uses. Default sliders reproduce the pipeline's 12.0% result."
)

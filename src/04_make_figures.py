"""
04_make_figures.py — RegretZero charts for the README and reports

Draws the static charts used in README.md and reports/. All cost numbers come
from optimizer.score, the same function the pipeline and the dashboard use,
so the charts can't drift from the headline.

Input : outputs/forecasts.csv     (stock_code, week_start_date, actual,
                                   p33, p50, p67, p82, p90)
        data/prices.csv           (stock_code, unit_price)
        data/demand.csv           (optional; only for demand_distribution.png)
Output: assets/tier_savings.png
        assets/holding_sweep.png
        assets/coverage.png
        assets/demand_distribution.png

Run from the project root:
    python src/04_make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write image files, no GUI needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

# Shared decision logic, as in 03_optimize.py.
from optimizer import AVAILABLE_QUANTILES, HOLDING_FRACTION, TIER_LABELS, score

# --------------------------------------------------------------------------
# Paths (resolved from project root, consistent with the other scripts).
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FORECAST_PATH = PROJECT_ROOT / "outputs" / "forecasts.csv"
PRICES_PATH = PROJECT_ROOT / "data" / "prices.csv"
DEMAND_PATH = PROJECT_ROOT / "data" / "demand.csv"
ASSETS_DIR = PROJECT_ROOT / "assets"

# --------------------------------------------------------------------------
# Style — shared by every chart.
# --------------------------------------------------------------------------
BASELINE_COLOR = "#8C8C8C"   # accuracy-first: order P50
DECISION_COLOR = "#2F6B9A"   # decision-aware: order the critical-ratio quantile
GRID_COLOR = "#E6E6E6"
FIGSIZE = (8, 4.6)
DPI = 150

SWEEP_HOLDING = np.round(np.arange(0.02, 0.501, 0.01), 2)
DASHBOARD_RANGE = (0.02, 0.30)   # the dashboard's holding slider range

pounds = FuncFormatter(lambda v, _: f"£{v:,.0f}")


def new_axes():
    """White figure, top/right spines removed, light horizontal gridlines."""
    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def save(fig, name: str) -> None:
    path = ASSETS_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path.relative_to(PROJECT_ROOT)}")


def total_saving_pct(forecasts: pd.DataFrame, prices: pd.DataFrame, holding: float) -> float:
    df, _ = score(forecasts, prices, holding=holding)
    acc, dec = df["cost_accuracy"].sum(), df["cost_decision"].sum()
    return 100 * (acc - dec) / acc


def plot_tier_savings(scored: pd.DataFrame) -> None:
    """Total test-window cost per tier, both strategies, % saving on top."""
    by_tier = scored.groupby("tier")[["cost_accuracy", "cost_decision"]].sum().reindex(TIER_LABELS)

    fig, ax = new_axes()
    x = np.arange(len(by_tier))
    width = 0.38
    ax.bar(x - width / 2, by_tier["cost_accuracy"], width,
           color=BASELINE_COLOR, label="Order the median forecast (P50)")
    ax.bar(x + width / 2, by_tier["cost_decision"], width,
           color=DECISION_COLOR, label="Order the critical-ratio quantile")

    top = by_tier.max(axis=1)
    for xi, (tier, row) in zip(x, by_tier.iterrows()):
        pct = 100 * (row["cost_accuracy"] - row["cost_decision"]) / row["cost_accuracy"]
        ax.text(xi, top[tier] + 0.02 * top.max(), f"{pct:+.1f}%",
                ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x, [t.capitalize() for t in by_tier.index])
    ax.set_xlabel("Price tier")
    ax.set_ylabel("Total cost over 12 test weeks (£)")
    ax.yaxis.set_major_formatter(pounds)
    ax.set_ylim(0, top.max() * 1.12)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "tier_savings.png")


def plot_holding_sweep(forecasts: pd.DataFrame, prices: pd.DataFrame) -> None:
    """Total saving % across holding fractions, with default and dashboard range."""
    pct = [total_saving_pct(forecasts, prices, h) for h in SWEEP_HOLDING]

    fig, ax = new_axes()
    ax.axvspan(*DASHBOARD_RANGE, color=DECISION_COLOR, alpha=0.07, linewidth=0)
    ax.axhline(0, color=BASELINE_COLOR, linewidth=1)
    ax.plot(SWEEP_HOLDING, pct, color=DECISION_COLOR, linewidth=1.8, marker="o", markersize=3)

    default_pct = total_saving_pct(forecasts, prices, HOLDING_FRACTION)
    ax.plot([HOLDING_FRACTION], [default_pct], "o", color=DECISION_COLOR, markersize=8)
    ax.annotate(f"default ({default_pct:.1f}%)", (HOLDING_FRACTION, default_pct),
                xytext=(8, 8), textcoords="offset points", fontsize=10)

    ymin, ymax = min(pct + [0]), max(pct)
    ax.text(sum(DASHBOARD_RANGE) / 2, ymax + 0.08 * (ymax - ymin), "dashboard range",
            ha="center", va="bottom", fontsize=9, color="#555555")
    ax.set_ylim(ymin - 0.08 * (ymax - ymin), ymax + 0.2 * (ymax - ymin))

    ax.set_xlim(SWEEP_HOLDING[0] - 0.01, SWEEP_HOLDING[-1] + 0.01)
    ax.set_xlabel("Holding cost (fraction of unit price per week)")
    ax.set_ylabel("Saving vs ordering P50 (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    save(fig, "holding_sweep.png")


def plot_coverage(forecasts: pd.DataFrame) -> None:
    """Nominal vs empirical coverage (share of actuals at or below each quantile)."""
    nominal = [100 * q for q in AVAILABLE_QUANTILES]
    empirical = [100 * (forecasts["actual"] <= forecasts[f"p{round(q * 100)}"]).mean()
                 for q in AVAILABLE_QUANTILES]

    fig, ax = new_axes()
    ax.plot([0, 100], [0, 100], linestyle="--", color=BASELINE_COLOR, linewidth=1,
            label="perfect calibration")
    ax.plot(nominal, empirical, "o-", color=DECISION_COLOR, linewidth=1.5, markersize=6,
            label="empirical coverage (test set)")
    for q, n, e in zip(AVAILABLE_QUANTILES, nominal, empirical):
        ax.annotate(f"P{round(q * 100)}: {e:.1f}%", (n, e), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=9)

    ax.set_xlim(25, 100)
    ax.set_ylim(25, 100)
    ax.set_xlabel("Nominal quantile level (%)")
    ax.set_ylabel("Actuals at or below forecast (%)")
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.legend(frameon=False, loc="lower right")
    save(fig, "coverage.png")


def plot_demand_distribution() -> None:
    """Histogram of weekly demand per product on a log x-axis, median marked."""
    if not DEMAND_PATH.exists():
        print(f"Skipping demand_distribution.png: {DEMAND_PATH.relative_to(PROJECT_ROOT)} "
              "not found (run src/01_data_prep.py first)")
        return
    d = pd.read_csv(DEMAND_PATH)["demand"]
    d = d[d > 0]  # log axis; demand.csv only holds weeks with sales anyway

    fig, ax = new_axes()
    # Demand is whole units, so bin edges sit halfway between integers; plain
    # log-spaced edges would leave empty bins between 1, 2, 3, ...
    bins = np.unique(np.round(np.logspace(0, np.log10(d.max() + 1), 60))) - 0.5
    ax.hist(d, bins=bins, color=DECISION_COLOR)
    ax.set_xscale("log")

    median = d.median()
    ax.axvline(median, color=BASELINE_COLOR, linestyle="--", linewidth=1.2)
    ax.annotate(f"median = {median:,.0f} units", (median, ax.get_ylim()[1] * 0.92),
                xytext=(6, 0), textcoords="offset points", fontsize=10)

    ax.set_xlabel("Weekly demand per product (units, log scale)")
    ax.set_ylabel("Product-weeks")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    save(fig, "demand_distribution.png")


def main() -> None:
    ASSETS_DIR.mkdir(exist_ok=True)
    forecasts = pd.read_csv(FORECAST_PATH, dtype={"stock_code": "string"},
                            parse_dates=["week_start_date"])
    prices = pd.read_csv(PRICES_PATH, dtype={"stock_code": "string"})

    scored, _ = score(forecasts, prices)
    plot_tier_savings(scored)
    plot_holding_sweep(forecasts, prices)
    plot_coverage(forecasts)
    plot_demand_distribution()


if __name__ == "__main__":
    main()

"""
02_demand_eda.py — quick distribution check on weekly demand

Reads the cleaned product x week demand table and reports the shape of the
weekly-demand distribution (median, skew, outliers) plus a saved plot.
This informs later modelling choices: a heavy right skew means we should
think about log-transforms and quantile-based service levels rather than
assuming anything Gaussian.

Run from the project root:
    python src/02_demand_eda.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write image files, no GUI needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMAND_PATH = PROJECT_ROOT / "data" / "demand.csv"
PLOT_PATH = PROJECT_ROOT / "outputs" / "weekly_demand_distribution.png"


def main() -> None:
    df = pd.read_csv(DEMAND_PATH)
    d = df["demand"]

    # --- Summary statistics ---
    median = d.median()
    mean = d.mean()
    skew = stats.skew(d)

    # Outliers via the standard 1.5*IQR rule (robust to skew).
    q1, q3 = d.quantile(0.25), d.quantile(0.75)
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    n_outliers = int((d > upper_fence).sum())
    pct_outliers = 100 * n_outliers / len(d)

    print("--- Weekly demand distribution ---")
    print(f"observations:        {len(d):,}")
    print(f"mean:                {mean:,.1f}")
    print(f"median:              {median:,.1f}")
    print(f"std:                 {d.std():,.1f}")
    print(f"min / max:           {d.min():,} / {d.max():,}")
    print("\nquantiles:")
    for q in [0.50, 0.90, 0.95, 0.99, 0.999]:
        print(f"  p{q*100:>5.1f}:           {d.quantile(q):,.0f}")
    print(f"\nskewness:            {skew:,.2f}  (0=symmetric, >1=strong right skew)")
    print(f"mean / median ratio: {mean / median:,.2f}  (>>1 confirms right skew)")
    print(
        f"upper IQR fence:     {upper_fence:,.0f}  "
        f"-> {n_outliers:,} outliers ({pct_outliers:.1f}% of rows)"
    )

    # --- Plot: raw vs log scale, since raw will be unreadable if skewed ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(d, bins=100, color="#4C72B0")
    axes[0].axvline(median, color="red", ls="--", lw=1, label=f"median={median:.0f}")
    axes[0].set_title("Weekly demand (raw)")
    axes[0].set_xlabel("units / product / week")
    axes[0].set_ylabel("count")
    axes[0].legend()

    # log1p handles the long tail; +1 keeps any zero-ish values valid.
    axes[1].hist(np.log1p(d), bins=100, color="#55A868")
    axes[1].axvline(np.log1p(median), color="red", ls="--", lw=1, label="median")
    axes[1].set_title("Weekly demand (log1p scale)")
    axes[1].set_xlabel("log1p(units / product / week)")
    axes[1].set_ylabel("count")
    axes[1].legend()

    fig.suptitle("RegretZero — weekly demand distribution", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=120)
    print(f"\nSaved plot -> {PLOT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

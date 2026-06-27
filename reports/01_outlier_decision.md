# Decision note — keeping extreme weekly-demand values

**Date:** 2026-06-27
**Status:** Decided (pending mentor review)
**Scope:** `data/demand.csv` (product × week demand, 183,543 rows, 3,219 products)

## Context

The weekly-demand distribution is extremely right-skewed:

| Statistic | Value |
|---|---|
| Median | 15 units |
| Mean | 58.5 units (mean / median = 3.9×) |
| Skewness | ≈ 158 |
| p99 | 619 |
| p99.9 | 1,879 |
| Max | 74,215 |

(See `outputs/weekly_demand_distribution.png` and `src/02_demand_eda.py`.)

A 1.5×IQR rule flags ~11.8% of rows as "outliers," but that is the IQR rule
misfiring on a skewed, heavy-tailed distribution — not 11.8% of rows being
data errors. The substantive feature is a long, genuine tail of high-volume
weeks (likely wholesale/bulk orders).

## Decision

**We do NOT cap or winsorize the extreme values.**

## Rationale

- Forecasting uses **LightGBM quantile regression (pinball loss)**, which is
  robust to outliers by design: quantile estimates depend on the *rank* of
  observations, not their magnitude. A few enormous weeks move a target
  quantile far less than they would move a mean.
- This is precisely the opposite of **mean-based safety stock** (mean ± k·σ),
  where σ = 245 on a median of 15 is dominated by the tail and would produce
  absurd stocking levels. Avoiding that fragility is the reason we chose the
  quantile approach in the first place.
- The extreme weeks are plausibly real demand (bulk/wholesale orders).
  Discarding them would bias high-quantile forecasts (P90/P95/P99) downward —
  exactly the quantiles that matter most for avoiding stockouts.

## Follow-up (not blocking)

- Spot-check a handful of the largest weeks to confirm they are genuine
  orders rather than data-entry artifacts. If any are clearly errors, handle
  them as data-quality fixes — separately from this modelling decision.

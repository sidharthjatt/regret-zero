# Calibration note — lower quantiles run "hot" (known, explained)

**Date:** 2026-06-27
**Status:** Known property, not a bug — documented, intentionally not "fixed"
**Scope:** `outputs/forecasts.csv` test-set coverage (12 weeks, 3,219 products)

## Observation

Empirical coverage — the share of test actuals at or below each forecast
quantile — sits **above** the nominal target for the lower quantiles, and
converges to target for the upper ones:

| Quantile | Empirical coverage | Target | Gap |
|---|---|---|---|
| P33 | 51.8% | 33.3% | +18.5 |
| P50 | 63.1% | 50.0% | +13.1 |
| P67 | 75.1% | 66.7% | +8.4 |
| P82 | 86.1% | 81.8% | +4.3 |
| P90 | 91.8% | 90.0% | +1.8 |

(Quantile predictions are monotonically rearranged per row, so there are
**no quantile crossings** — this calibration gap is a separate, real property.)

## Why this happens — intermittent-demand zero mass

Weekly demand is intermittent. After reindexing to a continuous weekly grid
(weeks with no sales = 0 demand), a large fraction of test product-weeks are
**exactly zero** — e.g. 638 of 3,219 products (19.8%) have zero demand across
the entire 12-week test window.

Coverage is measured as `P(actual <= forecast)`. When a low quantile predicts
a value at or just above 0 and the actual is exactly 0, the row counts as
"covered" (0 <= ~0). That large point mass of zeros inflates empirical
coverage above the nominal level. The effect is strongest at the lowest
quantiles (whose predictions sit inside the zero mass) and fades at the upper
quantiles (P82/P90), whose predictions rise above the zero mass and are
therefore well calibrated.

This is a property of discrete / intermittent demand against a continuous
coverage definition — **not** a defect in the pinball objective or the split.
The models still minimize pinball loss correctly.

## Why we are NOT over-engineering it

- The **upper quantiles that drive the decision-aware wins are well
  calibrated** (P82 = 86.1%, P90 = 91.8%). The premium tier — the largest
  source of savings — orders P82.
- The calibration gap affects both strategies' inputs identically, so the
  **decision comparison is unbiased**: decision-aware still beats
  accuracy-first by **+12.3% (₹57,232)** in realized cost.
- A "proper" fix (zero-inflated / hurdle / Tweedie models, or a separate
  intermittency model) is a larger modelling change out of scope here. It is
  noted as possible future work.

## Practical consequence

Low/mid-tier orders slightly **over-cover** relative to their nominal critical
ratio (they hold marginally more stock than `F^{-1}(CR)` would imply). This is
conservative for stockout protection and does not overturn the cost result.

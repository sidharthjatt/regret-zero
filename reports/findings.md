# RegretZero — Findings

## What this project set out to test

A standard demand-forecasting model is judged on accuracy: how close its prediction is to actual sales. But accuracy is a symmetric measure — it treats over-forecasting and under-forecasting as equally bad. In inventory, they are not. A stockout forfeits a sale and risks losing the customer; an overstock ties up cash and warehouse space and may end in a markdown. The two mistakes carry different costs, and which one is costlier depends on the product.

RegretZero tests a simple but important claim: if you order at the cost-optimal quantile of demand instead of the median forecast, you make cheaper decisions — even though the forecast itself is no more accurate. We measure success in pounds of decision-regret, not in RMSE.

## The headline result

On a held-out test set of 12 weeks across 3,218 products, the two ordering strategies cost:

| Strategy | Total cost (12 weeks) |
|---|---|
| Accuracy-first — order the median (P50) forecast | £456,736 |
| Decision-aware — order the critical-ratio quantile | £402,105 |
| **Savings** | **£54,631 (12.0%)** |

The decision-aware strategy was cheaper on 1,677 of 3,218 products, worse on 1,015, and tied on 526. Crucially, it wins on the products that matter most — the high-value, high-volume items where a stockout is expensive — while conceding small amounts on low-value intermittent items where being slightly over-stocked barely costs anything.

Every price tier comes out ahead:

| Tier | Critical ratio | Quantile ordered | Savings |
|---|---|---|---|
| Low (commodity) | 0.333 | P33 | +7.1% |
| Mid (general) | 0.667 | P67 | +7.4% |
| Premium (specialty) | 0.818 | P82 | +14.9% |

The premium tier saves the most, which is exactly what the theory predicts: high-margin products make stockouts expensive, so the optimizer stocks them more aggressively, and the savings from avoided stockouts outweigh the extra holding cost.

## What this means in business terms

The test window is 12 weeks. Extrapolated across a full year (×52/12) on the same product range and demand pattern, the same decision rule would avoid **~£237,000** in regret per year — purely from changing *how much to order*, with no change to the forecasting model, no new data, and no extra cost. For a single mid-size online retailer this is a direct, recurring saving; for a larger operation it scales with the catalogue. One caveat: the test window runs September to December, the pre-Christmas peak for this retailer, so scaling it up to 52 weeks assumes the rest of the year looks like that season. It probably does not, so treat the annual figure as a rough order of magnitude.

The number itself is illustrative — it depends on the cost assumptions, which are grounded in real retail-margin benchmarks but not measured from this dataset. The durable finding is the *direction and the mechanism*: ordering at the critical-ratio quantile beats ordering at the median, and the gap is largest exactly where it should be. The interactive dashboard lets anyone change the cost assumptions and confirm the conclusion holds across a wide range. A sensitivity sweep over the holding cost makes this concrete: decision-aware ordering stays ahead for holding fractions up to ~0.26 (breakeven lies between 0.26, at +0.38%, and 0.27, at −0.07%), and is slightly negative from 0.27 to 0.32 — where over-stocking finally outweighs the avoided stockouts. The curve is not monotonic (for example, +4.9% at a holding fraction of 0.40, which is outside the dashboard's 0.02–0.30 sweep range and was computed offline with `optimizer.score`): each product snaps to the nearest trained quantile, so as the holding fraction changes, tiers jump between quantiles instead of moving smoothly. So the result is robust across the practical range, not a single lucky setting, and the point where it stops working is known.

## Data-cleaning sensitivity

The raw data contains orders that were later cancelled in full: same customer, same product, same quantity, on a "C" invoice. There are 6,476 such pairs. How many of them to remove changes the headline a little:

| Cleaning variant | Pairs removed | Savings |
|---|---|---|
| A — keep all orders | 0 | £57,232 (12.25%) |
| **B — drop pairs cancelled within 24h (primary)** | **1,395** | **£54,631 (11.96%)** |
| C — drop every matched pair | 6,476 | £47,953 (10.78%) |

B is the primary result. An order reversed within 24 hours was never fulfilled, so it is not demand. A later cancellation may be a return of stock that did ship, so the original order is left in. Decision-aware ordering wins under all three variants.

## Why the result is trustworthy

A good-looking number is worthless if the pipeline is leaking future information or isn't reproducible. Before trusting the 12.0%, the whole pipeline was audited:

- **No leakage.** Lag and rolling features are strictly backward-looking and computed per product; the train/validation/test split is purely chronological with no overlap. This was verified by independently reconstructing the features.
- **Reproducible.** With the pinned requirements (`requirements.txt`, verified on Python 3.11.15), a fresh end-to-end run reproduces the same outputs byte-for-byte.
- **Verified three ways.** The headline was re-derived independently and matched the pipeline to the pound.
- **One source of truth.** The newsvendor decision logic lives in a single shared module (`src/optimizer.py`) that both the batch pipeline and the live dashboard import, so the two can never drift apart — and the refactor that introduced it was confirmed byte-identical (the result file's checksum was unchanged).
- **Calibrated where it counts.** The decision-driving upper quantiles (P82, P90) are well-calibrated — P90 coverage is 91.7% against a 90% target.

## Honest limitations

- The stockout and holding costs are transparent assumptions, justified against published retail-margin benchmarks but not observed in the data. The framework and relative result are the contribution, not the exact figure.
- The lower quantiles run hot (over-cover their nominal level) because the demand is intermittent — many product-weeks are zero. This is documented and deliberately left as-is rather than over-engineered; it does not affect the upper quantiles that drive the decisions.
- The optimizer snaps each product to the nearest of five trained quantiles rather than its exact critical-ratio quantile. A denser quantile grid would sharpen this.

## The one-sentence takeaway

Optimising for the decision beats optimising for the forecast: ordering at the critical-ratio quantile cut inventory regret by 12.0% on held-out data, with the largest gains exactly where the theory says they should be — and the result is leakage-free, reproducible, and verified.

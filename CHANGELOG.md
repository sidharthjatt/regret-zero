# Changelog

## 2026-09-18

An audit of the pipeline and docs turned up the problems below. The headline moved from 12.3% to 12.0%.

- **Currency.** Every money figure was labelled in rupees, but the source data is priced in pounds sterling. All labels are now £. The underlying numbers didn't change: the old "₹57,232" was always £57,232.
- **Cancelled orders.** The cleaning step removed cancellation lines but kept the orders they reversed. One was a 74,215-unit order cancelled 16 minutes after it was placed. It was the largest week in the data and produced most of the reported skewness (158 with it, about 21 without). Orders cancelled in full within 24 hours are now dropped together with their cancellation (1,395 of 6,476 matched pairs). Headline: £57,232 (12.3%) to £54,631 (12.0%). How other cleaning choices change the result is in [reports/findings.md](reports/findings.md).
- **Reproducibility.** Forecasts depend on library versions. The README now says so and lists the verified environment.
- **Dashboard.** The top-10 table shows product names, the sensitivity text is computed from the current slider settings instead of fixed wording, and colors match the report charts.
- **Docs.** README and findings rewritten, and four charts added (`src/04_make_figures.py`). The calibration note no longer claims the coverage gap affects both strategies identically.

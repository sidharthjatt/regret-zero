"""
01_data_prep.py — RegretZero data preparation pipeline

Turns the raw UCI "Online Retail II" transaction log into a clean
product x week demand table suitable for forecasting.

Input : data/online_retail_II.csv   (~1M transaction rows, ~90 MB)
Output: data/demand.csv             (stock_code, week_start_date, demand)
        data/sample.csv             (first ~1000 rows, git-tracked sample)
        data/prices.csv             (stock_code, unit_price — cost basis for
                                      the optimizer; lets 03 avoid the raw file)

Run from the project root:
    python src/01_data_prep.py
"""

from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Paths — resolved relative to the project root (parent of this file's dir),
# so the script works no matter what directory you run it from.
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "online_retail_II.csv"
DEMAND_PATH = PROJECT_ROOT / "data" / "demand.csv"
SAMPLE_PATH = PROJECT_ROOT / "data" / "sample.csv"
PRICES_PATH = PROJECT_ROOT / "data" / "prices.csv"

# Minimum number of distinct weeks a product must appear in to be kept.
# Below this, a demand series is too short for meaningful forecasting.
MIN_WEEKS = 20

# Drop non-product StockCodes (postage, fees, manual adjustments)? These are
# not real SKUs, so forecasting their "demand" is meaningless. Flip to False
# to keep them. Two rules are applied together:
#   1. An explicit blocklist of known admin codes (matched case-insensitively).
#   2. Any code with no digit at all — real product codes always have a
#      numeric core (e.g. '85048', '79323P'); admin codes like POST/M/D do not.
REMOVE_NON_PRODUCT_CODES = True
NON_PRODUCT_CODES = {
    "POST", "DOT", "M", "BANK CHARGES", "ADJUST", "D", "CARRIAGE",
}

# Drop orders that were later cancelled in full? A positive line is paired with
# a cancellation ('C' invoice) for the same Customer ID, StockCode and absolute
# Quantity, timestamped after it. Pairing is one-to-one: each cancellation takes
# the most recent earlier unmatched order. Both rows of a pair are dropped
# (subject to MAX_CANCEL_LAG_HOURS below), so e.g. the 74,215-unit order 541431
# (cancelled 16 min later by C541433) no longer counts as demand. Rows without
# a Customer ID are never paired.
REMOVE_MATCHED_CANCELLATIONS = True

# Only drop a matched pair if the cancellation came within this many hours of
# the order. Quick reversals look like entry errors; most later ones look like
# returns of goods that did ship. None = drop every matched pair.
MAX_CANCEL_LAG_HOURS = 24

# Number of rows to write to the small, git-tracked sample file.
SAMPLE_ROWS = 1000


def load_raw(path: Path) -> pd.DataFrame:
    """Load the raw transaction log efficiently.

    We read only the 6 columns we actually need (skipping Description /
    Country), and declare dtypes up front so pandas doesn't have to guess.
    StockCode and Invoice are kept as strings: StockCode is alphanumeric
    (e.g. '79323P') and the leading 'C' on Invoice marks cancellations,
    which we filter on later. Customer ID is only used to pair orders with
    their cancellations.
    """
    df = pd.read_csv(
        path,
        usecols=["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price",
                 "Customer ID"],
        dtype={
            "Invoice": "string",
            "StockCode": "string",
            "Quantity": "int64",
            "Price": "float64",
        },
        parse_dates=["InvoiceDate"],
    )
    return df


def matched_cancellation_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Orders that were cancelled in full, paired with their cancellations
    (see REMOVE_MATCHED_CANCELLATIONS). Returns one row per pair with the
    index labels of both rows and the lag between them.

    Each cancellation, in time order, is paired with the most recent earlier
    unmatched order sharing (Customer ID, StockCode, |Quantity|). Pairing is
    one-to-one, so a repeated identical order is only cancelled once.
    """
    key = ["Customer ID", "StockCode", "abs_qty"]
    is_cancel = df["Invoice"].str.startswith("C", na=False)
    has_customer = df["Customer ID"].notna()
    orders = df[~is_cancel & (df["Quantity"] > 0) & has_customer]
    cancels = df[is_cancel & (df["Quantity"] < 0) & has_customer]
    orders = orders.assign(abs_qty=orders["Quantity"])
    cancels = cancels.assign(abs_qty=-cancels["Quantity"])

    # Candidate orders per key, oldest first: [(index label, timestamp), ...].
    cancel_keys = pd.MultiIndex.from_frame(cancels[key])
    orders = orders[pd.MultiIndex.from_frame(orders[key]).isin(cancel_keys)]
    pool = {
        k: list(zip(g.index, g["InvoiceDate"]))
        for k, g in orders.sort_values("InvoiceDate").groupby(key)
    }

    matched = []
    for k, g in cancels.sort_values("InvoiceDate").groupby(key):
        candidates = pool.get(k, [])
        for c_idx, c_time in zip(g.index, g["InvoiceDate"]):
            # Most recent order placed strictly before this cancellation.
            for j in range(len(candidates) - 1, -1, -1):
                if candidates[j][1] < c_time:
                    matched.append((candidates.pop(j)[0], c_idx))
                    break
    pairs = pd.DataFrame(matched, columns=["order_idx", "cancel_idx"])
    pairs["lag"] = (df.loc[pairs["cancel_idx"], "InvoiceDate"].to_numpy()
                    - df.loc[pairs["order_idx"], "InvoiceDate"].to_numpy())
    return pairs


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning rules, printing a before/after count at each step
    so the funnel is transparent.
    """
    print("\n--- Cleaning ---")
    n_start = len(df)
    print(f"rows loaded:                      {n_start:>10,}")

    # 1. Drop rows we can't group on: missing StockCode or InvoiceDate.
    df = df.dropna(subset=["StockCode", "InvoiceDate"])
    print(f"after drop missing key fields:    {len(df):>10,}  (-{n_start - len(df):,})")

    # 2. Remove orders that were later cancelled in full, together with their
    #    cancellations. Must run before step 3 drops the cancellation rows.
    if REMOVE_MATCHED_CANCELLATIONS:
        prev = len(df)
        pairs = matched_cancellation_pairs(df)
        n_matched = len(pairs)
        if MAX_CANCEL_LAG_HOURS is not None:
            pairs = pairs[pairs["lag"] <= pd.Timedelta(hours=MAX_CANCEL_LAG_HOURS)]
        df = df.drop(pd.Index(pairs["order_idx"]).append(pd.Index(pairs["cancel_idx"])))
        lag_note = "no lag limit" if MAX_CANCEL_LAG_HOURS is None else f"lag <= {MAX_CANCEL_LAG_HOURS}h"
        print(f"after removing matched cancels:   {len(df):>10,}  (-{prev - len(df):,}; "
              f"{len(pairs):,} of {n_matched:,} matched pairs, {lag_note})")

    # 3. Remove cancelled invoices (Invoice number starts with 'C').
    prev = len(df)
    df = df[~df["Invoice"].str.startswith("C", na=False)]
    print(f"after removing cancellations:     {len(df):>10,}  (-{prev - len(df):,})")

    # 4. Remove non-sales lines: zero/negative quantity (returns) or
    #    zero/negative price (free items, adjustments).
    prev = len(df)
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    print(f"after removing qty<=0 / price<=0: {len(df):>10,}  (-{prev - len(df):,})")

    # 5. Remove non-product StockCodes (postage, fees, adjustments).
    if REMOVE_NON_PRODUCT_CODES:
        prev = len(df)
        code = df["StockCode"].str.strip().str.upper()
        is_blocklisted = code.isin(NON_PRODUCT_CODES)
        has_no_digit = ~df["StockCode"].str.contains(r"\d", na=False)
        df = df[~(is_blocklisted | has_no_digit)]
        print(f"after removing non-product codes: {len(df):>10,}  (-{prev - len(df):,})")

    return df


def to_weekly_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions into product x week demand.

    The week_start_date is the Monday of the week each sale falls in.
    We compute it explicitly: normalize the timestamp to midnight, then
    subtract its weekday offset (Monday=0 ... Sunday=6). This is clearer
    and less surprising than relying on period-anchor strings.
    """
    print("\n--- Weekly aggregation ---")

    ts = df["InvoiceDate"]
    week_start = ts.dt.normalize() - pd.to_timedelta(ts.dt.weekday, unit="D")

    weekly = (
        df.assign(week_start_date=week_start)
        .groupby(["StockCode", "week_start_date"], as_index=False)["Quantity"]
        .sum()
        .rename(columns={"StockCode": "stock_code", "Quantity": "demand"})
    )
    print(f"product-week rows before filter:  {len(weekly):>10,}")
    return weekly


def filter_min_history(weekly: pd.DataFrame) -> pd.DataFrame:
    """Keep only products observed in at least MIN_WEEKS distinct weeks."""
    weeks_per_product = weekly.groupby("stock_code")["week_start_date"].transform("size")
    kept = weekly[weeks_per_product >= MIN_WEEKS].copy()
    n_dropped_products = weekly["stock_code"].nunique() - kept["stock_code"].nunique()
    print(
        f"product-week rows after >= {MIN_WEEKS} wks:  {len(kept):>10,}"
        f"  (dropped {n_dropped_products:,} short-history products)"
    )
    return kept.sort_values(["stock_code", "week_start_date"]).reset_index(drop=True)


def summarize(weekly: pd.DataFrame) -> None:
    """Print a human-readable summary of the final demand table."""
    print("\n--- Summary ---")
    print(f"final rows:        {len(weekly):,}")
    print(f"unique products:   {weekly['stock_code'].nunique():,}")
    print(f"unique weeks:      {weekly['week_start_date'].nunique():,}")
    print(
        "date range:        "
        f"{weekly['week_start_date'].min().date()} "
        f"-> {weekly['week_start_date'].max().date()}"
    )
    print("\nsample of output:")
    print(weekly.head(8).to_string(index=False))


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data not found at {RAW_PATH}")

    print(f"Loading {RAW_PATH.name} ...")
    df = load_raw(RAW_PATH)

    df = clean(df)
    weekly = to_weekly_demand(df)
    weekly = filter_min_history(weekly)

    # Persist full cleaned demand + a small tracked sample.
    weekly.to_csv(DEMAND_PATH, index=False)
    weekly.head(SAMPLE_ROWS).to_csv(SAMPLE_PATH, index=False)
    print(f"\nWrote {DEMAND_PATH.relative_to(PROJECT_ROOT)} ({len(weekly):,} rows)")
    print(
        f"Wrote {SAMPLE_PATH.relative_to(PROJECT_ROOT)} "
        f"({min(SAMPLE_ROWS, len(weekly)):,} rows, git-tracked)"
    )

    # Per-product average unit price — the cost basis for the optimizer layer
    # (03_optimize.py). Computed from the cleaned sales rows (so it reflects
    # real selling prices) and restricted to the products that survive into
    # demand.csv. Emitting it here means 03 never has to touch the 90 MB raw.
    kept_codes = weekly["stock_code"].unique()
    prices = (
        df[df["StockCode"].isin(kept_codes)]
        .groupby("StockCode")["Price"]
        .mean()
        .rename("unit_price")
        .reset_index()
        .rename(columns={"StockCode": "stock_code"})
    )
    prices.to_csv(PRICES_PATH, index=False)
    print(f"Wrote {PRICES_PATH.relative_to(PROJECT_ROOT)} ({len(prices):,} products)")

    summarize(weekly)


if __name__ == "__main__":
    main()

# GDELT Bitcoin narrative source v2 transport amendment

## Reason for amendment

The frozen v1 downloader required every API response to contain all 1,461 daily
bins. Its first full broad-query run failed before writing any response cache or
final artifact because GDELT returned 1,459 bins.

Outcome-blind date-only diagnostics established:

- the full broad request covered the correct first/last dates and daily
  resolution but omitted `2020-10-20` and `2023-03-23`;
- independent two-year broad requests omitted the same date in each half;
- an unrelated `(economy OR government OR market)` anchor also omitted each
  target date in an 11-day daily window.

No article count, BTC candle, funding value, return, PnL, CAGR, or MDD was
inspected. The evidence identifies two global GDELT monitoring-outage days, not
a long-window truncation problem.

## Corrective transport

The original v1 source and preregistration remain immutable. V2 is implemented
in `training/download_gdelt_bitcoin_narrative_daily_v2.py` and cryptographically
binds its v1 dependency before import.

V2 preserves without change:

- all four feature queries;
- `[2020-01-01, 2024-01-01)`;
- daily resolution;
- the `+48h15m` availability clock;
- one full half-open request per query;
- all GNRC features, signs, windows, thresholds, holds, and support gates.

Only sparse-bin semantics change:

1. an omitted query/day bin means zero matching articles;
2. the global norm is the unanimous norm among available feature-query bins;
3. when all four bins are absent, global norm and all four counts are zero;
4. category counts may never exceed the broad count;
5. the all-query outage set must equal exactly
   `{2020-10-20, 2023-03-23}` or acquisition fails closed.

The pseudocount already frozen by GNRC makes an all-zero outage row finite. No
threshold or signal formula is repaired from these dates.

## Outcome boundary

V2 writes the same final daily/raw/manifest paths because v1 never finalized
them, but uses a separate cache directory and protocol version. After download,
an outer source seal containing the exact manifest SHA-256 must be committed
before the source-support evaluator may parse daily feature counts.

Official API semantics:

- [GDELT DOC 2.0 API](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/)
- [TimelineVolRaw announcement](https://blog.gdeltproject.org/gdelt-2-0-api-now-supports-raw-result-counts/)

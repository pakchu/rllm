# Cboe tail-risk source audit — 2026-07-18

## Scope

This freeze adds a source family that is independent of the repository's BTC
price, taker, funding, premium, OI, Kimchi, DXY, REX, CFTC, network, and
central-bank-liquidity inputs: **SPX option-implied tail risk and
volatility-of-volatility**.

Official daily histories:

- <https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv>
- <https://cdn.cboe.com/api/global/us_indices/daily_prices/VVIX_History.csv>
- <https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv>

Cboe describes SKEW as information about demand for tail-risk options and VVIX
as expected volatility of the 30-day forward VIX:

- <https://www.cboe.com/insights/posts/inside-volatility-trading-the-adventures-of-volatility-markets>
- <https://cdn.cboe.com/resources/indices/documents/SKEWwhitepaperjan2011.pdf>
- <https://cdn.cboe.com/resources/indices/documents/vvix-termstructure.pdf>

## Leakage boundary

- The builder reads only the three official Cboe CSV responses.
- It retains observations from `2018-01-01` through `2023-12-31`; later rows
  are discarded before panel construction.
- No BTC bar, return, funding, portfolio record, label, or model output is read.
- The exact three-way date intersection is used. Missing dates are never
  forward-filled.
- Raw-response, normalized-snapshot, panel, and manifest hashes are frozen.
- A snapshot rebuild must reproduce the panel byte for byte.

The historical CSVs are a frozen current Cboe vintage, not a point-in-time
archive of every publication. A downstream policy may therefore use a completed
source close only at the **next Cboe source date 09:35 America/New_York**. Live
promotion requires forward-vintage parity monitoring and fails closed on an
unexplained source change.

## Methodology-version note

Cboe published a 2025 consultation result saying modifications to the SKEW
methodology were being developed:
<https://cdn.cboe.com/resources/release_notes/2025/Consultation-Results-Regarding-Proposed-Changes-to-the-Cboe-SKEW-Index-SKEW-.pdf>.
The research panel ends on `2023-12-29`, before that notice. This freeze does not
assume that a future revised SKEW series is interchangeable with the frozen
2018–2023 history; any live adapter must pin and audit the active methodology.

## Frozen coverage

| Source | Rows | First | Last |
|---|---:|---|---|
| SKEW | 1,507 | 2018-01-02 | 2023-12-29 |
| VVIX | 1,509 | 2018-01-02 | 2023-12-29 |
| VIX | 1,521 | 2018-01-02 | 2023-12-29 |
| exact intersection | 1,507 | 2018-01-02 | 2023-12-29 |

Frozen panel SHA-256:
`cdde3f8d4bb1e23d00b192f5f9ef759aefba9087be5fd60653e9c02479dfa41a`.

This artifact freezes input data only. No post-signal BTC outcome has been
opened by the builder.

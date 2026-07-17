# Cboe volatility term-structure source audit — 2026-07-17

## Scope

This freeze introduces a source family not used by the existing BTC price,
taker, funding, premium, OI, Kimchi, DXY, REX, CFTC, network, or central-bank
liquidity sleeves: the **Cboe SPX implied-volatility term structure**.

Official daily histories:

- <https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv>
- <https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv>
- <https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv>

Cboe states that VIX9D, VIX, and VIX3M apply the VIX methodology at 9-day,
30-day, and three-month target horizons:
<https://www.cboe.com/tradable-products/vix/term-structure> and
<https://cdn.cboe.com/api/global/us_indices/governance/Volatility_Index_Methodology_Selected_SPX_Target_Expected_Volatility_Term_Indices.pdf>.

## Leakage boundary

- The builder reads only the three official Cboe CSV responses.
- It retains dates from `2018-01-01` through `2023-12-31`; later source rows are
  discarded before panel construction.
- No BTC bars, returns, funding, portfolio records, labels, or model outputs are
  read.
- The exact three-way date intersection is used; missing index dates are never
  forward-filled.
- Raw-response, normalized-snapshot, panel, and manifest hashes are frozen.
- An offline `--from-snapshot` rebuild must reproduce the panel byte for byte.

The historical CSV is a frozen current Cboe vintage, not an archived sequence
of every past file publication. The candidate therefore uses the prior trading
day's close only at the **next Cboe trading day 09:35 America/New_York**, well
after the source observation was publicly calculable. Any live promotion also
requires forward-vintage parity monitoring; silent source changes fail closed.

## Frozen coverage

| Source | Rows | First | Last |
|---|---:|---|---|
| VIX | 1,521 | 2018-01-02 | 2023-12-29 |
| VIX9D | 1,509 | 2018-01-02 | 2023-12-29 |
| VIX3M | 1,509 | 2018-01-02 | 2023-12-29 |
| exact intersection | 1,509 | 2018-01-02 | 2023-12-29 |

Frozen panel SHA-256:
`6f1b2f7f3a5b1e4d5001d673e6ff54374791879c278248ce27b3d610e4f75dc7`.

This artifact freezes input data only. No post-signal BTC outcome has been
opened by the builder.

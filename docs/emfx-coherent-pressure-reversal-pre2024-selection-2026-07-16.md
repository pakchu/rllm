# EM-FX coherent pressure reversal — pre-2024 selection

## Frozen mechanism

AUD, CNY, INR and MXN one-session USD returns are standardized with prior-only 252-session histories. Their equal-weight mean is multiplied by absolute common-mode coherence, suppressing idiosyncratic FX moves. When this score enters either fitted tail, the policy fades the already-completed BTC 24-hour move for seven days.

- feature: `em_coherent_pressure_1d`
- fitted tails: `0.2`; lower `-0.281018635241`, upper `0.395746853380`
- direction rule: `price_reversal`
- hold: `7` days, no overlap
- fixed FX panel: `USDAUD, USDCNY, USDHKD, USDINR, USDMXN`; HKD is tested separately and is not part of the chosen common factor.
- semantic availability: complete UTC day plus five minutes; completed 5-minute signal bar; next-open fill.
- source vintage: the local PostgreSQL rows were historically backfilled, not captured in a point-in-time database snapshot. The values are fixed-panel timestamped quotes, but promotion requires live forward validation.
- execution: 0.5x, 6 bp per side, realized funding, full-calendar CAGR, intratrade strict MDD.

## Selection evidence

The `1368`-cell family first requires at least five trades and positive absolute return in each 2021/2022/2023H1/2023H2 block, plus full-2023 CAGR/strict-MDD of at least `2.0`. Among eligible cells, selection maximizes the minimum subperiod ratio before the full-2023 tie-break.

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |
|---|---:|---:|---:|---:|---:|---:|
| fit_2021 | 28.0261% | 28.0478% | 30.0008% | 0.9349 | 34 | 20/14 |
| fit_2022 | 72.9943% | 73.0592% | 15.3973% | 4.7450 | 34 | 15/19 |
| select_2023_h1 | 16.2064% | 35.4038% | 14.9721% | 2.3647 | 15 | 8/7 |
| select_2023_h2 | 17.1067% | 36.8165% | 12.3461% | 2.9820 | 16 | 6/10 |
| select_2023 | 34.3650% | 34.3922% | 14.9721% | 2.2971 | 31 | 16/15 |

## Falsification controls

- Every-valid-FX-day BTC reversal loses in every core block.
- Direct/risk mappings and ±1/3/7/14 valid-session clock shifts are reported without altering the frozen choice.
- `2000` year- and count-matched random clocks: all-block-positive fraction `0.0430`; q99 minimum-block ratio `0.5625`; empirical p(minimum-block ratio) `0.003498`; empirical p(positive blocks and summed return) `0.000500`.

## Integrity boundary

- The exploratory family opened `1368` pre-2024 cells. Trade and random-clock p-values are descriptive post-selection diagnostics, not family-wise correction.
- PostgreSQL ingestion timestamps are later backfills. Inputs are timestamped market quotes with a fixed symbol panel rather than a reconstructed composition, but this is explicitly not point-in-time source evidence. Source and feature prefixes are hash-frozen before future replay.
- 2024+ BTC outcomes are globally research-seen in this repository; the manifest prevents reranking this family but does not create a pristine human holdout.
- Promotion requires positive frozen test/eval/holdout performance, doubled-cost survival, actual trade/PnL orthogonality, and a live point-in-time forward window.

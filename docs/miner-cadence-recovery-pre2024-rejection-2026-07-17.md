# MCR-7 pre-2024 rejection

MCR-7 is **rejected before orthogonality and before opening 2024**. No
threshold, side, hold, latency, cost, or control was repaired after outcomes.

## Frozen primary result

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Mean gross edge | Trades |
|---|---:|---:|---:|---:|---:|---:|
| 2021-03 through 2022 | -37.2817% | -22.4264% | 46.2578% | -0.4848 | -182.18 bp | 38 |
| 2021 | +2.0031% | +2.3956% | 32.7534% | 0.0731 | +131.33 bp | 13 |
| 2022 | -36.5363% | -36.5560% | 39.9944% | -0.9140 | -334.91 bp | 24 |
| 2023 | +8.8701% | +8.8764% | 14.5582% | 0.6097 | +97.31 bp | 27 |
| 2023 H1 | +12.4173% | +26.6425% | 7.7347% | 3.4446 | +269.46 bp | 10 |
| 2023 H2 | -4.1993% | -8.1635% | 12.0858% | -0.6755 | -19.84 bp | 16 |

At the frozen 10 bp/notional/side stress, train absolute return was
`-38.2295%` and 2023 absolute return was `+7.6988%`.

## Why this is not alpha

- The sign reversed by regime: mildly positive in 2021, deeply negative in
  2022, positive only in 2023 H1, and negative again in 2023 H2.
- The random-clock diagnostic beat the primary in both broad windows:
  `+29.2768%` train and `+28.2459%` in 2023, versus `-37.2817%` and `+8.8701%`.
- Constant weekly long exposure returned `+52.1792%` in 2023, showing that much
  of the apparent 2023 result was ordinary long-market beta.
- Direction flip was not a stable replacement: `+35.7401%` in train but
  `-12.6107%` in 2023. It remains a diagnostic and is not promoted.
- The extra five-minute delay reproduced the failure, so latency was not the
  cause: train `-37.4598%`, 2023 `+8.9628%`.

The mechanism therefore detects a state whose directional consequence depends
on the broader market regime; it does not provide persistent standalone edge.

## Integrity boundary

- Market parsing physically stopped before the first `2024-01-01` row.
- 2024, 2025, and 2026 YTD remain sealed.
- CAGR uses each full declared wall-clock window, including idle cash.
- Strict MDD uses the global/pre-entry HWM, favorable-before-adverse held OHLC,
  entry/exit/hypothetical-liquidation costs, and funding credits/debits.
- Funding settlement times and rates are exact frozen Binance observations.
  Historical settlement notional uses the uniformly frozen official 8h
  mark-kline-open proxy; the evaluator result's phrase “returned
  settlement_mark_price” is a metadata wording error, not the implemented
  source. Its measured worst funding-cash proxy error is
  `0.001348432 bp/notional`.
- Orthogonality and portfolio improvement were not evaluated because the
  primary performance gate failed first.

Result SHA-256:
`c7c3100847b3318fb0b2976a985042594be2b30086ab13ca032c77bc3c41e74f`

Result manifest hash:
`c5e9f3ecb7d7d5b24e01bc7ebcb4cce1c463e97d83bb722035466109d6b03e09`

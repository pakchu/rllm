# CMSR-36 source-support result

## Verdict

**PASS outcome-blind support and novelty gates.** CMSR-36 may advance to a
separately committed, hash-frozen strict train evaluator. No `BTCUSDT`
execution OHLC, funding row, return, PnL, CAGR, strict MDD, or hit rate was
loaded.

## Frozen clock support

| Split | Clocks | Long / short | Max month | Max contract-pair |
|---|---:|---:|---:|---:|
| Fit 2020-08 through 2022 | 93 | 41 / 52 | 9.68% | 16.13% |
| 2020H2 | 19 | 8 / 11 | — | — |
| 2021H1 | 16 | 5 / 11 | — | — |
| 2021H2 | 21 | 9 / 12 | — | — |
| 2022H1 | 12 | 4 / 8 | — | — |
| 2022H2 | 25 | 15 / 10 | — | — |
| 2023 support-only | 65 | 26 / 39 | 20.00% | 32.31% |
| 2023H1 / H2 | 35 / 30 | 16/19 · 10/20 | — | — |

Every frozen count, side-balance, month-concentration, and pair-concentration
gate passed. The q90 share/q80 flow/q80 lead-shock cell is unchanged from the
preregistration.

## Clock novelty

The nearest source-family comparator is the rejected single-bar COIN-M roll
policy. Exact signal Jaccard is 0.00075 in fit and 0.00421 in 2023; only 12.90%
and 10.77% of CMSR signals are within ten minutes of that dense clock, below
the frozen 25% cap.

All independent-clock exact-entry Jaccards are zero. CMSR entries within six
hours of each comparator are:

| Comparator | Fit | 2023 |
|---|---:|---:|
| COIN-M calendar compression | 9.68% | 7.69% |
| PSR-30/6 | 4.30% | 7.69% |
| CCPR-H4 clock family | 3.23% | 6.15% |
| CLBR-24 | N/A | 12.31% |
| EBLR-60/30 | N/A | 6.15% |
| CIPA-48 | 5.38% | 9.23% |

Every observed value is below the frozen 25% cap.

## Next evidence boundary

The next work unit must implement the strict evaluator, freeze its source and
schedule hashes while outcomes remain unopened, and only then parse the
`[2020-08-01, 2023-01-01)` BTCUSDT train window. The 2023 outcome stays sealed
unless every train gate passes. Threshold, direction, feature, latency, hold,
and support gates are immutable.

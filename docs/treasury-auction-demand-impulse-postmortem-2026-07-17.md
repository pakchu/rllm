# TADI-1 rejection postmortem — 2026-07-17

## Verdict

TADI-1 is rejected without parameter tuning. The untouched 2023 window remains
sealed and must not be opened for this family.

| Window / cost | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | p(two-sided) |
|---|---:|---:|---:|---:|---:|---:|
| 2021–2022, 6 bp/side | -4.2508% | -2.1500% | 13.0136% | -0.1652 | 28 | 0.7541 |
| 2021–2022, 10 bp/side | -5.3187% | -2.6975% | 13.8666% | -0.1945 | 28 | 0.6842 |
| 2021 | 0.0122% | 0.0122% | 8.5115% | 0.0014 | 13 | 0.9766 |
| 2022 | -4.2625% | -4.2653% | 10.1369% | -0.4208 | 15 | 0.6978 |

## What failed

- Gross edge was negative at -12.60 bp per underlying trade before costs.
- Results were not statistically distinguishable from zero (`p=0.7541`).
- The sign was unstable across contained years: flat in 2021 and negative in
  2022.
- Neither bid-to-cover nor indirect-share demand supplied a profitable standalone
  mechanism control.
- The one-auction-delay falsification control was positive, but weak
  (CAGR/MDD 0.35, `p=0.5948`) and therefore is not evidence of a tradable edge.

## Research consequence

Treasury coupon-auction demand is source-orthogonal, but this fixed mapping from
demand surprise to a 24-hour BTC position has no standalone alpha. Do not tune
rank thresholds, holding period, entry delay, tenor selection, or direction on
the opened outcome. The next search must use a different exogenous mechanism,
not a repaired TADI variant.

## Integrity evidence

- Only `[2021-01-01, 2023-01-01)` market and funding rows were physically parsed.
- Full-clock CAGR includes every idle day.
- Strict MDD includes intratrade adverse OHLC.
- Stage2 invocation fails before parsing 2023 with
  `TADI-1 Stage1 failed; 2023 remains sealed`.
- Frozen result manifest:
  `4d1525e039745619826e4ae0c8a5716730e7857a83b237e4386bdaacfb54b921`.

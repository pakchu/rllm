# ICDR-144 support result — rejected before returns

## Decision

`ICDR-144 — Inverse-Collateral Deleveraging Reclaim` is rejected at the
outcome-blind support stage. No execution OHLC, entry-to-exit return, funding
cash flow, absolute return, CAGR, strict MDD, or CAGR/strict-MDD was opened.
Consequently all performance statistics are **N/A by design**, not zero.

- preregistration commit: `a46e13c86daa706e01c28c6c186e57ca9ff93866`
- metrics source commit: `8d347432cd36d59458ad9a26c7c8aef1ec94b8ee`
- support manifest SHA-256:
  `20803dd46f1da0544ebd04ee174fa889edb0abe7318740e3efac49ffb07c4f79`
- empty rejected-clock SHA-256:
  `457399d0b1500cc18264eec4187b9d35e95e54fef3aa91f147551d57f26d26ca`
- decision: `reject_before_returns_no_train_support_q`
- 2023 outcome and all 2024+ data remain unopened for this candidate

## What was tested

The signal used only official Binance five-minute USD-M and COIN-M metrics:

- one-hour change in USD-M open-interest notional;
- one-hour change in COIN-M inverse-contract count;
- their relative contraction;
- three-bar USD-M and COIN-M taker long/short ratios;
- a COIN-M-specific sell gap;
- a source-only reclaim requiring taker recovery and nonnegative one-bar
  COIN-M OI change.

Every threshold used a strictly lagged 8,640-bar rolling window with at least
2,016 prior clean observations. Missing data was never filled; an unavailable
row and the following 24 rows were quarantined. Each policy reserved its own
globally non-overlapping 12-hour clock with a two-bar execution delay.

The purge quantile grid was fixed at
`{0.80, 0.85, 0.90, 0.925, 0.95}`. Selection inspected only 2021-07-08 through
2022-12-31 support. Because no quantile passed train support, 2023 support was
not used and no outcome evaluator was created.

## Train support evidence

| Q | non-overlap events | 2021 partial | 2022 | confirmation rate | largest month share | failed gates |
|---:|---:|---:|---:|---:|---:|---|
| 0.950 | 38 | 14 | 24 | 82.69% | 23.68% | total, both subperiods, confirmation, concentration |
| 0.925 | 66 | 26 | 40 | 79.00% | 18.18% | total, 2022, concentration |
| 0.900 | 89 | 33 | 56 | 80.58% | 19.10% | total, confirmation, concentration |
| 0.850 | 143 | 57 | 86 | 80.08% | 15.38% | confirmation, concentration |
| 0.800 | 193 | 73 | 120 | 82.45% | 15.03% | confirmation, concentration |

The least sparse variants did not express a selective reclaim state:
approximately four out of five accepted purges confirmed within one hour.
They were also concentrated in December 2022. The closest cases missed frozen
limits only narrowly, but relaxing `80%` confirmation or `15%` monthly
concentration after seeing these counts would be a prohibited post-result
repair.

## Structural-control evidence

The failure was not caused by excessive overlap with the preregistered
component controls. For example, at `Q=0.85` the exact-entry Jaccards were:

- CM-only OI: `0.2177`;
- no taker gap: `0.3754`;
- no reclaim: `0.0000`;
- no OI stop: `0.2392`;
- matched USD-M: `0.0017`;
- one-hour delay: `0.0000`;
- one-day shift: `0.0035`.

All were under their frozen limits. This only shows that the clocks were
distinct from their controls; it does not rescue insufficient selectivity and
time distribution, and it is not portfolio orthogonality. Existing-alpha
trade/PnL orthogonality is intentionally not measured for a candidate that
fails its standalone support gate.

## Interpretation and next research constraint

The relative inverse-collateral axis remains economically different from REX,
funding/premium, FX/Kimchi, price action, aggregate-trade, and attention
families. This exact purge-and-reclaim construction is nevertheless retired.
The next candidate must use a genuinely different observable or state
transition rather than:

- lowering the same purge quantile;
- weakening the source-only reclaim;
- widening the confirmation window;
- reversing the direction;
- adding a price/REX gate to repair this clock.

This preserves the negative result and prevents a near-threshold support miss
from turning into an outcome-mined variant family.

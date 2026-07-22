# IVFHR-72 source-support rejection

## Verdict

**Retire IVFHR-72 without opening a post-entry outcome.** The exact frozen
equal-notional flow-handoff event produced only one primary candidate across
2020–2023. It therefore fails density, yearly/half-year coverage, side balance,
calendar dispersion, and maximum-gap gates before economics are considered.

No post-entry price, future return, funding cash flow, PnL, absolute return,
CAGR, strict MDD, hit rate, LLM label, or 2024+ source row was loaded. A strict
evaluator and LLM abstention stage are not authorized.

## Frozen source-only result

| Window | Events | LONG | SHORT |
|---|---:|---:|---:|
| Train 2020–2022 | 1 | 1 | 0 |
| 2020 | 0 | 0 | 0 |
| 2021 | 0 | 0 | 0 |
| 2022 | 1 | 1 | 0 |
| Selection 2023 | 0 | 0 | 0 |
| 2023 H1 | 0 | 0 | 0 |
| 2023 H2 | 0 | 0 | 0 |
| All | **1** | **1** | **0** |

The source funnel explains the collapse:

- 1,197 valid equal-notional first-passage anchors;
- 1,107 anchors with the frozen strictly-prior reference warmup;
- 1,050 calendar-consecutive anchors;
- 259 anchors preceded by a three-anchor same-side state;
- 471 raw sign handoffs;
- 415 strong-flow anchors;
- 236 price-lag anchors; but
- only **one** anchor satisfying persistent state, sign handoff, q60 new flow,
  and price lag simultaneously.

## Frozen controls

| Clock | Events | LONG | SHORT | Max gap (days) | Longest side run |
|---|---:|---:|---:|---:|---:|
| primary | 1 | 1 | 0 | n/a | 1 |
| any handoff + price lag | 66 | 29 | 37 | 90.70 | 7 |
| no price lag | 15 | 9 | 6 | 259.83 | 5 |
| no flow strength | 18 | 10 | 8 | 161.85 | 4 |
| persistence level | 5 | 1 | 4 | 420.80 | 4 |
| fixed-noon handoff | 0 | 0 | 0 | n/a | 0 |

The result is not a coding artifact that can be repaired by deleting one
threshold. Both q60 and price-lag ablations remain far below the frozen primary
density requirement, while the broad `any_handoff` control is a materially
different event identity and still misses the frozen 90-day maximum gap by
0.70 day. Controls are report-only and cannot rescue IVFHR-72.

## Rejection boundary

Do not change the three-anchor state, q60 flow threshold, price-lag sign,
50%-volume anchor, UTC-day origin, 17:55 cutoff, next-open timing, or six-hour
hold and call it IVFHR-72. The exact candidate is terminally rejected.

The useful source-only finding is that daily flow signs switch often, but a
**mature same-side state followed by a strong opposite flow while price still
lags is an intersection of mostly disjoint conditions**. A successor should
not add more gates. It must either:

1. treat the broad handoff itself as a separately disclosed source-seen event;
   or
2. leave the equal-notional family and test a different causal data axis.

Neither route may inspect an IVFHR post-entry return, because none has been
opened.

## Integrity

- final preregistration commit: `d4ddf16`;
- preregistration file SHA-256:
  `e01e7f5af034adf98c0eef1e086ed1265c02998641f39d8cddd5137089f4153e`;
- preregistration manifest:
  `12bc72d42deca3382ade0456c64431ac3cfd4ca081aea453621c68834a4a9bbb`;
- support report manifest:
  `2dc150bbcb4ebfbf2c24bd902290771710d3aede70253bd62cb42c74455ad23e`;
- clock SHA-256:
  `ab12762dec9a93d41c293766e46dfc80ade81914fb32753a5923faa6437c338e`;
- clock rows across primary and controls: 107.

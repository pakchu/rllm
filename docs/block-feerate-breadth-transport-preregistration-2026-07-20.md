# BFRT-288 singleton preregistration — 2026-07-20

## Status

This document freezes exactly one **Block Fee-Rate Breadth Transport** policy
before any archived fee-rate value, derived feature, signal incidence, BTC bar,
funding value, return, PnL, CAGR, or MDD is opened. The policy is **BFRT-288**:
a 24-hour, non-overlapping BTC long/short rule based on broad transport across
the inner mined-block fee-rate percentiles.

The source is bound to manifest
`fe616bcf294e8b3b2abc6dec124e922f77df4bca47a86249fc270f2af6b46f21`
and normalized artifact
`007d13ba756fd29faae1ae87caa11554438b54bb5028f24b2f0c21ddf3a0e55d`.
The exact machine-readable policy will be generated only after its
implementation and tests are committed.

## Feature definition

For each retained 12-hour source bucket `t`, use only percentiles
`p = {10,25,50,75,90}`. Percentiles 0 and 100 remain source-quality fields and
never enter BFRT-288.

1. `x[p,t] = log1p(fee_p[t])`.
2. Require `t`, `t-1`, and `t-2` to be consecutive 12-hour buckets.
3. Define the 24-hour transport `d[p,t] = x[p,t] - x[p,t-2]`.
4. Define `L = median_p(d[p,t])` and `S = sum_p(abs(d[p,t]))`.
5. Reject the row when `S <= 0`, `L == 0`, or any input/result is non-finite.
6. Define signed coherence `C_s = sum_p(d[p,t]) / S` and coherence
   `C = abs(C_s)`.
7. Reject when `C_s == 0` or `sign(C_s) != sign(L)`.
8. Define tail divergence
   `T = abs((d[90]-d[75])-(d[25]-d[10])) / S`.

All operations use IEEE-754 binary64 `math.log1p`. There is no epsilon,
clipping, rounding, interpolation, forward fill, or imputation. Exact binary64
equality defines rank ties.

## Strict-prior normalization and singleton rule

At each valid row, calculate rolling empirical midranks over the most recent
180 **strict-prior valid feature rows**, requiring at least 120. Current-row
values are excluded, and every prior row must already be causally available.

`midrank = (count(prior < current) + 0.5*count(prior == current)) / prior_count`.

Calculate:

- `magnitude_rank` from `abs(L)`; and
- `tail_divergence_rank` from `T`.

The only primary eligibility rule is:

```text
magnitude_rank >= 0.75
and coherence >= 0.60
and tail_divergence_rank <= 0.75
```

The side is `sign(C_s)`, which must equal `sign(L)`: positive is long and
negative is short. No learned coefficient or threshold exists.

## Causal clock and non-overlap

- Source availability is fixed bucket end plus 48 hours.
- `entry_time = ceil_5m(source_available_at) + 5 minutes`.
- Enter at the open after that complete latency bar.
- `scheduled_exit_time = entry_time + 24 hours`, or 288 five-minute bars.
- Sort candidates by `(entry_time, bucket_start)`.
- Accept a candidate only when its entry is at or after the prior accepted
  exit; suppress every intervening signal without score priority or
  replacement.
- A trade is `[entry_time, scheduled_exit_time)` and may not cross a declared
  split boundary.

Execution uses 0.5 notional leverage, 6 bp per notional per side at base cost,
10 bp at stress cost, and exact entry-inclusive/exit-exclusive funding with
fixed entry quantity.

## Frozen calendar

All assignments use entry time, and the full hold must fit inside the window.

- warm-up source only: `2023-07-20T12:00:00Z` through 2023-10-31;
- train: `[2023-11-01T00:00:00Z, 2025-01-01T00:00:00Z)`;
- test: calendar 2025; and
- eval: `[2026-01-01T00:00:00Z, 2026-07-20T00:00:00Z)`.

Test and eval are report-only. They may not select, rerank, invert, refit, or
repair anything.

## Outcome-blind support gate

Count only accepted primary non-overlap entries. Before any market or funding
value is loaded, all of the following must pass:

| Window | Total | Long | Short | Dispersion |
| --- | ---: | ---: | ---: | --- |
| Train | >=80 | >=25 | >=25 | 2023 Nov-Dec >=8; 2024H1 >=14; 2024H2 >=14; max month share <=15% |
| Test | >=35 | >=12 | >=12 | each half >=14; each quarter >=6; max month share <=20% |
| Eval | >=20 | >=6 | >=6 | 2026H1 >=18; max month share <=25% |

The frozen source must also have zero missing 12-hour buckets. Any failure
rejects BFRT-288 without outcomes. Threshold, side, rank-window, latency, hold,
or calendar repair is forbidden.

## Controls

The evaluator must report these already-frozen controls:

- direction flip, diagnostic only;
- magnitude-only, removing coherence and tail filters but retaining sign
  agreement and an independent chronological non-overlap clock;
- vetoed-tail, using the same magnitude/coherence gates but
  `tail_divergence_rank > 0.75` and its own clock;
- constant long and constant short on the primary clock;
- the fully formed feature/ranks delayed by 14 source buckets;
- month-and-side-stratified random clocks, seed `20260720`; and
- one complete five-minute bar delayed entry and exit.

If a component control other than direction flip independently passes the full
performance gate, the specific BFRT breadth-transport mechanism is rejected;
the primary policy is not repaired around the control.

## Sequential performance gate

Only after support passes may a strict evaluator be committed and hash-frozen.
It then opens train, test, and eval in that order, stopping permanently on the
first failure. Every opened window must have:

- positive absolute return;
- full-calendar CAGR / global strict MDD at least 3;
- global/pre-entry-HWM strict MDD no greater than 15%;
- one-sided weekly cluster sign-flip `p <= 0.10`, 100,000 draws, seed
  `20260720`;
- mean gross return at least 20 bp per trade;
- positive stress-cost absolute return; and
- positive one-bar-delayed-entry absolute return.

Train 2024H1/H2, test 2025H1/H2, and eval 2026H1 must each be positive. Long
and short sleeves must each be positive in train and test. CAGR always uses the
entire declared wall-clock window, including idle cash. Strict MDD includes the
pre-entry high-water mark, all costs, exact funding, and the path ordering with
all favorable extremes before all adverse extremes.

## Hard limitations

This is **snapshot research only**. The rolling source is not historical-vintage
proof. BFRT cannot be promoted to shadow or live trading before 90 forward
shadow days pass frozen schema, freshness, and value-stability checks.

The branch has prior BTC outcome exposure, so even a complete pass is
candidate-level evidence rather than a pristine global holdout. No performance
claim exists at preregistration time.

# SDDR-12 outcome-blind preregistration — 2026-07-20

## Status and boundary

This document freezes one **Stablecoin Denominator Dislocation Reversion**
policy before calculating its real event incidence or reading any BTCUSDT
perpetual price, funding cash flow, future return, label, PnL, CAGR, or MDD.
Only the checksum-bound 2023 Spot cross-price source and, in the next work
unit, already frozen source-only SQFD timestamps may be opened.

The policy ID is `SDDR-12`. Its thresholds, direction, scheduler, support
gates, controls, and later outcome stopping rule cannot be repaired under this
name after any real support or return result is seen.

## Frozen observable and normalization

For completed UTC source hour `h`:

```text
r_usdc(h)  = log(BTCUSDC_close(h)  / BTCUSDT_close(h))
r_fdusd(h) = log(BTCFDUSD_close(h) / BTCUSDT_close(h))
```

For each ratio separately, calculate a rolling median and the 25th/75th
percentiles from at most the **720 strictly prior common hours**. The current
hour is excluded with `shift(1)`. At least 672 prior observations are required.

```text
scale(h) = (q75_prior(h) - q25_prior(h)) / 1.349
z(h)     = (r(h) - median_prior(h)) / scale(h)
```

A row fails closed when either scale is non-positive or non-finite. The
disagreement threshold is likewise the linear 80th percentile of the preceding
720 hours, with at least 672 observations and the current row excluded.

## Frozen primary clock

The primary state is active only when all conditions hold:

1. both robust z-scores are finite and have the same nonzero sign;
2. `min(abs(z_usdc), abs(z_fdusd)) >= 1.0`;
3. current `abs(r_usdc - r_fdusd)` is no greater than its strictly-prior 80th
   percentile.

Only a `false -> true` transition emits an event. The frozen direction is:

```text
side = sign((z_usdc + z_fdusd) / 2)
```

A positive relative ratio means the USDT quote is relatively rich in this
proxy and emits `LONG`; a negative ratio emits `SHORT`, seeking denominator
reversion. This interpretation is a hypothesis, not evidence of edge.

## Frozen execution clock

- Source hour: `[h, h+1h)`.
- Feature available: `h+1h`, after all three hourly Spot klines are final.
- Conservative entry: BTCUSDT USD-M perpetual open at `h+1h+5m`.
- Scheduled exit: open exactly 12 five-minute bars later, a one-hour hold.
- Reservation: candidates are considered chronologically and an entry is
  accepted only when it is at or after the previous accepted exit.
- Boundary: the exit must be no later than `2024-01-01 00:00 UTC` for initial
  support. Missing, late, duplicated, incomplete, or misaligned source rows
  fail closed.

## Frozen source-only controls

All controls keep the same false-to-true onset and scheduler:

- `no_disagreement`: removes only the two-book disagreement threshold;
- `usdc_only`: uses `abs(z_usdc) >= 1.0` and its sign;
- `fdusd_only`: uses `abs(z_fdusd) >= 1.0` and its sign;
- `stale_1h`: delays the complete primary active state and side by exactly one
  source hour while retaining the current decision scheduler.

No control may be selected as the policy after support is opened. They are
falsification diagnostics only.

## Frozen source-support gates

The primary clock must satisfy every gate on the initial 2023 source prefix:

- at least 30 accepted events;
- at least 5 accepted events in each full signal month from September through
  December 2023;
- LONG and SHORT each at least 30% of accepted events;
- no UTC entry month more than 45% of events;
- against each frozen SQFD comparator (`primary`, `no_usdt_lag`, and
  `no_participation`): exact-entry Jaccard at most 0.10 and bidirectional
  event containment within plus/minus six hours at most 0.35.

The comparator clock and support artifacts are SHA-256 bound in the generated
preregistration. Event counts and overlap values have not been calculated at
the time of this freeze. Failure of any gate rejects SDDR-12 before outcomes.

## Later outcome contract, not yet authorized

If and only if source support passes, a strict evaluator must be implemented,
tested, committed, and hash-frozen before opening outcomes. Stages then open
sequentially: train 2023, test 2024, evaluation 2025, final 2026H1. Stop at the
first failure; no threshold, side, hold, or feature repair is permitted.

The future evaluator must use 0.5 gross leverage, exact next-open execution,
realized funding, full-calendar CAGR, strict position-path MDD, and both 6 bp
base and 10 bp stress cost per notional side. The primary gates are base
`CAGR/MDD >= 3.0`, stress `CAGR/MDD >= 2.5`, strict MDD no greater than 15%,
mean gross move at least 20 bp, weekly clustered sign-flip `p <= 0.10`, positive
predeclared subperiods, and at least 0.25 ratio advantage over mandatory
direction, random-side, latency, single-book, coherence, and matched BTC
direction controls.

Gemma/RLLM remains outside the signal clock. It may be considered only after
the deterministic policy proves a gross edge above costs, and then only for
abstention or risk routing—not to repair timing or manufacture direction.

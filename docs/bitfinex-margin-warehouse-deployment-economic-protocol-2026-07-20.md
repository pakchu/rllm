# BFMWD-144 strict economic protocol — 2026-07-20

## Purpose

This document freezes the economic evaluation of the four BFMWD-144 variants
that passed the outcome-blind source-support and novelty gate.  No BTC price,
realized funding, return, or PnL row was inspected while choosing this protocol.

## One-way stage order

1. Freeze the evaluator, all control clocks, accounting rules, gates, and the
   parent execution-data identities without hashing or parsing execution-data
   bytes.
2. Materialize an exact physical **2021-01-01 through 2022-12-31** train slice.
   The transport may parse only the timestamp field while copying rows and must
   stop before the first 2023 row.  It computes no strategy outcome.
3. Evaluate all four source-supported variants on train.
4. Materialize or read the 2023 selection slice only if at least one variant
   passes every train gate.  Only train-passing variants advance, while the
   multiple-testing family remains the original four variants.
5. Any failed variant is retired without sign inversion, parameter repair,
   source extension, LLM/RL rescue, or replacement search inside this family.

No row at or after 2024-01-01 is in scope for this evaluator.

## Execution and accounting

- Instrument: Binance USD-M `BTCUSDT`, five-minute bars.
- Entry/exit: the already-frozen BFMWD clock, next five-minute open after source
  availability, followed by exactly 144 bars / 12 hours open-to-open.
- Exposure: fixed 0.5x equity notional; globally non-overlapping positions.
- Base friction: 6 bp per notional side.  Stress friction: 10 bp per side.
- Funding: exact realized Binance funding rate and the frozen settlement-mark
  mapping.  Interior events are symmetric; exact entry/exit credits are dropped
  while debits are retained.
- Absolute return and CAGR use the complete declared calendar, including idle
  cash.
- Strict MDD uses the global and pre-entry high-water mark, both execution
  costs, each funding settlement mark, and every held five-minute bar in
  favorable-then-adverse order.
- Stage boundaries are exclusive.  Every admitted exit must be strictly before
  the boundary; the one-bar-delay control is dropped if the shifted trade would
  violate that containment.

## Frozen candidate family and controls

The family order is the preregistered order:

1. `bfmwd_w12_d3_z10_h12`
2. `bfmwd_w24_d3_z10_h12`
3. `bfmwd_w12_d6_z10_h12`
4. `bfmwd_w24_d6_z10_h12`

For each primary schedule the evaluator derives these immutable diagnostics:

- `direction_flip`: same clocks, opposite side;
- `fUSD_only`: same primary clocks, fUSD / long rows only;
- `fBTC_only`: same primary clocks, fBTC / short rows only;
- `deterministic_random_side`: same clocks, side from the low bit of
  `SHA256("BFMWD-144|variant_id|entry_time")`;
- `extra_latency_one_bar`: entry and exit shifted five minutes, with strict
  stage containment;
- `ten_bp_per_side_stress`: primary clocks at 10 bp per side.

The source-only ablations are not new economic candidates and cannot replace a
failed primary variant.

## Stage gates

Train and selection apply the same gates independently:

- absolute return greater than zero;
- CAGR / strict MDD at least 3.0;
- strict MDD at most 15%;
- every contained calendar half-year has positive absolute return (four halves
  in train, two in selection);
- fUSD/long-only and fBTC/short-only full-stage contributions are each positive;
- mean gross side-adjusted underlying move at least 30 bp;
- 10 bp-per-side stress absolute return positive and stress CAGR/MDD at least
  2.5;
- one-extra-five-minute-latency absolute return positive;
- one-sided Romano–Wolf adjusted p-value at most 0.10.

## Multiple-testing inference

For each variant, compounded trade log returns are assigned to the UTC calendar
day of realized exit; all idle days are explicit zeros.  The vector therefore
has exactly 730 observations for train and 365 observations for selection, and
its sum equals the log ending equity from strict simulation.

The evaluator performs a synchronized one-sided Romano–Wolf step-down max-t
circular block bootstrap across the original four-variant family:

- block length: 7 calendar days;
- draws: 100,000;
- seed: 20,260,720;
- centered null series;
- identical sampled day indices across variants;
- equal observed t-statistics removed as one step-down group;
- zero-variance or non-advanced variants fail closed at adjusted p = 1.

The adjusted p-value, not the unadjusted weekly sign-flip diagnostic, controls
the statistical gate.

## Physical source seal

The parent containers are the already frozen Binance 2020–2023 BTCUSDT 5m and
realized-funding-mark artifacts.  Their official manifest hashes and declared
data hashes are bound during evaluator freeze, but execution-data bytes are not
opened or hashed during that freeze.  Train preparation deliberately does **not**
hash the full parent containers because those compressed bytes also contain the
sealed 2023 selection interval.  It trusts the already hash-bound official
manifest identity, copies exactly the frozen train row count without reading the
first excluded row, and records hashes for the resulting train-only slices.  The
full parent hashes are verified only after train passes and selection becomes
eligible; any mismatch then invalidates the chain fail-closed.

Each stage artifact is write-once, contains only its exact physical interval,
records its own SHA-256, and is tied to the frozen evaluator manifest. Existing
canonical slice or manifest paths are never deleted or overwritten.  A partial
prior preparation therefore blocks retry and preserves the evidence for audit.

The 2023 source cannot be prepared before a passing 2021–2022 report exists.
The evaluator has no command that can prepare or read 2024+ data.

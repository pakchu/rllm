# BFWC-288 outcome-blind write-once preregistration

Date frozen: 2026-07-30  
Policy: `BFWC-288`  
Status: singleton preregistration; source values previously seen; exact candidate
incidence, overlap, and economic outcomes unopened

## Purpose and evidence boundary

This document freezes one relational Bitcoin blockspace rule before its exact
incidence, comparator overlap, or market outcome is computed. There is one
feature definition, one threshold, one direction rule, and one 24-hour hold.
There is no threshold, sign, horizon, latency, or leverage grid.

The normalized BFRT fee values and WCTR size/weight values, and the individual
BFRT-288 and WCTR-288 incidences, had already been inspected in earlier work.
The broad BCRT relational family had also been seen. This is therefore not a
claim that the source concepts are pristine. The narrower claim is that the
exact BFWC primitives, exact joined incidence, exact comparator overlaps, and
all BFWC returns, funding, PnL, CAGR, and drawdowns remain unopened at freeze.

The executable preregistration may:

- hash frozen repository artifacts byte-for-byte;
- parse frozen JSON source manifests only to validate their internal hashes;
- decompress and decode exactly the first CSV record as a header; and
- validate exact repository-relative paths and dependency hashes.

It must not decode a CSV data row, derive source incidence, join source rows,
load market or funding rows, or compute overlap or economic statistics.

## Frozen source join and feature

The evaluator must exact-inner-join the frozen normalized BFRT and WCTR 12-hour
rows on both `bucket_start_utc` and `bucket_end_utc`. Duplicate keys, conflicting
availability fields within a source, or a many-to-many join fail. The common
domain is the inclusive key interval from the later first key through the
earlier last key, intersected with the strict research horizon. An exact outer
join over that common domain must have zero left-only and zero right-only rows.
There is no nearest-time join, tolerance join, interpolation, forward fill,
backfill, or row repair.

For a joined bucket indexed by `t`, the rows `t`, `t-1`, and `t-2` must be
strictly consecutive 12-hour buckets. Define, for
`p in {10,25,75,90}`:

```text
x[p,t] = log1p(fee_p[t])
Delta2 z[t] = z[t] - z[t-2]
R[t] = 0.5 * ((Delta2 x[10,t] + Delta2 x[25,t])
              - (Delta2 x[75,t] + Delta2 x[90,t]))
witness_share[t] = (4*avg_size[t] - avg_weight[t]) / (3*avg_size[t])
W[t] = Delta2 witness_share[t]
fullness[t] = avg_weight[t] / 4_000_000
U[t] = Delta2 fullness[t]
```

Arithmetic is IEEE-754 binary64 with no rounding before comparisons. Fee
percentiles must be finite and nonnegative; `avg_size` must be finite and
strictly positive; `avg_weight` must be finite and in `[0, 4_000_000]`.
`witness_share` must be finite and in `[0,1]`. Invalid primitives invalidate
the joined base row; there is no clipping or imputation.

A base-valid row has an exact one-to-one join, the required consecutive
`t,t-1,t-2` rows, and finite defined `R`, `W`, and `U`. Its rank history is the
latest at most 180 strictly prior base-valid joined rows in chronological key
order. The current row is excluded. At least 120 prior rows are required.
For `n` prior rows, with `L` prior `abs(R)` values strictly below current
`abs(R)` and `E` exactly equal under binary64 equality, the empirical midrank is
`(L + 0.5*E)/n`. Exact ties are not broken by time or randomness.

The only primary eligibility rule is:

```text
midrank(abs(R[t])) >= 0.75
R[t] != 0, W[t] != 0, U[t] != 0 exactly
sign(R[t]) = sign(W[t]) = sign(U[t])
```

The side is `LONG` for positive `R` and `SHORT` for negative `R`.

Economically, `R>0` means the lower fee percentiles have risen relative to the
upper fee percentiles over two buckets: fee pressure has flattened or broadened
rather than remaining concentrated only in the upper tail. `W>0` means witness
share rose, and `U>0` means average block fullness rose. Their positive
concordance therefore maps to `LONG`; the exact negative mirror maps to
`SHORT`. This interpretation fixes the polarity and is not a causal claim.

## Availability, execution, and reservation

Joint availability is the later of the exact BFRT and WCTR `available_at_utc`
values. `ceil_5m(a)` is the smallest Unix-epoch multiple of 300 seconds not
earlier than `a`. Entry is always `ceil_5m(joint_availability) + 5 minutes`;
an already aligned availability still waits one complete bar.

Entry and exit use the Binance USD-M BTCUSDT perpetual 5-minute open. The hold
is exactly 288 complete 5-minute bars, or 24 elapsed hours:
`exit_time = entry_time + 86_400 seconds`. Position leverage is `0.5x`
pre-entry equity and quantity is fixed through exit. Base execution cost is
6 bp per notional per side; stress cost is 10 bp per notional per side.
Realized funding is included exactly when
`entry_time <= funding_time < exit_time`.

Raw eligible candidates are sorted by `(entry_time, bucket_start_utc,
canonical_signal_id)`. An interval must be wholly contained in one frozen split;
an exit equal to the split's exclusive end is contained. Crossing intervals are
reported and rejected. The primary then applies one global chronological
reservation over `[entry_time, exit_time)`: accept only when entry is not before
the previous accepted exit. Suppressed candidates are never queued, shifted, or
reconsidered. Feature and rank state continue independently of suppression.

## Calendar and support gates

The report horizon is the strict UTC wall-clock interval
`[2023-06-01T00:00:00Z, 2026-06-01T00:00:00Z)`, exactly three anniversary
years. Idle warmup inside the horizon is part of the calendar return.

- Selection: `[2023-06-01, 2025-01-01)`.
- Future 2025: `[2025-01-01, 2026-01-01)`.
- Future 2026: `[2026-01-01, 2026-06-01)`.

Window attribution is by entry and every accepted interval must be
split-contained. Support is evaluated before novelty or outcomes.

- Selection: at least 45 accepted primary trades; at least 6 in
  2023 November-December; at least 12 in 2024 H1; at least 12 in 2024 H2;
  at least 14 per side; maximum calendar-month share at most 0.20.
- Future 2025: at least 30; at least 10 in each half; at least 10 per side;
  maximum calendar-month share at most 0.25.
- Future 2026: at least 15; at least 6 in Q1; at least 4 in April-May; at
  least 5 per side; maximum calendar-month share at most 0.30.
- Exact source-join gaps in the common source domain: zero.
- Future append invariance: rebuilding any completed prefix with only source
  rows available through that prefix must reproduce byte-identical base-valid
  status, rank numerators/denominators, raw candidates, accepted signals, sides,
  and clocks for that prefix.

Every denominator must be positive and every threshold comparison is inclusive
as written. Any support failure retires the exact rule unchanged.

## Frozen controls

Component controls build their own raw candidates and their own chronological
non-overlap clocks, with the same split containment, availability, entry, hold,
and accounting:

- `fee_rotation_only`: retain the primary `abs(R)` midrank threshold and exact
  `R != 0`; drop `W` and `U`; side is `sign(R)`.
- `witness_fullness_only`: define
  `Q=0.5*(abs(W)+abs(U))`, use the same 180/120 strict-prior midrank rule on
  `Q`, require rank at least 0.75 and exact nonzero equal signs for `W,U`;
  side is their common sign. It does not use `R`.
- `drop_witness`: retain the primary `abs(R)` rank, require exact nonzero
  equal signs for `R,U`, and drop `W`; side is `sign(R)`.
- `drop_fullness`: retain the primary `abs(R)` rank, require exact nonzero
  equal signs for `R,W`, and drop `U`; side is `sign(R)`.
- `one_bucket_stale_witness_fullness`: retain current `R` and its current rank,
  but substitute the fully formed `W,U` from joined bucket `t-1`; require
  current `R` and stale `W,U` exact nonzero equal signs. It retains current
  joint availability and builds its own reservation clock.

Same-parent controls retain the accepted primary signal set unless the stated
clock shift makes an interval cross a split:

- `exact_direction_flip`: multiply every primary side by `-1`.
- `deterministic_random_side`: SHA256 of
  `BFWC-288|<canonical_signal_id>|RANDOM_SIDE`; first digest byte below 128 is
  `LONG`, otherwise `SHORT`.
- `constant_long` and `constant_short`: fixed side on the primary clock.
- `one_bar_delayed_entry`: shift both entry and exit exactly one 5-minute bar;
  reject a shifted interval that is no longer split-contained; do not rerun
  reservation or admit a suppressed parent.

`fee_rotation_only` and `witness_fullness_only` are component-only controls.
If either passes the complete economic gate in selection and every future
period, joint-specificity is rejected. A control can never replace the primary.

## Novelty before candidate outcomes

After support passes, but before any BFWC market/funding outcome is joined, the
candidate is compared separately with the frozen BFRT-288 and WCTR-288 primary
clock artifacts over the strict three-year horizon:

- exact-entry Jaccard at most 0.20;
- fraction of candidate entries having any comparator entry within plus or
  minus 6 elapsed hours at most 0.50; and
- absolute Pearson correlation of signed occupied 5-minute exposure at most
  0.40.

Jaccard is intersection over union of distinct exact UTC entry timestamps.
Containment uses the candidate count as denominator and does not consume
matches. Signed occupied exposure is `+1` long, `-1` short, and `0` idle on
every complete 5-minute bar, with intervals `[entry,exit)`. Duplicate entries,
overlapping intervals within a clock, off-grid times, empty denominators, and
undefined/nonfinite correlation fail.

After those checks pass, a separately committed evaluator must deterministically
rebuild the five authoritative Gross9 sleeve clocks from the frozen config,
anchor, market/funding/premium artifacts, and bound builders. Before BFWC
outcomes, compare the candidate separately with every sleeve:

- exact-entry Jaccard at most 0.10;
- candidate-entry containment within plus or minus 6 hours at most 0.35;
- occupied-bar Jaccard at most 0.25; and
- absolute signed-exposure Pearson correlation at most 0.35.

All five sleeves must pass every metric. The authoritative weights are
`cand_rex_veto_7=1.6`, `fresh_kimchi_fx=2.0`,
`frozen_annual_rank7=3.0`, `markov_transition_long=2.0`, and
`rex_taker_low_range_position=0.4`, gross exactly 9.0. No comparator threshold
may be repaired after opening BFWC outcomes.

## Economic gates and portfolio marginal

Only after support, both novelty stages, and a separately committed evaluator
may BFWC outcomes be opened. Selection is evaluated first. Future 2025 and then
future 2026 are veto-only and cannot alter the rule or ranking.

For the primary, `fee_rotation_only`, `witness_fullness_only`, and the one-bar
delay, report base and stress accounting. For each required period the complete
primary economic gate is:

- standalone base and stress absolute return strictly positive;
- base and stress full-calendar CAGR divided by strict MDD at least 3.0;
- base and stress strict MDD at most 0.15;
- mean signed gross underlying move at least 20 bp;
- deterministic weekly-cluster one-sided sign-flip p-value at most 0.10;
- independently compounded long-only-trade and short-only-trade subaccounts
  each have strictly positive base and stress absolute return; and
- the `one_bar_delayed_entry` base and stress absolute returns are strictly
  positive.

The gross move is
`mean(side*(exit_open/entry_open-1)*10_000)` before costs and funding.
Weekly clusters are UTC ISO entry weeks of base-cost net trade PnL. Use 20,000
deterministic sign-flip draws indexed `00000` through `19999`; for each week
and draw, flip when the most significant bit of
`SHA256("BFWC-288|<PERIOD>|<DRAW>|<ISO_YEAR>-W<ISO_WEEK>")` is one.
The one-sided p-value is
`(1 + count(flipped_total >= observed_total)) / 20001`.

Strict MDD is measured on the full marked equity path from the global
pre-entry high-water mark: entry fee; funding credits before the favorable
held-bar extreme and funding debits before the adverse extreme; adverse
intrabar extreme before favorable; adverse virtual exit fee while held; and
scheduled exit fee. Zero MDD with positive CAGR produces positive infinity and
passes the ratio gate; nonpositive equity or any other nonfinite quantity fails.
Full-calendar CAGR uses the exact wall-clock duration of the period; the
stitched horizon uses exactly three anniversary years.

If the selection primary gate passes, evaluate candidate sleeve weights
`[0.25,0.50,0.75,1.00]`. At weight `w`, multiply every Gross9 baseline sleeve
weight by `(9-w)/9` and apply `w` to the canonical 0.5x BFWC path, so configured
gross remains exactly 9. Compare the combined portfolio with the unscaled
authoritative Gross9 baseline under matching base and stress accounting.
Selection requires:

- base CAGR/MDD improvement of at least 0.05 and stress CAGR/MDD improvement
  of at least 0.05, each in absolute ratio units;
- base and stress strict MDD no worse than baseline;
- base and stress absolute return strictly positive; and
- base and stress absolute return retention at least 0.97 of baseline.

Among passing selection cells only, rank by larger minimum base/stress ratio
improvement, then lower maximum base/stress MDD, then larger minimum return
retention, then lower candidate weight. Freeze top 1 only. No passing cell
retires the marginal candidate. The frozen top 1 is then evaluated unchanged
in future 2025 and future 2026; each future period must pass the complete
standalone gate and all same-gross requirements. Future cannot rerank, select
rank 2, repair, or change weights.

A stitched full-horizon report over the exact three-year calendar is mandatory
for the standalone primary, controls, authoritative Gross9 baseline, and frozen
combined top 1, under base and stress accounting. It is descriptive and cannot
repair a failed selection or future gate.

## Sequence and retirement

The immutable sequence is:

1. Commit this document and executable write-once preregistration.
2. Generate and commit the canonical JSON preregistration.
3. In a later separate commit, add the evaluator bound to the preregistration
   file SHA256 and internal `manifest_hash`.
4. Validate frozen dependencies and exact CSV headers without decoding rows.
5. Open source rows only for exact support and append-invariance checks.
6. If support passes, run BFRT/WCTR novelty.
7. If that passes, rebuild Gross9 clocks and run per-sleeve novelty.
8. If that passes, open selection outcomes and freeze top 1.
9. If selection passes, open future 2025, then future 2026.
10. Produce the stitched report.

The first failure retires the exact BFWC-288 rule. No polarity flip, threshold
change, alternative horizon, fallback control, lower-ranked portfolio cell, or
post-outcome repair is permitted.

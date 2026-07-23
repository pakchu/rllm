# CVICR-72 source-support implementation contract

## Scope and stop rule

This contract binds the first executable evaluator for
**CVICR-72 — Cross-Venue Intrinsic-Clock Resolution Relay**.

The evaluator may decode only the eight preregistered source columns. It may
construct causal event clocks, source-support diagnostics, and predecessor
clock overlap. It may not load OHLC, funding, returns, PnL, labels, rewards, or
any 2024-or-later source.

Execution is fail-closed:

1. validate the committed preregistration, source, manifest, audit, mechanism,
   boundary, evaluator source, tests, and this contract;
2. construct source-only clocks and evaluate every support/selectivity gate;
3. stop without decoding comparator rows if source support fails;
4. only a complete source-support pass may decode the frozen comparator clock
   columns and evaluate novelty;
5. any failure retires CVICR-72 unchanged before market outcomes.

The evaluator source, tests, and this contract must be committed and unchanged
from `HEAD` before the real source can be decoded.

## Exact source decoding

`pandas.read_csv` must receive the preregistered eight-column `usecols`
allowlist. Load-and-drop is forbidden.

The loader requires:

- the exact 420,768-row UTC five-minute grid on
  `[2020-01-01, 2024-01-01)`;
- unique monotonic timestamps;
- both availability timestamps exactly equal to bar start plus five minutes;
- a strict boolean `source_complete`;
- finite positive Spot and USD-M quote notional and finite signed quote
  notional bounded in absolute value by quote notional for an accepted row.

The grid itself must be complete. Invalid source rows remain present and are
marked unusable; they may not be filled, skipped, or repaired.

A historical complete day has exactly 288 usable rows. Its venue totals alone
may enter the exact preceding 28-calendar-day expected-volume windows.
Current-day event validity uses only the prefix through the later anchor plus
the one complete computation-buffer row. A defect after that prefix cannot
erase an event.

## Deterministic daily state

For each venue and UTC day:

1. take the float64 linear median of complete-day quote totals in the exact
   preceding 28 calendar positions when at least 21 are present;
2. multiply by 0.50;
3. find the first current-day completed bar whose cumulative quote notional
   reaches that target;
4. reject an anchor later than 17:50 UTC.

For a non-tied pair with a valid causal prefix, append its positive five-minute
anchor gap to the paired-gap history only after computing the current day.
The gap threshold is the float64 linear q60 of the last at most 180 prior
valid pairs and requires at least 90. Event flow signs use cumulative signed
quote notional divided by cumulative quote notional with no epsilon or
deadband.

The fixed-expected-time control uses each venue's lower median anchor index
from valid anchor observations in the exact preceding 28 calendar positions,
requires at least 21 per venue, and excludes the current day. It reuses the
current primary pair's strictly prior q60 threshold and current-day flows.

The stale-laggard control requires the immediately preceding UTC day to be a
complete day and reads that prior day's laggard cumulative flows at the
current pair's two anchor indices. It retains current leader flows and cannot
emit before the current primary decision time.

## Clock representation and reservation

The only emitted clock columns are:

```text
control
signal_id
source_day
causal_origin_time
resolution_time
signal_available_time
decision_time
entry_time
exit_time
side
leader
```

`side` is `LONG` or `SHORT`; `leader` is `spot` or `um`. `signal_id` is a
canonical SHA-256 identity over the policy, control, source day, causal and
execution timestamps, side, leader, and frozen source hash.

Each independent control is sorted by entry and greedily reserves one global
non-overlapping clock across the entire source before any split is inspected.
An entry equal to the previous exit is accepted. Same-clock direction and
random-side controls reuse accepted primary timestamps. Delayed controls shift
entry and exit by exactly one or twelve bars and then repeat global
reservation. Split statistics require entry and exit containment in the same
half-open split.

## Support diagnostics

The evaluator reports raw incidence, globally accepted incidence, split
statistics, side and leader shares, active months, month and quarter
concentration, maximum entry gap, same-side run, and same-leader run.

Mechanism count ratios and exact-entry Jaccards use each control's own globally
reserved, split-contained accepted clock. A zero denominator, empty required
clock, missing statistic, or non-finite value is a failed gate.

## Comparator decoding and novelty

Comparator files are hash- and exact-header-bound before row decoding.
Decoding uses only group, entry, exit or hold, and side columns. Selected
groups are extracted separately; unknown side encodings, duplicate entries,
invalid intervals, overlapping positions inside one selected group, or an
empty extraction in declared common coverage fail closed.

For each selected comparator group, intersect its declared coverage with the
CVICR source coverage. CVTT remains train-only. Compare the split-contained
accepted primary clock against the comparator using:

- exact entry-time Jaccard;
- one-bar, twelve-bar, and six-hour maximum-cardinality one-to-one tolerant
  Jaccard;
- absolute Pearson correlation of signed occupied exposure on the complete
  common five-minute grid; and
- report-only position-time Jaccard.

The two-pointer tolerant match and all thresholds are exactly those in the
committed preregistration. Undefined correlation or non-finite metrics fail.

## Deterministic artifacts

The clock is canonical LF CSV inside deterministic gzip (`mtime=0`, empty
filename). The JSON report is canonical sorted JSON with no wall-clock
timestamp. Both are write-once: an existing byte-identical artifact is
accepted, while drift is rejected. The report records hashes, decoded-row
counts, all checks, the first failing stage, and explicit zero counts for
outcome, funding, future-return, and network access.

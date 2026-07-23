# CXRT-288 source-support implementation contract

## Scope and sequencing

This contract binds the outcome-blind implementation of
**CXRT-288 — CBOE Cross-Surface Risk-Transfer Relay**.

The evaluator may decode only the three preregistered CBOE column allowlists.
It may construct strict-prior surface states, deterministic control clocks,
support/composition diagnostics, and frozen predecessor-clock novelty. It may
not load BTC OHLC, funding, returns, labels, rewards, PnL, portfolio results, or
any 2024-or-later source.

The evaluator source, tests, and this contract must be committed and unchanged
from `HEAD` before a real source row is decoded.

Execution order is fail-closed:

1. validate all preregistration, document, source, manifest, comparator, source
   header, evaluator, test, and contract hashes;
2. decode exact source allowlists and build source-only clocks;
3. evaluate source support and relational composition;
4. stop without comparator-row decoding on any source/support/composition
   failure;
5. only a complete pass may decode comparator clock columns and evaluate
   novelty;
6. any failure retires CXRT-288 before BTC outcomes.

## Source validation and feature construction

Each `pandas.read_csv` call receives its exact preregistered `usecols`.
Load-and-drop is forbidden.

Within each panel:

- `observation_date` is a unique, strictly increasing date before 2024;
- every retained numeric primitive is finite and strictly positive;
- a malformed primitive fails the full source before feature construction.

Each surface computes strict-prior midranks on its own source calendar before
the exact date intersection. The lookback is at most 252 preceding source
observations and requires at least 126. The current value is appended only
after all current ranks are fixed.

The term and tail `VIX_close` values must match exactly on common dates. Missing
dates are never carried, filled, interpolated, or synthesized.

The formulas, vote center, buckets, majority eligibility, and fixed direction
are exactly the committed preregistration. No pressure threshold or fitted
weight is introduced by the implementation.

`vote_relation` is `UNANIMOUS` when all three nonzero votes equal the
majority, `NEUTRAL_SUPPORTED` when the eligible majority contains a neutral
surface, and `SPLIT_MAJORITY` otherwise. `minority_surface` names the one
surface whose vote differs when the other two equal the nonzero majority;
otherwise it is `NONE`.

Surface and prior-majority transitions compare adjacent rank-complete common
dates. Their vocabulary is `NO_PRIOR`, `PERSIST`, `FLIP`, `TO_NEUTRAL`,
`FROM_NEUTRAL`, and `NEUTRAL_PERSIST`. An ineligible majority has neutral
value zero for this relation only. Calendar-gap buckets compare adjacent
rank-complete common source dates and are `NO_PRIOR`, `1D`, `2_3D`, and
`4P_D`.

## Timing and controls

For rank-complete common source date `D`, the state may schedule only at 09:35
America/New_York on the first later exact rank-complete common source date.
Signal availability is five minutes earlier. Exit is exactly 288 five-minute
bars after entry.

Raw clocks are built across the complete pre-2024 panel, globally
non-overlap-reserved, then split-contained. Entry equal to a previous exit is
accepted.

The emitted clock contains source-only relations:

```text
control
signal_id
source_date
signal_available_time
entry_time
exit_time
side
term_vote
tail_vote
option_vote
vote_relation
minority_surface
term_bucket
tail_bucket
option_bucket
term_transition
tail_transition
option_transition
prior_majority_transition
calendar_gap_bucket
```

No raw value, rank, price, return, label, or outcome is written.

Single-surface and pair controls build their own clocks and reserve
independently. `one_common_date_stale` is a subset of current primary timestamps
whose replacement previous-common-date majority is defined. It emits the
previous common date's votes, buckets, and transition relations on the current
primary clock while retaining the current primary `source_date`. Direction-flip
and random-side controls reuse accepted primary timestamps and current relation
tokens. The delayed control retains the original signal-availability timestamp,
shifts raw primary entry and exit by exactly 288 bars, and repeats reservation.

## Support and composition

All count, calendar, concentration, side-run, vote-share, minority-share,
unanimity, and reproduction gates are computed from globally accepted,
split-contained clocks.

For a split:

- surface RELIEF/STRESS shares use primary accepted dates as denominator;
- non-unanimous means not all three votes equal the same nonzero vote;
- a surface is the unique minority only when the other two votes equal the
  nonzero primary majority and its own vote differs;
- unique-minority shares use all primary non-unanimous dates as denominator;
- same-side reproduction is same-entry same-side matches divided by primary
  accepted entries.

An empty denominator, missing required control, or non-finite statistic fails.

## Comparator novelty

Comparator files are hash- and exact-header-validated before row decoding.
Decoding uses only control, entry, exit, and side columns. Selected comparator
groups are extracted separately and restricted to declared common coverage.

An empty extraction, invalid side, duplicate entry, invalid interval, or
self-overlap fails closed.

For every group:

- exact entry-time Jaccard is gated;
- same-entry same-side reproduction, divided by candidate primary count, is
  gated;
- absolute Pearson correlation of signed occupied exposure on the complete
  common five-minute grid is gated; and
- one-calendar-day maximum-cardinality one-to-one tolerant Jaccard is
  report-only.

Undefined correlation fails.

## Deterministic artifacts

The clock is canonical LF CSV inside deterministic gzip (`mtime=0`, empty
filename). The report is canonical sorted JSON with no wall-clock timestamp.
Both are write-once: byte-identical existing artifacts are accepted and drift
is rejected.

The report records source and implementation hashes, decoded-row counts,
feature funnel, every gate, the first failing stage/check, clock hashes, and
explicit zero counts for BTC market, funding, future-return, PnL, and network
access.

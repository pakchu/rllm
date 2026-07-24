# VARR-6 rejection — pre-support comparator parse breach

## Immutable decision

Retire the exact **VARR-6 — Venue Access Recovery Reversal** candidate before
mechanism commit, synthetic adaptation, historical incident incidence, or any
market outcome.

The candidate boundary in commit `8ff9adb` made this an absolute stage gate:

```text
comparator timestamp rows parsed = 0
until the unchanged source/semantic and causal-market support gates pass
```

That invariant was violated during mechanism drafting. Removing or replacing
the observed comparator afterward would be post-breach cohort repair, so the
candidate cannot continue.

## Detection

- breach date: `2026-07-24`
- detected and stopped at: `2026-07-24T10:48:59+09:00`
- worktree:
  `/tmp/rllm-alpha-orthogonal-20260718`
- branch: `codex/alpha-orthogonal-20260718`
- last committed candidate boundary: `8ff9adb`
- mechanism status: uncommitted draft, deleted after this report

## Exact prohibited access

A local diagnostic intended to find a pre-2024 premium comparator used
Python `gzip` and `csv.reader` to iterate every data row in:

1. `data/premium_compression_breakout_relay_clocks_2020_2026.csv.gz`
2. `data/premium_snapback_recenter_clocks_2020_2026.csv.gz`

The diagnostic:

- decoded both gzip streams;
- parsed both CSV headers;
- iterated 572 rows from the first file;
- iterated 1,147 rows from the second file;
- retained the first and last row of each file;
- printed each header, row count, first row, and last row; and
- therefore exposed pre- and post-2023 comparator timestamps before VARR
  causal-market support.

Total prohibited comparator rows parsed:

```text
1,719
```

Printed fields for the premium-compression clock:

```text
candidate
control
split
context_start_time
decision_time
feature_available_time
entry_time
exit_time
side
context_range
trigger_move
trigger_efficiency
terminal_location
outside_distance
```

Printed fields for the premium-snapback clock:

```text
candidate
split
path_start_time
decision_time
feature_available_time
entry_time
planned_exit_time
direction
prior_center
path_range
efficiency
turns
up_excursion
down_excursion
max_excursion
terminal_deviation
```

The output included the complete first and last row values. Those values are
not reproduced here and may not inform a successor candidate's comparator
cohort, support floor, clock, side, threshold, or hold.

## What remained unopened

At retirement:

- VARR 2020–2023 incident IDs enumerated: `0`
- VARR 2020–2023 update bodies decoded: `0`
- historical incident model calls: `0`
- VARR accepted source events: `0`
- VARR BTC market rows read: `0`
- VARR displacement values computed: `0`
- VARR sides or entries emitted: `0`
- funding rows read: `0`
- post-entry/future market rows read: `0`
- comparator return or PnL fields read: `0`
- VARR absolute return, CAGR, MDD, hit rate, or reward computed: `0`
- VARR 2024-or-later incident bodies or outcomes opened: `0`

The two prohibited files are clock/feature artifacts and contain no return or
PnL column. This limits the breach scope but does not cure the hard stage-gate
violation.

## Non-results

VARR-6 produced no semantic pass, support result, novelty result, backtest,
alpha statistic, adapter, checkpoint, or live artifact. It must not appear in
an alpha portfolio or candidate ranking.

The useful source feasibility facts remain research-process knowledge only:

- both official Statuspage archives reach well before the intended period;
- history rows require deterministic incident-versus-maintenance endpoint
  type resolution;
- update-prefix replay must avoid global incident metadata that can
  retroactively encode future reopens; and
- post-2023 update objects must be sealed rather than used to reject an
  earlier prefix.

These structural lessons may be used in a genuinely new source mechanism.
The observed premium-clock rows, coverage endpoints, counts, and values may
not.

## Required successor boundary

A successor must:

1. use a new candidate name and causal mechanism;
2. freeze its source, split, controls, and comparator cohort before parsing
   any cohort row;
3. bind comparator files by raw-byte hash only until causal support passes;
4. use a tested prefix loader that proves zero sealed-row materialization;
5. avoid the two observed premium-clock artifacts entirely; and
6. repeat the immutable retirement rule on any future stage breach.

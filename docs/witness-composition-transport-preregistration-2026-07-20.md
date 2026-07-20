# WCTR-288 singleton preregistration — 2026-07-20

## Status

This document freezes exactly one **Witness Composition Transport** policy
before any derived witness-share feature, complete signal incidence, BTC bar,
funding value, return, PnL, CAGR, or MDD is opened. The source loader necessarily
parsed size/weight values to enforce its frozen source contract; this
preregistration writer reads only manifest metadata and artifact byte hashes.
The policy is **WCTR-288**: a 24-hour, non-overlapping BTC long/short rule based
on unusually large seven-day changes in the witness share of mined blocks,
confirmed over 24 hours while blockspace remains utilized.

The source is bound to manifest
`55914b3ec31fe8fb66d8a8dc31acb3784a10b256625073a5aeff1d317660ea8d`
and normalized artifact
`ee761e813085dfdee675ca9d420516f814c4c2824f3f5cef604acc3871d46c61`.
This is one falsifiable singleton. There is no sign, lag, threshold, rank
window, hold, leverage, latency, or split grid.

## Feature definition

For each retained 12-hour source bucket `t`, require `t` through `t-14` to be
an exact contiguous 12-hour sequence. Define:

```text
witness_share[t] = (4 * avg_size[t] - avg_weight[t]) / (3 * avg_size[t])
fullness[t]      = avg_weight[t] / 4_000_000
transport_7d[t]  = witness_share[t] - witness_share[t-14]
impulse_24h[t]   = witness_share[t] - witness_share[t-2]
```

The base feature row is valid when every required source value is finite and
positive, every required witness share is finite and in `[0,1]`, and current
fullness is finite and in `[0,1]`. The primary event then additionally requires
both changes to be nonzero and
`sign(transport_7d) == sign(impulse_24h)`. The source loader already
enforces the explicit four-byte tolerance needed for integer-rounded averages;
the feature calculation itself applies no tolerance, epsilon, clipping,
rounding, interpolation, forward fill, or imputation. All arithmetic is
IEEE-754 binary64.

For component controls only, also define seven-day and 24-hour changes in
`log(avg_size)` and `log(avg_weight)`. These fields never enter the primary
WCTR side or eligibility rule.

## Strict-prior normalization and singleton rule

At each base-valid row, calculate rolling empirical midranks over the latest
180 **strict-prior base-valid feature rows**, requiring at least 120. When more
than 180 exist, use exactly the most recent 180. The current row is excluded
and every prior row must have `available_at` strictly before the current row.

```text
midrank = (count(prior < current) + 0.5 * count(prior == current)) / prior_count
```

The primary ranks are:

- `magnitude_rank`: midrank of `abs(transport_7d)`; and
- `fullness_rank`: midrank of `fullness`.

The only primary eligibility rule is:

```text
magnitude_rank >= 0.75
and fullness_rank >= 0.50
and sign(transport_7d) == sign(impulse_24h)
```

The side is `sign(transport_7d)`: positive is long and negative is short. No
learned coefficient, score priority, threshold search, or post-incidence repair
exists.

## Causal clock and non-overlap

- Source availability is fixed bucket end plus 48 hours.
- `ceil_5m(x) = ((unix_seconds(x) + 299) // 300) * 300`.
- `entry_time = ceil_5m(source_available_at) + 5 minutes`; even an already
  aligned availability timestamp receives this additional full bar.
- Enter at the open after that complete latency bar.
- `scheduled_exit_time = entry_time + 24 hours`, or 288 five-minute bars.
- Sort candidates by `(entry_time, bucket_start)`.
- Accept a candidate only when its entry is at or after the previous accepted
  exit; suppress every intervening signal without replacement.
- A trade is `[entry_time, scheduled_exit_time)` and both endpoints must be
  contained in one declared split.

Execution uses 0.5 notional leverage, 6 bp per notional per side at base cost,
10 bp at stress cost, and exact entry-inclusive/exit-exclusive funding with a
fixed entry quantity.

## Frozen calendar

All assignments use entry time and the full hold must fit inside the window.

- warm-up source only: `2022-07-20T12:00:00Z` through 2022-10-31;
- train: `[2022-11-01T00:00:00Z, 2024-01-01T00:00:00Z)`;
- test: calendar 2024;
- eval: calendar 2025; and
- forward: `[2026-01-01T00:00:00Z, 2026-07-20T00:00:00Z)`.

No period selects, reranks, inverts, refits, or repairs a later period. The
stitched full horizon is the fixed wall-clock interval from 2022-11-01 through
2026-07-20 and is opened only after all four constituent windows pass.

## Outcome-blind support gate

Count only accepted primary non-overlap entries. Before any market or funding
value is loaded, all of the following must pass:

| Window | Total | Long | Short | Dispersion |
| --- | ---: | ---: | ---: | --- |
| Train | >=45 | >=14 | >=14 | 2022 Nov-Dec >=5; 2023H1 >=16; 2023H2 >=16; max month share <=20% |
| Test | >=35 | >=10 | >=10 | each half >=14; each quarter >=5; max month share <=20% |
| Eval | >=35 | >=10 | >=10 | each half >=14; each quarter >=5; max month share <=20% |
| Forward | >=18 | >=5 | >=5 | 2026H1 >=16; max month share <=28% |

Month share is the largest calendar-month count divided by total window count;
it is not a side-specific cap. The frozen source must also have zero missing
12-hour buckets. July rows whose conservative availability or entry is outside
the forward window are omitted rather than backfilled. Any failure
rejects WCTR-288 without outcomes. Threshold, side, rank window, latency, hold,
calendar, or support-floor repair is forbidden.

## Controls

The evaluator must report these already-frozen controls:

- direction flip on the same primary clock, diagnostic only;
- transport-only: retain the nonzero seven-day `>=0.75` magnitude threshold but
  remove 24-hour confirmation and the fullness floor, using its own clock;
- impulse-only: use an independently ranked 24-hour witness-share magnitude,
  the same 0.75 magnitude and 0.50 fullness floors, and the impulse sign;
- low-fullness complement: retain transport magnitude and sign confirmation
  but require `fullness_rank < 0.50`, using its own clock;
- serialized-size-only and block-weight-only controls, each using its own
  nonzero seven-day magnitude rank, nonzero 24-hour sign confirmation, the same
  frozen floors, and its own clock;
- constant long and constant short on the primary clock;
- the fully formed primary feature and ranks delayed by 14 source buckets,
  while retaining the current source availability and execution clock;
- deterministic month-and-side-stratified random rank-ready clocks, seed
  `20260720`; and
- one complete five-minute bar delayed entry and exit.

Control ranks use the same strict-prior base-valid rows. Each independent-clock
control applies the same split containment and chronological non-overlap. The
random control first creates a feature-agnostic, rank-ready, split-contained
non-overlap pool; within each split-month it SHA-256-orders candidates using
`20260720|window|month|entry_time`, takes the primary month's total count, and
assigns its first primary-long count long and the remainder short. A control can
challenge specificity only after independently passing the same applicable
support floors. If a mechanism/component control other than direction flip
then passes the full performance gate, the specific witness-composition
transport claim is rejected. The primary policy is not repaired around a
control. The delayed-entry control drops an interval if the shifted exit is no
longer split-contained; it never moves a split boundary.

## Sequential performance gate

Only after support passes may a strict evaluator be committed and hash-frozen.
It then opens train, test, eval, and forward in that order, stopping permanently
on the first failure. Every opened constituent window must have:

- positive absolute return;
- full-calendar CAGR / global strict MDD at least 3;
- global/pre-entry-high-water strict MDD no greater than 15%;
- one-sided weekly cluster sign-flip `p <= 0.10`, 100,000 draws, seed
  `20260720`;
- mean gross return at least 20 bp per trade;
- positive stress-cost absolute return; and
- positive one-bar-delayed-entry absolute return.

Train 2023H1/H2, test 2024H1/H2, eval 2025H1/H2, and forward 2026H1 must each
be positive. Long and short sleeves must each be positive in train, test, and
eval. After all windows pass, the stitched 2022-11-01 through 2026-07-20 path
must also have positive absolute return, CAGR/strict-MDD at least 3, and strict
MDD no greater than 15%.

CAGR always uses the full declared wall-clock interval, including idle cash.
Strict MDD includes the global and pre-entry high-water mark, every held
five-minute OHLC/funding path, entry/exit costs, and virtual adverse exit cost.
The adverse within-bar ordering places favorable extremes and funding credits
before adverse extremes and funding debits.

## Stop and promotion boundary

Stop permanently at the first support, train, test, eval, forward, or stitched
gate failure. No failed WCTR result may be inverted or repaired under the same
candidate ID. This branch has broad prior BTC exposure, so even a pass is only
a candidate-level frozen result. Live or shadow promotion additionally
requires 90 forward-shadow days of frozen schema, freshness, revision, and
value-stability monitoring.

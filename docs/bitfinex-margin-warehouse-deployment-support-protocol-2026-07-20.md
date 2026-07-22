# BFMWD-144 source-support protocol — 2026-07-20

This protocol was written before any Bitfinex funding amount, utilization,
tenor, candidate incidence, comparator incidence, BTC price, return, or PnL
was inspected. The source-support stage is outcome blind.

## Immutable inputs

- BFMWD preregistration SHA-256:
  `6e478bac6becb58d282867f4ee612d9d13e803d01985474477d6e3073cd49e58`;
- novelty comparator freeze SHA-256:
  `37ee403d33b5361c752b84ef94d46a05d991d82e1b0a77338b41a6c49e8410de`;
- official Bitfinex `fUSD` and `fBTC` hourly statistics for
  `[2020-01-01, 2024-01-01)`; and
- exactly the six primary comparator clocks named in the comparator freeze.

The source is the fail-closed transport-v2 artifact. Its only amendment is the
availability rule
`max(floor(observation_time, 1h) + 15m, ceil(observation_time, 5m))`; ordinary
rows retain HH:15 availability and a late official row is never backdated.

The source-access seal binds the source manifest, canonical source, raw source,
transport-v2 amendment, SQFD prefix transport and artifacts, this document,
and the evaluator bytes before the evaluator parses any source feature value.
Every bound hash is rechecked before `pandas.read_csv` opens the canonical
source. Its header must exactly equal the nine-field source allowlist; unknown
or extra fields fail before row parsing.

## Rolling calculation

The signal grid is the complete UTC hourly grid. Each official row is assigned
to `floor(observation_time, 1h)`. A timestamp-only audit found six duplicated
symbol-hours; the latest official snapshot in each duplicated hour is retained
because it is the final state available to the fixed poll. Every absent hour is
inserted with `NaN` source fields. Exact lags are joins at `t-d` and `t-d-w`; a
missing exact lag invalidates that feature. There is no forward fill, backward
fill, nearest match, interpolation, or cross-symbol fill.

For each raw feature at hour `t`, robust location and scale use the preceding
1,440 physical hourly grid slots only. The current hour is excluded. At least 1,080
finite prior feature values are required inside that fixed window. Location is
the finite median and scale is `1.4826 × finite MAD`; a zero/non-finite scale
invalidates the z-score. The tenor threshold is the finite median over the
same strictly-prior fixed row window and minimum count.

This interpretation is frozen before incidence because it preserves elapsed
clock time when the provider has missing rows; it does not silently reach
farther back to collect 1,440 valid values.

## Trigger and controls

The four preregistered `(warehouse, deployment)` variants are the only
candidate family. The primary trigger requires all four robust z-scores at or
above `1.0` plus current tenor at or above its prior median.

Only these diagnostic controls are built, and none can be promoted:

1. omit warehouse-charge z;
2. omit unused-draw z;
3. omit tenor confirmation; and
4. retain the primary trigger but delay decision availability by 24 hours.

Triggers are converted to same-symbol false-to-true onsets only when the exact
preceding one-hour anchor exists and is false; a missing preceding anchor is
not filled and cannot create an onset. Opposing `fUSD` and `fBTC` onsets with
the same UTC source-hour anchor are both discarded even if provider latency
would otherwise give them different executable entry times.
Entry is decision availability plus five minutes and exit is 12 hours later.
Within each variant, control, and split, accepted events are globally
non-overlapping. Equality with the preceding scheduled exit is allowed.

Train (`2021-01-01` to `2023-01-01`) and selection (`2023-01-01` to
`2024-01-01`) are scheduled independently. An event is admissible only when
entry is at or after the split start and scheduled exit is strictly before the
exclusive split end. This prevents a
train event from consuming a selection outcome and keeps every 2024 value
sealed.

## Incidence gate

Only primary clocks determine promotion. The train/selection count gates use
their named split. Side and concentration gates use the union of contained
train and selection primary clocks:

- train events `>= 60` and selection events `>= 30`;
- each of 2021 and 2022 `>= 20` events;
- each half of 2023 `>= 12` events;
- each side share in `[0.20, 0.80]`;
- maximum calendar-month share `<= 0.20`;
- maximum weekday share `<= 0.25`; and
- maximum count in any trailing closed 14-day interval, divided by all union
  events, `<= 0.20`.

The trailing interval at event `i` is `[entry_i - 14 days, entry_i]`.

## Novelty gate

For each variant, primary entry timestamps from both contained splits are
compared with each frozen comparator over its declared common interval. Exact
duplicates are removed independently on both sides.

The SQFD registry member is read only through the independently frozen
`control == primary`, 2023-only prefix. Its manifest binds the original
2023–2026 clock artifact and reports the discarded later rows; the support
evaluator never parses those post-2023 rows.

A pair is sufficient only with at least ten candidate and five comparator
events. Every sufficient pair must have exact-entry Jaccard `<= 0.10` and both
directions of nearest-event containment inside symmetric ±6 hours `<= 0.35`.
At least four comparators must be sufficient, including one of CPR/CCIPA and
one of AMTR/SQFD. An insufficient comparator does not count as novelty evidence
and cannot be replaced.

## Stop rule

If no preregistered variant passes every incidence and novelty gate, the whole
BFMWD-144 family is retired without opening BTC outcomes. Threshold, sign,
window, hold, source, comparator, feature, or control repair is forbidden under
this identifier. An LLM/RL gate cannot rescue a failed deterministic family.

If at least one variant passes, only those passing variants may be named in a
separately frozen 2021–2022 economic evaluator. The 2023 selection outcome
remains sealed until train passes; 2024+ remains sealed until selection passes.

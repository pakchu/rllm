# LCDP-D1 source-support implementation contract — 2026-07-25

## Authority

This contract implements but cannot alter these committed artifacts:

```text
docs/london-cash-derivative-path-boundary-2026-07-25.md
SHA256 2be38b0181d0269af4159a70070dc3a9eef340a4e277965d09cf082eea848b7e
commit 0fd5e738f27b034a914853145bc8434d5d928502

training/preregister_london_cash_derivative_path.py
SHA256 a20118ab1b7cfe1a12c050bfc4a612689286383d9fa8fd2043dfcce62fd7368f
producer commit a1e5435ea43fb17337ce76b80ccf53d5c26e9b0b

results/london_cash_derivative_path_preregistration_2026-07-25.json
SHA256 da0dd2f24236c3b64e31604268b0ad9d9b342723629790d1ecf061d0a02f4ad4
manifest_hash 0cbeeaad957c67187381405681e8e7935c7039c7c9f9e2d0a19cbe5e912d5dac
artifact commit 044a7a78ee7ff1e71d287b78f486c23961ea1a76
```

The implementation may expose source incidence, validity, categorical token
support, and frozen source-only controls. It may not open funding, execution
prices, post-boundary returns, rewards, model rows, actions, trades, PnL,
CAGR, MDD, or any at-or-after-2023 non-date source value.

## Exact future implementation files

```text
runner
  training/build_london_cash_derivative_path_source_support.py
tests
  tests/test_build_london_cash_derivative_path_source_support.py
pass token output
  data/lcdp_d1_source_support/token_support.csv.gz
pass report
  results/lcdp_d1_source_support_2026-07-25.json
failure report
  results/lcdp_d1_source_support_rejection_2026-07-25.json
```

The runner and tests must be committed and hash-bound by a separate execution
seal before the first real source row is decoded. Synthetic fixtures are
allowed before that seal.

## Physical parser

The runner first verifies every source, physical-header, source-manifest,
boundary, preregistration producer, and preregistration artifact hash.

For each physical CSV row:

1. split and parse the first `date` field before parsing any other field;
2. require strictly increasing unique UTC five-minute timestamps;
3. parse projected non-date fields only when the timestamp is inside the
   authorized prefix source range;
4. continue parsing date fields alone after the prefix cutoff to prove that no
   later non-date value was opened; and
5. reject malformed selected rows rather than filling or skipping them.

For the full build, the non-date cutoff is the UTC instant corresponding to:

```text
2022-12-31 16:00 Europe/London
```

The 2022-12-31 LCDP source line consumes bars only through 15:55 London.
Physical rows from 16:00 London through the end of 2022 are post-boundary for
this source-support transaction and remain date-only. All rows at or after
`2023-01-01` also remain date-only.

Coinbase empty numeric fields are allowed physically only when
`source_complete == 0`; they invalidate the containing day. A
`source_complete == 1` row must contain finite valid numerics. Every selected
Binance row must contain finite valid numerics.

## Daily assignment and DST

Each parsed UTC row belongs to the London date whose 16:00 local boundary
first follows it. The runner independently constructs every expected UTC
five-minute timestamp in:

```text
[B_(D-1), B_D)
```

and requires exact equality with the selected physical timestamps for both
venues. Expected counts are computed from UTC elapsed time between adjacent
localized boundaries:

```text
276 on spring DST transition
288 on ordinary days
300 on autumn DST transition
```

The full calendar emits exactly 1,096 lines from 2020-01-01 through
2022-12-31. The first line is always `SOURCE_INVALID_START`, even if an
unauthorized pre-2020 file later appears.

Every other invalid joint window emits one `SOURCE_INVALID` line. No invalid
day is deleted from rank history, transition adjacency, sequence history, or
gate denominators.

## Primitive builder

For each valid venue window, preserve ordered arrays sufficient to compute the
frozen return, path length, efficiency, range, quote notional, start/end
price, and equal first/second arc returns. Do not persist raw primitives in the
token output.

All comparisons use IEEE finite `float64` values and exact `>`, `<`, `==`
branching. No epsilon or rounding precedes token assignment.

Cash participation quantiles use this exact deterministic rule:

1. select finite `cash_share` values from the previous 126 emitted calendar
   lines; the current line is excluded;
2. require at least 63 values;
3. sort ascending;
4. for quantile `q`, set `h=(n-1)q`, `lo=floor(h)`, `hi=ceil(h)`, and return
   `x[lo] + (h-lo) * (x[hi]-x[lo])`;
5. low is current `<q1/3`, high is current `>q2/3`, and equality/between is
   mid.

The 126-line window advances across invalid lines; invalid values are absent
from its finite subset rather than time-compressed.

## Primary token stream

Token mapping and ordered serialization are inherited exactly from the
boundary and preregistration. Additional implementation invariants:

- a ready line contains ready vocabulary only;
- a safety line has the same safety token in every market field;
- `SOURCE_INVALID_START` is legal only on 2020-01-01;
- a ready current line compares transitions only with the immediately
  preceding emitted calendar-day line;
- a safety predecessor produces `PARTICIPATION_UNKNOWN`,
  `ALIGNMENT_MIXED`, and `LEAD_MIXED`;
- sequence history is the current line plus exactly twenty preceding emitted
  lines; and
- December 31 can be sequence-ready for source statistics but remains a
  deterministic terminal flat in every later economic split.

For support-gate counts, `model_eligible` means:

```text
current line ready
and twenty prior emitted lines exist
and London month/day is not December 31
```

This definition prevents the source-support count from claiming a model action
on a frozen terminal-flat line.

## Source-only controls

Controls rebuild affected primitives, ranks, transitions, and sequences from
their transformed inputs:

- `cash_perp_role_swap`: swap complete venue paths and their quote notionals;
- `cash_stale_one_day`: current perpetual plus immediately previous emitted
  cash path;
- `perp_stale_one_day`: current cash plus immediately previous emitted
  perpetual path;
- `lag_7_calendar_days`: both paths from exactly seven emitted lines earlier,
  with current calendar context;
- `calendar_context_mask`: primary stream with only calendar context replaced
  by `CALENDAR_MASKED`;
- `cash_only_language`: current cash return sign in the frozen cash-only
  `daily_alignment` vocabulary and all other market fields masked; and
- `perp_only_language`: symmetric perpetual-only construction.

If a stale or lag control lacks a required path or that required path is
invalid, every market field is `CONTROL_UNREADY`. `CONTROL_UNREADY` lines stay
in the control clock and rank history.

`calendar_context_mask`, `cash_only_language`, and `perp_only_language` are
derived only after the primary line has been classified as ready or safety.
Primary safety is preserved rather than converted into a mask.

## Append replay

The runner executes four independent physical parses:

```text
full end exclusive    2023-01-01
prefix end exclusive  2021-01-01
prefix end exclusive  2022-01-01
prefix end exclusive  2023-01-01
```

Each prefix's non-date cutoff is its final calendar date's 16:00 London
boundary. The prefix still scans all later physical date fields but parses
zero later non-date fields.

For every prefix date, compare the UTF-8 canonical JSON representation of the
complete source-support record against the corresponding full-build record.
Canonical JSON uses sorted keys, compact separators, UTF-8, no NaN, and one
record per date. The comparison includes safety/readiness, all primary fields,
the serialized primary line, all serialized controls, and the primary sequence
hash.

Any mismatch fails before output materialization.

## Exact support metrics

### Source validity

Denominator: every emitted calendar line in the named year or quarter,
including `SOURCE_INVALID_START`.

```text
minimum each year     0.97
minimum each quarter  0.95
```

### Readiness

Using the exact `model_eligible` definition above:

```text
2020 minimum                       280
2021 minimum                       350
2022 minimum                       350
each quarter after 2020Q1 minimum  80
```

2020Q1 remains in source-validity accounting but has no readiness minimum.

### Token diversity

Denominator: model-eligible primary lines in the named year. For every
non-calendar field:

- at least two ready categories each have share at least 0.03; and
- no category share exceeds 0.94.

Cash and perpetual full-window directions must each contain at least one
strictly positive and one strictly negative return in every year. The report
stores only direction counts, never numeric returns.

### Control distinctness

For role-swap, both stale controls, and lag-7:

- jointly ready means neither compared serialized line contains a primary
  safety token, mask token, or `CONTROL_UNREADY`;
- difference share is unequal serialized lines divided by jointly ready
  dates; and
- each difference share must be at least 0.05.

The calendar and source mask controls must differ from every ready primary
line on the field they are required to mask.

## Gate order and failure semantics

Run exactly:

1. protocol/source/header/manifest/type/clock integrity;
2. calendar/DST integrity;
3. annual and quarterly source validity;
4. annual and quarterly model eligibility;
5. annual token/direction diversity;
6. source-control existence and distinctness;
7. append replay;
8. forbidden-access counters.

The first failed gate stops later gates and writes only:

```text
results/lcdp_d1_source_support_rejection_2026-07-25.json
```

It must not write the pass token output or pass report. Exact failure action:

```text
retire_lcdp_d1_unchanged_before_outcomes
```

If every gate passes, write deterministic gzip token support with `mtime=0`
and then:

```text
results/lcdp_d1_source_support_2026-07-25.json
```

Exact pass action:

```text
authorize_economic_rllm_evaluator_freeze_only
```

Neither decision is a profitability result.

## Output schema

The pass token CSV has exact columns:

```text
london_date
boundary_utc
expected_slots
source_state
rank_ready
model_eligible
calendar_context
daily_alignment
daily_leader
relative_basis_path
arc_transfer
path_efficiency
range_relation
participation_state
participation_transition
alignment_transition
leader_transition
primary_line
primary_sequence_hash
cash_perp_role_swap
cash_stale_one_day
perp_stale_one_day
lag_7_calendar_days
calendar_context_mask
cash_only_language
perp_only_language
```

Exact scalar encoding:

```text
london_date       YYYY-MM-DD
boundary_utc      YYYY-MM-DDTHH:MM:SSZ
expected_slots    base-10 integer
source_state      SOURCE_INVALID_START | SOURCE_INVALID |
                  RANK_UNREADY | READY
rank_ready        true | false
model_eligible    true | false
```

`primary_sequence_hash` is lowercase SHA-256 of the 21 serialized primary
lines joined by one ASCII newline, without a trailing newline. It is empty
until twenty prior emitted lines exist. It is still computed for a
21-line safety sequence even though that current line is not model-eligible.

CSV uses UTF-8, the exact header order above, `\n` line endings, Python
minimal quoting, and deterministic gzip with filename omitted and `mtime=0`.

No raw primitive, rank boundary, price, return, volume, notional, funding,
reward, model target, action, trade, PnL, CAGR, or MDD is persisted.

The report binds:

- authority and source hashes;
- runner/test commit and SHA-256;
- gate decisions in executed order;
- annual/quarterly source and readiness counts;
- field-year category counts/shares;
- direction counts;
- control jointly-ready/difference counts;
- append replay hashes;
- token-output hash when passed; and
- every forbidden counter.

The report itself is deterministic and contains no wall-clock creation time.

## Required synthetic verification

Before sealing the runner, tests must cover:

- ordinary, spring-DST, and autumn-DST source windows;
- exact first-line invalidity;
- complete, missing, duplicate, malformed, nonfinite, bad-OHLC, and incomplete
  source rows;
- strictly prior 126-line/63-value quantiles with exact interpolation;
- current-row exclusion;
- safety-day transition adjacency;
- ready and safety serialization invariants;
- every token branch;
- every control and its unready prefix;
- 21-line sequence readiness and December 31 terminal exclusion;
- append replay equality and deliberate corruption detection;
- forbidden post-cutoff non-date parsing detection;
- deterministic gzip bytes;
- first-failure output behavior; and
- pass-output schema without forbidden fields.

The official source run is unauthorized until these tests pass on the exact
committed runner and an independent review reports no blocker.

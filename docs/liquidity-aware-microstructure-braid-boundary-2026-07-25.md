# LAMB-21 boundary — liquidity-aware microstructure braid

Date: 2026-07-25

## Decision

Freeze one new candidate:

**LAMB-21 — Liquidity-Aware Microstructure Braid, twenty-one-line
seven-day-context eight-hour rolling target-position RLLM.**

At each fixed eight-hour decision the future policy chooses exactly one next
target:

```text
SHORT
FLAT
LONG
```

The target persists until the next decision. Deterministic code owns source
clocks, exact joins, numeric transforms, strictly-prior ranks, categorical
tokens, execution, transaction costs, realized funding, reward, and strict
drawdown. The model may see only the frozen symbolic history and current
position.

This boundary authorizes only an outcome-blind source/token-support
preregistration and implementation. It does not authorize execution OHLC,
funding cash-flow evaluation, future returns, rewards, model fitting,
checkpoint selection, trades, or profitability claims.

## Identity and prior-work boundary

The mechanism-selection audit is:

```text
docs/post-tracer-alpha-mechanism-audit-2026-07-25.md
```

LAMB is a fixed-clock relational MDP, not a repaired event strategy:

- QLCD's scalar score, threshold, fixed side, and fixed hold are forbidden;
- SMCC's rare-event threshold, event clock, side, and hold are forbidden;
- BAFR's frustration score and two-hour reversal rule are absent;
- DCLB's H.8 anchor, exact fresh-release intersection, fixed macro side, and
  fixed multi-day event are absent;
- IVPLH's intrinsic-volume anchor and fixed handoff side are absent; and
- TRACER's four-hour leadership/premium conjunction is absent.

The two microstructure views share a frozen Binance USD-M aggTrade origin.
They are not claimed to be independent data sources. Their distinct
deterministic transforms are used to test a relation between exact quantity
cohorts and within-millisecond concentration.

LAMB intentionally reuses the outcome-blind QLCD cohort definitions:

```text
coarse  quantity_mbtc % 100 == 0
medium  quantity_mbtc % 10 == 0 and not coarse
fine    every remaining exact 1 mBTC increment
```

That primitive reuse is disclosed contamination, not a QLCD repair or a
novelty claim. Transform bytes and cohort boundaries are immutable. LAMB never
uses `qlcd_score`, QLCD's event threshold, side, selected side flip, or hold.
A later economic evaluator must score lattice-only and `no_lattice` killer
baselines; failure to add value beyond the known weak lattice family retires
LAMB.

Prior source and market outcomes make this a contaminated research family.
The exact LAMB joint state and candidate-specific outcomes remain unopened at
this boundary. A failed identity cannot be renamed or repaired after source
incidence, reward, or outcome inspection.

## Frozen source containers

Every support loader must use the exact physical header and exact projected
allowlist. Loading all columns and dropping forbidden values afterward is not
allowed.

### Source A — Federal Reserve H.4.1 net liquidity

```text
path
  data/federal_reserve_h41_net_liquidity_2018_2023/
  federal_reserve_h41_net_liquidity_2018-01-04_2023-12-28.csv.gz
SHA256
  224883dad01b9d7f17d52eb87f3d7ef9890c8dd055a6c36577a534d2afe69621
header SHA256
  4bd522eddda52fefa94c9722f6015596fcde80769c59441046bc0438e1d314d9
manifest
  data/federal_reserve_h41_net_liquidity_2018_2023/build_manifest.json
manifest-file SHA256
  1ec212a85de0e49c5a0c2d35b8b22be86eb7d62989f7a0098be1bb1274b2a99b
builder SHA256
  822ab0602549d71f50834da4abf13c5a81dd6af7e58147b9b37aa9940355dc7d
```

Exact allowlist and order:

```text
release_date
observation_date
available_at_utc
net_liquidity_usd_millions
```

### Source B — New York Fed overnight RRP

```text
path
  data/new_york_fed_overnight_rrp_2018_2023/
  new_york_fed_overnight_rrp_2018-01-01_2023-12-31.csv.gz
SHA256
  49f67ed44b7eb81fd35c17a8209cf14d6a8019d7e9f77fce8c343d1a7fb66b27
header SHA256
  81a388d6e36c5e84c166b5fe111d3766ea5c6b56ac83895ed3541a6c05a01e9c
manifest
  data/new_york_fed_overnight_rrp_2018_2023/build_manifest.json
manifest-file SHA256
  4f87e2219da71c94832c8708086ba01387efc145e3488b62cd3b3d07c62d8fee
builder SHA256
  0567157dde18b1c6ccfb37b669ceead521360f23dd0b73033fccc08e37c0d42c
```

The manifest's internal `manifest_hash` is not the physical manifest-file
hash. The support runner binds the physical file hash above.

Exact allowlist and order:

```text
operation_date
result_available_at_utc
total_amount_accepted_usd
participating_counterparties
accepted_counterparties
source_complete
quarantine_reason
```

### Source C — Binance BTC quantity-lattice cohorts

```text
path
  data/binance_um_quantity_lattice_btc_2020_2023/
  BTCUSDT_quantity_lattice_5m_2020-01-01_2023-12-31.csv.gz
SHA256
  3ca945f134115fc7b58086405fd881db3e3b70087bd9da54ffc293f6b658072e
header SHA256
  1021675e0998dfaf49a13d46af7365b0762719917817d8719c2d8d99116f47ed
manifest
  data/binance_um_quantity_lattice_btc_2020_2023/build_manifest.json
manifest SHA256
  bcdf89924f54a5b97d4219749c2094d2a4c08d8473a37bc5367d9b8e5791284f
transform SHA256
  8e7503dfb518bdd6515d255dc1ae4a1ac8b47cc078f39bee24cd2d52561a8e1b
```

Exact allowlist and order:

```text
date
source_observed
source_complete
source_gap_day
verified_zero_volume_empty
post_gap_quarantine
agg_trade_count
total_quantity_mbtc
coarse_quantity_mbtc
coarse_signed_quantity_mbtc
fine_quantity_mbtc
fine_signed_quantity_mbtc
```

`qlcd_score` is deliberately excluded.

### Source D — Binance BTC same-millisecond cascade primitives

```text
path
  data/binance_um_same_millisecond_cascade_btc_2020_2023/
  BTCUSDT_same_millisecond_5m_2020-01-01_2023-12-31.csv.gz
SHA256
  8fa03b0d7f58db9d0ba6c889e99ce87ba668f55a3c7f0ab5638a374c4584bfd1
header SHA256
  1af1937bd53b900960f12c73af0701a86990fcb688d6727c9715c9330b1f6090
manifest
  data/binance_um_same_millisecond_cascade_btc_2020_2023/build_manifest.json
manifest SHA256
  e6ba3fbf74bc9bc1a7c1b35873e9ff430e5bc0a7b7edcc7e082f3f397362c805
transform SHA256
  cfc1c1c587236e1458465955c133b240a6c4f4748c2e7260519e9cdbea3a16de
```

Exact allowlist and order:

```text
date
source_observed
source_complete
source_gap_day
verified_zero_volume_empty
post_gap_quarantine
first_transact_time_ms
last_transact_time_ms
agg_trade_count
first_price
last_price
quote_notional
collision_quote_notional
max_ms_quote_notional
max_ms_signed_quote_notional
```

`max_ms_score` is deliberately excluded.

No support process may decode an execution kline, funding rate, future return,
label, target, reward, model prediction, PnL, CAGR, MDD, portfolio weight, or
post-2023 source value.

## Frozen eight-hour process clock

Canonical boundaries are:

```text
00:00 UTC
08:00 UTC
16:00 UTC
```

For boundary `B`:

```text
micro source window    [B-8h, B)
latest five-minute bar B-5m, complete at B
macro as-of cutoff     B
state complete         B
policy decision        B+5m
execution              B+10m at the USD-M five-minute open
next execution         B+8h+10m
```

Source-C and Source-D rows are labeled by five-minute start. A contributing
row must have `date` in `[B-8h,B)` and is considered complete at `date+5m`.
Source-C and Source-D must contain exactly the same 96 ordered timestamps.
Every timestamp must be unique and every row must be available by `B`.

Source A selects the latest release with:

```text
available_at_utc <= B
0 <= B - available_at_utc <= 10 elapsed days
```

Its delta uses that release and the immediately preceding release in the
physical source. Both must be finite and causal. No later release may be used.

Source B selects the latest normal operation with:

```text
result_available_at_utc <= B
0 <= B - result_available_at_utc <= 5 elapsed days
source_complete = true
quarantine_reason = blank
```

Amount and breadth deltas use the immediately preceding physical operation
only when it is also complete and no quarantined row lies between the two.
A quarantined operation resets the RRP delta segment. There is no bridge,
interpolation, calendar fill, or skip to an older convenient predecessor.

A boundary with a missing, stale, quarantined, malformed, or rank-unready core
source is represented by the safety state and forces target `FLAT`. It remains
in wall-clock accounting. There is no event filter, cooldown, queued signal,
fixed side, TP, SL, or variable hold.

`core_source_valid` is determined before any rank or token mapping. All 96
Source-C and Source-D rows must have `source_complete=true`,
`source_gap_day=false`, and `post_gap_quarantine=false`. Each row must be
either observed (`source_observed=true`,
`verified_zero_volume_empty=false`) or a verified empty bucket
(`source_observed=false`, `verified_zero_volume_empty=true`), never both or
neither. Source-C and Source-D flags must match at every timestamp. The
selected macro predecessors and all required numeric cells must also pass
their source rules.

`rank_ready` means `core_source_valid` and every ranked micro primitive has at
least 180 strictly prior valid observations inside the 270-boundary reference.
`sequence_ready` means the current state is rank-ready and twenty prior
canonical state lines exist. Those prior lines may include a safety line; time
is never compressed by skipping an invalid boundary.

## Frozen causal transforms

All scale-sensitive ranks use only prior valid canonical boundaries:

```text
micro reference length  270 prior valid boundaries = 90 days
micro minimum history   180 prior valid boundaries
cut points              q33 and q67
current included        false
tie rule                LOW: x<=q33; MID: q33<x<=q67; HIGH: x>q67
```

Macro primitives use only mathematical sign, relation, and explicit source
age. They are not ranked. Repeating a carried release at multiple eight-hour
boundaries therefore cannot manufacture additional macro observations.

The ranked micro primitives are exactly:

```text
coarse_share
coarse_coherence
fine_conviction
collision_share
cascade_share
cascade_coherence
```

For each valid boundary define:

```text
h41_delta
  latest net_liquidity_usd_millions
  - immediately prior release net_liquidity_usd_millions

rrp_amount_delta
  latest total_amount_accepted_usd
  - immediately prior in-segment operation amount

rrp_breadth_delta
  latest accepted_counterparties
  - immediately prior in-segment accepted_counterparties

coarse_flow
  sum(coarse_signed_quantity_mbtc)

fine_flow
  sum(fine_signed_quantity_mbtc)

coarse_share
  sum(coarse_quantity_mbtc) / sum(total_quantity_mbtc)

coarse_coherence
  abs(sum(coarse_signed_quantity_mbtc)) / sum(coarse_quantity_mbtc)

fine_signed_share
  sum(fine_signed_quantity_mbtc) / sum(fine_quantity_mbtc)

fine_conviction
  abs(fine_signed_share)

cascade_flow
  sum(max_ms_signed_quote_notional)

collision_share
  sum(collision_quote_notional) / sum(quote_notional)

cascade_share
  sum(max_ms_quote_notional) / sum(quote_notional)

cascade_coherence
  sum(abs(max_ms_signed_quote_notional)) / sum(max_ms_quote_notional)

source_price_response
  log(last last_price / first first_price)
```

Zero denominators produce an invalid boundary, not a neutralized ratio.
Required notionals, quantities, counts, and prices must be finite and
nonnegative, with quantities/counts integral where declared by the source.
Prices and required denominators must be strictly positive. Signed quantities
and notionals may be negative. Absolute signed values may not exceed their
paired unsigned totals. Shares and coherence must lie in `[0,1]`.
`accepted_counterparties` must not exceed `participating_counterparties`;
`collision_quote_notional` and `max_ms_quote_notional` must not exceed
`quote_notional`; and absolute `max_ms_signed_quote_notional` must not exceed
`max_ms_quote_notional`.

For an observed Source-C or Source-D row, `agg_trade_count` is a positive
integer. For a verified-empty row it and every projected numeric primitive are
exactly zero. On an observed Source-D row:

```text
date_ms <= first_transact_time_ms <= last_transact_time_ms < date_ms+300000
```

The transaction timestamps and counts are validation-only fields. They never
enter a rank or prompt.

The immediately prior state is the immediately prior canonical boundary.
Transitions never skip an invalid or rank-unready state.

## Frozen categorical relation language

The model input contains 21 ordered state lines, covering exactly seven
calendar days, followed by current position. Exact state-field order:

```text
h41_impulse
rrp_impulse
macro_sponsorship
macro_age
lattice_relation
lattice_concentration
cascade_impact
cascade_intensity
micro_braid
macro_transition
micro_transition
```

Exact vocabularies:

```text
h41_impulse
  H41_EXPANDS | H41_CONTRACTS | H41_FLAT

rrp_impulse
  RRP_RELEASES | RRP_DRAINS | RRP_FLAT

macro_sponsorship
  LIQUIDITY_SUPPORTS | LIQUIDITY_RESTRICTS |
  MACRO_SPLIT | MACRO_NEUTRAL

macro_age
  BOTH_FRESH | H41_AGING | RRP_AGING | BOTH_AGING

lattice_relation
  COHORTS_BUY | COHORTS_SELL |
  COARSE_BUY_FINE_SELL | COARSE_SELL_FINE_BUY |
  LATTICE_NEUTRAL

lattice_concentration
  COARSE_DOMINANT | FINE_DOMINANT | LATTICE_MIXED

cascade_impact
  CASCADE_BUY_FOLLOWTHROUGH | CASCADE_BUY_ABSORBED |
  CASCADE_SELL_FOLLOWTHROUGH | CASCADE_SELL_ABSORBED |
  CASCADE_NEUTRAL

cascade_intensity
  CASCADE_BROAD | CASCADE_LOCAL | CASCADE_MIXED

micro_braid
  MICRO_CONFIRMS_BUY | MICRO_CONFIRMS_SELL |
  LATTICE_BUY_CASCADE_SELL | LATTICE_SELL_CASCADE_BUY |
  MICRO_NEUTRAL

macro_transition
  SUPPORT_PERSISTS | RESTRICTION_PERSISTS |
  ROTATES_TO_SUPPORT | ROTATES_TO_RESTRICTION |
  MACRO_TRANSITION_MIXED

micro_transition
  BUY_PRESSURE_PERSISTS | SELL_PRESSURE_PERSISTS |
  PRESSURE_FLIPS | PRESSURE_DISSIPATES |
  MICRO_TRANSITION_MIXED

current_position
  SHORT | FLAT | LONG
```

Exact mappings:

```text
h41_impulse
  sign(h41_delta) > 0 -> H41_EXPANDS
  sign(h41_delta) < 0 -> H41_CONTRACTS
  otherwise           -> H41_FLAT

rrp_impulse
  rrp_amount_delta<0 and rrp_breadth_delta<=0 -> RRP_RELEASES
  rrp_amount_delta>0 and rrp_breadth_delta>=0 -> RRP_DRAINS
  otherwise                                   -> RRP_FLAT

macro_sponsorship
  H41_EXPANDS   and RRP_RELEASES -> LIQUIDITY_SUPPORTS
  H41_CONTRACTS and RRP_DRAINS   -> LIQUIDITY_RESTRICTS
  either primitive flat          -> MACRO_NEUTRAL
  otherwise                      -> MACRO_SPLIT

macro_age
  H.4.1 age <= 4d and RRP age <= 2d -> BOTH_FRESH
  H.4.1 age > 4d and RRP age <= 2d  -> H41_AGING
  H.4.1 age <= 4d and RRP age > 2d  -> RRP_AGING
  otherwise                          -> BOTH_AGING

lattice_relation
  coarse_flow>0 and fine_flow>0 -> COHORTS_BUY
  coarse_flow<0 and fine_flow<0 -> COHORTS_SELL
  coarse_flow>0 and fine_flow<0 -> COARSE_BUY_FINE_SELL
  coarse_flow<0 and fine_flow>0 -> COARSE_SELL_FINE_BUY
  otherwise                     -> LATTICE_NEUTRAL

lattice_concentration
  coarse_share HIGH and coarse_coherence HIGH -> COARSE_DOMINANT
  coarse_share LOW or fine_conviction HIGH -> FINE_DOMINANT
  otherwise -> LATTICE_MIXED

cascade_impact
  cascade_flow>0 and source_price_response>0 -> CASCADE_BUY_FOLLOWTHROUGH
  cascade_flow>0 and source_price_response<=0 -> CASCADE_BUY_ABSORBED
  cascade_flow<0 and source_price_response<0 -> CASCADE_SELL_FOLLOWTHROUGH
  cascade_flow<0 and source_price_response>=0 -> CASCADE_SELL_ABSORBED
  cascade_flow=0 -> CASCADE_NEUTRAL

cascade_intensity
  collision_share HIGH and cascade_share HIGH and cascade_coherence HIGH
    -> CASCADE_LOCAL
  collision_share LOW and cascade_share LOW
    -> CASCADE_BROAD
  otherwise
    -> CASCADE_MIXED

micro_braid
  sign(coarse_flow)>0 and sign(cascade_flow)>0 -> MICRO_CONFIRMS_BUY
  sign(coarse_flow)<0 and sign(cascade_flow)<0 -> MICRO_CONFIRMS_SELL
  sign(coarse_flow)>0 and sign(cascade_flow)<0 -> LATTICE_BUY_CASCADE_SELL
  sign(coarse_flow)<0 and sign(cascade_flow)>0 -> LATTICE_SELL_CASCADE_BUY
  otherwise -> MICRO_NEUTRAL
```

`macro_transition` compares the immediately prior and current
`macro_sponsorship` categories:

```text
both LIQUIDITY_SUPPORTS  -> SUPPORT_PERSISTS
both LIQUIDITY_RESTRICTS -> RESTRICTION_PERSISTS
current LIQUIDITY_SUPPORTS, prior differs  -> ROTATES_TO_SUPPORT
current LIQUIDITY_RESTRICTS, prior differs -> ROTATES_TO_RESTRICTION
otherwise -> MACRO_TRANSITION_MIXED
```

`micro_transition` maps `micro_braid` to BUY, SELL, CONFLICT, or NEUTRAL and
compares the immediately prior canonical state:

```text
BUY then BUY   -> BUY_PRESSURE_PERSISTS
SELL then SELL -> SELL_PRESSURE_PERSISTS
BUY then SELL or SELL then BUY -> PRESSURE_FLIPS
BUY/SELL then CONFLICT/NEUTRAL -> PRESSURE_DISSIPATES
otherwise -> MICRO_TRANSITION_MIXED
```

The exact safety line is:

```text
SOURCE_INVALID|RRP_FLAT|MACRO_NEUTRAL|BOTH_AGING|LATTICE_NEUTRAL|
LATTICE_MIXED|CASCADE_NEUTRAL|CASCADE_MIXED|MICRO_NEUTRAL|
MACRO_TRANSITION_MIXED|MICRO_TRANSITION_MIXED
```

`SOURCE_INVALID` is a safety-only `h41_impulse` value. It cannot be emitted by
a valid state or selected as a learned category. If the immediately prior
canonical state is invalid, current primitive fields are preserved but both
transition fields use their mixed value.

No raw level, delta, rank percentile, rank numerator, timestamp, date, price,
return magnitude, funding value, reward, previous reward, action probability,
equity, drawdown, or portfolio statistic may enter the prompt.

## Source-only controls

Each control independently rebuilds primitives, ranks, transitions, and token
lines from the frozen physical sources:

1. `h41_stale_one_release` — replace the selected H.4.1 release with its
   immediately prior physical release while preserving the primary clock;
2. `rrp_stale_one_operation` — replace the selected RRP operation with its
   immediately prior in-segment operation;
3. `lattice_cohort_swap` — swap coarse and fine signed quantities and paired
   unsigned quantities before aggregation;
4. `cascade_delay_37` — replace each Source-D row with the row exactly 37
   five-minute positions earlier inside the same UTC month; the first 37
   positions are control-invalid safety rows and may not borrow from the prior
   or next month; and
5. `macro_relation_mask` — on every history line replace `h41_impulse` with
   `H41_FLAT`, `rrp_impulse` with `RRP_FLAT`, `macro_sponsorship` with
   `MACRO_NEUTRAL`, and `macro_age` with `BOTH_AGING`, then recompute
   `macro_transition=MACRO_TRANSITION_MIXED` before serializing the control.

No control may replace the primary after a failure.

## Frozen source-support gates

Source support is conjunctive. The first failure retires LAMB unchanged:

1. Exact paths, bytes, gzip `mtime=0`, physical headers, manifests, allowlists,
   types, monotonicity, clocks, quarantine rules, finite-value rules, and
   source identities reproduce.
2. Source-C and Source-D exact five-minute timestamp joins exist for at least
   `99.0%` of the canonical 2020–2023 grid in every calendar year.
3. At least `95.0%` of nominal eight-hour boundaries are core-source valid in
   every year 2020–2023.
4. Sequence-ready boundaries are at least `750` in 2020 and `1,000` in each
   of 2021, 2022, and 2023; every quarter after the 90-day warm-up has at least
   `225`.
5. Forced-`FLAT` source-invalid or rank-unready share is at most `8.0%` in
   every full post-warm-up quarter.
6. Every one of the eleven relation fields has at least two categories with at
   least `3.0%` annual share; no category exceeds `94.0%`.
7. `macro_sponsorship` has at least `5.0%` `LIQUIDITY_SUPPORTS` and `5.0%`
   `LIQUIDITY_RESTRICTS` support in every year.
8. `micro_braid` has at least `10.0%` aggregate buy support and `10.0%`
   aggregate sell support in every year.
9. `cascade_impact` has at least `7.5%` follow-through and `7.5%` absorption
   support in every year.
10. Each year's exact eleven-field signatures contain at least `120` distinct
    values and no signature exceeds `10.0%` share.
11. Jensen-Shannon divergence for every field between adjacent years is at
    most `0.30`, using the complete frozen vocabulary and base-2 logarithms.
12. Every source-only control differs from the primary token sequence and no
    control is byte-identical to another.
13. Prefix reconstruction and later-row append replay leave all already
    formed primitives, ranks, macro selections, transitions, and tokens
    byte-identical.
14. No forbidden-source or outcome counter is nonzero.

Gate 6 through 11 denominators contain only sequence-ready, core-valid states
in the named UTC year. Safety lines and `current_position` are excluded.
Signature rows contain exactly the eleven relation fields.

For Jensen-Shannon divergence, absent vocabulary values receive probability
zero, `M=(P+Q)/2`, logarithms are base two, and every
`0*log2(0/M)` term is zero.

## RLLM boundary

The source stage does not choose a model. After a complete support pass, a
separate immutable economic protocol must bind one compact causal language
model, tokenizer, quantization, adapter method, RL objective, seed set,
checkpoint rule, invalid-output action, and inference budget before any reward
is materialized.

Allowed prompt information is exactly:

- 21 ordered categorical state lines;
- current target `SHORT|FLAT|LONG`;
- source-validity represented only by the safety line; and
- a fixed action grammar.

The model must return exactly one of:

```text
TARGET_SHORT
TARGET_FLAT
TARGET_LONG
```

Invalid, late, ambiguous, multi-action, or unavailable inference is
`TARGET_FLAT`. The model cannot change leverage, place an order, set a stop,
choose a hold, or access the database directly.

The intended model advantage is conditional temporal deduction:

```text
slow liquidity transition
  + fast cohort/cascade agreement or conflict
  + persistence/rotation over seven days
  + current position and turnover consequence
  -> one target position
```

Raw-number forecasting is outside the model boundary.

## Contingent economic chronology

Only a complete source-support pass may authorize a separately committed
reward/evaluator freeze. That freeze must bind:

- exact execution kline and realized-funding sources;
- action timing, terminal flattening, target-delta fees, and delay stress;
- leverage, base/stress costs, full-calendar CAGR, and held-path strict MDD;
- one-step reward, discount, terminal rule, replay construction, and all
  algorithm/model/checkpoint choices;
- always-flat, always-long, always-short, exact action-flip, macro-mask,
  micro-mask, reward-shuffle, stale-source, and deterministic-random controls;
- selection correction and weekly clustered inference;
- write-once schedules produced before each evaluated year's outcomes; and
- a fail-flat live-parity contract.

Research-history labels:

```text
2020-2022  development/history; component outcomes already contaminated
2023       candidate-specific transfer only, not globally pristine
2024       first candidate-specifically sealed historical annual test
2025       sealed historical annual evaluation after an unchanged 2024 pass
2026-YTD   sealed recent confirmation after unchanged 2024 and 2025 passes
```

The 2024–2026 schedules are transaction-style historical holdouts, not
real-time prospective observations: their market outcomes already exist and
other repository research has seen parts of those calendars. The evaluator
must nevertheless materialize each LAMB schedule before decoding that stage's
LAMB execution/funding rows. A live profitability claim requires a later
forward shadow/live interval.

The intended minimum annual transfer gate, to be made no weaker by the later
freeze, is:

- positive base, stress-cost, and +5-minute-delay absolute return;
- base CAGR/strict-MDD at least `3.0`;
- stress and delay CAGR/strict-MDD at least `2.5`;
- strict MDD at most `15%`;
- at least `120` non-flat target intervals;
- at least `20%` long and `20%` short non-flat interval share;
- positive return in both equal-duration halves of the evaluated window;
- required-control defeat; and
- familywise weekly-clustered `pmax < 0.10`.

The final historical transfer claim requires 2024, 2025, and 2026-YTD to pass
unchanged. That still does not satisfy a full three-calendar-year claim until
2026-12-31 is available and evaluated without repair. A pass in 2023 alone is
not sufficient.

## Failure and live parity

Any source-support failure retires LAMB before rewards. Any later economic year
failure retires the frozen economic identity before the next year. No failed
year may select a new token, rank horizon, freshness limit, model, seed,
checkpoint, reward, cost, leverage, action grammar, or gate.

Live operation must reproduce the same publication delays, source-complete
flags, quarantine resets, maximum ages, five-minute completion rule,
eight-hour boundaries, model input, and fail-flat behavior. Missing, late,
schema-drifted, stale, quarantined, or non-finite data means `TARGET_FLAT`; it
never means carrying an unverified new target.

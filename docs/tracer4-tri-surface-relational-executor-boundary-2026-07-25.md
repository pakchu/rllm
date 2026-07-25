# TRACER-4H boundary — tri-surface relational auction-crowding executor

Date: 2026-07-25

## Decision

Freeze one new candidate:

**TRACER-4H — Tri-surface Relational Auction-Crowding Executor, four-hour
rolling target-position policy.**

TRACER asks whether an RLLM can infer a persistent BTC position from the
ordered relation among three causal but individually weak market surfaces:

1. Spot versus USD-M leadership and cash/perpetual sponsorship;
2. USD-M aggregate-trade participation, impact, and arrival topology; and
3. completed premium-index path and crowding pressure.

At each fixed four-hour decision it chooses exactly one next target:

```text
SHORT
FLAT
LONG
```

The target persists until the next decision. Deterministic code owns source
clocks, numeric transforms, trailing ranks, token construction, transaction
costs, realized funding, reward, execution, and strict drawdown. The model sees
only the frozen categorical relation language, its previous two states, and the
current target.

This commit authorizes only a source/token support implementation. It does not
authorize execution OHLC, funding cash-flow evaluation, future return, reward,
model training, policy selection, or profitability claims.

## New identity and prior-work boundary

TRACER is not a repair of SABLE-8, CARTA, CSPR, LURI, CATCH, BAFR, PIVOT, or
BCTP.

- SABLE was an eight-hour funding-boundary process over fused OHLCV, OI,
  funding, premium, Kimchi, FX, and DXY. It retired when its fresh-OI source
  invariant failed. TRACER contains no OI, macro, Kimchi, funding-boundary
  clock, or SABLE token.
- CARTA was a sparse event-conditioned `ABSTAIN/FOLLOW/FADE` contextual bandit
  over one aggregate-trade state. TRACER is a dense action-complete rolling
  target-position MDP over three independently timestamped surfaces.
- CSPR, LURI, and CATCH imposed fixed event sides and fixed holds. TRACER has no
  event side, entry gate, fixed directional rule, TP, SL, or one-shot hold.
- PIVOT used paired equal-notional intrinsic clocks and failed a 2023 source
  coverage gate. TRACER uses a fixed UTC clock and no intrinsic-volume anchor.
- BCTP used delayed Bitcoin block-topology states. TRACER uses exchange auction
  and crowding relations and shares no feature or clock.

Repository search before this boundary found no existing training evaluator
that references all three frozen TRACER source families. Prior source and
market outcomes elsewhere in the repository mean this is not a pristine
source-family discovery. The claim under test is candidate-specific
composition and annual transfer, not clean-room global novelty.

A failed TRACER identity may not be renamed after threshold, token, clock,
reward, model, or outcome inspection. A successor requires a new mechanism,
source boundary, and preregistration.

## Frozen source containers

The support runner must use exact physical projections and may not load all
columns and drop forbidden values afterward.

### Surface A — Spot/USD-M minute leadership

```text
path
  data/binance_cross_venue_minute_leadership_btc_2020_2023/
  BTCUSDT_cross_venue_minute_leadership_5m_2020-01_2023-12.csv.gz
SHA256
  00ab6a55fc7bfeb3012584db5bc97a7d7b98dd995491acfd3f865c6bd41f92cc
header SHA256
  b7c730d6fc2c37d6e94f6a436478fd09ff42d15d7fd81bf521c4ca36465ff49f
manifest
  data/binance_cross_venue_minute_leadership_btc_2020_2023/build_manifest.json
manifest SHA256
  544c2945a2b56be478a1edc4abbb93b762bda5afc32cbd0658dd6822ff6b70fa
```

Exact allowlist and order:

```text
date
feature_available_time_utc
spot_quote_notional
um_quote_notional
spot_signed_quote_notional
um_signed_quote_notional
spot_to_um_lagged_flow_response_bp
um_to_spot_lagged_flow_response_bp
open_basis_bp
close_basis_bp
source_complete
cross_venue_feature_valid
```

### Surface B — USD-M aggregate-trade topology

```text
path
  data/binance_um_aggtrade_microstructure_btc_2020_2023/
  BTCUSDT_aggtrade_5m_2020-01-01_2023-12-31.csv.gz
SHA256
  c2bb0e6742f8cdc4e13315e7f0a13d6ab9cd536fb40d9cb4484b7a6ba30131cf
header SHA256
  fbdbd489b8d0b01262a8f8c73f19ea0ecf4dfb0de86040c1f2933e0374ea2507
manifest
  data/binance_um_aggtrade_microstructure_btc_2020_2023/build_manifest.json
manifest SHA256
  6eec40460a6146c58994e52f1af9ace4eecc0c085887d97af5ef17c30b9f7e73
```

Exact allowlist and order:

```text
date
first_transact_time_ms
last_transact_time_ms
agg_trade_count
quote_notional
signed_quote_notional
micro_log_return
event_notional_hhi
normalized_effective_event_count
sign_flip_rate
max_same_sign_run_share
interarrival_mean_ms
interarrival_burstiness
```

`micro_log_return` is a completed same-window causal source feature; it is not
a post-decision outcome.

### Surface C — USD-M premium path

```text
path
  data/binance_um_premium_path_btc_2020_2026/
  BTCUSDT_premium_path_1m_2020-01-01_2026-06-30.csv.gz
SHA256
  7fbaae1f85482b9fc9e148af357c7315e4e7fc4b4e3ae36c31f27545109f8aa9
header SHA256
  8efbf5700dc24aadf216da08b3c74712ceaec0b1a52e21ef01a877b5fbe26274
```

Exact allowlist and order:

```text
date
source_close_time
feature_available_time
source_valid
premium_open
premium_high
premium_low
premium_close
```

The premium loader must stream chronologically and stop at the first row whose
`date >= 2024-01-01T00:00:00Z` before converting any numeric field from that
row. It must materialize a deterministic gzip cut (`mtime=0`) under:

```text
data/tracer4_source_cuts/pre2024/premium.csv.gz
```

Surface A and B are already physically bounded before 2024. Their exact
projected cuts must be materialized beside the premium cut:

```text
data/tracer4_source_cuts/pre2024/leadership.csv.gz
data/tracer4_source_cuts/pre2024/aggtrade.csv.gz
results/tracer4_source_cut_manifest_pre2024_2026-07-25.json
```

No support process may decode an execution kline, funding rate, future return,
label, target, reward, prior model prediction, PnL, CAGR, MDD, portfolio weight,
or post-2023 numeric source value.

## Frozen four-hour process clock

Canonical boundaries are:

```text
00:00 UTC
04:00 UTC
08:00 UTC
12:00 UTC
16:00 UTC
20:00 UTC
```

For boundary `B`:

```text
source window          [B-4h, B)
latest five-minute bar B-5m, complete at B
five-minute cutoff     B
premium cutoff         B+61s
unified state complete B+61s
policy decision        B+5m
execution              B+10m at the USD-M five-minute open
next execution         B+4h+10m
```

Only Surface-A rows with `feature_available_time_utc <= B`, Surface-B rows
whose `last_transact_time_ms < B`, and Surface-C rows with
`feature_available_time <= B+61s` may contribute. Every contributing `date`
must lie in `[B-4h,B)`. The premium source contract fixes availability to
`date+61s`; therefore the final row at `date=B-1m` is available at `B+1s`.
The additional sixty-second margin is conservative and cannot admit a row
whose `date >= B`.

The policy is queried at every source-ready boundary. A boundary with an
invalid core surface or insufficient trailing rank history executes the
safety target `FLAT`; it remains in wall-clock accounting and is not silently
removed. There is no event filter, cooldown, overlap selector, queued signal,
TP, SL, or dynamic hold.

At each evaluation-year boundary:

- target starts flat;
- causal pre-year source context may build the first state;
- equity and high-water mark start at one;
- no reward transition crosses the year end;
- the final target is flattened at the exact year-end five-minute open; and
- funding stamped at the year end belongs to the position immediately before
  flattening.

## Frozen causal primitive and rank rules

Each boundary aggregates only its exact four-hour source window. All sums,
weighted means, signs, ratios, and transitions are deterministic.

A core-valid boundary contains exactly 48 ordered Surface-A rows, 48 ordered
Surface-B rows, and 240 ordered Surface-C rows. Every Surface-A row must set
both validity flags, every Surface-C row must set `source_valid`, all projected
numeric cells must be finite, every notional/count/range field that is
mathematically nonnegative must be nonnegative, and all three surfaces must
span the exact half-open source window without duplicate timestamps. Otherwise
the boundary is source-invalid.

Surface A additionally requires each absolute signed notional not to exceed its
paired quote notional. Surface B requires absolute signed notional not to exceed
quote notional, positive aggregate-trade count, `event_notional_hhi` in
`[0,1]`, positive effective participation, `sign_flip_rate` and
`max_same_sign_run_share` in `[0,1]`, nonnegative interarrival mean, and finite
burstiness. Surface C requires
`premium_high >= max(premium_open,premium_close)`,
`premium_low <= min(premium_open,premium_close)`, and finite OHLC; premium may
be negative and is not subject to a positivity rule.

For a core-valid boundary define:

```text
cash_flow
  sum(spot_signed_quote_notional)

leverage_flow
  sum(um_signed_quote_notional) from Surface A

auction_flow
  sum(signed_quote_notional) from Surface B

auction_return
  sum(micro_log_return) from Surface B

sponsor_score
  arithmetic mean(
    spot_to_um_lagged_flow_response_bp
    - um_to_spot_lagged_flow_response_bp
  )

participation_hhi
  median(event_notional_hhi)

effective_participation
  median(normalized_effective_event_count)

flow_flip
  median(sign_flip_rate)

flow_run
  median(max_same_sign_run_share)

arrival_burst
  median(interarrival_burstiness)

arrival_wait
  median(interarrival_mean_ms)

basis_change
  last(close_basis_bp) - first(open_basis_bp)

premium_change
  last(premium_close) - first(premium_open)

premium_range
  max(premium_high) - min(premium_low)
```

The immediately prior state is the prior canonical boundary, including an
invalid boundary represented by the safety token line. Transitions do not skip
calendar time to find a more convenient prior valid observation.

Continuous scale-sensitive primitives are converted with a strictly prior
rolling empirical reference:

```text
reference length      540 prior valid boundaries = 90 days
minimum observations  360
cut points            q33 and q67
current row included  false
tie rule              LOW: x<=q33; MID: q33<x<=q67; HIGH: x>q67
```

No full-sample, future-year, centered, expanding-from-future, or outcome-based
rank is allowed. Sign dead zones use no optimized numeric threshold: a signed
primitive is `NEUTRAL` only when it is exactly zero or both opposing source
notionals are zero; otherwise its mathematical sign owns direction.

The exact primitive families are:

1. cash-to-perpetual lagged response difference;
2. cash/perpetual signed-flow pair;
3. USD-M signed flow and same-window price response;
4. aggregate-event concentration and effective participation;
5. event sign persistence and arrival burstiness;
6. Spot/USD-M basis change;
7. premium close change and premium intrawindow range; and
8. transition of each relation from the immediately prior canonical boundary.

A source-invalid boundary never contributes to a future rolling reference.
Append replay must reproduce every already formed primitive, rank, token, and
availability byte-for-byte.

## Frozen categorical relation language

The model receives three ordered state lines (`t-2`, `t-1`, `t`) plus current
position. Exact field order:

```text
sponsor
flow_consensus
impact_relation
participation
flow_persistence
auction_tempo
premium_price_relation
basis_premium_relation
sponsor_transition
impact_transition
crowding_transition
```

Exact vocabularies:

```text
sponsor
  CASH_LEADS | LEVERAGE_LEADS | BALANCED

flow_consensus
  CONSENSUS_BUY | CONSENSUS_SELL |
  CASH_BUY_LEVERAGE_SELL | CASH_SELL_LEVERAGE_BUY | FLOW_NEUTRAL

impact_relation
  BUY_FOLLOWTHROUGH | BUY_ABSORBED | SELL_FOLLOWTHROUGH |
  SELL_ABSORBED | RESPONSE_NEUTRAL

participation
  BROAD | MIXED | CONCENTRATED

flow_persistence
  PERSISTENT | ROTATING | MIXED

auction_tempo
  BURST | STEADY | SLOW

premium_price_relation
  CROWDING_CONFIRMS_UP | CROWDING_CONFIRMS_DOWN |
  PREMIUM_DIVERGES_FROM_UP | PREMIUM_DIVERGES_FROM_DOWN | PREMIUM_NEUTRAL

basis_premium_relation
  BOTH_EXPAND | BOTH_COMPRESS | BASIS_ONLY | PREMIUM_ONLY |
  CROSS_DISAGREE | BOTH_NEUTRAL

sponsor_transition
  STABLE_CASH | STABLE_LEVERAGE | ROTATED_TO_CASH |
  ROTATED_TO_LEVERAGE | SPONSOR_MIXED

impact_transition
  FOLLOWTHROUGH_PERSISTS | ABSORPTION_PERSISTS |
  FOLLOWTHROUGH_TO_ABSORPTION | ABSORPTION_TO_FOLLOWTHROUGH |
  IMPACT_MIXED

crowding_transition
  CROWDING_BUILDS | CROWDING_RELEASES | CROWDING_FLIPS | CROWDING_STABLE

current_position
  SHORT | FLAT | LONG
```

Exact token mappings:

```text
sponsor
  sponsor_score HIGH -> CASH_LEADS
  sponsor_score LOW  -> LEVERAGE_LEADS
  otherwise          -> BALANCED

flow_consensus
  sign(cash_flow)=+ and sign(leverage_flow)=+ -> CONSENSUS_BUY
  sign(cash_flow)=- and sign(leverage_flow)=- -> CONSENSUS_SELL
  sign(cash_flow)=+ and sign(leverage_flow)=- -> CASH_BUY_LEVERAGE_SELL
  sign(cash_flow)=- and sign(leverage_flow)=+ -> CASH_SELL_LEVERAGE_BUY
  otherwise                                      FLOW_NEUTRAL

impact_relation
  auction_flow>0 and auction_return>0  -> BUY_FOLLOWTHROUGH
  auction_flow>0 and auction_return<=0 -> BUY_ABSORBED
  auction_flow<0 and auction_return<0  -> SELL_FOLLOWTHROUGH
  auction_flow<0 and auction_return>=0 -> SELL_ABSORBED
  auction_flow=0                        -> RESPONSE_NEUTRAL

participation
  participation_hhi HIGH and effective_participation LOW -> CONCENTRATED
  participation_hhi LOW and effective_participation HIGH -> BROAD
  otherwise                                               -> MIXED

flow_persistence
  flow_flip LOW and flow_run HIGH -> PERSISTENT
  flow_flip HIGH and flow_run LOW -> ROTATING
  otherwise                       -> MIXED

auction_tempo
  arrival_burst HIGH                         -> BURST
  arrival_burst not HIGH and arrival_wait HIGH -> SLOW
  otherwise                                  -> STEADY

premium_price_relation
  auction_return>0 and premium_change>0 -> CROWDING_CONFIRMS_UP
  auction_return<0 and premium_change<0 -> CROWDING_CONFIRMS_DOWN
  auction_return>0 and premium_change<0 -> PREMIUM_DIVERGES_FROM_UP
  auction_return<0 and premium_change>0 -> PREMIUM_DIVERGES_FROM_DOWN
  otherwise                             -> PREMIUM_NEUTRAL

basis_premium_relation
  basis_change>0 and premium_change>0 -> BOTH_EXPAND
  basis_change<0 and premium_change<0 -> BOTH_COMPRESS
  basis_change!=0 and premium_change=0 -> BASIS_ONLY
  basis_change=0 and premium_change!=0 -> PREMIUM_ONLY
  basis_change*premium_change<0         -> CROSS_DISAGREE
  otherwise                             -> BOTH_NEUTRAL

sponsor_transition
  prior/current both CASH_LEADS     -> STABLE_CASH
  prior/current both LEVERAGE_LEADS -> STABLE_LEVERAGE
  current CASH_LEADS and prior differs     -> ROTATED_TO_CASH
  current LEVERAGE_LEADS and prior differs -> ROTATED_TO_LEVERAGE
  otherwise                                -> SPONSOR_MIXED

impact_transition
  prior/current both FOLLOWTHROUGH -> FOLLOWTHROUGH_PERSISTS
  prior/current both ABSORBED      -> ABSORPTION_PERSISTS
  prior FOLLOWTHROUGH, current ABSORBED -> FOLLOWTHROUGH_TO_ABSORPTION
  prior ABSORBED, current FOLLOWTHROUGH -> ABSORPTION_TO_FOLLOWTHROUGH
  otherwise                            -> IMPACT_MIXED

crowding_transition
  sign(premium_change) flips from nonzero to opposite nonzero -> CROWDING_FLIPS
  same nonzero sign and premium_range rank rises              -> CROWDING_BUILDS
  same nonzero sign and premium_range rank falls              -> CROWDING_RELEASES
  otherwise                                                   -> CROWDING_STABLE
```

`FOLLOWTHROUGH` means either directional `*_FOLLOWTHROUGH` token and
`ABSORBED` means either directional `*_ABSORBED` token. A source-invalid state
uses the exact safety line:

```text
SOURCE_INVALID|FLOW_NEUTRAL|RESPONSE_NEUTRAL|MIXED|MIXED|STEADY|
PREMIUM_NEUTRAL|BOTH_NEUTRAL|SPONSOR_MIXED|IMPACT_MIXED|
CROWDING_STABLE
```

`SOURCE_INVALID` is an additional safety-only `sponsor` value. It may never be
emitted for a core-valid boundary or selected as a learned value.
Rank comparisons use the frozen order `LOW < MID < HIGH`. Category-incidence,
signature, and Jensen-Shannon support statistics exclude source-invalid and
rank-warm-up safety lines; their availability is governed separately by the
core-valid and sequence-ready gates.

If the current boundary is source-invalid, all eleven fields use the safety
line above. If the current boundary is core-valid but the immediately prior
canonical boundary is source-invalid or rank-unready, the eight current-state
fields are computed normally while:

```text
sponsor_transition  = SPONSOR_MIXED
impact_transition   = IMPACT_MIXED
crowding_transition = CROWDING_STABLE
```

No transition skips backward to a prior valid state.

No raw value, rank percentile, timestamp, date, price, return magnitude,
funding value, reward, previous reward, action probability, equity, drawdown,
or portfolio statistic may enter the prompt.

## Frozen source-support gates

Source support is outcome-blind and conjunctive. All must pass:

1. Exact paths, SHA-256 values, physical headers, allowlists, manifests, source
   clocks, types, monotonicity, positivity, and finite-value rules reproduce.
2. Exact UTC join rows exist for at least 99.0% of canonical five-minute rows
   in every calendar year 2020–2023.
3. At least 95.0% of the 2,190/2,196 nominal annual boundaries are core-source
   valid in every year.
4. Sequence-ready boundaries are at least 1,500 in 2020 and 2,000 in each of
   2021, 2022, and 2023; every quarter after warm-up has at least 450.
5. Forced-`FLAT` source-invalid share is at most 5.0% in every year.
6. Every one of the eleven source-relation fields has at least two categories
   with at least 5.0% annual share in every year; no one category exceeds
   90.0%.
7. `flow_consensus` has at least 15.0% aggregate buy support and 15.0% aggregate
   sell support in every year.
8. `impact_relation` has at least 10.0% follow-through and 10.0% absorption
   support in every year.
9. `sponsor` has at least 10.0% `CASH_LEADS` and 10.0%
   `LEVERAGE_LEADS` support in every year.
10. Each year's exact eleven-field signatures contain at least 300 distinct
    values and no signature exceeds 5.0% share.
11. Jensen-Shannon divergence for every field between adjacent years is at
    most 0.25 using the complete frozen vocabulary and base-2 logarithms.
12. Three deterministic controls are not identical to the primary token
    sequence: Surface C shifted exactly 1,440 minutes stale, paired cash and
    perpetual Surface-A columns swapped before aggregation, and Surface-B rows
    circularly rotated forward by exactly 37 five-minute rows independently
    inside each UTC calendar month.
13. Prefix reconstruction and later-row append replay leave all already formed
    states byte-identical.
14. No forbidden source or outcome counter is nonzero.

A gate-6 through gate-11 denominator contains exactly the sequence-ready,
core-valid states in the named UTC calendar year. Safety lines and
`current_position` are excluded. Signature rows contain exactly the eleven
source-relation fields. For Jensen-Shannon divergence, absent vocabulary values
receive probability zero, `M=(P+Q)/2`, logarithms are base two, and every
`0*log2(0/M)` term is defined as zero.

A support failure retires TRACER-4H unchanged. Gates may not be relaxed and
controls may not replace the primary.

## Contingent economic chronology

Only a complete source-support pass may authorize a separately committed
Stage 0.5 reward/evaluator freeze. Before any future return or reward is built,
Stage 0.5 must bind:

- every policy family, algorithm, feature mask, hyperparameter, seed, fit
  window, checkpoint, tie-break, invalid-output action, and schedule schema;
- exact one-step reward, Bellman target, discount, terminal rule, transaction
  cost, realized-funding convention, strict-MDD path order, and delay stress;
- exact 2021 selection statistic, algorithm-selection order, familywise test,
  promotion gate, and report schema;
- every comparator path and novelty calculation;
- source/evaluator/test hashes and clean committed runner revision; and
- write-once schedule/report manifests for each annual transaction.

No algorithm grid, checkpoint choice, threshold, or selection statistic may be
introduced after a reward is constructed. The intended chronology, which
Stage 0.5 may narrow but not weaken, is:

```text
fit 2020 only
seal every 2021 target schedule before 2021 outcomes
open 2021 test once
if and only if the full frozen gate passes:
  freeze one selected algorithm id
  refit that algorithm on 2021 only
  seal 2022 schedule before 2022 outcomes
  open 2022 eval once
if and only if the full frozen gate passes:
  refit the same algorithm on 2022 only
  seal 2023 schedule before 2023 outcomes
  open 2023 confirmation once
```

No failed year may be repaired with a later year. The selected algorithm may
not change after 2021. Each annual refit is rolling one-year prior-only,
not expanding, monthly, or outcome-weighted.

Execution economics, to be bound before the first reward is built:

- fixed `0.5x` target exposure;
- `6 bp` base and `10 bp` stress cost per changed unit of target notional;
- exact realized funding with no synthetic weekend fill;
- +5-minute entry-delay sensitivity;
- full-calendar absolute return and CAGR including flat time;
- held-path favorable-first/adverse-second strict MDD;
- terminal flattening;
- weekly Monday-UTC clustered familywise max-stat inference; and
- exact always-flat, always-long, always-short, direction-flip, stale-premium,
  sponsorship-swap, token-mask, reward-shuffle, and circular-reward controls.

The minimum transfer gate is positive base/stress/delay return, base
CAGR/strict-MDD at least `3.0`, stress and delay ratios at least `2.5`, strict
MDD at most `15%`, at least 120 non-flat intervals, at least 20% long and 20%
short share, positive return in both calendar halves, required-control defeat,
and familywise `pmax < 0.10`.

Mandatory score-bearing killer baselines and ablations are:

```text
always_flat
always_long
always_short
current_state_only
reversed_three_state_order
sorted_three_state_order
surface_a_only
surface_b_only
surface_c_only
no_surface_a
no_surface_b
no_surface_c
categorical_linear_fqi
categorical_ridge_fqi
extra_trees_fqi
scratch_sequence_model
exact_signature_memory
reward_shuffle
circular_21_reward
```

The primary must beat every single-surface, masked-surface, order-destruction,
memory, and shuffled-reward baseline on the frozen minimum transfer score. A
Gemma policy may not be promoted merely because aggregate-trade tokens were
already useful in CARTA train or because another primary failed.

Before the 2021 schedule is sealed, Stage 0.5 must also bind committed
candidate-clock comparators from CARTA, CSPR, LURI, CATCH, and any available
PIVOT source clock. No comparator outcome is needed. On active comparator
timestamps, no single prior comparator may reproduce more than 80% of
TRACER's non-flat target signs. After mapping each comparator's scheduled
exposure to the fixed four-hour intervals and filling inactive intervals with
zero, the absolute Pearson correlation with TRACER targets must also remain
below 0.80. A constant series has correlation zero by definition. Failure is a
novelty retirement before opening 2021 outcomes.

## RLLM role

The later evaluator may compare frozen cheap relational policies with one
compact Gemma 4 policy, but Gemma is not allowed to see raw numeric features.
The language model's intended advantage is compositional deduction across
ordered relation changes and current position, not arithmetic, threshold
selection, or direct exchange execution.

Any Gemma run requires a separate pre-training freeze fixing base revision,
quantization, adapter shape, optimizer steps, checkpoint selection, prompt,
parser, invalid-output action, data chronology, pair construction, and memory
limits. A cheap-policy failure does not automatically authorize Gemma, and a
Gemma pass cannot override failed source, execution, control, or statistical
gates.

## Stop rule

Stop immediately and record immutable rejection if any source hash, schema,
clock, support, append-invariance, contamination, economic, control, or
familywise gate fails. Do not inspect a later year's outcome after failure.
Do not use a failed result to change TRACER's source, clock, rank window,
category vocabulary, action space, rolling-fit rule, costs, risk accounting,
thresholds, model family, or promotion gate.

## Pre-support structural correction record

After the initial boundary commit and before source incidence, token
distribution, future return, reward, or outcome access, review against the
already committed premium builder and artifact tests found that premium
availability is exactly `date+61s`, not `date+60s`. One read-only review also
checked a single timestamp row and no feature magnitude. This amendment moves
only the source completion cutoff to `B+61s`, removes projected columns that no
frozen token consumes, makes invalid-prior transitions total, corrects raw
header hashes to include the terminating LF byte, and adds the missing Stage
0.5 and killer-baseline constraints. No incidence threshold, category,
direction, reward, market outcome, funding value, PnL, CAGR, or MDD informed
these corrections.

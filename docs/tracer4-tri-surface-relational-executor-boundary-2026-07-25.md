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
  ed7949cad9ee5396f110678ff85fad19a12466f99195a6e585acef6e2c46f242
manifest
  data/binance_cross_venue_minute_leadership_btc_2020_2023/build_manifest.json
```

Exact allowlist and order:

```text
date
feature_available_time_utc
spot_quote_notional
um_quote_notional
spot_signed_quote_notional
um_signed_quote_notional
spot_flow_coherence
um_flow_coherence
spot_log_return_5m
um_log_return_5m
spot_activity_time_centroid
um_activity_time_centroid
spot_flow_time_centroid
um_flow_time_centroid
spot_to_um_lagged_flow_response_bp
um_to_spot_lagged_flow_response_bp
flow_transfer_asymmetry
return_leadership_asymmetry
simultaneous_flow_sign_agreement
simultaneous_return_sign_agreement
open_basis_bp
close_basis_bp
basis_change_bp
source_complete
cross_venue_feature_valid
```

The same-bar returns are causal source features after
`feature_available_time_utc`; they are not post-decision outcomes.

### Surface B — USD-M aggregate-trade topology

```text
path
  data/binance_um_aggtrade_microstructure_btc_2020_2023/
  BTCUSDT_aggtrade_5m_2020-01-01_2023-12-31.csv.gz
SHA256
  c2bb0e6742f8cdc4e13315e7f0a13d6ab9cd536fb40d9cb4484b7a6ba30131cf
header SHA256
  224a65d10568adc024f66022bf47344ac783691a18e09b6ef773eb0946bfbb5f
manifest
  data/binance_um_aggtrade_microstructure_btc_2020_2023/build_manifest.json
```

Exact allowlist and order:

```text
date
first_transact_time_ms
last_transact_time_ms
agg_trade_count
underlying_trade_count
quote_notional
buy_quote_notional
sell_quote_notional
signed_quote_notional
flow_coherence
micro_log_return
signed_price_response
event_notional_hhi
normalized_effective_event_count
underlying_trades_per_agg_event
signed_event_imbalance
sign_flip_rate
mean_same_sign_run_length
max_same_sign_run_share
interarrival_mean_ms
interarrival_std_ms
interarrival_burstiness
buy_sell_event_size_log_ratio
```

### Surface C — USD-M premium path

```text
path
  data/binance_um_premium_path_btc_2020_2026/
  BTCUSDT_premium_path_1m_2020-01-01_2026-06-30.csv.gz
SHA256
  7fbaae1f85482b9fc9e148af357c7315e4e7fc4b4e3ae36c31f27545109f8aa9
header SHA256
  dae25782c3e4478e1d60323957af7ffb2cb09fa8d72297e5d9a48e7216614076
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
state cutoff           B
policy decision        B+5m
execution              B+10m at the USD-M five-minute open
next execution         B+4h+10m
```

Only Surface-A rows with `feature_available_time_utc <= B`, Surface-B rows
whose `last_transact_time_ms < B`, and Surface-C rows with
`feature_available_time <= B` may contribute. Every contributing timestamp
must lie in `[B-4h,B)` by its source observation clock.

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
4. response per absolute signed notional;
5. aggregate-event concentration and effective participation;
6. event sign persistence and arrival burstiness;
7. Spot/USD-M activity and flow centroid difference;
8. Spot/USD-M basis change;
9. premium close change and premium intrawindow range; and
10. transition of each relation from the immediately prior canonical boundary.

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
6. Every field has at least two categories with at least 5.0% annual share in
   every year; no one category exceeds 90.0%.
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

A support failure retires TRACER-4H unchanged. Gates may not be relaxed and
controls may not replace the primary.

## Contingent economic chronology

Only a complete source-support pass may authorize a separately committed
reward/evaluator freeze. The intended chronology, which that later freeze may
narrow but not weaken, is:

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

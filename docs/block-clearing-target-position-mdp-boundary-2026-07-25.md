# BCTP-12H boundary — block-clearing target-position RLLM

## Decision

Freeze one new candidate identity:

**BCTP-12H — Block-Clearing Target-Position RLLM**.

BCTP is a continuous target-position Markov decision process over causally
delayed Bitcoin block-clearing topology. At each source-state release, one
compact policy chooses exactly one target:

```text
TARGET_SHORT
TARGET_FLAT
TARGET_LONG
```

The target remains active until the next source-state release or the declared
split end. This is not an event sleeve with an isolated fixed hold.

This boundary opens no new Bitcoin-ledger row, BCTP sequence, market row,
funding row, reward, label, PnL, or post-2023 value. The exact source loader,
sequence builder, support gates, accounting, offline-RL method, Gemma snapshot,
and evaluator must be committed before the corresponding data class is opened.

## Why BCRT-72 stays retired

`BCRT-72` remains terminally retired. It reserved sparse, action-independent
six-hour trade intervals and required every interval to be contained in a
calendar split. Its only source-support failure was a preregistered
three-calendar-day maximum entry-gap rule: the conservative source delay plus
year-boundary containment produced deterministic gaps of roughly five days.
No BCRT market outcome or model was opened.

BCTP does not:

- increase BCRT's failed gap limit;
- release a BCRT reservation;
- shorten BCRT's 288-successor closure or 48-hour embargo;
- alter a BCRT primitive, rank, relation token, or source value;
- reuse the `BCRT-72` identity; or
- report a repaired BCRT statistic.

BCTP changes the predictive and execution object:

- every causally released topology state is a decision;
- actions are persistent target positions, not isolated six-hour trades;
- current sleeve position is part of the MDP state;
- source chronology remains exactly BCRT-continuous; BCTP changes the
  execution object by resetting sleeve position and accounting at declared
  split boundaries while carrying every blackout as idle cash;
- split boundaries force the sleeve flat and idle time remains in the
  denominator;
- the next exogenous source state is unaffected by the action; and
- transaction cost depends on target-position change.

The known boundary blackout is therefore reported as idle cash, not recast as
missing source data and not made into a pass/fail support threshold.

## Research-history disclosure

The source family is not globally clean-room:

- several earlier base-chain mechanisms exposed source-support evidence;
- BATE opened and failed pre-2023 economics;
- BCRT opened exact source integrity, aggregate token support, and source-state
  counts; and
- BCRT's immutable support artifact disclosed 2,918 formed buckets, 2,792
  rank-complete states, and 2,791 token-ready states.

No BCTP-specific sequence, position-conditioned state, reward, policy,
checkpoint, return, or 2023 market outcome has been opened. BCTP can establish
only candidate-specific sequential evidence.

## Immutable source representation

BCTP reuses the source validation, causal bucket, primitive, rank, and
twelve-token grammar of BCRT as an immutable source representation. Reuse is
permitted only after byte-for-byte replay against all bindings below.

Primary source:

```text
data/bitcoin_utxo_fee_block_stats_2020_2023.csv.gz
SHA256 8d5160b59bd104819adb34752f2bd5b01e9c07d7aa108188abcd9e6d323d102f
header SHA256 66cefecff20d70a4229285fc5b93a7cd6126dfd53173acc9c8bffe805638c342
```

Source manifest:

```text
results/bitcoin_utxo_fee_block_stats_source_manifest_2026-07-20.json
file SHA256 ceb096b7bfeaf309729c4cab9f5cd4d2e0d526e261b1e460e9d03cd4f24e8084
manifest_hash 98a84b0bd0338300f62eaa047b87498cc5a8d9505a03f6bd1912d1deb9564e8c
```

Independent basic-field reference:

```text
data/bitcoin_block_summaries_2020_2023.csv.gz
SHA256 1f8d1c0153717f2c18d8fa6f09428c780a850b2aecc8fb42bc497e16e68e1833
header SHA256 afb4c6b31bc7909918b78f60ef1e14b9f59bd2e619f2c297fe1b3ce31f02d2fe
```

BCRT source-support artifact:

```text
results/block_clearing_relational_topology_support_2026-07-24.json
SHA256 9ccccf7a3176fcf86baddacb65c11bbde78ea73ed7ab18d3594b0e6327567055
manifest_hash e2b2d7301d204043f2df33f4453da82112fb5db7bfb9aed66a74bee6ec76932b
```

The exact source allowlist remains:

```text
height
id
previousblockhash
timestamp
mediantime
tx_count
size
weight
total_fees
total_inputs
total_outputs
utxo_set_change
```

Load-and-drop is forbidden. A BCTP source-support implementation must replay
all BCRT path, hash, header, schema, integer, height, parent-link, reference,
UTXO-identity, prefix-closure, clock, rank, token, and future-append
invariants. No market or outcome column may be decoded in this stage.

## Immutable causal source clock

BCTP retains BCRT's source clock exactly:

1. twelve-hour UTC source bucket;
2. median-time boundary anchor;
3. exact 288th successor for immutable prefix closure;
4. maximum prefix timestamp/median-time clock;
5. additional 48 elapsed hours;
6. upward five-minute rounding;
7. one complete five-minute inference/order buffer; and
8. live availability equal to the later of this historical clock and actual
   Bitcoin Core receipt/validation time.

The resulting `entry_time` is a **decision-transition time**, not the start of
a fixed six-hour BCRT trade. BCTP consumes every token-ready state in strict
chronological order.

If multiple source states share one entry time, only the state with the latest
bucket start and then greatest confirmation height is actionable. Earlier
same-release states remain causal sequence predecessors. This deterministic
batch rule must be frozen before real BCTP sequence incidence is constructed.

## Immutable source tokens

The eight primitives and twelve current-state relation tokens remain exactly
those frozen for BCRT:

```text
cadence
utilization
packing
fee_burden
utxo_pressure
witness_discount
load_dispersion
fee_dispersion
```

```text
cadence_utilization
utilization_fee
packing_witness
utxo_fee
load_fee_dispersion
high_leader
low_leader
rank_breadth
extreme_occupancy
relation_breadth
order_transition
leader_transition
```

Each primitive rank uses only the at-most-252 immediately prior source-valid
buckets and requires 126. BCTP may not change any primitive, threshold,
category, tie rule, history, or vocabulary after the prior source-support
distribution is known.

## New MDP state and prompt

One BCTP decision state consists of:

1. the twelve source tokens at the current release `S0`;
2. the same ordered tokens from the immediately prior actionable release
   `S-1`;
3. the same ordered tokens from the release before that `S-2`; and
4. current BCTP sleeve position:
   `POSITION_SHORT | POSITION_FLAT | POSITION_LONG`.

The first two actionable releases are sequence warm-up and cannot trade.
Suppressed same-release states remain source predecessors but are not
actionable predecessors.

The policy prompt may contain only:

- one frozen task identifier;
- the 36 categorical source-token lines, labeled only by relative sequence
  position;
- one categorical position line; and
- neutral action codes whose permutation is evaluated as a mandatory control.

Forbidden model inputs include:

```text
raw source values or numeric ranks
timestamp, date, year, month, weekday, height, id, or row identity
BTC price, return, funding, premium, OI, Kimchi, DXY, or future path
split identity, elapsed calendar label, prior reward, PnL, CAGR, or MDD
another alpha's signal, side, state, action, or outcome
free-form analyzer output, chain of thought, generated feature, or arithmetic
```

This is one policy model, not an analyzer/trader pair. Deterministic code owns
source validation, tokenization, position state, order size, costs, funding,
reward, strict MDD, and exchange execution.

## Target-position transition semantics

Let target exposure be:

```text
TARGET_SHORT = -0.5 account gross
TARGET_FLAT  =  0.0
TARGET_LONG  = +0.5 account gross
```

At an actionable entry time:

1. observe the frozen three-state token sequence and current sleeve position;
2. infer one target;
3. rebalance at that exact five-minute open;
4. keep the target until the next actionable entry; and
5. rebalance again from the current target to the next target.

The action cannot alter, release, delay, or create a later source state.
Model failure, unknown vocabulary, stale live source, missing market bar,
non-finite score, or order infeasibility targets `FLAT`.

For every declared evaluation split:

- start flat;
- define `terminal_flat_time = split_end-5m`;
- permit only entry times inside `[start,terminal_flat_time)`;
- include source history before `start` only as causal token warm-up;
- force flat at the open of `terminal_flat_time` and remain flat through the
  final complete bar;
- charge entry, transition, and terminal flattening costs;
- include all idle time in absolute return and CAGR; and
- do not reset source ranks or source-token chronology at the boundary.

This boundary-aware rule is fixed before BCTP sequence decoding. It neither
deletes nor fills the known source-delay blackout around calendar boundaries.

## Frozen chronological roles

```text
algorithm fit             2020
algorithm transfer gate   2021
final cheap-policy fit    2020-2021
Gemma/checkpoint select   2022
untouched candidate eval  2023
sealed source/outcome     2024+
```

No 2023 outcome may choose a feature, sequence length, reward, algorithm,
hyperparameter, prompt, model, checkpoint, action threshold, cost, or control.
No 2024+ source or outcome may be opened unless the unchanged 2023 gate passes.

The already published BCRT 2023 source-token report is prior contamination and
may be used only to enforce known-vocabulary fail-closed behavior. It may not
select a BCTP parameter.

## Required support stage

Before market or funding access, BCTP must:

- replay the exact raw source and independent reference bindings;
- reproduce BCRT's 2,918 buckets, 2,792 rank-complete states, and 2,791
  token-ready states byte-for-byte before BCTP sequencing;
- prove current and prior sequence tokens never change after future source
  append;
- prove every actionable entry is strictly increasing after same-release
  batching;
- prove the first two actionable releases are warm-up only;
- report, but not gate on, calendar-boundary idle gaps;
- retain at least 500 actionable sequence states in each of 2020, 2021, and
  2022;
- emit 2023 incidence only under a report-only namespace that cannot change a
  Boolean support decision, retirement, threshold, sequence rule, or
  parameter; its only operational effect is the already frozen fail-closed
  behavior for an unknown vocabulary value;
- retain all twelve active months in 2021 and 2022;
- reproduce every BCRT train/2022 marginal-token and vocabulary-support pass;
- show no exact 36-token **source** signature above 5% before any policy action
  is generated; position-conditioned concentration is evaluated only after a
  frozen policy exists and cannot gate source support; and
- emit no raw, rank, action, side, market, funding, reward, return, or PnL
  column.

Because BCRT source incidence is already known, this is an integrity and
sequence-support gate, not a new source-discovery claim.

## Economic and RL boundary

Market/funding accounting must be separately frozen before access. It must
use:

- exact Binance BTCUSDT perpetual five-minute opens and held OHLC paths;
- exact realized funding;
- 0.5 maximum gross exposure;
- 6 bp per notional side base cost and 10 bp stress cost;
- costs proportional to the absolute BTC-quantity change valued at execution;
- full-calendar CAGR including idle time;
- one global pre-entry high-water strict MDD;
- favorable then adverse held-bar ordering; and
- forced split-end flattening.

For consecutive actionable entries `t_i < t_{i+1}`, the position selected at
the open of `t_i` owns exactly the five-minute bars whose open times are in:

```text
[t_i, min(t_{i+1}, terminal_flat_time))
```

At `t_i`, let `E_pre`, `q_old`, `P`, `c`, and target `f` denote pre-rebalance
equity, old BTC quantity, exact open, per-side cost, and one of
`{-0.5,0,+0.5}`. For `f=0`, `q_new=0`. Otherwise, `q_new` is the unique
same-sign admissible solution with positive post-cost equity to:

```text
q_new*P / (E_pre - c*abs(q_new-q_old)*P) = f
```

The implementation must use a closed-form piecewise solution and test it
against an independent monotone root solver. If no admissible solution exists,
the only action is flat. Rebalancing cost is charged once on
`abs(q_new-q_old)*P`, including a same-sign target whose quantity changes.
There is no separate close-and-open double charge. This keeps post-cost gross
exposure at or below the exact target instead of silently exceeding 0.5x.

At an interior funding timestamp, the held quantity immediately before that
settlement receives the exact funding cash flow. At a timestamp equal to an
entry or rebalance boundary, compute funding cash under both the old and new
quantities and retain exactly:

```text
min(0, old_quantity_funding_cash, new_quantity_funding_cash)
```

Thus an ambiguous boundary credit is dropped, at most one debit is retained,
and backtest/live use the same convention. Funding is never counted twice.

At `terminal_flat_time`, flatten at the exact open of the final complete
five-minute bar and remain flat for that bar; no post-split market row may be
loaded. Charge one terminal cost on the remaining absolute quantity. A funding
settlement exactly at this boundary uses the same old-versus-flat conservative
minimum rule. Strict MDD marks every owned bar after its open, favorable
extreme first and adverse extreme second, then the rebalance or terminal-open
equity. These conventions must be copied byte-for-byte into base, stress,
delayed-entry, and control simulations.

The train-only per-transition utility may combine log account multiplier,
held-path downside, and turnover only after its coefficients are committed.
Future-best action replay is an oracle diagnostic, never policy performance.

The mandatory learner order is:

1. constants, persistence, exact-memory, categorical linear/ridge,
   Extra-Trees fitted-Q, shuffled-reward, circular-shift, current-only, and
   sequence-order controls;
2. one frozen full-information offline fitted-Q policy using only labels
   available in its fit interval;
3. only if cheap transfer gates pass, one compact Gemma policy;
4. Gemma optimization by a frozen offline RL objective over action-token
   probabilities, not supervised future-best target echo; and
5. one immutable 2023 evaluation.

Gemma may improve selection among the same three deterministic targets. It may
not alter source features, clocks, accounting, or rewards.

Every policy, control, action-code permutation, ablation, and Gemma checkpoint
opened in 2021 or 2022 belongs to one frozen development family. Selection must
use a shared weekly-cluster max-stat null with identical week-level Rademacher
signs across the whole family. Development reports must include both local
`p` and familywise `p_max`; omitted or failed policies remain in the family.
The immutable single 2023 policy uses the ordinary one-policy weekly-cluster
`p` because no 2023 family is selected.

## Minimum continuation and final gates

Before GPU training, an unchanged cheap policy must:

- have positive 2021 transfer absolute return;
- have 2021 CAGR/strict-MDD at least 1.0;
- remain positive under 10 bp stress and one extra five-minute delay;
- execute at least 80 non-flat target intervals;
- execute both long and short targets, each at least 20% of non-flat
  intervals; and
- beat shuffled, circular-shift, exact-memory, and strongest single-snapshot
  controls;
- have familywise weekly-cluster `p_max < 0.25` in 2021; and
- after the unchanged 2020-2021 refit, have `p_max < 0.10` in 2022.

After refit on 2020-2021, the same algorithm must meet the same conditions in
2022 with CAGR/strict-MDD at least 1.5 before Gemma is authorized.

The single 2022-selected Gemma checkpoint advances only if 2022 has:

- positive absolute return;
- CAGR/strict-MDD at least 3.0;
- strict MDD at most 15%;
- at least 100 non-flat intervals;
- both long and short at least 20% of non-flat intervals;
- positive H1 and H2 absolute return;
- positive 10 bp stress and delayed-entry returns; and
- familywise weekly-cluster `p_max < 0.05`.

The immutable 2023 candidate evaluation must independently satisfy the same
economic, risk, activity, direction, half-year, stress, and delay gates, with
ordinary one-policy weekly-cluster `p < 0.10` replacing development `p_max`.
A 2023 failure retires BCTP-12H unchanged. It cannot change the 2022 checkpoint
or open 2024+ for repair.

## Mandatory controls

Every economic report must include:

- always flat, always long, always short;
- previous-target persistence;
- exact-signature memory with unseen-state flat;
- current snapshot only;
- reversed temporal sequence;
- shuffled rewards;
- circularly shifted rewards;
- masked-source-token prior;
- neutral action-code permutation;
- one additional five-minute execution delay;
- 10 bp per-side cost stress;
- exact target-direction flip; and
- exact BCRT six-hour reservation clock as a non-promotable historical-shape
  comparator.

Controls diagnose mechanism contribution. No control may replace a failed
primary.

## Production boundary

Historical source support is insufficient for live launch. Production
requires:

- an owned Bitcoin Core node;
- actual block receipt and validation timestamps;
- raw-response and derived-state hashes;
- parity checks against the frozen allowlist and token builder;
- a live decision time equal to the later of actual availability and the
  frozen historical clock;
- stale-state, duplicate-release, open-order, and position-reconciliation
  guards; and
- paper execution before any capital-bearing order.

A live-source mismatch forces `TARGET_FLAT`; it does not trigger a fallback to
stale REST data.

## One-way sequence

1. Commit this boundary.
2. Commit preregistration, source-sequence builder, and synthetic tests.
3. Run source/sequence support exactly once.
4. Retire unchanged on any integrity or sequence-support failure.
5. Commit the economic evaluator and cheap-policy family before market access.
6. Open 2020, then 2021, then 2022 only through the frozen gates.
7. Train one compact Gemma only after cheap 2022 authorization.
8. Freeze one checkpoint on 2022.
9. Open 2023 once.
10. Keep 2024+ sealed unless the unchanged 2023 policy passes.

## Boundary ledger

At this commit:

```text
prior BCRT source-support aggregates read = yes
prior BCRT token marginals read           = yes
new raw source rows decoded               = 0
BCTP three-state sequences built          = 0
BCTP actionable incidence opened          = 0
BTC market rows opened                    = 0
funding rows opened                       = 0
future returns opened                     = 0
rewards or labels created                 = 0
model training runs                       = 0
post-2023 source rows opened              = 0
```

Selection status:

```text
selected_for_preregistration
```

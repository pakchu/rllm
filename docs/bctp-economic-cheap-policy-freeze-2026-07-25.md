# BCTP-12H economic accounting and cheap-policy freeze

## Decision

This document freezes the economic evaluator, reward, cheap-policy family,
familywise inference, chronological access order, and the conditional Gemma 4
specification for `BCTP-12H`.

No BTC OHLC row, funding row, forward return, reward, policy fit, or economic
metric was opened while choosing this contract.  The already committed BCTP
source sequences and support report are the only candidate-specific data used.

The implementation is:

```text
training/freeze_block_clearing_target_position_evaluator.py
```

The official freeze is write-once:

```text
results/block_clearing_target_position_mdp_evaluator_freeze_2026-07-25.json
```

The later economic runner must refuse to run unless the freeze manifest,
implementation hash, source-support bindings, and all configuration objects
match exactly.

## Bound execution sources

The freeze may read and hash the two source manifests below.  It may not open
or hash the bound OHLC or funding payloads.

```text
market
  path:
    data/binance_um_kline_reference_btc_2020_2023/
    BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz
  expected payload SHA256:
    e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d
  manifest:
    data/binance_um_kline_reference_btc_2020_2023/build_manifest.json
  manifest SHA256:
    c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e

funding
  path:
    data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz
  expected payload SHA256:
    3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6
  manifest:
    results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json
  manifest SHA256:
    a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b
```

Each economic stage streams only its half-open calendar year and stops before
the next year.  A stage reports a hash of the exact decoded lines.  It does not
hash the four-year payload as a shortcut.  This preserves the chronological
opening order:

```text
2020 labels and fit
2021 target schedules, then 2021 outcomes
2020-2021 refit
2022 target schedules, then 2022 outcomes
conditional Gemma training and 2022 checkpoint selection
one immutable 2023 target schedule, then 2023 outcomes
```

Before a transfer year is opened, every target schedule for that year,
including controls, must be written and hashed.  Policy inference uses source
tokens and current target position only, so this pre-outcome schedule is
possible.  Any loader call before schedule sealing is a protocol error.

## Split and target semantics

Each calendar stage starts with equity `1`, cash `1`, and BTC quantity `0`.
Source history before the stage remains available only as causal token
warm-up.  The stage has:

```text
terminal_flat_time = split_end - 5 minutes
```

Only decisions in `[split_start, terminal_flat_time)` are executable.  The
selected target owns bars whose open timestamps are:

```text
[decision_time, min(next_decision_time, terminal_flat_time))
```

At `terminal_flat_time`, the evaluator flattens at the exact open, charges one
quantity-change cost, and remains flat for the final complete five-minute bar.
No row at or after `split_end` may be loaded.

The action set and exact account-gross targets are:

```text
TARGET_SHORT = -0.5
TARGET_FLAT  =  0.0
TARGET_LONG  = +0.5
```

Malformed input, unknown vocabulary, non-finite model output, missing market
or funding data, stale source, or an infeasible quantity solution always maps
to `TARGET_FLAT`.

## Exact quantity and cost equation

At a rebalance open, let:

```text
E = pre-rebalance marked equity
y = q_old * P
x = q_new * P
c = changed-notional cost rate
f = target account gross
```

For `f = 0`, `x = 0`.  Otherwise the implementation tries the two analytic
branches of:

```text
x / (E - c*abs(x-y)) = f
```

For `x >= y`:

```text
x = f*(E + c*y)/(1 + f*c)
```

For `x <= y`:

```text
x = f*(E - c*y)/(1 - f*c)
```

A branch is admissible only when its inequality is satisfied, its post-cost
equity is strictly positive, its sign matches `f`, and substitution satisfies
the original equation within `1e-12` relative tolerance.  Coincident
`x == y` boundary branches are deduplicated.  Exactly one unique admissible
notional solution must remain.  Otherwise the result is flat.

The cost is charged once:

```text
c * abs(q_new-q_old) * P
```

A reversal is not represented as a separately charged close and open.
Same-target rebalancing still pays for the actual quantity change.  Synthetic
tests compare the analytic result with an independent monotone bisection
solver over signed target notional.

Base cost is `0.0006`; stress cost is `0.0010`.

## Funding boundary

Funding cash for quantity `q`, settlement mark `M`, and rate `r` is:

```text
-q*M*r
```

An interior event applies that exact cash once.  At a timestamp equal to a
rebalance or terminal-flatten boundary, the evaluator computes cash under the
old and new quantities and applies:

```text
min(0, old_cash, new_cash)
```

Thus an ambiguous credit is discarded, at most one debit is retained, and the
event cannot be double counted.  A funding timestamp inside a five-minute bar
is applied at its exact settlement mark before that bar's favorable/adverse
path marks.

## Strict path accounting

The evaluator uses cash plus BTC quantity and marks every owned bar in this
order:

1. boundary open under the old quantity;
2. one rebalance cost and conservative boundary funding;
3. post-cost/post-funding equity;
4. favorable held-bar extreme;
5. adverse held-bar extreme after a virtual flattening cost;
6. any interior funding settlement mark after its cash flow and a virtual
   flattening cost; and
7. the next boundary open, followed by its rebalance.

Within a held bar, favorable is intentionally marked before adverse:

```text
long:  high, then low
short: low, then high
flat:  open-equity, then open-equity
```

The high-water mark begins at `1` before the first entry and never resets.
Strict MDD is the largest loss from this single global high-water mark.
Terminal liquidation cost is included.  CAGR uses exactly
`365.2425 * 86400` seconds per year and the complete declared calendar,
including warm-up, blackouts, flat time, and the final flat bar.

## Frozen transition utility

The economic account multiplier remains the headline result.  Training only
uses this frozen per-transition utility:

```text
log(max(E_end/E_pre, 1e-12))
- (1/3) * held_path_downside_fraction
- 0.0010 * abs(target_new-target_old)
```

`held_path_downside_fraction` is the maximum conservative path loss from
`E_pre` inside the transition, including entry cost, funding, adverse OHLC,
and virtual flattening cost.  Costs are already present in the account
multiplier; the final term is a separate target-churn regularizer.  It is 5 bp
for a flat/non-flat change and 10 bp for a direct long/short reversal.

The reward has no future-best action label.  An oracle replay may be reported
only under an explicitly non-policy diagnostic namespace.

## Full-information transition table

The exogenous source sequence is action independent.  For every fit-year
decision after the first stage decision, the evaluator constructs all nine
counterfactual rows:

```text
current position in {SHORT, FLAT, LONG}
next target       in {SHORT, FLAT, LONG}
```

The first executable decision has only the reachable `FLAT` current position.
For a hypothetical non-flat current position at a later decision, the exact
pre-rebalance quantity/equity ratio is reconstructed by entering that target
at the immediately prior decision and carrying it through the prior interval.
Because target exposure and all cash flows are homogeneous in equity, this
ratio is independent of wealth before the prior rebalance.  This avoids a
behavior-policy leak while preserving exact quantity-change costs.

A row is terminal when there is no later executable decision before
`terminal_flat_time`.  Its reward includes the forced terminal flatten.
Every fit label's entire held path must end inside the fit stage.

## Frozen cheap learner family

The three promotable algorithms are full-information categorical linear,
categorical ridge, and Extra-Trees fitted-Q policies.  Each receives:

- the frozen 36 categorical source tokens;
- the categorical current position; and
- no price, timestamp, height, split, return, reward, or outcome field.

All categorical columns use a fixed one-hot vocabulary from the BCRT token
schema.  Unknown values fail closed to flat.

Fitted-Q configuration:

```text
discount                         0.99 per source transition
Bellman iterations               25
internal action order             FLAT, SHORT, LONG
linear                            deterministic lstsq(rcond=1e-12)
ridge                             alpha=100, intercept unpenalized
ridge solver                      deterministic closed-form
Extra-Trees estimator             ExtraTreesRegressor
n_estimators                     512
max_depth                        6
min_samples_split                24
min_samples_leaf                 12
max_features                     sqrt
bootstrap                        false
criterion                        squared_error
random_state                     20260725
n_jobs                           1
```

One multi-output estimator predicts the three action values.  Iteration zero
uses the immediate utility matrix.  Later iterations use:

```text
r(s,p,a) + 0.99 * max_a' Q(previous_fit, next_source, position=a, a')
```

with a zero continuation value at a terminal row.  Training order is canonical
`entry_time, current_position`; action output order is `FLAT, SHORT, LONG`.

Inference chooses the greatest finite Q.  Ties within `1e-12` choose, in
order:

1. flat;
2. the existing position;
3. short;
4. long.

All three algorithms are fixed.  There is no hyperparameter search.  They fit
2020 and transfer unchanged to 2021.  Only the three full-sequence primary
algorithms may qualify.  One passing algorithm is selected by the frozen rule
below, then that algorithm and the complete control family refit from scratch
on 2020-2021 if the 2021 gate passes.

## Frozen baseline and falsification family

Every opened variant remains in the shared development family even when it
fails, is flat, or has zero variance:

```text
always_flat
always_long
always_short
previous_target_persistence
exact_signature_memory
categorical_linear_fqi
categorical_ridge_fqi
extra_trees_fqi
categorical_linear_fqi_current_only
categorical_linear_fqi_reversed_sequence
categorical_linear_fqi_masked_source
categorical_ridge_fqi_current_only
categorical_ridge_fqi_reversed_sequence
categorical_ridge_fqi_masked_source
extra_trees_fqi_current_only
extra_trees_fqi_reversed_sequence
extra_trees_fqi_masked_source
categorical_linear_fqi_shuffled_reward
categorical_linear_fqi_circular_21_reward
categorical_ridge_fqi_shuffled_reward
categorical_ridge_fqi_circular_21_reward
extra_trees_fqi_shuffled_reward
extra_trees_fqi_circular_21_reward
categorical_linear_fqi_direction_flip
categorical_linear_fqi_action_code_permutation
categorical_ridge_fqi_direction_flip
categorical_ridge_fqi_action_code_permutation
extra_trees_fqi_direction_flip
extra_trees_fqi_action_code_permutation
bcrt_exact_six_hour_always_long
bcrt_exact_six_hour_always_short
```

This is exactly 31 return series: five simple controls; three full-sequence
primaries; three feature ablations for each primary; two reward-null controls
for each primary; two inference controls for each primary; and two exact BCRT
clock-shape controls.

The current-snapshot ablation receives only `S0` plus position.  The masked
ablation replaces `order_transition` and `leader_transition` in all three
sequence slots with one frozen `MASKED` category.  The reversed ablation fits
and infers with oldest/newest snapshot values exchanged while preserving the
relative field labels.  The shuffled control permutes complete reward tensors
within calendar month using seed `20260725`.  The circular control rotates
complete reward tensors forward by exactly 21 source decisions without
rotating source states.

The exact-memory control stores train mean immediate utility by exact
36-token signature and current position; unseen keys choose flat.  Previous
target persistence starts flat at every split.  The neutral permutation uses
the code order `(LONG, SHORT, FLAT)` internally and must map back to a target
schedule byte-identical to its corresponding primary.  Any mismatch is an
evaluator failure.

The BCRT comparators use the exact committed BCRT six-hour reservation clock,
hold always-long or always-short for each frozen reservation, and stay flat
between reservations.  They are non-promotable and avoid inventing an
unfrozen BCRT side rule.

The 10 bp and +5 minute gate variants are not additional max-stat family
members.  They reuse the selected primary target sequence.  Delay
shifts every non-terminal rebalance by exactly one complete five-minute bar;
terminal flatten time never moves.

## Shared weekly max-stat test

For each stage, each variant emits full-calendar weekly log account returns,
including zero-return weeks.  Weeks are Monday 00:00 UTC half-open clusters.
All family members must have the same ordered week keys.

For a vector `w` with at least two weeks:

```text
t(w) = sqrt(n) * mean(w) / sample_std(w)
```

Zero-variance policies have internal statistic negative infinity and receive
local and adjusted `p=1`.

The null uses identical Rademacher signs for every family member and week.
With at most 20 weeks it enumerates every sign vector.  Otherwise it uses
`100000` deterministic Monte Carlo draws, seed `20260725`, in batches of
`2000`.  For candidate `i`:

```text
local p_i = P(t(sign*w_i) >= t(w_i))
pmax_i    = P(max_j t(sign*w_j) >= t(w_i))
```

Monte Carlo p-values use the plus-one correction.  Missing or failed variants
remain as all-zero vectors; they are never removed after outcomes are seen.

## Frozen gates

At least one of the three primaries must satisfy all 2021 transfer gates:

- positive absolute return;
- CAGR/strict-MDD at least `1.0`;
- positive 10 bp stress return;
- positive +5 minute delay return;
- at least `80` non-flat target intervals;
- long and short each at least `20%` of non-flat intervals;
- absolute return and CAGR/strict-MDD both exceed exact memory, its
  algorithm-matched shuffled/circular controls, and the stronger of its
  current-snapshot/reversed/masked controls;
- familywise `pmax < 0.25`; and
- its neutral action-code permutation is byte-identical.

Among passing primaries, the 2021 winner is selected lexicographically by:

1. largest minimum of base, 10 bp stress, and delayed CAGR/strict-MDD;
2. largest base CAGR/strict-MDD;
3. largest absolute return;
4. lower strict MDD; and
5. lexical policy ID.

Only a complete pass permits a from-scratch 2020-2021 refit and opening 2022.
The same selected algorithm must meet the same 2022 gates, with ratio at least
`1.5` and `pmax < 0.10`, before any Gemma training.  There is no 2022
reselection.

## Conditional Gemma 4 policy

Gemma is not authorized by this freeze.  The following specification merely
prevents a post-outcome model or hyperparameter choice.

Official base:

```text
repository  google/gemma-4-E4B
revision    9f9f0f28c85251b6616672841d041635e1763f13
license     Apache-2.0
```

Google describes E4B as a Gemma 4 dense model with 4.5B effective parameters,
8B total parameters including per-layer embeddings, and a 128K context
window:

- https://ai.google.dev/gemma/docs/core
- https://huggingface.co/google/gemma-4-E4B
- https://huggingface.co/google/gemma-4-E4B/commit/9f9f0f28c85251b6616672841d041635e1763f13

The frozen Gemma 4 tokenizer identity is:

```text
vocabulary size  262144
tokenizer.json SHA256
  cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f
```

The E4B tokenizer must match this identity before training or BCTP retires.
Neutral action tokens and IDs are:

```text
A  236776
B  236799
C  236780
```

The mapping ensemble contains all six action-code permutations.  A checkpoint
policy averages target probabilities after mapping each permutation back to
`SHORT/FLAT/LONG`; this removes a fixed token preference.

The prompt contains only task ID `BCTP-12H`, the 36 relative token lines, one
position line, and `ACTION=`.  Thinking is disabled.  No generated text is
decoded.  The logits of the three frozen action tokens at the final prompt
position define the policy.

Training uses 4-bit NF4 double quantization, BF16 compute, LoRA rank 16,
alpha 32, dropout 0.05, and targets:

```text
q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
```

Maximum sequence length is 768.  Micro-batch size is 2, gradient accumulation
is 8, learning rate is `2e-5`, weight decay is `0.01`, AdamW betas are
`(0.9,0.95)`, warm-up is 10%, gradient clip is 1.0, and seeds are
`20260725` and `20260726`.

The objective is not a future-best target label.  For frozen full-information
action values `Q` and mapped action-token probabilities `pi`:

```text
loss = -sum(pi * stop_gradient(Q))
       + 0.01 * KL(pi || uniform_three_actions)
```

Q is centered per state and clipped to `[-0.10,+0.10]` before the actor loss.
Fixed checkpoints are optimizer steps `80`, `160`, and `240`; no other
checkpoint may be opened.  Every seed, checkpoint, and permutation is part of
the same development family.  Selection on 2022 is lexicographic:

1. pass every Gemma gate;
2. largest minimum of base/stress/delay CAGR/strict-MDD;
3. largest base CAGR/strict-MDD;
4. lower strict MDD;
5. smaller checkpoint step;
6. lexical checkpoint ID.

The selected checkpoint must have positive return, ratio at least `3`, MDD at
most `15%`, at least `100` non-flat intervals, both directions at least 20%,
positive H1/H2, positive stress/delay, and familywise `pmax < 0.05`.

The unchanged selected policy is evaluated once in 2023.  It must meet the
same economic, risk, activity, direction, half-year, stress, and delay gates,
with a one-policy weekly-cluster `p < 0.10`.  Failure retires BCTP unchanged.
No 2024+ source or outcome may be opened for repair.

## Freeze ledger

The official freeze must record:

```text
market rows parsed                  0
funding rows parsed                 0
market/funding payload bytes hashed false
future returns created              0
rewards created                     0
models fit                          0
economic metrics computed           0
2023 outcomes opened                false
2024+ source/outcomes opened        false
mutable parameters                  []
```

Any nonzero field invalidates the freeze.

# PIVOT-72 mechanism decision — paired intrinsic venue orderflow topology

## Decision

**Selected mechanism:** `PIVOT-72`

**Full name:** Paired Intrinsic Venue Orderflow Topology Policy

PIVOT-72 converts every valid, non-tied Binance Spot versus USD-M intrinsic
volume first-passage pair into one compact causal relation state. One policy
chooses exactly one of:

```text
LONG
SHORT
ABSTAIN
```

The position, when selected, is Binance USD-M `BTCUSDT` perpetual at fixed
`0.5x` account gross for exactly `72` five-minute bars.

This document freezes the complete mechanism before:

- decoding a PIVOT source row;
- measuring PIVOT incidence or token support;
- loading a PIVOT execution price or funding value;
- fitting a PIVOT baseline or language model;
- parsing a comparator row; or
- opening a 2024-or-later source or outcome value.

PIVOT is a new policy object, not a repair of retired CVICR. CVICR required a
q60 clock gap, early conflict, leader persistence, laggard resolution, and a
source-owned side. PIVOT admits every valid non-tied paired anchor and asks one
frozen policy to infer `LONG`, `SHORT`, or `ABSTAIN` from relational topology.
No CVICR threshold, failed state, or outcome is modified.

## Frozen evidence boundary

The mechanism was selected using repository history, source schemas, source
hashes, manifests, aggregate predecessor reports, and model runtime probes.
No PIVOT row, PIVOT token, PIVOT action, comparator row, or PIVOT outcome was
opened to choose any definition below.

The binding boundary is:

| Artifact | SHA-256 |
|---|---|
| `docs/paired-intrinsic-venue-orderflow-topology-boundary-2026-07-24.md` | `dd06a3aea17596e2d1e451b5c3f8f3d98af5691d9ffe5f1ac26af59d8e8fcacb` |

The boundary commit is `00f9d3e`.

## Frozen source contract

### Predictor source

| Artifact | SHA-256 |
|---|---|
| `data/binance_cross_venue_minute_leadership_btc_2020_2023/BTCUSDT_cross_venue_minute_leadership_5m_2020-01_2023-12.csv.gz` | `00ab6a55fc7bfeb3012584db5bc97a7d7b98dd995491acfd3f865c6bd41f92cc` |
| source CSV header | `b7c730d6fc2c37d6e94f6a436478fd09ff42d15d7fd81bf521c4ca36465ff49f` |
| `data/binance_cross_venue_minute_leadership_btc_2020_2023/build_manifest.json` | `544c2945a2b56be478a1edc4abbb93b762bda5afc32cbd0658dd6822ff6b70fa` |
| `results/binance_cross_venue_minute_leadership_audit_2026-07-14.json` | `ffe0124ac9c5c0c3f1d1c284b672618cf910dc16cae36e65c1efe79710f039af` |

The predictor loader must use `pandas.read_csv(usecols=SOURCE_ALLOWLIST)`.
Load-and-drop is forbidden. The exact allowlist is:

```text
date
feature_available_time_utc
trade_earliest_time_utc
spot_quote_notional
um_quote_notional
spot_signed_quote_notional
um_signed_quote_notional
source_complete
```

The physical source must remain exactly `420,768` monotone five-minute rows on
`[2020-01-01T00:00:00Z, 2024-01-01T00:00:00Z)`. For every accepted row:

```text
feature_available_time_utc == date + 5m
trade_earliest_time_utc    == date + 5m
source_complete            == True
quote_notional             is finite and >= 0
signed_quote_notional      is finite
abs(signed_quote_notional) <= quote_notional
```

There is no imputation, forward fill, backward fill, epsilon, clipping,
rounding, or alternate venue substitution.

### Execution market source

| Artifact | SHA-256 |
|---|---|
| `data/binance_um_kline_reference_btc_2020_2023/BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz` | `e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d` |
| market CSV header | `5e8d51e7e1218929db6a54ca59280eb4306171b81d5d0880467a85cf9d23eff2` |
| `data/binance_um_kline_reference_btc_2020_2023/build_manifest.json` | `c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e` |

No market value may be loaded until the source-support builder and the strict
economic evaluator have both been committed and hash-frozen.

### Funding source

| Artifact | SHA-256 |
|---|---|
| `data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz` | `3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6` |
| funding CSV header | `71b2b1395313f631969674c43e569c8f1619a9fb23c8316e2e0478c32f01d61f` |
| `results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json` | `a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b` |

No funding value may be loaded before the same evaluator freeze.

## Base paired state

### Complete reference day

For UTC calendar day `D` and venue `v`, a reference day is complete only when
all `288` five-minute rows exist, satisfy the source invariants above, and
belong to `D`.

Define exact daily quote notional:

```text
V[v,D] = sum(quote_notional[v,t] for all 288 bars t in D)
```

For current day `D`, the reference calendar is exactly:

```text
D-28 calendar days, ..., D-1 calendar day
```

Missing or invalid calendar positions remain missing and the window never
widens. At least `21` complete reference days are required independently for
each venue.

```text
expected[v,D] = numpy.median(float64 complete V[v,d] in exact prior 28 days)
target[v,D]   = 0.50 * expected[v,D]
```

`expected[v,D]` and `target[v,D]` must be finite and strictly positive. The
current day is always excluded.

### Current causal prefix

For each venue, cumulative quote notional through completed bar start `t` is:

```text
Q[v,D,t] = sum(quote_notional[v,s], D 00:00 <= s <= t)
```

The venue anchor `A[v,D]` is the first bar start satisfying:

```text
Q[v,D,A[v,D]] >= target[v,D]
```

Both anchors must exist and have bar starts no later than `23:50 UTC`.
`23:50` is frozen prospectively and may never be compared with another cutoff.
An exact anchor tie is ineligible.

```text
A_early = min(A[spot,D], A[um,D])
A_late  = max(A[spot,D], A[um,D])
leader  = venue at A_early
laggard = other venue
gap_bars = (A_late - A_early) / 5m
```

`gap_bars` is a positive integer.

Every source row from `D 00:00` through the computation-buffer bar starting
`A_late+5m`, inclusive, must exist and satisfy every source invariant. A defect
in that prefix cancels the base state. A later source defect cannot retroactively
cancel a state after entry.

The buffer bar completes at `A_late+10m`. If `A_late=23:50`, the buffer starts
at `23:55`, completes at next-day `00:00`, and remains part of source day `D`.

### Cumulative flow

For venue `v` and completed bar start `t`:

```text
F[v,D,t] =
  sum(signed_quote_notional[v,s], D 00:00 <= s <= t)
  /
  sum(quote_notional[v,s], D 00:00 <= s <= t)
```

The denominator must be finite and strictly positive. There is no epsilon,
deadband, clipping, or rounding. Exact zero is a valid state and has sign
`ZERO`; it is not rejected.

### Raw measures

Every base paired state stores these outcome-blind raw measures:

```text
gap_bars
early_anchor_start_minute
laggard_progress_at_early =
  Q[laggard,D,A_early] / target[laggard,D]
spot_flow_early = F[spot,D,A_early]
um_flow_early   = F[um,D,A_early]
spot_flow_late  = F[spot,D,A_late]
um_flow_late    = F[um,D,A_late]
spot_abs_flow_late = abs(spot_flow_late)
um_abs_flow_late   = abs(um_flow_late)
```

`laggard_progress_at_early` must be finite and in `[0,1)`. No price, basis,
funding, return, PnL, comparator, or future bar participates.

## Strictly prior state transform

### Reference population

A **base paired state** is a source-valid, reference-ready, non-tied state
defined above before:

- ordinal-history readiness;
- policy action;
- opportunity reservation;
- split containment;
- external position conflict; or
- any market outcome.

For each current base state, the prior reference is the immediately preceding
at most `180` base paired states in chronological `(source_day, A_late)` order.
The current state is excluded. At least `90` prior base states are required.

States later suppressed by opportunity reservation or split containment remain
inside future prior references. This makes every transform policy-independent.
The “immediately previous valid state” is the immediately preceding base paired
state in this same order, not the preceding trade or preceding accepted action.

### Quartiles

For every ordinal raw measure, compute:

```python
q25, q50, q75 = numpy.quantile(
    numpy.asarray(previous_values, dtype=numpy.float64),
    [0.25, 0.50, 0.75],
    method="linear",
)
bucket = numpy.searchsorted(
    numpy.asarray([q25, q50, q75], dtype=numpy.float64),
    numpy.float64(current_value),
    side="right",
)
```

The resulting labels are `Q0`, `Q1`, `Q2`, `Q3`. Exact equality maps upward.
Duplicate thresholds remain duplicated; empty levels are not repaired.
Non-finite current values or thresholds reject the state.

Future source appends must leave every earlier raw state and token byte-identical.

## Exact 12-token policy state

The canonical token order and vocabulary are:

1. `leader`: `SPOT | UM`
2. `gap_q`: `Q0 | Q1 | Q2 | Q3`
3. `early_session`: `S00_06 | S06_12 | S12_18 | S18_24`
4. `laggard_progress_q`: `Q0 | Q1 | Q2 | Q3`
5. `spot_early_sign`: `NEG | ZERO | POS`
6. `um_early_sign`: `NEG | ZERO | POS`
7. `spot_late_sign`: `NEG | ZERO | POS`
8. `um_late_sign`: `NEG | ZERO | POS`
9. `spot_late_abs_flow_q`: `Q0 | Q1 | Q2 | Q3`
10. `um_late_abs_flow_q`: `Q0 | Q1 | Q2 | Q3`
11. `gap_change`: `NARROW | SAME | WIDEN`
12. `leader_change`: `SAME | SWITCH`

`early_session` uses the early anchor bar start:

```text
S00_06 = [00:00,06:00)
S06_12 = [06:00,12:00)
S12_18 = [12:00,18:00)
S18_24 = [18:00,24:00)
```

Each sign token is the exact mathematical sign of its cumulative flow.
`gap_change` compares current `gap_bars` with the immediately previous base
state. `leader_change` compares the current leader with that previous leader.

The four quartile tokens use separate strictly prior distributions:

```text
gap_q                 <- prior gap_bars
laggard_progress_q    <- prior laggard_progress_at_early
spot_late_abs_flow_q  <- prior spot_abs_flow_late
um_late_abs_flow_q    <- prior um_abs_flow_late
```

No date, year, month, quarter, row ID, event ID, raw timestamp, raw number, raw
price, return, basis, funding, premium, OI, Kimchi, DXY, reward, action, PnL,
comparator identity, free-form rationale, or post-2023 value enters the prompt.

Current position is deliberately not a model token. Strategy reservation and
external-position conflicts are deterministic pre-model guards. A non-flat or
otherwise unexecutable live state becomes `ABSTAIN`.

Any token level not observed in the 2020–2021 training vocabulary forces
deterministic `ABSTAIN`. It may not be merged, renamed, or rebucketed.

## Opportunity reservation and execution

The action-independent opportunity clock is:

```text
causal origin          = A_early
state completion       = A_late + 5m
buffer completion      = A_late + 10m
decision deadline      = A_late + 15m
entry                  = USD-M BTCUSDT open at A_late + 15m
exit                   = USD-M BTCUSDT open at entry + 72*5m
```

The extra bar between buffer completion and entry is intentional. It gives a
complete five-minute inference/order window and prevents a backtest from
claiming the exact open that occurred at the same instant as buffer completion.

The policy must finish before the `A_late+15m` open. Failure, timeout, stale
state, non-finite score, unknown token, model-load error, or position conflict
means deterministic `ABSTAIN`; no late fill is backdated.

Other rules:

- at most one base state per UTC source day;
- fixed exposure `0.5x` account gross;
- exactly `72` held five-minute bars;
- scheduled exit only;
- no stop, take-profit, trailing, pyramiding, model exit, or dynamic sizing;
- reserve `[entry,exit)` before asking the policy;
- an abstention does not release the interval;
- suppress, never queue or replace, a later opportunity with
  `later_entry < previous_reserved_exit`;
- build and reserve the complete pre-2024 opportunity clock globally before
  applying temporal splits;
- a split-crossing reserved state is omitted from that split's economics but
  remains reserved and cannot free a later state; and
- origin, both anchors, buffer, decision window, entry, all held bars, and exit
  must remain inside the same half-open split.

Primary execution delay controls preserve the frozen action and use:

```text
one_bar_delay: entry+5m,  exit+5m
one_hour_delay: entry+60m, exit+60m
```

Each delay control recomputes global reservation and split containment on its
own delayed clock.

## Temporal roles

```text
train/model fit   [2020-01-01, 2022-01-01)
selection         [2022-01-01, 2023-01-01)
untouched eval    [2023-01-01, 2024-01-01)
sealed            [2024-01-01, ...)
```

`2022` is a model-selection window and must never be described as untouched
test evidence. All state transforms remain online and strictly prior, but all
policy parameters use only 2020–2021 outcomes. Hyperparameters are frozen here;
2022 selects only among the exact frozen cheap policies and exact frozen DPO
checkpoints.

No monthly, rolling, continuous, or eval-label adaptation is allowed.

## Source-only support gate

The source-support builder may load only the predictor allowlist. It must not
load market values, funding values, comparator rows, labels, rewards, or
post-2023 source values.

Counts below use token-ready, globally reserved, split-contained opportunities.
The strictly prior transforms continue to use all base paired states, including
suppressed states.

All gates must pass:

### Global and train

- at least `750` opportunities globally through 2023;
- at least `350` train opportunities;
- at least `150` train opportunities in each of 2020 and 2021;
- at least `7` active calendar months in warm-up year 2020;
- all `12` calendar months active in 2021;
- maximum train single-month share at most `15%`; and
- maximum gap between train entries at most `14` calendar days.

### Selection 2022 and eval 2023

Each year independently must have:

- at least `200` opportunities;
- at least `85` opportunities in each half-year;
- at least `35` opportunities in every quarter;
- all 12 months active;
- maximum single-month share at most `15%`; and
- maximum entry gap at most `14` calendar days.

### Token support in train, selection, and eval

Each split independently must satisfy:

- `SPOT` and `UM` leader share each at least `20%`;
- `NEG` and `POS` share each at least `20%` for each of the four sign tokens;
- `ZERO` is reported but has no minimum share;
- every quartile level share is between `10%` and `40%`, inclusive, for every
  quartile token;
- at least three session levels occur and the largest session share is at most
  `65%`;
- `NARROW` and `WIDEN` each have share at least `20%`;
- `SWITCH` share is at least `15%`;
- the largest exact 12-token signature share is at most `3%`;
- no token is missing or invalid; and
- every token level appearing in 2022 or 2023 already appears in train.

The builder must additionally pass synthetic and real-prefix tests for:

- venue swap;
- sign mirror;
- future append;
- current-value exclusion from every prior quartile;
- suppressed-state inclusion in prior history;
- missing prefix;
- exact anchor tie;
- exact-zero sign preservation;
- duplicate quartile boundaries;
- option-order independence at serialization; and
- action-independent reservation.

Any failure retires PIVOT-72 unchanged. The target fraction, anchor cutoff,
prior length, token schema, support floor, latency, or hold may not be repaired.

These floors do not assume unseen incidence from the new `23:50` cutoff. The
already committed CVICR source-only aggregate disclosed `954` non-tied valid
prefixes and `864` strictly-prior-reference-ready pairs under its stricter
`17:50` cutoff. PIVOT uses the same target/reference coordinate, retains every
such reference-ready pair without applying CVICR's event conjunction, and also
permits later anchors; no count from the later-anchor extension is assumed. The
`750` global floor leaves more than 10% attrition against the disclosed `864`
reference-ready precedent for the different token warm-up, reservation, and
split-containment rules.

The reduced 2020 month floor is structural, not incidence-selected. The source
begins on `2020-01-01`; the exact prior-28-calendar-day reference and subsequent
90-base-state ordinal warm-up make the first months incapable of producing a
token-ready opportunity. Requiring 12 active months in 2020 would therefore be
impossible before any row is decoded.

## Frozen economic accounting

### Costs and quantity

```text
leverage = 0.5
base fee + slippage = 0.0006 of notional per side
stress replacement cost = 0.0010 of notional per side
```

Stress replaces, rather than adds to, the base cost.

At entry:

```text
quantity = entry_equity * leverage / entry_open
entry_cost = quantity * entry_open * cost_notional_per_side
```

Quantity remains fixed through the trade. Exit cost uses scheduled exit
notional:

```text
exit_cost = quantity * exit_open * cost_notional_per_side
```

### Funding

For each funding settlement with `entry_time <= funding_time <= exit_time`:

```text
funding_cash = -side * quantity * settlement_mark_price * funding_rate
```

`side=+1` is LONG and `side=-1` is SHORT. Because exact entry/exit timestamp
ordering cannot guarantee receipt of a credit, a positive funding contribution
exactly at entry or exit is dropped. A negative contribution exactly at either
boundary is retained. All interior contributions are retained.

### Realized equity

```text
realized_equity =
    entry_equity
  - entry_cost
  + side * quantity * (exit_open - entry_open)
  + retained_funding_cash
  - exit_cost
```

Account equity is floored at zero. Trades compound chronologically.

### Held-path strict MDD

Strict MDD carries one global high-water mark across the full declared
calendar, including idle cash and pre-entry history.

For each trade:

1. mark equity after entry cost;
2. over market bars `[entry_position, exit_position)`, including the entry bar
   and excluding the scheduled exit bar, mark the favorable extreme first;
3. add all retained funding credits to that favorable mark;
4. mark the adverse extreme second;
5. include all retained funding credits and debits plus a virtual exit cost at
   the adverse price;
6. mark realized scheduled-exit equity after exit cost.

For LONG, favorable is maximum held high and adverse is minimum held low. For
SHORT, favorable is minimum held low and adverse is maximum held high.

This deliberately assumes favorable-before-adverse within the aggregate held
path and therefore cannot hide intratrade drawdown behind OHLC ordering.

### Full-calendar CAGR

For half-open calendar `[start,end)`:

```text
years = (end-start) / (365.25 days)
CAGR = final_equity ** (1/years) - 1
```

Warm-up, abstention, and idle cash remain in the denominator.

### Ratio and gross-edge definitions

Metrics are expressed in percentage points. Define:

```text
if strict_mdd_pct > 1e-12:
    cagr_to_strict_mdd = cagr_pct / strict_mdd_pct
elif cagr_pct > 0:
    cagr_to_strict_mdd = 1e12
    zero_mdd_ratio_cap_applied = True
else:
    cagr_to_strict_mdd = 0
```

Every other case sets `zero_mdd_ratio_cap_applied=False`. Non-finite equity,
CAGR, strict MDD, or ratio is a hard failure and cannot enter selection.

For each executed trade:

```text
gross_underlying_return =
    side * (scheduled_exit_open / entry_open - 1)
```

`mean_gross_underlying_move_bp` is the arithmetic mean of these signed,
pre-cost, pre-funding returns times `10,000`; it is zero for no trades. It is
not the mean absolute price move.

### Weekly cluster sign-flip

Use each trade's net compounded account return after base costs and funding.
Assign it to the UTC ISO week of entry, Monday `00:00`, and sum within nonempty
weeks. Retain zero-return trades.

```text
observed = sum(weekly_sums) / trade_count
```

Initialize `numpy.random.default_rng(20260724)` independently for each reported
split and policy. Draw `100,000` independent Rademacher sign vectors, one sign
per nonempty week. For each draw:

```text
null = sum(sign[week] * weekly_sum[week]) / trade_count
```

The one-sided positive-mean p-value is:

```text
(1 + count(null >= observed)) / 100001
```

Return `1.0` for no trades or no nonempty cluster.

## Frozen trade utility and labels

For each action on each train opportunity, compute the exact local accounting
above from normalized entry equity `1.0`.

```text
U(ABSTAIN) = 0
U(trade) =
    log(max(account_multiplier, 1e-12))
  - (1/3) * local_held_path_strict_drawdown
  - 0.0010
```

The `0.0010` term is a 10-bp **account-level utility hurdle** applied only to
LONG and SHORT. It is not an execution cost and is never added to reported
returns.

The oracle action is the maximum-utility action with exact tie priority:

```text
ABSTAIN, then LONG, then SHORT
```

Only 2020–2021 outcomes may create SFT labels or preference pairs.

For every unordered pair of actions, emit one preference pair only when:

```text
chosen_utility - rejected_utility >= 0.0005
```

The higher-utility action is chosen. Pairs below the margin are omitted. All
qualifying pairs are retained; there is no outcome-dependent oversampling,
downsampling, class balancing, or hard-negative mining. Rows sort by
`(entry_time, symmetry_index, option_permutation_index, chosen_priority,
rejected_priority)` before deterministic epoch shuffling.

### Training-only relational symmetry

Every train state receives two deterministic symbolic views:

1. identity; and
2. sign mirror.

Sign mirror applies `NEG <-> POS`, preserves `ZERO`, and leaves all non-sign
tokens unchanged. It swaps LONG and SHORT utilities and targets while
preserving ABSTAIN. Applying the transform twice must reproduce the original
row byte-for-byte.

The two symmetry views are training augmentation only. Reported support,
actions, trades, and economics use original physical states only.

Venue swap remains a source-builder equivariance test, not training
augmentation and not a policy-invariance claim. Swapping the physical Spot and
USD-M source columns must produce the mechanically transformed token record:

```text
leader SPOT <-> UM
spot_early_sign <-> um_early_sign
spot_late_sign <-> um_late_sign
spot_late_abs_flow_q <-> um_late_abs_flow_q
```

All other tokens remain unchanged. Applying this source transform twice must
restore the original token record. No action or utility equality is asserted,
because Spot and USD-M have economically distinct identities and execution
occurs only in USD-M.

## Frozen cheap causal baselines

Baselines fit only on 2020–2021 and are evaluated on 2022. Nominal tokens never
receive ordinal integer encodings.

### Common representation

The common design matrix contains:

- one-hot indicators for every train-observed main token level;
- one-hot indicators for all `66` unordered token-pair conjunctions;
- one unpenalized intercept where supported; and
- only features occurring at least `3` times in train.

Unknown downstream token levels force `ABSTAIN`.

### Policies and controls

1. `always_abstain`
2. `always_long`
3. `always_short`
4. `exact_signature_memory`
   - majority oracle action per exact train signature;
   - tie priority `ABSTAIN`, `LONG`, `SHORT`;
   - unseen signature abstains.
5. `categorical_naive_bayes`
   - field-wise categorical likelihood;
   - Laplace alpha `1.0`;
   - oracle-action target.
6. `ridge_contextual_value`
   - separate LONG and SHORT utility regressions;
   - ridge alpha `100.0`;
   - intercept unpenalized;
   - ABSTAIN value fixed at zero;
   - trade only when the best fitted trade utility is strictly above zero.
7. `extra_trees_contextual_value`
   - same binary one-hot matrix;
   - separate LONG and SHORT regressors;
   - `n_estimators=512`;
   - `criterion="squared_error"`;
   - `max_depth=5`;
   - `min_samples_split=20`;
   - `min_samples_leaf=10`;
   - `max_features="sqrt"`;
   - `bootstrap=False`;
   - `random_state=20260724`;
   - ABSTAIN value fixed at zero;
   - trade only when best fitted trade utility is strictly above zero.
8. `shuffled_oracle_label`
   - 32 Naive Bayes controls;
   - independent train-label permutations with seeds `20260724..20260755`.
9. `shuffled_action_utility`
   - 32 ridge controls;
   - independently permute each trade action's train utility with seeds
     `20260724..20260755`.
10. twelve single-token ridge policies and twelve leave-one-token-out ridge
    ablations using the unchanged ridge hyperparameters.

The strongest shuffled control is the maximum 2022
`CAGR/strict-MDD`, then absolute return, across all 64 shuffles.
The strongest single-token control is selected by the same ordering.

### Cheap learnability gate

At least one of `categorical_naive_bayes`, `ridge_contextual_value`, or
`extra_trees_contextual_value` must satisfy all 2022 conditions:

- positive absolute return;
- `CAGR/strict-MDD >= 1.0`;
- strict MDD at most `15%`;
- positive H1 and H2 absolute return;
- at least `40` trades;
- at least `15` trades in each half;
- at least `10` LONG and `10` SHORT trades;
- no action exceeds `90%` of opportunities;
- positive 10-bp/notional/side stress return;
- positive one-bar-delay return;
- weekly-cluster one-sided `p < 0.20`;
- strictly higher absolute return and ratio than the strongest shuffled
  control; and
- strictly higher absolute return and ratio than the strongest single-token
  policy.

Among qualifying learned policies, select by:

1. higher 2022 `CAGR/strict-MDD`;
2. higher absolute return;
3. lower strict MDD;
4. lexicographically smaller policy ID.

If none qualifies, retire PIVOT before GPU training.

## Frozen single-Gemma RLLM

### Model and runtime

```text
base model = google/gemma-4-E2B-it
revision   = 3e22461f65e89153144f8adb70e3b8c2cc9845a7
loader     = transformers.AutoModelForCausalLM
tokenizer  = transformers.AutoTokenizer
trust_remote_code = False
```

The workload is text only. A multimodal processor, image token, image tensor,
analyzer model, second trader model, or free-form rationale is forbidden.

Runtime versions are frozen to the current lock/environment:

```text
torch             2.9.0
transformers      git 5d7ff4393ab99aa7cadf4cccd1f814dbb799f2bb
trl               0.29.0
peft              0.18.1
bitsandbytes      0.49.2
numpy             2.2.6
pandas            2.3.3
scikit-learn      1.7.2
```

Quantization:

```text
load_in_4bit=True
bnb_4bit_quant_type="nf4"
bnb_4bit_use_double_quant=True
bnb_4bit_compute_dtype=torch.bfloat16
```

LoRA:

```text
r=16
alpha=32
dropout=0.05
bias="none"
task_type="CAUSAL_LM"
target_modules=["q_proj","k_proj","v_proj","o_proj"]
```

Memory gates:

- 4-bit inference peak reserved CUDA memory at most `7.5 GiB`;
- training peak reserved CUDA memory at most `24 GiB`;
- training peak allocated CUDA memory at most `20 GiB`;
- each adapter/checkpoint directory at most `256 MiB`; and
- retained SFT plus selected DPO artifacts together at most `1 GiB`.

Any memory failure rejects the RLLM path without changing quantization, model,
LoRA targets, prompt length, or optimizer steps.

### Prompt and option-order invariance

Tokens serialize in the exact canonical order above as one `KEY=VALUE` line
each. The instruction contains no prose analysis and ends with:

```text
OPTIONS=<one permutation of LONG,SHORT,ABSTAIN>
Return exactly ACTION=<one option>.
```

Every state is represented in all six permutations of the three actions, in
lexicographic permutation order. This sixfold expansion depends on no date,
ID, outcome, or model score.

The exact completions are:

```text
ACTION=ABSTAIN
ACTION=LONG
ACTION=SHORT
```

Generation is forbidden. For each prompt permutation and completion, compute
the conditional sum of completion-token log probabilities, excluding prompt
and special tokens, divided by the number of completion tokens. Average each
action's normalized score over all six prompt permutations.

Choose the maximum mean score. Scores tied within absolute tolerance `1e-12`
use priority `ABSTAIN`, `LONG`, `SHORT`. Missing, malformed, non-finite, or
incomplete scores force `ABSTAIN`.

This makes inference invariant to any single displayed option order.

Maximum tokenized prompt plus completion length is `384`. Longer rows are
invalid and may not be truncated.

### SFT warm-up

All six prompt permutations receive the same train-only oracle action.

```text
optimizer                  = AdamW
learning_rate              = 2e-4
betas                      = (0.9,0.999)
epsilon                    = 1e-8
weight_decay               = 0.01
scheduler                   = cosine
warmup_steps               = 8
max_grad_norm               = 1.0
optimizer_steps            = 64
per_device_batch_size      = 1
gradient_accumulation      = 8
packing                    = False
completion_only_loss       = True
bf16                       = True
seed                       = 20260724
```

No SFT checkpoint is selected on 2022. Only the final SFT adapter initializes
DPO.

### DPO preference stage

Use the exact train-only pairs above and the SFT adapter as initialization.
The reference policy is the frozen SFT adapter with DPO updates disabled.

```text
loss                       = DPO sigmoid
beta                       = 0.1
label_smoothing            = 0.0
optimizer                  = AdamW
learning_rate              = 5e-6
betas                      = (0.9,0.999)
epsilon                    = 1e-8
weight_decay               = 0.01
scheduler                   = cosine
warmup_steps               = 8
max_grad_norm               = 1.0
optimizer_steps            = 96
per_device_batch_size      = 1
gradient_accumulation      = 8
bf16                       = True
seed                       = 20260724
checkpoints                = [24,48,72,96]
```

Every checkpoint is evaluated exactly once on 2022. A checkpoint qualifies
only when it satisfies:

- positive absolute return;
- `CAGR/strict-MDD >= 2.0`;
- strict MDD at most `15%`;
- positive H1 and H2 absolute return;
- at least `50` trades;
- at least `20` trades in each half;
- at least `15` LONG and `15` SHORT trades;
- no action exceeds `90%`;
- positive stress return;
- positive one-bar-delay return;
- weekly-cluster one-sided `p < 0.15`;
- absolute return strictly above the strongest selected cheap policy; and
- ratio at least `0.25` above that cheap policy.

Qualifying checkpoints are selected lexicographically by:

1. higher 2022 `CAGR/strict-MDD`;
2. higher absolute return;
3. lower strict MDD;
4. earlier optimizer step.

If no checkpoint qualifies, retire PIVOT before 2023. Record hashes for every
checkpoint, retain only the selected DPO adapter and final SFT adapter, and
delete rejected checkpoint directories.

## Novelty contract

Comparator rows remain closed until:

1. source support passes;
2. the complete baseline/evaluator/model code is committed and hash-frozen;
3. one final policy/checkpoint is selected using 2022 only; and
4. that frozen policy emits its complete pre-2024 action clock, including a
   separately hash-frozen 2023 action-clock view, without opening 2023 market
   or funding values.

The frozen comparator cohort is:

| Comparator | Artifact/code binding | SHA-256 | Groups |
|---|---|---|---|
| CVICR primary and components | `data/cross_venue_intrinsic_clock_resolution_clocks_2020_2023.csv.gz` | `9f05b372686805539dbf56fb9b7ea7a8f90f8887d6731e1a8e1b1c1db14d8c0e` | `primary`, `gap_only`, `initial_conflict_only`, `late_alignment_only`, `no_leader_persistence`, `no_gap_tail`, `fixed_expected_time_clocks`, `stale_laggard_flow_24h`, `exact_direction_flip`, `deterministic_random_side`, `one_bar_execution_delay`, `one_hour_execution_delay` separately |
| CATCH-12 | `results/cash_auction_transfer_catchup_handoff_clock_2026-07-14.csv` | `066bf8e08267a043cc191eb436f0aa33105ab948de9f9f1edfde4d1c30de46d1` | `catch12` |
| CLASP-24 | `results/cash_late_arrival_spillover_propagation_clock_2026-07-14.csv` | `e166f4bd24afd5a2f129bcc26393ad4293ad0bc5792686b3b0fc4a805d53f9d5` | `clasp24` |
| LURI-48 | `results/leveraged_um_inventory_release_handoff_clock_2026-07-14.csv` | `50765cfed0c3ec6a0d1df18857c4e0a3e574d1aa449538c9b89cfac1fff67095` | `luri48` |
| CVTT | `data/cross_venue_temporal_torsion_v2_support_clocks_2020_2022.csv.gz` | `8f933b9d387fbcb764645a7002a5eefa9ee159c9c1ce7e007dca0dc4c16ebe33` | V01–V04 separately |
| IVLIR | `data/intrinsic_volume_latent_impact_relay_clocks_2020_2023.csv.gz` | `523f24a0d955fe99cfb86c62078532c5fc9091234e6669ab9acff2a8f3367788` | primary |
| IVFHR | `data/intrinsic_volume_flow_handoff_relay_clocks_2020_2023.csv.gz` | `ab12762dec9a93d41c293766e46dfc80ade81914fb32753a5923faa6437c338e` | primary and any_handoff separately |
| IVPLH | `data/intrinsic_volume_price_lag_handoff_clocks_2020_2023.csv.gz` | `2efca3b44b0512a9423da90171f43babcadec2316dc6148796f3e61f98138e80` | primary |
| CARTA state | `training/preregister_causal_adaptive_relational_tokens.py` | `a3a0be1c8c4401bfb707176d9def951938471805597d51c66f92500bafc4f4af` | binding |
| CARTA policy | `training/causal_adaptive_relational_bandit.py` | `7cb4428b39c923dc909fbd380cef6bb8647c47a5acef099d75c8d5c22d518b68` | ridge and NB emitted/executed clocks |
| CARTA evaluator | `training/evaluate_causal_adaptive_relational_baselines.py` | `130bc08767d6f4d71541215a66b4a88fdc160081e14849ab0000066bb7f3dc21` | binding |
| CARTA support | `results/causal_adaptive_relational_tokens_support_2026-07-14.json` | `77dfd1d0b0ad444744157972aa437f805901bc56428a4e5d76029bf64100d339` | binding |
| CARTA result | `results/causal_adaptive_relational_baseline_selection_2026-07-14.json` | `b17ef30fd97bc8054a49e42c84d406439c547b97fbd8fb94f0baf59625c55a75` | binding |
| Frozen live sleeves, physically pre-2024 clock | `results/cchr_live_portfolio_pure_clocks_2026-07-21.csv.gz` | `73d6efbd35b3be64b0fa04fa9c8cb2db25866ef884f19b1ae673949e22a42b08` | every candidate_id |
| Live clock manifest | `results/cchr_live_portfolio_pure_clock_manifest_2026-07-21.json` | `6c53ae482cf72bba0f286a47626842bf43070276ff5fe359be718e44864af57d` | binding |

After the PIVOT policy is frozen, a separately committed CARTA action-only
exporter may reproduce `relational_ridge` and `naive_bayes` actions from the
hash-bound code. It must write only policy ID, decision, entry, exit, action,
side, and execution-admission status. PIVOT reads only that action-only export.
Failure to reproduce both clocks retires PIVOT before eval.

The frozen live-sleeve clock contains exactly `440` action-only rows and its
manifest declares source end exclusive `2024-01-01T00:00:00Z`, with no
post-2023 rows loaded. The novelty loader must verify the file and manifest
hashes, schema, row count, and that every decision/entry/exit timestamp is
strictly before `2024-01-01T00:00:00Z`. Any row at or after that boundary is a
hard contamination failure. No raw 2024–2026 live source named by the manifest
may be opened.

For every required comparator, use the intersection of its declared coverage
and PIVOT's pre-2024 action clock. CVTT coverage ends at
`2023-01-01T00:00:00Z`; its absence in 2023 is not a missing-coverage failure.
For every required nonempty comparator over common coverage:

- exact entry-time Jaccard at most `0.10`;
- one-bar tolerant Jaccard at most `0.20`;
- twelve-bar/one-hour tolerant Jaccard at most `0.35`;
- absolute Pearson correlation of signed occupied exposure on the complete
  common five-minute grid at most `0.40`; and
- position-time Jaccard and incremental live-portfolio occupied time are
  reported but are not gates.

For IVLIR, IVFHR, IVPLH, and each CVICR intrinsic-clock group, six-hour tolerant
Jaccard must also be at most `0.60`.

Tolerant matching uses maximum-cardinality chronological one-to-one matching.
For sorted entry sets `a,b` and tolerance `w`, advance the earlier unmatched
entry when outside tolerance; otherwise match and advance both. Then:

```text
tolerant_jaccard =
  matched_pairs / (count_pivot + count_comparator - matched_pairs)
```

Hash mismatch, empty required common coverage, undefined signed-exposure
correlation, or any threshold failure retires PIVOT before 2023 outcomes.

These permanently forbidden artifacts must never be read:

```text
data/premium_compression_breakout_relay_clocks_2020_2026.csv.gz
data/premium_snapback_recenter_clocks_2020_2026.csv.gz
```

## Untouched 2023 eval gate

Only a complete source-support, cheap-baseline, RLLM-selection, invariance, and
novelty pass authorizes one 2023 outcome evaluation.

The frozen 2023 policy must satisfy all:

- positive absolute return;
- `CAGR/strict-MDD >= 3.0`;
- strict MDD at most `15%`;
- at least `60` trades;
- at least `20` trades in each half;
- at least `15` LONG and `15` SHORT trades;
- positive H1 and H2 absolute return;
- at least `8` active execution months;
- maximum single execution-month share at most `20%`;
- no action exceeds `90%` of opportunities;
- at least `20` nonempty UTC entry-week clusters;
- weekly-cluster one-sided `p < 0.10`;
- mean gross underlying move at least `20 bp` per trade;
- positive 10-bp/notional/side stress return;
- positive one-bar-delay return;
- positive base return under every action-option permutation audit;
- absolute return strictly above the strongest frozen cheap policy; and
- ratio at least `0.50` above the strongest frozen cheap policy.

The one-hour delay is mandatory reporting but not a gate.

Any failure retires PIVOT without token, prompt, action, threshold, checkpoint,
cost, delay, or hold repair.

## Sealed years

Only an unchanged 2023 pass may authorize an official-source extension. The
extension must reproduce all pre-2024 rows and tokens byte-for-byte before
opening later values.

Open later years sequentially:

1. 2024;
2. 2025 only after 2024 passes;
3. 2026 YTD as report-only forward evidence only after 2025 passes.

Each full sealed year independently must meet the unchanged 2023 economic,
risk, direction, half-year, stress, delay, and significance gates. Combined
2024–2025 weekly-cluster one-sided `p` must be below `0.05`. A later year may
not repair an earlier failure.

No leverage increase is authorized here. Passing at `0.5x` establishes risk
efficiency. Any later leverage decision is a separate deployment protocol.

## Mandatory implementation sequence

1. commit this mechanism;
2. commit a canonical preregistration manifest and synthetic tests;
3. commit the source-only state/support builder;
4. execute the source-support gate exactly once;
5. retire unchanged on any source/support failure;
6. if support passes, commit and hash-freeze the economic evaluator, baseline
   runner, action clock, model runner, and all controls;
7. open only 2020–2022 outcomes;
8. retire before GPU work if the cheap gate fails;
9. train SFT and the four DPO checkpoints only if authorized;
10. select one checkpoint on 2022 and freeze its 2023 action clock;
11. run novelty before opening 2023 outcomes;
12. evaluate 2023 exactly once;
13. open later years sequentially only after every prior gate passes; and
14. commit every completed unit with hashes and fresh tests.

No waiting, exploratory parameter sweep, alternate state, or post-failure
repair is part of PIVOT-72.

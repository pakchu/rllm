# BCRT-72 mechanism decision — block-clearing relational topology

## Decision

Freeze one new source-seen, outcome-unseen candidate:

**BCRT-72 — Block-Clearing Relational Topology Policy**.

BCRT is one text-only policy over a dense, delayed Bitcoin blockspace state.
It chooses:

```text
ABSTAIN
LONG
SHORT
```

No source feature, threshold, vote, or previous policy owns a direction. The
policy receives only a fixed twelve-token relational grammar. Raw values,
numeric ranks, dates, prices, returns, labels, prior actions, and portfolio PnL
are forbidden model inputs.

This document is binding to:

```text
docs/block-clearing-relational-topology-boundary-2026-07-24.md
SHA256 ab3d71d5b7f52254a3d25f4eeada35acab16ec7a2460528e2727b56ae8039560
commit d4d28645d57f3af04bfc89f80c0bb268f6e1fa26
```

The mechanism opens no BCRT feature, rank, token, incidence, market, funding,
comparator, return, PnL, label, or post-2023 value.

## Research-history boundary

Bitcoin base-chain source values and some other base-chain policy outcomes are
globally seen:

- BATE-288 opened 2021–2022 outcomes and failed strict train economics;
- UFCP-1, FETD-288, BFRT-288, WCTR-288, EMFC-864, and other fixed-rule
  candidates disclosed source-only support evidence; and
- the source manifest disclosed row count, height range, timestamps, and
  integrity results.

BCRT therefore makes no clean-room source-family claim. The following remain
unopened:

```text
BCRT primitive values
BCRT rolling ranks
BCRT token distribution
BCRT opportunity incidence
BCRT labels or actions
BCRT market outcomes
BCRT 2023 outcomes
```

Prior failures may motivate a different predictive object but may not supply a
BCRT side, threshold, token repair, or selected horizon. The already committed
BCRT boundary fixed the twelve-hour state, 48-hour minimum embargo, six-hour
hold, one-policy action space, and new candidate identity before this
mechanism.

## Frozen source

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

Exact primary allowlist and order:

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

Load-and-drop is forbidden. The loader must use the exact allowlist and fail on
path, hash, header, schema, type, or order drift.

The source must reproduce the manifest-bound facts:

```text
height range                 610691..823785 inclusive
rows                         213095
end timestamp exclusive      1704067200
unique block ids             true
contiguous heights           true
parent links                 213094
basic reference rows         213095
basic reference equality     true
utxo_set_change identity     total_outputs-total_inputs on every row
```

Validation also requires:

- integer heights, timestamps, median times, counts, sizes, weights, fees,
  inputs, outputs, and UTXO changes;
- `tx_count >= 1`, `size > 0`, `weight > 0`, and `total_fees >= 0`;
- `size <= weight <= 4*size`;
- timestamps and median times strictly before the frozen cutoff;
- lowercase 64-character hexadecimal block ids and parent ids; and
- no duplicate height or id.

No BTC price, funding, premium, OI, liquidation, order-book, return, label,
action, reward, PnL, portfolio, comparator, or 2024+ field may be decoded in
source support.

The historical transport is a frozen private Mempool REST cache. Production
requires an owned Bitcoin Core node, actual first-seen timestamps, raw-response
hashing, and forward field parity. A live mismatch blocks BCRT rather than
rewriting historical values.

## Exact twelve-hour bucket

Let:

```text
bucket_start = floor(header_timestamp / 43200) * 43200
bucket_end   = bucket_start + 43200
```

The bucket is **formed from a causal height prefix**, not from all later rows.
For each nominal `bucket_end`, define:

```text
anchor_height =
    first height whose mediantime >= bucket_end

confirmation_height =
    anchor_height + 288

bucket_members =
    rows with height <= confirmation_height
    and bucket_start <= timestamp < bucket_end
```

The anchor depends only on monotone chain progression past the bucket boundary.
The exact 288th successor closes membership permanently. A later block with a
backdated header timestamp inside an already formed bucket is ignored for that
bucket by definition and recorded as a late-member diagnostic; it never
rewrites the state.

A bucket is source-valid only when:

- `[bucket_start,bucket_end)` is inside `[2020-01-01,2024-01-01)`;
- `anchor_height` and exact `confirmation_height` both exist;
- `bucket_members` contains at least one block;
- every contributing row passed source validation;
- anchor and confirmation heights are strictly larger than those of the
  previous bucket; and
- the resulting signal-availability time is nondecreasing by bucket start.

The 288-successor condition is a causal prefix-closure device, not a tunable
event threshold. A latest bucket lacking its exact anchor or successor is
omitted.

The previous bucket for features and transitions is the immediately preceding
source-valid bucket by `bucket_start`, including a bucket later suppressed by
reservation or split containment.

## Exact causal clock

Define raw historical availability:

```text
prefix_max_timestamp =
    max(timestamp for every source row with height <= confirmation_height)

prefix_max_mediantime =
    max(mediantime for every source row with height <= confirmation_height)

raw_available_seconds =
    max(
        bucket_end,
        prefix_max_timestamp,
        prefix_max_mediantime,
    )
    + 172800
```

Define:

```text
ceil_5m(t) = 300 * ceil(t / 300)
signal_available = ceil_5m(raw_available_seconds)
entry            = signal_available + 300 seconds
exit             = entry + 72*300 seconds
```

If `raw_available_seconds` is exactly on a five-minute boundary, the ceiling
does not add another boundary. The separate complete five-minute inference and
order buffer always remains.

The prefix maximum prevents a forward-dated intermediate successor from being
ignored merely because the exact confirmation row has a lower header time.
The additional 48 hours begin only after the latest bucket boundary or
observed prefix clock, so the historical schedule cannot use the successor
header/MTP as if it were an exact receipt time. Historical header/median times
remain conservative proxies, not first-seen proof. Live availability is:

```text
max(frozen historical clock, actual node receipt/validation clock)
```

Execution:

- exact Binance BTCUSDT perpetual five-minute open at entry;
- fixed `0.5x` account gross exposure;
- exactly 72 held five-minute bars;
- scheduled exit only;
- no stop, take-profit, trailing, pyramiding, dynamic sizing, or model exit;
- reserve `[entry,exit)` globally before policy inference;
- abstention does not release a reservation;
- suppress rather than queue an overlapping later state; and
- require source bucket, confirmation, availability, latency, entry, all held
  bars, and exit to stay inside one half-open evaluation split.

The corrected clock depends on the exact anchor and 288th successor. Once that
successor forms the state, later source appends must not change the state,
rank, token, availability, or reservation.

The mandatory prefix-replay invariant is:

1. rebuild each formed bucket using only rows with
   `height <= confirmation_height`;
2. reproduce membership, primitives, ranks, tokens, availability, and
   reservation byte-for-byte;
3. append every later source prefix in turn; and
4. verify that no formed artifact changes.

Failure is terminal. Edge buckets lacking enough prefix or successor history
are omitted rather than repaired.

## Exact eight primitives

For one bucket, define:

```text
n             = block count
W             = sum(weight)
S             = sum(size)
T             = sum(tx_count)
F             = sum(total_fees)
U             = sum(utxo_set_change)
```

Then:

```text
cadence =
    log(n)

utilization =
    log((W + 1) / (4000000*n + 1))

packing =
    log((T + 1) / (W + 1))

fee_burden =
    log((F + 1) / (W + 1))

utxo_pressure =
    U / (T + 1)

witness_discount =
    (4*S - W) / (4*S)
```

For each contributing block:

```text
block_utilization_i = weight_i / 4000000
block_fee_density_i = log((total_fees_i + 1) / (weight_i + 1))
```

Let `MAD(x) = median(abs(x-median(x)))`. Then:

```text
load_dispersion = MAD(block_utilization)
fee_dispersion  = MAD(block_fee_density)
```

No clipping, winsorization, z-score, full-series normalization, weekday
conditioning, price conditioning, or future-row substitution is allowed.

## Strictly prior ranks

Primitive order:

```text
CADENCE
UTILIZATION
PACKING
FEE
UTXO
WITNESS
LOAD_DISPERSION
FEE_DISPERSION
```

Each primitive uses at most 252 immediately preceding source-valid buckets and
requires at least 126. For current finite value `x` and prior reference `R`:

```text
rank(x;R) =
    (count(R<x) + 0.5*count(R==x)) / len(R)
```

The current value is appended only after every current primitive rank is
fixed. Rank histories are independent by primitive. Missing or non-finite
current primitives invalidate that bucket but never widen a window or fill a
value.

A **rank-complete state** has all eight finite ranks in `[0,1]`. The first
rank-complete state has no policy token row but remains the predecessor of the
second.

## Exact twelve-token relational grammar

This grammar deliberately avoids per-feature absolute LOW/MID/HIGH buckets.
Most tokens compare two rolling ranks or describe topology across all eight
ranks.

### Pair relation

For a named pair `(left,right)`:

```text
d = rank(left) - rank(right)

d >  1/6 -> LEFT_LEADS
d < -1/6 -> RIGHT_LEADS
otherwise -> BALANCED
```

Exact `+1/6` and `-1/6` map to `BALANCED`.

Five pair tokens use semantic vocabulary:

```text
cadence_utilization:
    CADENCE_LEADS | BALANCED | UTILIZATION_LEADS

utilization_fee:
    UTILIZATION_LEADS | BALANCED | FEE_LEADS

packing_witness:
    PACKING_LEADS | BALANCED | WITNESS_LEADS

utxo_fee:
    UTXO_LEADS | BALANCED | FEE_LEADS

load_fee_dispersion:
    LOAD_WIDER | BALANCED | FEE_WIDER
```

### High and low leaders

`high_leader` is the identity of the unique maximum rank and `low_leader` the
unique minimum rank. Any exact tie for the relevant extreme maps to `TIE`.
There is no epsilon.

Vocabulary:

```text
CADENCE
UTILIZATION
PACKING
FEE
UTXO
WITNESS
LOAD_DISPERSION
FEE_DISPERSION
TIE
```

### Rank breadth

```text
high = count(rank > 0.5)
low  = count(rank < 0.5)
breadth = high-low

breadth >=  2 -> HIGH_BROAD
breadth <= -2 -> LOW_BROAD
otherwise     -> MIXED
```

Exact rank `0.5` contributes to neither side.

### Extreme occupancy

Count ranks strictly below `1/6` or strictly above `5/6`:

```text
0..2 -> COMPACT
3..5 -> FOCUSED
6..8 -> FRACTURED
```

Exact `1/6` and `5/6` are not extreme.

### Relation breadth

Score the five pair tokens:

```text
left semantic leader  +1
balanced               0
right semantic leader -1

sum >=  2 -> LEFT_BROAD
sum <= -2 -> RIGHT_BROAD
otherwise -> MIXED
```

For `load_fee_dispersion`, `LOAD_WIDER` is left and `FEE_WIDER` is right.

### Order transition

For every one of the 28 unordered primitive pairs, encode current and previous
strict rank order as `-1`, `0`, or `+1`. Count pair states that changed,
including entry into or out of an exact tie:

```text
0..6  -> STABLE
7..13 -> ROTATING
14..28 -> RESET
```

### Leader transition

Compare current and previous high/low leaders:

```text
any current/previous leader is TIE -> TIE_INVOLVED
neither changed                    -> BOTH_STABLE
only high changed                  -> HIGH_ROTATED
only low changed                   -> LOW_ROTATED
both changed                       -> BOTH_ROTATED
```

### Canonical order

The exact prompt token order is:

1. `cadence_utilization`
2. `utilization_fee`
3. `packing_witness`
4. `utxo_fee`
5. `load_fee_dispersion`
6. `high_leader`
7. `low_leader`
8. `rank_breadth`
9. `extreme_occupancy`
10. `relation_breadth`
11. `order_transition`
12. `leader_transition`

The prompt may contain only these twelve tokens, one frozen task identifier,
and one neutral-code option order.

Forbidden model inputs:

```text
raw source values or numeric ranks
bucket timestamp, date, year, month, weekday, height, id, or row identity
BTC price, return, funding, premium, OI, Kimchi, DXY, or future path
prior action, action history, reward, PnL, CAGR, MDD, or split identity
BATE/UFCP/FETD/BFRT/WCTR/EMFC signal, side, state, or outcome
source path, hash, or transport identity
free-form rationale, chain of thought, or generated feature
```

Current BCRT sleeve position is deterministically flat at every accepted
opportunity because the reservation clock precedes inference. External
portfolio conflicts are deterministic execution guards, not model tokens.
Unknown tokens, stale state, model errors, non-finite scores, or unexecutable
orders force `ABSTAIN`.

## Temporal roles

```text
initial fit                 [2020-01-01,2021-01-01)
transfer validation         [2021-01-01,2022-01-01)
final fit                   [2020-01-01,2022-01-01)
selection                   [2022-01-01,2023-01-01)
untouched candidate eval    [2023-01-01,2024-01-01)
sealed                      [2024-01-01,...)
```

Cheap policies first fit 2020 and transfer unchanged to 2021. A qualifying
cheap algorithm is refit on 2020–2021 and evaluated on 2022. Gemma fits only
2020–2021 and selects exactly one checkpoint on 2022.

2020–2022 are development evidence. 2023 is untouched by threshold, prompt,
feature, model, checkpoint, control, or market-outcome selection. The
hash-bound 2023 source covariates may be decoded for operational validity and
later immutable inference, but every 2023 incidence and token-distribution
statistic is report-only and cannot authorize continuation, retirement,
repair, or selection. There is no monthly, rolling, continuous, or eval-label
adaptation.

## Source-only support gate

The source-support builder may decode only the source allowlist and binding
files. It may not load market, funding, comparator, label, action, reward,
return, PnL, or post-2023 values.

Counts use token-ready, globally reserved, split-contained opportunities.
Suppressed states remain feature/rank/token predecessors.

### Integrity and incidence

All must pass:

- every frozen source, manifest, header, schema, height, link, reference, and
  identity check;
- exact bucket algebra and exact 288-successor confirmation;
- nondecreasing bucket maximum height and availability;
- strict-prior current exclusion, ties, 252 cap, and 126 minimum;
- future-append invariance for every earlier formed state and clock;
- at least 2,000 development opportunities in `[2020-01-01,2023-01-01)`;
- at least 1,250 opportunities in 2020–2021 combined;
- at least 570 opportunities in 2020;
- at least 700 opportunities in each of 2021 and 2022;
- at least nine active execution months in 2020;
- all twelve active months in each of 2021 and 2022;
- at least 340 opportunities per half-year in each of 2021 and 2022;
- at least 165 opportunities per quarter in each of 2021 and 2022;
- maximum 2020 single-month share at most 13%;
- maximum single-month share at most 10% in each of 2021 and 2022;
- maximum entry gap at most three calendar days in 2020–2022;
- exact six-hour hold and action-independent global reservation; and
- no raw, rank, action, side, market, return, or outcome column in the clock.

### Token support

Train means 2020–2021 combined. Train and 2022 must each satisfy:

- every value of every pair-relation token has at least 5% share;
- no pair-relation value exceeds 80%;
- at least five non-tie identities occur for each leader token;
- no non-tie leader identity exceeds 40%;
- `TIE` is at most 20% for each leader token;
- every rank-breadth value has at least 5% share and none exceeds 80%;
- every extreme-occupancy value has at least 2% share and none exceeds 90%;
- every relation-breadth value has at least 5% share and none exceeds 80%;
- every order-transition value has at least 3% share and none exceeds 85%;
- at least four leader-transition values occur and none exceeds 75%;
- largest exact twelve-token signature share is at most 5%;
- no token is missing or invalid; and
- every value appearing in 2022 already appears in train.

The same 2023 incidence, calendar, marginal-token, exact-signature, and
train-vocabulary statistics are emitted under a separate
`eval_source_report_only` namespace. They never enter a Boolean support gate.
An unseen 2023 token value forces `ABSTAIN` under the already frozen policy; it
does not permit a vocabulary change or candidate retirement before outcomes.

Required synthetic and real-prefix tests:

- exact path/hash/header/allowlist enforcement;
- height order, uniqueness, parent linkage, reference equality, UTXO identity,
  integer constraints, weight bounds, and pre-2024 rejection;
- exact 12-hour assignment at boundaries;
- 288-successor lookup and latest-bucket omission;
- current-value exclusion, midrank ties, fixed history, and primitive
  independence;
- pair threshold equality;
- exact leader ties;
- breadth, extreme occupancy, pair-order transition, and leader transition;
- predecessor inclusion despite reservation/split suppression;
- confirmation-aware future-append invariance;
- five-minute ceiling and latency;
- split containment; and
- action-independent half-open reservation.

Any failure retires BCRT-72 unchanged before market outcomes.

## Frozen accounting

Market:

```text
data/binance_um_kline_reference_btc_2020_2023/
  BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz
SHA256 e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d

data/binance_um_kline_reference_btc_2020_2023/build_manifest.json
SHA256 c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e
```

Funding:

```text
data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz
SHA256 3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6

results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json
SHA256 a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b
```

Costs and quantity:

```text
leverage                         0.5
base cost/notional/side          0.0006
stress cost/notional/side        0.0010
quantity                         entry_equity*0.5/entry_open
```

Stress replaces base cost. Quantity remains fixed until scheduled exit.

For funding settlement `entry <= time <= exit`:

```text
funding_cash =
    -side * fixed_quantity * settlement_mark_price * funding_rate
```

Positive funding exactly at entry or exit is dropped; negative boundary
funding is retained. Interior funding is retained.

Strict MDD uses one global/pre-entry high-water mark and marks:

1. post-entry cost;
2. favorable held OHLC extreme;
3. adverse held OHLC extreme with retained funding and virtual adverse exit
   cost; and
4. scheduled-exit equity.

Favorable before adverse is deliberately conservative. Full-calendar CAGR uses
the complete half-open declared interval, including warm-up, idle cash, and
abstention:

```text
years = (end-start)/(365.25 days)
CAGR  = final_equity**(1/years)-1
```

Every report includes absolute return, full-calendar CAGR, strict MDD,
CAGR/strict-MDD, trades, LONG/SHORT counts, action shares, side contributions,
mean signed gross move, months, halves, quarters, funding, stress cost, delay
controls, and weekly-cluster significance.

Weekly-cluster sign flip:

- group net compounded account returns by UTC ISO entry week;
- independent `numpy.random.default_rng(20260724)` per split/policy;
- 100,000 Rademacher sign draws; and
- `p=(1+count(null>=observed))/100001`.

No-trade/no-cluster policies return `1.0`.

### Familywise selection correction

Every development-stage policy comparison also uses one shared max-stat null.
For each policy, place weekly net returns on the union of nonempty UTC weeks,
using zero for a week in which that policy is flat, and compute:

```text
t_policy =
    mean(weekly_return)
    / (std(weekly_return,ddof=1)/sqrt(number_of_union_weeks))
```

Zero variance returns `-infinity`. For each of 100,000 draws, apply the same
Rademacher sign to the same week for every policy and retain the maximum null
`t` across the complete frozen family. The adjusted one-sided p-value is:

```text
p_max =
    (1 + count(max_null_t >= observed_selected_t)) / 100001
```

The family contains every emitted primary, exact-memory, prior-only,
quarter-phase-prior, single-token, group-only, leave-one-token-out,
leave-one-group-out, shuffled-label, circular-block-shift, final-SFT, and DPO
checkpoint policy that exists at that stage. A policy cannot be omitted after
its result is seen.

The ordinary policy-local sign-flip p-value remains reported. Development
selection gates use `p_max`; the one immutable 2023 policy uses the ordinary
one-policy p-value because no 2023 candidate family is selected.

## Train-only utility and labels

For each fit opportunity and action:

```text
U(ABSTAIN) = 0

U(trade) =
    log(max(account_multiplier,1e-12))
  - (1/3)*local_held_path_strict_drawdown
  - 0.0005
```

The final term is an account-level hurdle, not an execution cost.

Tie priority:

```text
ABSTAIN, LONG, SHORT
```

One model performs two sequential tasks:

```text
ADMISSION: ABSTAIN versus TRADE
DIRECTION: LONG versus SHORT, only after TRADE
```

Admission preference exists only when:

```text
abs(max(U(LONG),U(SHORT))-U(ABSTAIN)) >= 0.0003
```

Direction preference exists only when the best trade beats abstention and:

```text
abs(U(LONG)-U(SHORT)) >= 0.0003
```

Retain every qualifying row. Outcome-dependent sampling, class/direction
balancing, hard-negative mining, source symmetry, and synthetic labels are
forbidden.

Before GPU work:

- neither admission target exceeds 90%;
- LONG and SHORT each form at least 20% of direction targets; and
- both preference sets are nonempty.

## Cheap causal baselines

Tokens are nominal, never integer-ordinal encoded.

Representation:

- one-hot every train-observed token value;
- one-hot all 66 unordered token-pair conjunctions;
- retain features occurring at least five times in fit data; and
- one unpenalized intercept where supported.

Unknown downstream values force `ABSTAIN`.

Frozen policies:

1. always abstain;
2. always long;
3. always short;
4. exact-signature majority oracle memory, unseen abstains;
5. categorical Naive Bayes, Laplace alpha `1.0`;
6. separate LONG/SHORT ridge utility regressions, alpha `100.0`;
7. separate LONG/SHORT Extra Trees utility regressions:
   - 512 estimators,
   - squared-error criterion,
   - max depth 6,
   - min split 24,
   - min leaf 12,
   - sqrt max features,
   - no bootstrap,
   - seed 20260724;
8. fit-majority three-action prior;
9. fit admission-prior plus direction-prior constant policy;
10. quarter-phase prior:
    - fit one majority oracle action for each UTC calendar quarter number
      `Q1..Q4`;
    - apply only the matching quarter-number prior downstream;
    - diagnostic control only because calendar identity is forbidden to BCRT;
11. 32 shuffled-label Naive Bayes controls, seeds `20260724..20260755`;
12. 32 independently shuffled-utility ridge controls, same seeds;
13. 16 circular block-shift label/utility controls over chronological fit rows,
    with exact positive offsets:

    ```text
    [62,93,124,155,186,217,248,279,
     310,341,372,403,434,465,496,527]
    ```

14. twelve single-token ridge policies;
15. twelve leave-one-token-out ridge policies;
16. five group-only ridge policies:
    - pair relations only,
    - leaders only,
    - breadth/occupancy only,
    - transitions only,
    - current topology without transitions; and
17. five leave-one-group-out ridge policies over the same groups.

Circular controls preserve label prevalence and much of temporal
autocorrelation while destroying state alignment. None of the prior, seasonal,
shuffled, or circular controls may qualify as BCRT.

### 2020 to 2021 transfer gate

At least one learned primary algorithm must satisfy on unchanged 2021:

- positive absolute return;
- `CAGR/strict-MDD >= 0.5`;
- strict MDD at most 15%;
- positive H1 and H2 return;
- at least 200 trades;
- at least 60 LONG and 60 SHORT trades;
- positive LONG and SHORT net contribution separately;
- no action above 90%;
- positive stress-cost return;
- positive one-bar-delay return; and
- familywise weekly-cluster `p_max < 0.25`;
- higher return and ratio than every prior-only, quarter-phase, shuffled, and
  circular control; and
- higher return and ratio than the strongest single-token or group-only policy.

### 2022 cheap learnability gate

The refit 2020–2021 algorithm must satisfy on 2022:

- positive absolute return;
- `CAGR/strict-MDD >= 1.0`;
- strict MDD at most 15%;
- positive H1 and H2 return;
- at least 240 trades;
- at least 75 LONG and 75 SHORT trades;
- positive LONG and SHORT net contribution separately;
- no action above 85%;
- positive stress-cost return;
- positive one-bar-delay return;
- familywise weekly-cluster `p_max < 0.10`;
- higher return and ratio than every prior-only, quarter-phase, shuffled, and
  circular control;
- higher return and ratio than the strongest single-token or group-only
  policy; and
- no single-token majority-action policy reproduces more than 70% of selected
  non-abstain actions on matching opportunities.

Select by higher ratio, higher return, lower MDD, then lexicographically smaller
policy id. Failure retires BCRT before GPU.

## Single-Gemma RLLM

Model:

```text
google/gemma-2-2b-it
revision 299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8
AutoModelForCausalLM
AutoTokenizer
trust_remote_code=False
```

Runtime-used file hashes:

| File | SHA-256 |
|---|---|
| `config.json` | `eacec6c5ca317a87ed2c46789d9705b9274db5027e7ba59da739bfae23addb55` |
| `generation_config.json` | `a543a5d299bc2b20c52bd87ed174f561266510b57a392e12b5b5d758d798ce05` |
| `tokenizer_config.json` | `cb32b7929c62608d46572e813112b3ad8a841fb98fdd6a4da8559e368a951c89` |
| `tokenizer.json` | `3f289bc05132635a8bc7aca7aa21255efd5e18f3710f43e3cdb96bcd41be4922` |
| `model.safetensors.index.json` | `ada0043f3e3b2e5ab2f445cad9c0fbbf9d91ad444675e6a82b822591c63abf5a` |
| `model-00001-of-00002.safetensors` | `532d792c9178805064170a3ec485b7dedbfccc6fd297b92c31a6091b6c7e41bf` |
| `model-00002-of-00002.safetensors` | `6d6d9ce84db398fb6e0191f91542e5da0a73da2cb695e172a24edc2146dc8d20` |
| `special_tokens_map.json` | `baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351` |

Environment:

```text
torch             2.9.0
transformers git  5d7ff4393ab99aa7cadf4cccd1f814dbb799f2bb
trl               0.29.0
peft              0.18.1
bitsandbytes      0.49.2
numpy             2.2.6
pandas            2.3.3
scikit-learn      1.7.2
```

Quantization:

```text
4-bit NF4
double quantization true
compute dtype float16
```

LoRA:

```text
r=16
alpha=32
dropout=0.05
bias=none
task_type=CAUSAL_LM
targets=[q_proj,k_proj,v_proj,o_proj]
```

Memory/artifact gates:

- inference reserved CUDA at most 6.5 GiB;
- training reserved CUDA at most 16 GiB;
- training allocated CUDA at most 14 GiB;
- each adapter/checkpoint directory at most 256 MiB; and
- retained final SFT plus selected DPO at most 1 GiB.

Text only. Images, multimodal processors, analyzer/trader pairs, free-form
rationales, generated features, and hidden-reasoning targets are forbidden.

### Neutral-code scoring

Serialize tokens `KEY=VALUE`, one per line, in canonical order:

```text
TASK=ADMISSION
OPTIONS=<Q1,Q2 or Q2,Q1>
Return exactly CHOICE=<one option>.

TASK=DIRECTION
OPTIONS=<Q1,Q2 or Q2,Q1>
Return exactly CHOICE=<one option>.
```

Mapping:

```text
ADMISSION Q1=ABSTAIN Q2=TRADE
DIRECTION Q1=LONG    Q2=SHORT
```

Generation is forbidden. For each task and code:

1. score normalized conditional completion-token log probability;
2. average both displayed option orders;
3. score the same prompt with the adapter disabled;
4. compute `adapter_delta=adapted-base`;
5. subtract one task/code mean adapter delta over original 2020–2021 fit
   states; and
6. freeze the resulting offset with the checkpoint.

Offsets are never recomputed on 2022, 2023, or later data.

The train-mean adapter-delta subtraction removes global code bias but is not
treated as sufficient evidence against label-prior learning. One frozen
prior-only null adapter is trained with the identical SFT/DPO recipe and
labels, except every input line is replaced by its key plus literal
`MASKED`. It receives no BCRT state. The null is evaluated on 2022 and its
pre-2024 predictions are frozen. It can never qualify as BCRT; the selected
BCRT checkpoint must beat it on return, ratio, and familywise significance.

Admission trades only when calibrated `Q2>Q1`. Direction is LONG when
calibrated `Q1>Q2` and SHORT when `Q2>Q1`. Any tie or failure abstains.

Maximum prompt plus completion length is 384 tokens. Truncation is forbidden.

### SFT

```text
optimizer             AdamW
learning_rate         2e-4
betas                 (0.9,0.999)
epsilon               1e-8
weight_decay          0.01
scheduler             cosine
warmup_steps          16
max_grad_norm         1.0
optimizer_steps       128
per_device_batch      1
gradient_accumulation 8
packing               false
completion_only_loss  true
fp16                  true
bf16                  false
seed                  20260724
```

Both option orders receive the same target. Final SFT initializes DPO; no SFT
checkpoint is selected.

### DPO

```text
loss                  sigmoid
beta                  0.1
label_smoothing       0.0
reference             final SFT with DPO updates disabled
optimizer             AdamW
learning_rate         5e-6
betas                 (0.9,0.999)
epsilon               1e-8
weight_decay          0.01
scheduler             cosine
warmup_steps          16
max_grad_norm         1.0
optimizer_steps       192
per_device_batch      1
gradient_accumulation 8
fp16                  true
bf16                  false
seed                  20260724
checkpoints           [48,96,144,192]
```

Each checkpoint is evaluated once on 2022. It qualifies only with:

- positive absolute return;
- `CAGR/strict-MDD >= 2.0`;
- strict MDD at most 15%;
- positive H1 and H2 return;
- at least 240 trades;
- at least 75 LONG and 75 SHORT trades;
- positive LONG and SHORT net contribution separately;
- no action above 85%;
- positive stress-cost return;
- positive one-bar-delay return;
- familywise weekly-cluster `p_max < 0.05`;
- return and ratio above every frozen cheap primary, prior-only,
  quarter-phase, shuffled, circular, single-token, group-only, and ablation
  policy under common coverage;
- return and ratio above the masked-token prior adapter; and
- no single token value contains more than 60% of non-abstain actions;
- no single-token majority-action policy reproduces more than 70% of selected
  non-abstain actions; and
- ratio at least 0.25 above the strongest frozen non-RLLM policy.

Select by higher ratio, higher return, lower MDD, then earlier optimizer step.
Failure retires BCRT before 2023. Retain final SFT and selected DPO only.

## Novelty gate

Comparator rows remain closed until source support, cheap transfer, evaluator,
model training, scoring invariance, and 2022 checkpoint selection pass. The
selected policy first emits one immutable pre-2024 action clock without
opening 2023 outcomes.

Bind:

| Comparator | Path | SHA-256 |
|---|---|---|
| BATE-288 | `results/block_arrival_throughput_elasticity_clock_2026-07-20.csv` | `cd4fbd01c104bd969ca1c12a53b8da82dd0e9376990e233c286ff009a5115c02` |
| UFCP-1 | `results/utxo_fee_clearing_polarity_primary_clock_2026-07-20.csv` | `8338c290d63b522531c8d55c8a79ba73cc13915c936733ec03ffcf6ab0e86c1b` |
| MCR-7 | `results/miner_cadence_recovery_clock_2026-07-17.csv` | `2535244889b046ff00c369ee854973a91c23429dff82a6dd3c1a293a01352b0b` |
| NTB-7 | `results/network_topology_broadening_clock_2026-07-17.csv` | `6b1bd7c7458cffa062e40872c3ad1730007c01426790b1ba8e52c6eb853de42f` |
| BFC-3 | `results/blockspace_fee_confirmation_clock_2026-07-17.csv` | `edda7bb8ae8a1de4e51a3b86e98d533748e73d203125a3ded1a487e9a0e93632` |
| WCTR-288 | `results/witness_composition_transport_primary_clock_2026-07-20.csv.gz` | `7a6b56a3024d0d087322fad7b3229276c539b93374691cd2812af0630dc752b1` |
| BFRT-288 | `results/block_feerate_breadth_transport_primary_clock_2026-07-20.csv.gz` | `33428d29c2ace9b23672b2dc9dc3e9ba0e3020fa1a6e3845d55fa5d75230d64a` |
| EMFC-864 | `results/exact_maturity_fee_cadence_polarity_clocks_2026-07-20.csv` | `31af41f42ffe4dc73f0ff35ccf278e38c856d224184e802e46b370650d35951d` |
| frozen live sleeves | `results/cchr_live_portfolio_pure_clocks_2026-07-21.csv.gz` | `73d6efbd35b3be64b0fa04fa9c8cb2db25866ef884f19b1ae673949e22a42b08` |

Load only policy/group id, decision/admission when present, entry, exit, and
side/action. Raw comparator source features and outcomes are forbidden.

FETD-288 did not publish row-level clocks. Bind its sealed support report:

```text
results/fee_endpoint_topology_disagreement_support_2026-07-20.json
SHA256 03ba910a314ba6efb647f6588dff603261d414e5114680ca33bdc27d59aed035
```

Do not reconstruct or decode its unpublished event rows. Its mechanism overlap
is instead challenged by the frozen fee/utilization pair-only, group-only, and
single-token reducibility controls.

For every nonempty comparator over required common coverage:

- exact and one-hour tolerant entry Jaccard are reported;
- unsigned time containment is reported;
- absolute signed occupied-exposure Pearson correlation must be at most 0.35;
- zero variance, undefined correlation, missing coverage, duplicate entry, or
  hash drift fails; and
- against each frozen live sleeve, exact entry Jaccard must be at most 0.10
  and one-hour tolerant Jaccard at most 0.25.

Failure retires BCRT before 2023 outcomes.

## Untouched 2023 gate

The unchanged selected BCRT policy must satisfy:

- positive absolute return;
- `CAGR/strict-MDD >= 3.0`;
- strict MDD at most 15%;
- positive H1 and H2 return;
- at least 240 trades;
- at least 75 LONG and 75 SHORT trades;
- positive LONG and SHORT net contribution separately;
- at least ten active execution months;
- maximum single execution-month share at most 15%;
- no action above 85%;
- at least 30 nonempty UTC entry-week clusters;
- one-policy weekly-cluster one-sided `p < 0.05`;
- mean signed gross underlying move at least 20 bp per trade;
- positive stress-cost return;
- positive one-bar-delay return;
- positive return under every neutral-code option-order audit;
- return and ratio above every frozen cheap primary, prior-only,
  quarter-phase, shuffled, circular, single-token, group-only, and ablation
  policy under common coverage;
- return and ratio above the masked-token prior adapter;
- no single token value contains more than 60% of non-abstain actions;
- no single-token majority-action policy reproduces more than 70% of selected
  non-abstain actions; and
- ratio at least 0.50 above the strongest frozen non-RLLM policy.

One-hour delay is mandatory reporting, not a gate.

Any failure retires BCRT-72 unchanged.

## Sealed years

Only an unchanged 2023 pass authorizes official-source extension. An extension
must reproduce all pre-2024 buckets, ranks, tokens, clocks, and actions
byte-for-byte.

Open sequentially:

1. 2024;
2. 2025 only after 2024 passes; and
3. 2026 YTD as report-only after 2025 passes.

Each full year independently must pass the unchanged 2023 economic, risk,
direction, half, stress, delay, and significance gates. Combined 2024–2025
weekly-cluster `p` must be below 0.05. No leverage increase is authorized.

## Mandatory sequence

1. commit this mechanism;
2. commit canonical preregistration and synthetic tests;
3. commit source-only builder and tests;
4. execute source support once;
5. retire unchanged on any source/token failure;
6. if support passes, commit and hash-freeze evaluator and cheap baselines;
7. open only 2020–2022 market/funding outcomes;
8. retire before GPU on transfer or cheap learnability failure;
9. train one frozen Gemma SFT and four DPO checkpoints only if authorized;
10. select one checkpoint on 2022;
11. freeze pre-2024 actions and run novelty before 2023;
12. evaluate 2023 exactly once; and
13. open sealed years sequentially only after every prior pass.

## Outcome boundary

At this mechanism commit:

```text
source artifact bytes hashed            yes
source/manifest aggregate metadata read yes
source CSV header read                  yes
BCRT source values decoded              0
BCRT buckets derived                    0
BCRT primitive/rank values derived      0
BCRT token rows derived                 0
BCRT opportunity rows derived           0
market rows loaded                      0
funding rows loaded                     0
comparator rows decoded                 0
future-return rows loaded               0
return or PnL fields                    0
post-2023 rows loaded                   0
model labels created                    0
model training runs                     0
```

Status:

```text
mechanism_frozen_before_BCRT_values
```

# CSPG-288 mechanism decision — CBOE cross-surface pressure grammar

## Decision

Freeze one singleton:

**CSPG-288 — CBOE Cross-Surface Pressure Grammar Policy, 24-hour hold**.

Every valid rank-complete common CBOE source date creates an
action-independent opportunity. One text-only policy receives twelve compact
relation tokens and chooses:

```text
ABSTAIN
LONG
SHORT
```

No source surface, vote, majority, threshold event, or prior candidate owns the
direction.

This commit opens no new CSPG source value, rank, token, opportunity, market
row, funding row, comparator row, return, PnL, model label, or post-2023 row.
It authorizes only an immutable preregistration followed by one source-only
support build.

## Binding boundary

```text
docs/cboe-cross-surface-pressure-grammar-boundary-2026-07-24.md
SHA256 0b6feb15d1e7b616b5b65bb266b15db7e3fdcf82765b5848c76d68e804cb39f2
clock-correction commits f428adb, e210951
```

The boundary's prior-failure, contamination, source-vintage, live-parity,
alternative, and stopping-rule disclosures remain binding.

## Research-history quarantine

CSPG deliberately reuses the CBOE source family and primitive causal pressure
coordinates whose source-only incidence was opened by CXRT and OPRR. It is not
an independent clean-room discovery.

The admissible difference is a new predictive object:

- CXRT fixed a three-vote majority side and tested its run/composition;
- OPRR required a sparse rank-rotation event;
- CSPG creates no vote, majority side, or event eligibility and instead learns
  a three-action policy from the dense relation grammar.

The prior reports explicitly required any successor to change observable/state
geometry rather than loosen their failed rules. No CXRT/OPRR market outcome was
opened, and no CSPG token distribution or outcome informed the frozen grammar.

To quarantine the shared-source conditioning:

- CXRT and OPRR action groups remain mandatory novelty comparators;
- excessive same-side reproduction or signed-exposure correlation retires
  CSPG before 2023 outcomes;
- CSPG cannot claim a globally pristine holdout;
- the first failed CSPG gate is terminal; and
- no CSPG control may be promoted after inspection.

## Immutable predictor sources

### Term structure

```text
data/cboe_volatility_term_structure_2018_2023/
  cboe_vix_term_structure_2018-01-01_2023-12-31.csv.gz
SHA256 6f1b2f7f3a5b1e4d5001d673e6ff54374791879c278248ce27b3d610e4f75dc7

data/cboe_volatility_term_structure_2018_2023/build_manifest.json
SHA256 42b2a35ad131bd63574d2adcf684e28766dc3060fa645fc749df10dd3fb27f27
```

Exact ordered allowlist:

```text
observation_date
VIX9D_close
VIX_close
VIX3M_close
```

### Tail risk

```text
data/cboe_tail_risk_2018_2023/
  cboe_tail_risk_2018-01-01_2023-12-31.csv.gz
SHA256 cdde3f8d4bb1e23d00b192f5f9ef759aefba9087be5fd60653e9c02479dfa41a

data/cboe_tail_risk_2018_2023/build_manifest.json
SHA256 9ef80ef3034c93d97c5b2a8160b2502527287d570d15f9d7166d631d9866c7bd
```

Exact ordered allowlist:

```text
observation_date
SKEW_close
VVIX_close
VIX_close
```

### Option flow

```text
data/cboe_option_flow_2020_2023/
  cboe_option_flow_2020-01-01_2023-12-31.csv.gz
SHA256 35ef106ef01e3abadbcb4a6227187dd1d7cf2722191bd146bac06d08d1684a78

data/cboe_option_flow_2020_2023/build_manifest.json
SHA256 0a513b146ad5857d9ab7311e978152c308de64db8ef29c4d463eb07ea503089e
```

Exact ordered allowlist:

```text
observation_date
total_volume
index_call_volume
index_put_volume
index_volume
equity_call_volume
equity_put_volume
vix_call_volume
vix_put_volume
```

Each CSV loader must receive its exact `usecols` allowlist. Load-and-drop is
forbidden. File, manifest, header, schema, or data-range drift fails before a
state is constructed.

Within each panel:

- dates are unique and strictly increasing;
- dates are strictly before 2024;
- every retained numeric primitive is finite and strictly positive; and
- unknown or missing fields fail closed.

Term and tail `VIX_close` must match exactly on common dates.

Each surface builds its own causal history before the exact date intersection.
Missing dates are never filled, carried, interpolated, zeroed, or substituted
from a later release.

The source-only builder may load no BTC price, return, funding, premium, OI,
Kimchi, FX, label, action, reward, PnL, portfolio, comparator, or 2024+ field.

These artifacts are frozen current historical vintages, not point-in-time
revision archives. Live promotion requires forward raw response capture,
retrieval timestamps, content hashes, schema/value comparison, and revision
alarms. A forward discrepancy blocks live CSPG rather than rewriting history.

## Strictly prior primitive rank

Every primitive rank uses at most 252 immediately preceding observations from
its own source and at least 126.

For current finite value `x` and strictly prior reference `R`:

```text
rank(x; R) =
    (count(R < x) + 0.5*count(R == x)) / len(R)
```

The current value is appended only after every current primitive rank is fixed.
A missing primitive invalidates only that source date before intersection; no
window widening, expanding normalization, z-score, clipping, winsorization,
weekday conditioning, or later-row substitution is allowed.

## Exact three pressure coordinates

### Term pressure

```text
front_slope = log(VIX9D_close / VIX_close)
broad_slope = log(VIX_close / VIX3M_close)

front_rank = strict_prior_rank(front_slope)
broad_rank = strict_prior_rank(broad_slope)

term_pressure = 0.5 * (front_rank + broad_rank)
```

### Tail pressure

```text
skew_level    = log(SKEW_close / 100)
vvix_relative = log(VVIX_close / VIX_close)

skew_rank = strict_prior_rank(skew_level)
vvix_rank = strict_prior_rank(vvix_relative)

tail_pressure = 0.5 * (skew_rank + vvix_rank)
```

### Option-flow pressure

```text
institutional_gap =
    log((index_put_volume + 0.5) / (index_call_volume + 0.5))
  - log((equity_put_volume + 0.5) / (equity_call_volume + 0.5))

vix_call_pressure =
    log((vix_call_volume + 0.5) / (vix_put_volume + 0.5))

index_share =
    log((index_volume + 1.0) / (total_volume + 1.0))
```

For each option primitive, take current level minus the immediately preceding
option-source level. Rank each delta against its own strictly prior delta
history:

```text
option_pressure =
    mean(
        delta_institutional_gap_rank,
        delta_vix_call_pressure_rank,
        delta_index_share_rank,
    )
```

The fixed pseudocounts only define log ratios. They are not tunable.

All three pressure coordinates must be finite and in `[0,1]`.

## Common state and predecessor

A **rank-complete common state** is the exact date intersection of independently
rank-complete term, tail, and option states.

Sort by `observation_date`. The previous common state is the immediately
preceding rank-complete common date, including a state later suppressed by
reservation or split containment. The first common state has no CSPG token row
but remains the predecessor of the second.

Future source appends must leave every earlier pressure and token
byte-identical.

## Exact twelve-token grammar

### Pressure level

For each pressure `p`:

```text
p < 1/3  -> LOW
p > 2/3  -> HIGH
otherwise -> MID
```

Exact equality to either boundary maps to `MID`.

### Per-surface change

Compare the current pressure level index `LOW=0, MID=1, HIGH=2` with the
immediately previous common state:

```text
current < previous -> DOWN
current = previous -> SAME
current > previous -> UP
```

### Cross-surface extremes

`stress_leader` is the identity of the unique maximum raw pressure.
`relief_leader` is the identity of the unique minimum raw pressure.

Vocabulary:

```text
TERM
TAIL
OPTION
TIE
```

Any exact tie for the relevant extreme maps to `TIE`; no epsilon is used.

### Dispersion

```text
spread = max(term_pressure, tail_pressure, option_pressure)
       - min(term_pressure, tail_pressure, option_pressure)

spread < 1/6 -> COMPRESSED
spread < 1/3 -> SEPARATED
otherwise    -> FRACTURED
```

Exact `1/6` maps to `SEPARATED`; exact `1/3` maps to `FRACTURED`.

### Agreement topology

Use the three pressure-level indices:

```text
max(level)-min(level) == 0 -> UNISON
max(level)-min(level) == 1 -> ADJACENT
max(level)-min(level) == 2 -> POLARIZED
```

### Topology transition

Compare current and previous values of:

```text
term_level
tail_level
option_level
stress_leader
relief_leader
```

Count changed fields:

```text
0..1 -> STABLE
2..3 -> ROTATING
4..5 -> RESET
```

### Pressure breadth

From the three per-surface change tokens:

```text
breadth = count(UP) - count(DOWN)

breadth < 0 -> FALLING
breadth = 0 -> BALANCED
breadth > 0 -> RISING
```

### Canonical order

The exact token order and vocabulary are:

1. `term_level`: `LOW | MID | HIGH`
2. `tail_level`: `LOW | MID | HIGH`
3. `option_level`: `LOW | MID | HIGH`
4. `term_change`: `DOWN | SAME | UP`
5. `tail_change`: `DOWN | SAME | UP`
6. `option_change`: `DOWN | SAME | UP`
7. `stress_leader`: `TERM | TAIL | OPTION | TIE`
8. `relief_leader`: `TERM | TAIL | OPTION | TIE`
9. `dispersion`: `COMPRESSED | SEPARATED | FRACTURED`
10. `agreement`: `UNISON | ADJACENT | POLARIZED`
11. `topology_transition`: `STABLE | ROTATING | RESET`
12. `pressure_breadth`: `FALLING | BALANCED | RISING`

The prompt may contain only these tokens, one frozen task identifier, and one
neutral-code option order.

Forbidden model inputs include:

```text
raw values or ranks
CXRT votes, majority, side, eligibility, or run state
OPRR rotation eligibility
date, year, month, weekday, timestamp, row number, or identifier
source name, path, hash, or split identity
BTC price, return, funding, premium, OI, Kimchi, DXY, or future path
reward, action history, PnL, CAGR, MDD, or portfolio state
free-form rationale or generated feature
```

Current position is not a token. Strategy reservation makes CSPG flat at each
permitted opportunity; external-position conflicts are deterministic guards.
An unknown downstream token level or unexecutable state forces `ABSTAIN`.

## Causal availability and execution

The corrected binding boundary freezes this future-row-independent clock before
any CSPG value is opened.

For source observation date `D`, define `D+1` as the next **calendar** date,
not the next business, exchange, or source date:

```text
source state                 = completed close values from D
signal availability          = calendar D+1 09:30 America/New_York
decision/order/entry         = calendar D+1 09:35 America/New_York
entry                        = exact UTC-converted BTCUSDT 5m open
exit                         = entry + 288*5 minutes
```

The five minutes from availability to entry are a complete inference/order
buffer. DST conversion uses Python `zoneinfo.ZoneInfo("America/New_York")`;
fixed UTC offsets are forbidden.

Weekend and holiday entries are valid because BTCUSDT trades continuously.
Missing future CBOE rows cannot create, suppress, or move an already formed
clock. The fixed next-calendar-day clock is a conservative frozen-vintage proxy,
not proof of live publication time. Live production must use the later of this
clock and actual fully captured availability.

Other execution rules:

- fixed account gross exposure `0.5x`;
- exactly 288 held five-minute bars;
- scheduled exit only;
- no stop, take-profit, trailing, pyramiding, dynamic sizing, or model exit;
- reserve `[entry,exit)` before policy inference;
- abstention does not release the interval;
- suppress, never queue or replace, any later overlapping opportunity;
- build and reserve the complete pre-2024 clock globally before splits;
- a split-crossing reserved state remains reserved but is omitted from that
  split's economics; and
- source date, availability, inference buffer, entry, all held bars, and exit
  must stay inside one half-open split.

Inference timeout, model error, stale data, missing token, non-finite score, or
position conflict means `ABSTAIN`; a late fill is never backdated.

Execution-delay controls preserve the frozen action and rebuild reservation:

```text
one_bar_delay:  entry+5m,  exit+5m
one_hour_delay: entry+60m, exit+60m
```

## Temporal roles

```text
initial fit / transfer origin [2020-01-01, 2021-01-01)
transfer validation           [2021-01-01, 2022-01-01)
final model fit               [2020-01-01, 2022-01-01)
selection                     [2022-01-01, 2023-01-01)
untouched eval                [2023-01-01, 2024-01-01)
sealed                        [2024-01-01, ...)
```

Cheap algorithms first fit on 2020 and must transfer unchanged to 2021. A
qualifying algorithm is refit on 2020–2021 and evaluated on 2022. Gemma fits
only 2020–2021 and selects one checkpoint only on 2022.

2020, 2021, and 2022 are development evidence. 2023 is the sole untouched
candidate-level outcome window. No monthly, rolling, continuous, or eval-label
adaptation is permitted.

## Source-only support gate

The support builder may load only the predictor allowlists and binding files.
It may not load market, funding, comparator, label, reward, action, PnL, or
post-2023 values.

Counts use token-ready, globally reserved, split-contained opportunities.
Suppressed states remain predecessors.

### Incidence and time support

All must pass:

- exact source hashes, manifests, headers, schemas, date order, positivity,
  cross-panel VIX equality, and pre-2024 limits;
- at least 820 opportunities globally;
- at least 330 opportunities in 2020–2021 combined;
- at least 100 opportunities in 2020;
- at least 230 opportunities in each of 2021, 2022, and 2023;
- at least six active months in warm-up year 2020;
- all twelve months active in each of 2021, 2022, and 2023;
- at least 108 opportunities per half-year in each of 2021–2023;
- at least 48 opportunities per quarter in each of 2021–2023;
- maximum 2020 single-month share at most 20%;
- maximum single-month share at most 12% in each of 2021–2023; and
- maximum entry gap at most ten calendar days in every year.

The prior outcome-blind CXRT/OPRR reports already disclosed 879 rank-complete
common dates and 878 schedulable dates. The 820 global floor allows more than
6% attrition for the new predecessor, fixed-clock, reservation, and split
rules; it does not assume an undisclosed CSPG token distribution.

### Token support

Train means 2020–2021 combined. Train, 2022, and 2023 each must satisfy:

- `LOW` and `HIGH` each have at least 8% share for every pressure-level token;
- no pressure-level token level exceeds 80%;
- `DOWN` and `UP` each have at least 8% share for every change token;
- no change-token level exceeds 80%;
- `TERM`, `TAIL`, and `OPTION` each occur as stress leader and relief leader;
- no non-tie leader level exceeds 80%;
- all three dispersion levels occur and each has at least 3% share;
- all three agreement levels occur and each has at least 3% share;
- all three topology-transition levels occur and each has at least 2% share;
- `FALLING` and `RISING` each have at least 12% share;
- no pressure-breadth level exceeds 75%;
- largest exact twelve-token signature share is at most 5%;
- no token is missing or invalid; and
- every level appearing in 2022 or 2023 already appears in train.

The implementation must pass synthetic and real-prefix tests for:

- exact path/hash/header/allowlist enforcement;
- date order, uniqueness, positivity, and pre-2024 rejection;
- cross-panel VIX equality;
- independent histories before exact-date intersection;
- current-value exclusion from ranks;
- exact midrank ties;
- fixed 252/126 history;
- option delta history;
- missing-date non-fill;
- future-append invariance;
- prior-common transition semantics;
- exact level-boundary behavior;
- exact extreme ties;
- DST-aware fixed next-calendar-date clock independent of future row existence;
- neutral-code option-order serialization invariance for both tasks; and
- action-independent global reservation.

Any support failure retires CSPG-288 unchanged. No source, rank, threshold,
token, clock, support floor, latency, or hold repair is allowed.

## Frozen execution accounting

### Costs and quantity

```text
leverage = 0.5
base fee + slippage = 0.0006 of notional per side
stress replacement cost = 0.0010 of notional per side
```

Stress replaces rather than adds to base cost.

```text
quantity   = entry_equity * leverage / entry_open
entry_cost = quantity * entry_open * cost_per_side
exit_cost  = quantity * exit_open * cost_per_side
```

Quantity is fixed until scheduled exit.

### Funding

For settlement `entry_time <= funding_time <= exit_time`:

```text
funding_cash = -side * quantity * settlement_mark_price * funding_rate
```

`side=+1` for LONG, `-1` for SHORT. Positive funding exactly at entry or exit
is dropped; negative boundary funding is retained. Interior funding is
retained.

### Equity, strict MDD, and CAGR

Trades compound chronologically. Realized equity is entry equity less entry
cost plus directional scheduled-exit PnL plus retained funding less exit cost,
floored at zero.

Strict MDD keeps one high-water mark across the full declared calendar,
including idle cash and pre-entry history. For every trade, mark post-entry
cost, then favorable held extreme, then adverse held extreme with all applicable
funding and a virtual adverse exit cost, then scheduled-exit equity. Favorable
before adverse is deliberately conservative for aggregate OHLC drawdown.

For half-open calendar `[start,end)`:

```text
years = (end-start) / (365.25 days)
CAGR  = final_equity ** (1/years) - 1
```

Idle, warm-up, and abstention time remain in the denominator.

Every report includes absolute return, full-calendar CAGR, strict MDD,
CAGR/strict-MDD, trades, LONG/SHORT counts, action shares, each side's net
contribution, mean signed gross underlying move, active months, halves,
quarters, stress cost, delay controls, and weekly-cluster significance.

### Weekly cluster sign-flip

Assign each trade's net compounded account return to its UTC ISO entry week and
sum by week. Initialize `numpy.random.default_rng(20260724)` independently per
split/policy. Draw 100,000 Rademacher sign vectors over nonempty weeks.

```text
p = (1 + count(null >= observed)) / 100001
```

No-trade or no-cluster policies return `1.0`.

## Train-only utility and labels

For each action at each fit opportunity, compute local accounting from entry
equity `1.0`:

```text
U(ABSTAIN) = 0

U(trade) =
    log(max(account_multiplier, 1e-12))
  - (1/3) * local_held_path_strict_drawdown
  - 0.0010
```

The last term is a ten-basis-point account-level trade hurdle, not an execution
cost.

The deterministic three-action oracle uses tie priority:

```text
ABSTAIN, LONG, SHORT
```

The RLLM is one model with two sequential binary tasks, not two models:

```text
ADMISSION: ABSTAIN versus TRADE
DIRECTION: LONG versus SHORT, evaluated only after TRADE
```

For admission:

```text
trade_utility = max(U(LONG), U(SHORT))

emit one preference only when abs(trade_utility - U(ABSTAIN)) >= 0.0005
target TRADE   when trade_utility > U(ABSTAIN)
target ABSTAIN when trade_utility < U(ABSTAIN)
```

For direction:

```text
emit one preference only when:
    trade_utility > U(ABSTAIN)
    and abs(U(LONG)-U(SHORT)) >= 0.0005

target LONG  when U(LONG) > U(SHORT)
target SHORT when U(SHORT) > U(LONG)
```

SFT emits the same two targets where defined. Retain every qualifying row.
Outcome-dependent oversampling, downsampling, direction balancing, class
balancing, hard-negative mining, and synthetic source symmetry are forbidden.

Before GPU work:

- neither ADMISSION target exceeds 85% of qualifying 2020–2021 admission rows;
- LONG and SHORT each form at least 20% of qualifying direction rows; and
- admission and direction preference sets are both nonempty.

Failure retires the RLLM path rather than rebalancing labels.

## Cheap causal baselines

Tokens are nominal and never integer-ordinal encoded.

### Common representation

- one-hot indicators for every train-observed token level;
- one-hot indicators for all 66 unordered token-pair conjunctions;
- only features occurring at least three times in fit data; and
- one unpenalized intercept where supported.

Unknown downstream levels force `ABSTAIN`.

### Frozen policies

1. `always_abstain`
2. `always_long`
3. `always_short`
4. `exact_signature_memory`
   - majority fit-oracle action by exact signature;
   - tie priority `ABSTAIN`, `LONG`, `SHORT`;
   - unseen signature abstains.
5. `categorical_naive_bayes`
   - field-wise categorical likelihood;
   - Laplace alpha `1.0`;
   - oracle-action target.
6. `ridge_contextual_value`
   - separate LONG and SHORT utility regressions;
   - ridge alpha `100.0`;
   - unpenalized intercept;
   - ABSTAIN value zero.
7. `extra_trees_contextual_value`
   - same binary matrix;
   - separate LONG and SHORT utility regressions;
   - `n_estimators=512`;
   - `criterion="squared_error"`;
   - `max_depth=5`;
   - `min_samples_split=20`;
   - `min_samples_leaf=10`;
   - `max_features="sqrt"`;
   - `bootstrap=False`;
   - `random_state=20260724`;
   - ABSTAIN value zero.
8. 32 shuffled-label Naive Bayes controls, seeds `20260724..20260755`.
9. 32 independently shuffled-action-utility ridge controls, same seeds.
10. twelve single-token ridge policies.
11. twelve leave-one-token-out ridge policies.

### 2020 to 2021 transfer gate

Fit learned policies on 2020 only and evaluate unchanged on 2021. At least one
of Naive Bayes, ridge, or Extra Trees must have:

- positive absolute return;
- `CAGR/strict-MDD >= 0.5`;
- strict MDD at most 15%;
- positive H1 and H2 return;
- at least 50 trades;
- at least 15 LONG and 15 SHORT trades;
- positive LONG and SHORT net contribution separately;
- no action above 90%;
- positive stress-cost return;
- positive one-bar-delay return; and
- weekly-cluster one-sided `p < 0.25`.

Only a passing algorithm may be refit on 2020–2021.

### 2022 cheap learnability gate

The refit algorithm must have:

- positive absolute return;
- `CAGR/strict-MDD >= 1.0`;
- strict MDD at most 15%;
- positive H1 and H2 return;
- at least 60 trades;
- at least 20 LONG and 20 SHORT trades;
- positive LONG and SHORT net contribution separately;
- no action above 85%;
- positive stress-cost return;
- positive one-bar-delay return;
- weekly-cluster one-sided `p < 0.15`;
- strictly higher return and ratio than the strongest shuffled control; and
- strictly higher return and ratio than the strongest single-token policy.

Select by higher ratio, higher return, lower MDD, then lexicographically
smaller policy ID. If none qualifies, retire before GPU work.

## Single-Gemma RLLM

### Model and runtime

```text
base model = google/gemma-2-2b-it
revision   = 299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8
loader     = transformers.AutoModelForCausalLM
tokenizer  = transformers.AutoTokenizer
trust_remote_code = False
```

This exact causal-LM revision already has a repository-validated 4-bit text
runtime and a 3060-Ti-class deployment path. Gemma 4 E2B is not used because
the frozen repository run required multimodal runtime files and observed
11.78 GiB reserved inference memory, above the intended live budget.

Runtime-used base files are hash-bound:

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

Text only. Images, multimodal processors, analyzer/trader pairs, free-form
rationales, hidden reasoning targets, and generated features are forbidden.

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
bnb_4bit_compute_dtype=torch.float16
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

- inference reserved CUDA at most 6.5 GiB;
- training reserved CUDA at most 16 GiB;
- training allocated CUDA at most 14 GiB;
- each adapter/checkpoint directory at most 256 MiB; and
- retained final SFT plus selected DPO artifacts at most 1 GiB.

Failure retires the RLLM path without changing runtime or recipe.

### Two-task prompt and label-prior correction

Serialize canonical tokens one `KEY=VALUE` per line. The same single model is
queried sequentially:

```text
TASK=ADMISSION
OPTIONS=<Q1,Q2 or Q2,Q1>
Return exactly CHOICE=<one option>.

TASK=DIRECTION
OPTIONS=<Q1,Q2 or Q2,Q1>
Return exactly CHOICE=<one option>.
```

Neutral code mapping is fixed:

```text
ADMISSION Q1=ABSTAIN, Q2=TRADE
DIRECTION Q1=LONG,    Q2=SHORT
```

The direction task is scored only if admission selects TRADE. Generation is
forbidden. For each task/code:

1. score conditional completion-token log probability normalized by completion
   token count;
2. average over both displayed option orders;
3. repeat with the selected adapter disabled on the exact same prompt;
4. define `adapter_delta = adapted_score - base_score`;
5. compute one train-only offset as the mean adapter delta over every original
   2020–2021 fit state for that task/code; and
6. define `calibrated_score = adapter_delta - train_offset`.

Offsets are computed once per checkpoint from 2020–2021 only, hash-frozen with
the checkpoint, and never recomputed on 2022, 2023, or later data. This removes
both base completion-token prior and a checkpoint's global code bias while
preserving state-conditional adaptation.

Admission trades only when calibrated `Q2 > Q1`. A tie abstains. After a trade
admission, direction is LONG when calibrated `Q1 > Q2` and SHORT when
calibrated `Q2 > Q1`; a tie abstains.

Missing, malformed, non-finite, incomplete, offset-missing, or overlength
scores force `ABSTAIN`.

Maximum prompt plus completion length is 384 tokens; truncation is forbidden.

### SFT

Both option orders receive the same defined 2020–2021 admission or direction
target.

```text
optimizer             = AdamW
learning_rate         = 2e-4
betas                 = (0.9,0.999)
epsilon               = 1e-8
weight_decay          = 0.01
scheduler             = cosine
warmup_steps          = 8
max_grad_norm          = 1.0
optimizer_steps       = 64
per_device_batch_size = 1
gradient_accumulation = 8
packing               = False
completion_only_loss  = True
fp16                  = True
bf16                  = False
seed                  = 20260724
```

The final SFT adapter initializes DPO. No SFT checkpoint is selected.

### DPO

Use all frozen 2020–2021 admission and direction preference pairs. The
reference is final SFT with DPO updates disabled.

```text
loss                   = DPO sigmoid
beta                   = 0.1
label_smoothing        = 0.0
optimizer              = AdamW
learning_rate          = 5e-6
betas                  = (0.9,0.999)
epsilon                = 1e-8
weight_decay           = 0.01
scheduler              = cosine
warmup_steps            = 8
max_grad_norm           = 1.0
optimizer_steps         = 96
per_device_batch_size   = 1
gradient_accumulation   = 8
fp16                    = True
bf16                    = False
seed                    = 20260724
checkpoints             = [24,48,72,96]
```

Every checkpoint is evaluated exactly once on 2022. It qualifies only with:

- positive absolute return;
- `CAGR/strict-MDD >= 2.0`;
- strict MDD at most 15%;
- positive H1 and H2 return;
- at least 70 trades;
- at least 25 LONG and 25 SHORT trades;
- positive LONG and SHORT net contribution separately;
- no action above 85%;
- positive stress-cost return;
- positive one-bar-delay return;
- weekly-cluster one-sided `p < 0.10`;
- return strictly above the selected cheap policy; and
- ratio at least 0.25 above the selected cheap policy.

Select by higher ratio, higher return, lower MDD, then earlier optimizer step.
If none qualifies, retire before 2023. Retain final SFT and selected DPO only.

## Novelty gate

Comparator rows remain closed until source support passes, all evaluator/model
code is committed and hash-frozen, and one 2022-selected policy emits its
immutable pre-2024 action clock without opening 2023 outcomes.

Bind:

| Comparator | Artifact | SHA-256 |
|---|---|---|
| VTR | `results/cboe_volatility_term_rotation_clocks_2026-07-17.csv.gz` | `47f4ca447daa2b03a0827ad243ed1107eb34a37e5d7bab18ecd3c4331736959d` |
| THD | `results/cboe_tail_hedge_disagreement_clocks_2026-07-18.csv.gz` | `0e19455e2fb5ab2d36cc996c9adf514adc85c69dd1a325562344a8015464d546` |
| IHM | `results/cboe_institutional_hedge_migration_clocks_2026-07-18.csv.gz` | `5e04cffacb1754c3111fcc32b09d72f06b546a4803b40c77d655a9787b015c0b` |
| CXRT groups | `data/cboe_cross_surface_risk_transfer_clocks_2020_2023.csv.gz` | `b3cc6f3d6a19cb39ef63ec0ba9908c983ce03c56a0c7dd8786e51c2ef1c0885f` |
| OPRR groups | `data/cboe_option_pressure_rank_rotation_clocks_2020_2023.csv.gz` | `a5c15e0d6444f79239276fb9c3da0555dea27a52eda254e7425d9b223d30d46c` |
| Frozen live sleeves | `results/cchr_live_portfolio_pure_clocks_2026-07-21.csv.gz` | `73d6efbd35b3be64b0fa04fa9c8cb2db25866ef884f19b1ae673949e22a42b08` |

Comparator loaders may read only policy/group ID, decision, entry, exit,
action/side, and admission status.

Because CBOE policies share a daily source clock, timing Jaccard against CBOE
comparators is report-only. For each nonempty CBOE group:

- overlapping-entry same-side reproduction at most 70%;
- absolute signed occupied-exposure correlation at most 0.60.

Against each frozen live sleeve over common coverage:

- exact entry Jaccard at most 0.10;
- one-hour tolerant Jaccard at most 0.25;
- absolute signed occupied-exposure correlation at most 0.40.

Twenty-four-hour tolerant overlap and position-time Jaccard are reported, not
gates. Hash drift, undefined correlation, missing required common coverage, or
threshold failure retires CSPG before 2023 outcomes.

Never read:

```text
data/premium_compression_breakout_relay_clocks_2020_2026.csv.gz
data/premium_snapback_recenter_clocks_2020_2026.csv.gz
```

## Untouched 2023 gate

Only complete source, transfer, cheap-policy, RLLM-selection, scoring
invariance, and novelty passes authorize one 2023 outcome evaluation.

The frozen CSPG policy must satisfy all:

- positive absolute return;
- `CAGR/strict-MDD >= 3.0`;
- strict MDD at most 15%;
- positive H1 and H2 return;
- at least 70 trades;
- at least 25 LONG and 25 SHORT trades;
- positive LONG and SHORT net contribution separately;
- at least ten active execution months;
- maximum single execution-month share at most 15%;
- no action above 85%;
- at least 20 nonempty UTC entry-week clusters;
- weekly-cluster one-sided `p < 0.10`;
- mean signed gross underlying move at least 20 bp per trade;
- positive stress-cost return;
- positive one-bar-delay return;
- positive return under every admission/direction neutral-code order audit;
- return strictly above the strongest frozen cheap policy; and
- ratio at least 0.50 above the strongest frozen cheap policy.

The one-hour delay is mandatory reporting but not a gate.

Any failure retires CSPG-288 unchanged.

## Sealed years

Only an unchanged 2023 pass may authorize official-source extension. Extension
must reproduce all pre-2024 ranks, tokens, and action clocks byte-for-byte.

Open sequentially:

1. 2024;
2. 2025 only after 2024 passes;
3. 2026 YTD as report-only after 2025 passes.

Each full year independently must pass the unchanged 2023 economic, risk,
direction, half, stress, delay, and significance gates. Combined 2024–2025
weekly-cluster one-sided `p` must be below 0.05.

No leverage increase is authorized here.

## Mandatory sequence

1. commit this mechanism;
2. commit canonical preregistration and synthetic tests;
3. commit source-only state/support builder;
4. execute source support once;
5. retire unchanged on any source/support failure;
6. if support passes, commit and hash-freeze evaluator, cheap baselines,
   action-clock exporter, model runner, and controls;
7. open only 2020–2022 outcomes;
8. retire before GPU if transfer or cheap learnability fails;
9. train frozen SFT and four DPO checkpoints only if authorized;
10. select one checkpoint on 2022;
11. emit and novelty-check the pre-2024 action clock;
12. open 2023 once;
13. open later years sequentially only after every prior gate; and
14. commit every completed unit with fresh tests and hashes.

## Outcome boundary

```text
new CSPG source values read       = 0
CSPG pressures derived            = 0
CSPG token rows created           = 0
CSPG opportunity clocks opened    = 0
BTC market rows read              = 0
funding rows read                 = 0
comparator rows read              = 0
future-return rows read           = 0
return or PnL fields read         = 0
post-2023 source rows read        = 0
model labels created              = 0
model training runs               = 0
```

Status:

```text
frozen_for_preregistration
```

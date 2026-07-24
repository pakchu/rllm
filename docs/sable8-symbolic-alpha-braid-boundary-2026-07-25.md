# SABLE-8 boundary — symbolic alpha-braid target-position policy

Date: 2026-07-25

## Decision

Freeze one new candidate:

**SABLE-8 — Symbolic Alpha-Braid Language Executor, eight-hour target-position
policy**.

SABLE-8 asks a different question from the repository's prior event gates:

> Can one compact Gemma 4 policy infer a persistent BTC target position from
> the ordered interaction of several causal weak-signal families, while
> deterministic code owns every numeric transform, source clock, order,
> funding cash flow, cost, reward, and risk calculation?

The candidate is a dense sequential control problem, not a new conjunction,
threshold scan, fixed-hold event sleeve, analyzer/trader pipeline, text
classifier, or theorem prover. At each canonical Binance funding boundary it
chooses the next target position from `SHORT`, `FLAT`, and `LONG`. The position
persists until a later decision changes it.

SABLE-8 is explicitly a **contaminated-history research MDP**. Its component
families have already been exposed to broad BTC outcomes elsewhere in this
repository. A historical pass can establish only candidate-specific
learnability and execution consistency; it cannot create a pristine holdout or
authorize capital.

This commit authorizes only implementation of the source/token support gate.
It does not authorize reward construction, GPU training, post-2023 market
evaluation, or a profitability claim.

## Why the prior block-height proposal is rejected

A provisional successor based on `floor(height/144)` was considered and
rejected before implementation. The repository already contains absolute
height packets based on `floor(height/72)` in FETD and BLSR. Merely doubling
the packet width and adding a model would be a parameterized repair, not a new
predictive object.

BCRT also showed that its block source and relational grammar were dense, but
its frozen multi-day clock failed the source-support gap gate. BCRT remains
retired unchanged. SABLE-8 does not reuse BCRT's:

- twelve-hour timestamp buckets;
- 288-successor closure;
- 48-hour historical embargo;
- six-hour fixed hold;
- global action-independent reservation;
- twelve-token block topology; or
- candidate side, threshold, or support failure.

## Why the governance-text proposal is deferred

DeFi governance text plus executable payloads is a genuine semantic source,
but it is not selected for the next run:

- on-chain proposals are sparse and protocol-concentrated relative to a
  target-position process;
- public proposal prose has material pretrained-text contamination risk;
- immutable text, executable payloads, archive RPC, IPFS content, and live
  parity would need a new source contract before any alpha work; and
- it resembles the recently retired ECRL semantic-classifier-to-RL shape.

Governance remains an independent future source axis. It is not a SABLE
fallback or control.

## Relation to prior weak-signal work

SABLE-8 uses causal primitive families that survived only as usable source
representations or weak/context signals in prior audits:

1. completed price path and rolling range;
2. realized jump versus bipower variation;
3. causal volume-clock flow;
4. price/liquidity recovery efficiency;
5. realized funding and completed premium;
6. open-interest versus price pressure; and
7. fresh Kimchi/FX/DXY context.

Prior experiments mostly selected sparse conjunctions, fixed sides, fixed
holds, or allow gates. The post-funding static scan also used settlement as an
entry event and failed. SABLE-8 does not repair those rules:

- settlement is only a stable decision clock, not a directional trigger;
- no prior alpha threshold, side, TP, SL, or hold enters the policy;
- every decision is present, including flat periods;
- the action is a persistent portfolio target rather than a one-shot event
  direction; and
- turnover cost makes maintaining, flattening, and reversing distinct actions.

Some of these families failed as standalone alphas. SABLE does not relabel
those failures as evidence. Its only new claim is that the ordered,
position-aware composition may contain information not captured by static
rules. That claim must beat deterministic unions, non-LLM sequence models, and
the existing sleeve universe before it is retained.

## Frozen historical source containers

The three files below are immutable historical containers, not production
provider contracts. Their hashes bind Stage 0 extraction. The support runner
must materialize physically bounded pre-2024 cuts under
`data/sable8_source_cuts/pre2024/` and a tracked manifest under `results/`.
Every cut is gzip-compressed deterministically (`mtime=0`, fixed column order,
UTF-8, LF) and receives its own SHA-256.

Exact Stage 0 outputs:

```text
data/sable8_source_cuts/pre2024/market.csv.gz
data/sable8_source_cuts/pre2024/funding.csv.gz
data/sable8_source_cuts/pre2024/premium.csv.gz
results/sable8_source_cut_manifest_pre2024_2026-07-25.json
results/sable8_symbolic_alpha_braid_support_2026-07-25.json
```

### BTC market, external context, and open interest

```text
path
  data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz
SHA256
  dbc9e53b09551b469168fe19cc750c5c3ea86278db3055d079103f7654050192
```

Exact physical header:

```text
date,open,high,low,close,volume,quote_asset_volume,number_of_trades,
taker_buy_base,taker_buy_quote,tic,day,dxy,kimchi_premium,usdkrw,btckrw,
dxy_available,kimchi_available,usdkrw_available,external_any_available,
dxy_zscore,dxy_momentum,kimchi_premium_zscore,kimchi_premium_change,
usdkrw_zscore,usdkrw_momentum,open_interest,open_interest_value,
cmc_circulating_supply,open_interest_available
header SHA256
  c306861dde4024d44622d34e664188f41636c8bab6f544db740213dee71ab58b
```

Exact support-cut projection:

```text
date
open
high
low
close
quote_asset_volume
taker_buy_quote
dxy
kimchi_premium
usdkrw
dxy_available
kimchi_available
usdkrw_available
open_interest
open_interest_available
```

The loader must fail on path, hash, exact physical header and order, timestamp,
numeric, positivity, or availability drift. It may tokenize unprojected CSV
cells only to preserve row framing; it may not convert, retain, hash by value,
aggregate, or expose any unprojected cell. The bounded output contains exactly
the support-cut projection above. Pandas-style load-all-then-drop is forbidden.

### Realized BTCUSDT funding

```text
path
  data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz
SHA256
  4d381be086e275bacaf31df431dc31307a71a26b3947b7082efffc10bb129dd7
allowlist
  date,funding_rate,funding_time
physical header
  date,symbol,funding_rate,funding_time,mark_price
header SHA256
  1c09a5cc3f8b5e7f0c06f0055e364d0dc97a9677dd505535e6f59d3cb9b48202
```

### Completed BTCUSDT premium

```text
path
  data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz
SHA256
  b45fcc5a3cf75c8e594effe61a698c4652f841b1d304107e9669524e0fc9d0d7
allowlist
  date,close,close_time
physical header
  date,symbol,open,high,low,close,close_time
header SHA256
  22e5715846fbfa49646f0c7c9078d40455e2e726d8e316fcdd49dbfebbd626ed
```

The support gate may parse only source rows strictly before
`2024-01-01T00:00:00Z`. It may not parse a post-2023 numeric value. Gzip
streaming must stop at the first ordered post-2023 timestamp before converting
any other field in that row. All support calculations then read only the
physically bounded cuts, never the full-history containers.

No source-support code may load a future return, label, action, reward, PnL,
CAGR, MDD, prior model prediction, prior alpha activation, portfolio weight,
or post-2023 numeric source value.

### Field-level provenance and live boundary

The fused market container is accepted for historical support only under these
field identities:

| field family | historical identity | causal timestamp rule | live admission |
|---|---|---|---|
| OHLCV and taker quote | Binance USD-M `BTCUSDT` five-minute bars | `date` is bar open; row becomes usable only at `date+5m` | completed `bars_binance` parity |
| open interest | `public.open_interest_binance`, `BTCUSDT`, `period=5m` | exact current and lagged observation; no stale as-of promotion | same table/period and `open_interest_available=1` |
| funding | official Binance USD-M realized funding archive | exact `funding_time`; no inferred schedule value | realized funding row parity |
| premium | official Binance USD-M premium-index one-hour kline | only after exact `close_time` | completed `bars_binance_premium` parity |
| Kimchi | completed Binance BTCUSDT, Upbit KRW-BTC, and fresh USDKRW-derived cache field | both current and lag availability must be true | recomputed constituents plus freshness parity |
| USDKRW | fused external FX field | current and lag availability must be true | live FX row; weekends/holidays fail to `STALE` |
| DXY | synthetic completed FX-component field | current and lag availability must be true | same component basket and completion clock |

No fused-cache value is sufficient for live promotion. Before shadow, each row
above needs a separately committed adapter contract, provider/table identity,
freshness threshold, restart behavior, and field-by-field offline/live parity
test. Missing, stale, provider-disagreed, or schema-drifted inputs force
`FLAT`; numeric neutral fill may preserve tensor shape but may not authorize a
non-flat action.

### Sealed post-2023 source extension

Stage 0 and all cheap/Gemma specification work remain physically pre-2024.
After complete Stage 2 and Stage 2.5 passes, each later source interval is
opened by a separate transaction:

1. materialize only the next interval's bounded source cuts;
2. write and commit paths, hashes, row bounds, schemas, freshness counts, and
   prefix-replay hashes;
3. close the tree cleanly;
4. only then run the already frozen evaluator for that interval.

The required order and half-open source intervals are:

```text
2024
  data/sable8_source_cuts/2024/{market,funding,premium}.csv.gz
  results/sable8_source_cut_manifest_2024.json
  [2024-01-01T00:00:00Z, 2025-01-01T00:00:00Z)

2025
  data/sable8_source_cuts/2025/{market,funding,premium}.csv.gz
  results/sable8_source_cut_manifest_2025.json
  [2025-01-01T00:00:00Z, 2026-01-01T00:00:00Z)

2026H1
  data/sable8_source_cuts/2026h1/{market,funding,premium}.csv.gz
  results/sable8_source_cut_manifest_2026h1.json
  [2026-01-01T00:00:00Z, 2026-06-01T00:00:00Z)
```

Source decoding and outcome evaluation must be separate commits and separate
process invocations. The prefix portion reconstructed from each extension must
byte-match the previous cut's canonical rows and token hashes. A mismatch
retires SABLE without opening that interval's rewards.

## Frozen eight-hour process clock

Canonical funding boundaries are:

```text
00:00 UTC
08:00 UTC
16:00 UTC
```

For boundary `B`:

```text
state cutoff       B + 5 minutes
decision time      B + 10 minutes
execution time     B + 15 minutes, at the five-minute open
next execution     B + 8 hours + 15 minutes
```

The market `date` field is the five-minute bar open. The state may use only
rows satisfying `date+5m <= state_cutoff`; therefore the newest eligible row at
boundary `B` has `date=B`. Funding may use only exact returned funding events
whose event time is no later than the cutoff. Premium may use only rows whose
`close_time` is no later than the cutoff.

The extra five-minute bar after the funding boundary and the following
five-minute decision latency are fixed operational delays, not tunable holds.
The policy is queried at every sequence-ready boundary. There is no event
filter, cooldown, overlap selector, TP, SL, or fixed event exit. A boundary
whose core source line or six-line sequence is invalid executes the
deterministic safety action `FLAT`; it is not silently skipped. That forced
transition remains in wall-clock PnL and risk accounting but is not a model
training decision.

At split boundaries:

- the initial target is flat;
- state history may use causal pre-split source context;
- equity starts at one and high-water mark starts at one;
- reward and strict path accounting may not cross the exact split end;
- the final target is liquidated at the exact split-end five-minute open;
- funding stamped exactly at the split end belongs to the position held
  immediately before that forced liquidation;
- no replay transition crosses the terminal boundary.

## Frozen causal primitives

Let `t` be the last completed market bar at the state cutoff, `c` close,
`r = log(c/c[-1])`, `q` quote notional, and
`a = 2*taker_buy_quote-q`. Every rolling window includes only completed rows
not later than `t`.

The exact primitive order is:

```text
1  price_return_1d
2  range_location_7d
3  volatility_ratio_1d_30d
4  jump_share_1d
5  signed_jump_1d
6  volume_clock_flow_speed_25
7  liquidity_signed_efficiency_6h
8  taker_flow_recovery_1h_6h
9  funding_sum_24h
10 premium_mean_8h
11 oi_price_divergence_1d
12 kimchi_change_12h
13 usdkrw_change_12h
14 dxy_change_1d
```

Exact equations:

```text
price_return_1d =
    log(c_t / c_t-288)

range_location_7d =
    (c_t - min(c_t-2015..t)) /
    (max(c_t-2015..t) - min(c_t-2015..t))
```

If the range denominator is exactly zero, location is `0.5`.

```text
RV_1d = sum(r^2 over t-287..t)
RV_30d = sum(r^2 over t-8639..t)

volatility_ratio_1d_30d =
    log((RV_1d + 1e-18) / (RV_30d/30 + 1e-18))

BV_1d =
    (pi/2) * sum(abs(r_i)*abs(r_i-1) over t-287..t)

jump_share_1d =
    max(RV_1d-BV_1d,0) / RV_1d

signed_jump_1d =
    sum(r^3 over t-287..t) / RV_1d^(3/2)
```

Zero `RV_1d` invalidates both jump primitives.

For the volume clock, first compute target notional from the complete 24 hours
strictly before the current five-minute bar:

```text
target = 0.25 * sum(q over t-288..t-1)
```

Walk backward from `t` and stop at the first bar that makes the inclusive
notional reach or exceed `target`. Equivalently, choose the largest index `j`
such that `sum(q over j..t) >= target`. Then:

```text
duration = t-j+1
volume_clock_flow_speed_25 =
    (sum(a over j..t) / sum(q over j..t)) / duration
```

No future-volume target, interpolation, or partial bar is allowed.

```text
liquidity_signed_efficiency_6h =
    log(c_t/c_t-72) /
    sum(abs(r) over t-71..t)

taker_imbalance_i = a_i/q_i

taker_flow_recovery_1h_6h =
    mean(taker_imbalance over t-11..t)
    - mean(taker_imbalance over t-71..t)
```

Zero path length or quote notional invalidates the corresponding primitive.

```text
funding_sum_24h =
    sum of exact funding_rate where cutoff-24h < funding_time <= cutoff

premium_mean_8h =
    arithmetic mean of completed premium close values where
    cutoff-8h < close_time <= cutoff

oi_price_divergence_1d =
    log(open_interest_t/open_interest_t-288) - price_return_1d

kimchi_change_12h = kimchi_premium_t-kimchi_premium_t-144
usdkrw_change_12h = log(usdkrw_t/usdkrw_t-144)
dxy_change_1d = log(dxy_t/dxy_t-288)
```

Funding requires all three canonical events in the prior 24 hours. Premium
requires eight completed hourly rows. OI requires current and lagged
`open_interest_available`. External fields require their current availability
flag and the lagged observation's availability flag. A stale external value
is never promoted to a fresh current primitive.

No clipping, full-series normalization, centered window, backward fill, future
fill, label-conditioned transform, calendar-conditioned transform, or
post-2023-fitted threshold is allowed.

## Strictly-prior ordinal language

Each primitive has an independent history of at most the previous 540
source-valid SABLE decision states, equivalent to 180 days at three decisions
per day. At least 180 prior valid values are required.

For current value `x` and strictly prior reference multiset `R`:

```text
rank(x;R) =
    (count(R<x) + 0.5*count(R==x)) / len(R)
```

The current value is appended only after every current rank is fixed. External
history is updated only when that source is fresh.

Finite ranks map to exactly five bands:

```text
[0.0,0.2)  EXTREME_LOW
[0.2,0.4)  LOW
[0.4,0.6]  MIDDLE
(0.6,0.8]  HIGH
(0.8,1.0]  EXTREME_HIGH
```

Exact `0.4` and `0.6` map to `MIDDLE`; exact `0.2` maps to `LOW`; exact `0.8`
maps to `HIGH`. Primitives 1-10 are core and must all be rank-ready. Primitives
11-14 are availability-guarded context: a currently unavailable value emits
`STALE`, does not update its rank history, and does not invalidate an otherwise
valid state.

The canonical line prints the fourteen names in the order above, followed by
their band token. A model state contains exactly six **consecutive** canonical
boundary lines, oldest first. Their endpoints span exactly 40 hours
(`B-40h,...,B`); no invalid boundary may be skipped to assemble a longer,
irregular sequence. If any one of the six core lines is invalid, the current
boundary is not sequence-ready and the deterministic safety action is `FLAT`.
It then appends:

```text
POSITION = SHORT | FLAT | LONG
POSITION_AGE = ZERO | ONE | TWO | THREE_PLUS
STRICT_DRAWDOWN = ZERO | UNDER_2 | TWO_TO_5 | OVER_5
```

Position and age refer to the target held at the state cutoff, before the
current decision. Position age counts complete SABLE decision intervals since
the latest executed target change. Strict drawdown is the causal strategy
drawdown marked at the state cutoff, including realized funding at boundary
`B` and a virtual exit cost, but excluding the not-yet-known execution open at
`B+15m`.

Forbidden model inputs:

```text
raw numbers or numeric ranks
timestamp, date, year, month, weekday, hour, row id, or split identity
future price/path/funding/premium/OI/external value
reward, future utility, PnL, CAGR, MDD target, or oracle action
prior alpha name, prior model prediction, portfolio weight, or manual regime
source path, source hash, or transport identity
free-form analyzer prose, rationale, chain of thought, or generated feature
```

## Frozen augmented MDP and accounting order

Actions are target positions:

```text
SHORT = -1
FLAT  =  0
LONG  = +1
```

The historical research leverage is exactly `0.5x`. The live candidate, if it
ever qualifies, must first shadow at the same leverage.

For step `k`, let:

```text
B_k        canonical funding boundary
C_k        state cutoff B_k+5m
D_k        decision B_k+10m
E_k        execution B_k+15m
p_k-1      target held at C_k
a_k        newly selected target
```

The transition order is fixed:

1. the prior target receives price PnL through `C_k` and receives/pays funding
   stamped at `B_k`;
2. the state at `C_k` is formed from source tokens plus the resulting
   `POSITION`, `POSITION_AGE`, and `STRICT_DRAWDOWN`;
3. the model chooses `a_k` at `D_k`;
4. `p_k-1` remains held from `C_k` through the open at `E_k`;
5. at `E_k`, deterministic execution changes the target to `a_k` and charges
   turnover cost;
6. `a_k` is held through `C_k+1`, including funding stamped at `B_k+1`;
7. the next state is formed at `C_k+1`.

This delayed-action ordering keeps the next dynamic state and step reward on
the same cutoff. The action cannot receive or avoid the funding already
settled at its own boundary `B_k`.

At execution:

```text
turnover_k = abs(a_k-p_k-1)
base_cost_k = turnover_k * 0.5 * 0.0006
stress_cost_k = turnover_k * 0.5 * 0.0010
```

The base rate is six basis points and the stress rate replaces it with ten
basis points per changed notional unit; the two are never added. A direct
`SHORT` to `LONG` reversal pays two changed units.

For positive Binance funding `f`, a long pays and a short receives. Every
funding event contributes:

```text
funding_cash_return = -0.5 * held_target * f
```

The event is charged to the target held immediately before its exact
`funding_time`. An action executed at `B+15m` therefore first receives/pays
funding at the next returned boundary, not at `B`.

The source-only artifact contains only exogenous primitive lines, readiness,
price-path locators, and realized funding locators. It contains no
`POSITION`, age, drawdown, action, reward, or policy-dependent transition. The
augmented state is generated inside one deterministic environment from the
current trajectory.

Before any reward value is constructed, a separate committed evaluator freeze
must specify:

- the exact wealth equation at every five-minute open/high/low/close;
- the per-step reward equation and terminal reward;
- the drawdown-token update order;
- discount, n-step target, quantiles, risk functional, replay order, and
  terminal mask;
- train episode starts/resets and seed list;
- every cheap learner and Gemma hyperparameter;
- the exogenous interval-ledger schema/hash and environment code hash; and
- a hand-calculated synthetic transition table covering all three actions,
  funding signs, invalid-state flattening, reversal, and split termination.

No reward or model process may run while that freeze is absent, dirty, or
hash-mismatched. This boundary deliberately freezes the state/action timing
now while leaving the reward functional unopened until source support is known
to exist.

Strict MDD must include:

- the pre-entry high-water mark;
- every held five-minute adverse path;
- entry/rebalance cost;
- realized funding;
- virtual adverse liquidation cost at every mark; and
- split-end liquidation.

CAGR uses the complete wall-clock interval, including warm-up, flat, invalid,
and unchanged-position time. No active-time annualization is allowed.

## Why an LLM is admissible

Deterministic code calculates every primitive and band. Gemma is not asked to
do arithmetic, prove predicates, decode source schemas, or generate a
trade explanation.

The model's only proposed advantage is sequence composition: recognizing how
several weak states change together across six ordered steps while conditioning
on the current portfolio state. This claim is falsifiable against models that
receive the identical tokens.

SABLE uses one model. There is no analyzer/trader split.

The proposed model family is the repository's locally runnable compact Gemma 4
family in 4-bit QLoRA form with a non-generative quantile action-value head.
The exact model repository/revision, tokenizer hash, parameter size, adapter
targets, quantile count, optimizer, replay procedure, seeds, update count, and
deterministic snapshot contract must be committed in the evaluator freeze
after source support and **before any reward inspection or training**. No model
swap is allowed after 2021 internal-selection outcomes are opened. No generated
action token or parser is permitted.

Gemma is admissible only if ordered relational composition adds measurable
value. A pretrained parameter count is not evidence. The final ordered model
must beat a parameter-matched scratch sequence model and the strongest cheap
causal baseline on paired, identical decision paths. Reversing or shuffling
the six source lines must materially reduce that incremental advantage; a
reversed model that performs equally well falsifies the claimed LLM mechanism.

## Mandatory controls

Before Gemma training, the identical source clock and token table must test:

1. always flat;
2. always long;
3. always short;
4. previous-position persistence;
5. price/range/volatility-only policy;
6. one-hot linear fitted-Q policy;
7. ExtraTrees fitted-Q policy;
8. two-layer MLP QR-DQN;
9. a small transformer trained from scratch on the same token sequence;
10. shuffled reward within calendar month;
11. circularly shifted reward by 21 decision steps;
12. reversed six-step token order;
13. masked primitive groups;
14. external-context-only;
15. derivatives-context-only; and
16. price/order-flow-only.

Gemma must beat the strongest causal non-LLM policy. Beating only a constant,
shuffled, or price-only control is insufficient.

The evaluator freeze must also bind these repository-level killer baselines
before any 2022/2023 gate is opened:

1. deterministic union/vote of the individual weak-signal experts;
2. frozen funding/premium sleeves;
3. frozen Markov-transition sleeve;
4. frozen Fresh-Kimchi/FX sleeve;
5. frozen OI-divergence sleeves;
6. frozen REX and REX-veto sleeves; and
7. the then-current live/shadow portfolio target series.

Each baseline needs a committed artifact path/hash, active interval, leverage,
clock-mapping rule, and fail-closed behavior. Missing runtime artifacts do not
become zero-return baselines; they make the novelty gate unavailable.

Novelty is measured without retuning either side:

```text
exact action-change overlap
action-change overlap within +/- one SABLE boundary
occupied-position correlation
daily net-PnL correlation
identical-target share
marginal portfolio CAGR/strict-MDD at fixed total gross
```

The fixed Stage 0.5 contract must choose the exact aggregation and bootstrap
before those values are calculated.

## Frozen staged gates

### Stage 0 — source and language support

Only 2020-2023 source values may be parsed. Required development support:

- at least 3,000 token-ready decisions in 2020-2022;
- at least 900 token-ready decisions in each of 2021 and 2022;
- at least 900 report-only token-ready decisions in 2023;
- every calendar month from 2020-05 through 2022-12 has at least one
  token-ready decision;
- none of primitives 1-10 is missing on more than 1% of otherwise eligible
  decisions;
- every core primitive has at least four occupied bands in both 2020-2022 and
  2023;
- the OI primitive has at least four occupied bands among its fresh rows in
  each period;
- no core primitive's largest band exceeds 45%;
- the OI primitive is fresh on at least 50% of decisions;
- the Kimchi primitive is fresh on at least 80% of decisions;
- each of USDKRW and DXY is fresh on at least 55% and stale on at least 5% of
  decisions;
- at least 95% of adjacent token-ready states change one or more primitive
  token;
- exact six-line signatures have maximum share below 1%; and
- prefix replay proves that appending later source rows cannot alter an
  earlier primitive, rank, token, cutoff, or execution time.

Failure retires SABLE-8 before reward construction. A failed support threshold
may not be weakened.

### Stage 0.5 — evaluator and model freeze

Stage 0 passing authorizes specification, not outcome inspection. Before any
forward interval return, reward, or policy result is calculated, one clean
commit must freeze:

- the complete augmented environment and reward items listed above;
- exact cheap model families, hyperparameter grids, seeds, and tie breaks;
- exact Gemma repository/revision and all training parameters;
- deterministic baseline-union and prior-sleeve artifact hashes;
- stationary paired bootstrap block length, replicate count, confidence level,
  and familywise max-statistic correction;
- all economic, diversity, novelty, and ablation thresholds;
- write-once output paths and expected protocol hashes.

The chronological roles are fixed:

```text
2020          fit
2021          internal model/hyperparameter selection only
2022          untouched confirmation gate
2023          untouched candidate gate
```

Model family, hyperparameters, seeds, thresholds, action semantics, and
baseline definitions may use 2020-2021 only. A single cheap specification is
selected on 2021 after the preregistered familywise correction. It is then
retrained from scratch on 2020-2021 and tested once on 2022. Only a 2022 pass
allows the same specification to be retrained from scratch on 2020-2022 and
run once on 2023. Neither 2022 nor 2023 chooses a model, seed, checkpoint,
threshold, feature, token, or reward.

### Stage 1 — cheap learnability

Only after Stage 0 passes and the Stage 0.5 freeze is committed may the
evaluator construct rewards.

```text
fit                  2020
internal selection   2021
confirmation         2022
candidate gate       2023
```

All learners fit only transitions whose reward path ends inside the named fit
period. The single train-only-selected cheap specification must pass 2022 and,
after the allowed from-scratch refit on 2020-2022, full 2023:

- positive absolute return;
- positive H1 and H2 absolute return;
- full-calendar CAGR/strict-MDD at least 1.5;
- strict MDD no greater than 15%;
- at least 24 non-flat exposure episodes;
- at least six long and six short episodes;
- no target position occupies more than 80% of decisions;
- positive 10-bp-per-changed-unit stress return; and
- superiority to shuffled and circular controls under the frozen familywise
  paired test.

Failure retires SABLE-8 before GPU use. No cheap hyperparameter, token, history,
clock, reward, or action repair follows.

### Stage 2 — Gemma 4 RLLM

Gemma may train only after Stage 1 passes. Its complete specification is
already fixed at Stage 0.5. The final-step fixed-seed ensemble, not a selected
checkpoint or selected seed, is fit on 2020-2022 and evaluated once on 2023.
It must:

- pass every Stage 1 economic and action-diversity gate;
- improve the strongest causal cheap policy's CAGR/strict-MDD by at least 20%;
- not increase strict MDD by more than two percentage points;
- have a positive lower confidence bound for paired daily net-return
  improvement versus the strongest cheap policy and the parameter-matched
  scratch transformer;
- beat every single-group mask on the preregistered aggregate test;
- lose at least 10% of its ratio advantage when the six lines are reversed or
  order-shuffled; and
- fail the complete economic gate under shuffled or circular rewards.

If reversed/order-shuffled input retains the ordered model's advantage, the
claimed sequence-composition mechanism is falsified even when raw return is
positive.

### Stage 2.5 — prior-universe novelty

Using only the already opened 2023 candidate gate, SABLE must satisfy all of:

- exact action-change overlap with every prior sleeve no greater than 25%;
- `+/-8h` action-change overlap no greater than 45%;
- absolute occupied-position correlation with every sleeve no greater than
  `0.65`;
- absolute daily net-PnL correlation with every sleeve and the aggregate
  portfolio no greater than `0.55`;
- identical-target share versus the deterministic all-expert union below
  `80%`;
- positive lower confidence bound for daily net-PnL improvement over the
  deterministic union; and
- at fixed total gross, adding SABLE and proportionally rescaling existing
  sleeves improves portfolio CAGR/strict-MDD by at least 10% without raising
  strict MDD by more than two percentage points.

These are rejection gates, not optimization targets. No overlap threshold,
portfolio weight, baseline membership, or clock mapping may be changed after
the values are known.

### Stage 3 — sequential historical evaluation

Only complete Stage 2 and Stage 2.5 passes authorize the source-extension then
candidate-outcome transactions in this order:

```text
2024
2025
2026-01-01 through 2026-06-01
```

Each full year must independently have:

- positive absolute return;
- full-calendar CAGR/strict-MDD at least 3;
- strict MDD no greater than 15%;
- positive 10-bp-per-changed-unit stress return;
- at least 18 non-flat episodes;
- both long and short contribution; and
- positive first- and second-half return.

Before each line, its bounded source cut and prefix replay must be committed as
specified above. The shorter 2026 report requires positive absolute return,
ratio at least 3, strict MDD at most 15%, at least eight episodes, and both
sides. Stop at the first failure. Later source cuts and outcomes remain
unopened for SABLE after a failure.

### Stage 4 — forward shadow

Historical passage cannot promote SABLE directly because its primitive
families were selected in a globally outcome-seen research history. Live
capital requires an unchanged forward shadow of at least:

- 60 calendar days;
- 18 non-flat episodes;
- both long and short exposure;
- positive net return at production costs;
- strict MDD at most 15%;
- exact offline/live token parity; and
- no stale source authorizing an external-context token.

## Mandatory implementation tests

Before real support decoding, synthetic source tests must cover:

- exact file/hash/header/allowlist binding;
- physical stop before the first post-2023 numeric row;
- deterministic physically bounded cut creation and cut-hash verification;
- ordered unique five-minute and eight-hour grids;
- one full-bar source cutoff, decision latency, and execution open;
- all fourteen primitive equations and zero-denominator failures;
- volume target shifted strictly before the current bar;
- funding and premium interval inclusion boundaries;
- OI and external current/lag availability;
- no weekend FX forward-fill authorization;
- independent strict-prior rank histories, cap, minimum, and ties;
- every exact band boundary;
- six consecutive-state oldest-first sequence construction and invalid-gap
  rejection;
- prefix replay invariance;
- forbidden-column rejection;
- deterministic gzip/JSON outputs and write-once drift rejection; and
- refusal to run real support while protocol files are uncommitted.

Before reward construction, synthetic evaluator tests must additionally cover:

- all transition-order steps from `C_k` through `C_k+1`;
- position, age, and drawdown tokens at the pre-action cutoff;
- split reset, final truncation, and terminal containment;
- turnover for hold, flatten, enter, and direct reversal;
- funding assignment before and after an action boundary;
- base cost versus replacement stress cost;
- five-minute strict held-path MDD and virtual exit cost;
- all three counterfactual actions from one identical exogenous state;
- deterministic replay under every frozen seed;
- chronological train/selection/confirmation/gate enforcement;
- model/checkpoint/threshold selection refusal on 2022 or 2023; and
- prior-sleeve artifact/hash/clock binding for novelty tests.

## Independent-review disposition

An independent architecture review initially blocked this boundary. The
blocking findings are resolved as follows:

| finding | disposition |
|---|---|
| future source was not separately frozen | mandatory per-interval physical cuts, manifests, commits, and prefix replay |
| source table mixed exogenous and policy state | source artifact is exogenous only; policy state is generated by the frozen environment |
| reward/Q transition was absent | Stage 0.5 reward/environment freeze is mandatory before any forward return |
| 2023 could select the cheap champion | 2020 fit, 2021 selection, 2022 confirmation, 2023 gate |
| prior weak-signal repair risk | deterministic-union, prior-sleeve, overlap, correlation, and marginal-value killer gates |
| fused cache was treated as live source | field-level historical identities plus separate live adapters/parity and fail-flat |
| funding ownership at the action boundary was ambiguous | prior target owns boundary funding; new action begins at `B+15m` |
| Gemma sequence claim was too weak | ordered Gemma must beat scratch/cheap controls and degrade under order destruction |

The review changes the verdict from implementation-blocked to
**support-only admissible**. It does not pre-approve Stage 0 support or any
economic result.

## Outcome boundary at this commit

For SABLE-8 at this decision:

```text
candidate source values parsed       0
candidate token incidence calculated 0
future return labels built           0
rewards built                        0
market outcomes evaluated            0
funding cash flows evaluated         0
models trained                       0
2023 candidate outcomes opened       false
2024 candidate outcomes opened       false
2025 candidate outcomes opened       false
2026 candidate outcomes opened       false
```

Prior research disclosed broad component-family outcomes. That history is
declared contamination context, not silently relabeled as SABLE evidence.

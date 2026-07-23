# CXRT-288 mechanism decision — CBOE cross-surface risk-transfer relay

## Decision

Freeze one candidate:
**CXRT-288 — CBOE Cross-Surface Risk-Transfer Relay**.

CXRT observes three dense CBOE weak states on an exact common source date:

1. volatility-term pressure;
2. tail-hedge pressure; and
3. option-flow pressure.

Each surface votes `RELIEF`, `STRESS`, or `NEUTRAL` from a strictly-prior rank.
At least two non-neutral surfaces are required. The majority fixes:

```text
RELIEF -> LONG BTCUSDT
STRESS -> SHORT BTCUSDT
```

The position begins only on the first later exact common CBOE source date at
09:35 America/New_York and holds exactly 288 five-minute bars. A later RLLM may
trade the fixed side or abstain.

This document freezes source fields, feature algebra, direction, latency,
hold, controls, support/selectivity gates, comparator cohort, and RLLM input
boundary before any exact CXRT common-date state, vote, side, timestamp, or
post-entry return is computed.

## Falsifiable mechanism

The three source surfaces describe different observable layers:

- term structure: whether short-horizon implied volatility is rich or cheap
  relative to longer horizons;
- tail surface: whether SKEW and volatility-of-volatility are elevated relative
  to their own histories; and
- option flow: whether index/equity put-call separation, VIX call pressure, and
  index share are increasing relative to prior sessions.

CXRT does not claim that any layer identifies investors or causal BTC demand.
Its testable hypothesis is narrower: when at least two independently normalized
CBOE layers indicate relief or stress, the resulting cross-surface state
contains enough next-day BTC directional information for a causal RLLM to
select a profitable subset.

The deterministic majority is deliberately dense. It avoids the exact-tail
and exact-transition sparsity that invalidated recent candidates. The RLLM is
responsible for deductive abstention from relation tokens, not for inventing
direction.

## Immutable source contract

### Term panel

```text
data/cboe_volatility_term_structure_2018_2023/
cboe_vix_term_structure_2018-01-01_2023-12-31.csv.gz
SHA256 6f1b2f7f3a5b1e4d5001d673e6ff54374791879c278248ce27b3d610e4f75dc7
```

Manifest SHA-256:
`42b2a35ad131bd63574d2adcf684e28766dc3060fa645fc749df10dd3fb27f27`.

Allowed columns:

```text
observation_date
VIX9D_close
VIX_close
VIX3M_close
```

### Tail panel

```text
data/cboe_tail_risk_2018_2023/
cboe_tail_risk_2018-01-01_2023-12-31.csv.gz
SHA256 cdde3f8d4bb1e23d00b192f5f9ef759aefba9087be5fd60653e9c02479dfa41a
```

Manifest SHA-256:
`9ef80ef3034c93d97c5b2a8160b2502527287d570d15f9d7166d631d9866c7bd`.

Allowed columns:

```text
observation_date
SKEW_close
VVIX_close
VIX_close
```

### Option-flow panel

```text
data/cboe_option_flow_2020_2023/
cboe_option_flow_2020-01-01_2023-12-31.csv.gz
SHA256 35ef106ef01e3abadbcb4a6227187dd1d7cf2722191bd146bac06d08d1684a78
```

Manifest SHA-256:
`0a513b146ad5857d9ab7311e978152c308de64db8ef29c4d463eb07ea503089e`.

Allowed columns:

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

The source-support loader must pass exact column allowlists to each CSV reader.
It may load no BTC price, return, funding, label, reward, PnL, portfolio,
calendar-alpha, or 2024-or-later field.

Every retained numeric value must be finite and strictly positive. Dates must
be unique and increasing. The exact date intersection is used after each
surface computes its own causal history. Missing dates are never filled,
carried, interpolated, or treated as zeros.

The term and tail `VIX_close` values must match exactly on common dates. Any
disagreement fails closed.

## Strict-prior rank

Every primitive rank uses:

```text
lookback observations = at most 252 prior observations of that source
minimum prior observations = 126
```

For current value `x` and strictly prior reference `R`, the midrank is:

```text
(count(R < x) + 0.5 * count(R == x)) / len(R)
```

The current value is appended only after all current ranks are fixed. A missing
or non-finite primitive makes that source state unavailable. No expanding
future normalization, z-score, clipping, winsorization, weekday conditioning,
or later-source substitution is allowed.

## Surface algebra

### Term pressure

For each term observation:

```text
front_slope = log(VIX9D_close / VIX_close)
broad_slope = log(VIX_close / VIX3M_close)
front_rank  = strict_prior_rank(front_slope)
broad_rank  = strict_prior_rank(broad_slope)
term_pressure = 0.5 * (front_rank + broad_rank)
```

### Tail pressure

For each tail observation:

```text
skew_level    = log(SKEW_close / 100)
vvix_relative = log(VVIX_close / VIX_close)
skew_rank     = strict_prior_rank(skew_level)
vvix_rank     = strict_prior_rank(vvix_relative)
tail_pressure = 0.5 * (skew_rank + vvix_rank)
```

There is no VIX-level subtraction and no second-layer hidden-pressure rank.

### Option-flow pressure

For each option-flow observation:

```text
institutional_gap =
    log((index_put_volume + 0.5) / (index_call_volume + 0.5))
  - log((equity_put_volume + 0.5) / (equity_call_volume + 0.5))

vix_call_pressure =
    log((vix_call_volume + 0.5) / (vix_put_volume + 0.5))

index_share =
    log((index_volume + 1.0) / (total_volume + 1.0))
```

Take each current level minus the immediately preceding option-source level.
Rank each delta against its own strict-prior delta history:

```text
option_pressure =
    mean(
        delta_institutional_gap_rank,
        delta_vix_call_pressure_rank,
        delta_index_share_rank
    )
```

The half-contract and one-contract pseudocounts exist only to keep the
log-ratios defined; they are not tunable.

## Votes and deterministic side

For surface pressure `p`:

```text
p < 0.5  -> RELIEF vote (+1)
p > 0.5  -> STRESS vote (-1)
p == 0.5 -> NEUTRAL vote (0)
```

At an exact common source date:

```text
vote_sum = term_vote + tail_vote + option_vote
nonzero_votes = count(vote != 0)
```

The date is eligible only when:

```text
nonzero_votes >= 2
vote_sum != 0
```

Side is `LONG` when `vote_sum > 0` and `SHORT` when `vote_sum < 0`. There is no
tail threshold, fitted weight, volatility gate, BTC regime gate, confidence
minimum, or outcome-selected tie breaker.

Each surface pressure is also mapped to one immutable relation token:

```text
[0.00, 0.25)  RELIEF_STRONG
[0.25, 0.50)  RELIEF_WEAK
0.50          NEUTRAL
(0.50, 0.75]  STRESS_WEAK
(0.75, 1.00]  STRESS_STRONG
```

The exact boundary values `0.25` and `0.75` belong to the weak buckets.

## Availability, execution, and hold

For source observation date `D`:

- source-close values from `D` are treated as unavailable for trading on `D`;
- let `D_next` be the first later date that exists in the exact three-panel
  common CBOE source calendar;
- signal availability: `D_next` at `09:30` America/New_York;
- decision/order/entry: `D_next` at `09:35` America/New_York;
- instrument: Binance USD-M BTCUSDT;
- entry: the first five-minute open at the exact UTC-converted decision time;
- exit: exactly 288 five-minute bars after entry;
- leverage: fixed `0.5x`;
- stop, take-profit, trailing exit: none.

No weekend, holiday, or date absent from any of the three frozen source panels
receives an entry. A missing/no-data CBOE date is not synthesized and does not
receive a stale carried state. The historical clock uses the already-frozen
exact common source calendar; live promotion must use the corresponding
predeclared CBOE session calendar and fail flat when expected same-session
source parity later disagrees. Consecutive common source dates may produce an
entry exactly equal to the previous exit; equality is accepted. An entry before
the previous accepted exit is suppressed, never queued.

Raw candidates are built across the complete pre-2024 source, globally
non-overlap-reserved, and only then split-contained.

## Frozen source-only controls

Every independent clock repeats global reservation before split containment.

1. `primary` — three-surface majority.
2. `term_only` — term vote alone when non-neutral.
3. `tail_only` — tail vote alone when non-neutral.
4. `option_only` — option vote alone when non-neutral.
5. `term_tail_agreement` — emit only when term and tail agree non-neutrally.
6. `term_option_agreement` — emit only when term and option agree.
7. `tail_option_agreement` — emit only when tail and option agree.
8. `one_common_date_stale` — current entry clock with all three votes replaced
   by the immediately preceding eligible common-date votes.
9. `exact_direction_flip` — primary timestamps with opposite side.
10. `deterministic_random_side` — primary timestamps; SHA-256 of
    `CXRT-288|entry_time_utc`, LONG iff first byte is below 128.
11. `one_day_execution_delay` — primary source state and side; entry and exit
    delayed exactly 288 bars with overlap and containment recomputed.

Single-surface and pair-agreement controls are mechanism diagnostics only; they
may not replace a failed primary.

## Source-support gates

Warm-up uses 2020 only. Primary support is measured on:

```text
train     [2021-01-01, 2023-01-01)
selection [2023-01-01, 2024-01-01)
```

### Train

- at least 400 globally accepted split-contained events;
- at least 190 in each of 2021 and 2022;
- at least 22 active months;
- LONG and SHORT each at least 20%;
- maximum single-month share at most 7%;
- maximum single-quarter share at most 18%;
- maximum entry gap at most 10 calendar days;
- maximum same-side run at most 30.

### Selection

- at least 190 events;
- at least 90 in each half-year;
- at least 40 in every quarter;
- at least 11 active months;
- LONG and SHORT each at least 20%;
- maximum single-month share at most 11%;
- maximum entry gap at most 10 days;
- maximum same-side run at most 20.

### Relational composition

Across train and selection separately:

- every surface has at least 15% RELIEF and 15% STRESS votes;
- each surface is the unique minority vote on at least 8% of non-unanimous
  dates;
- unanimous non-neutral dates are between 10% and 80% of primary dates;
- same-side reproduction of primary by each single-surface control is at most
  80%;
- same-side reproduction by `one_common_date_stale` is at most 85%;
- same-side reproduction by deterministic random side is at most 60%.

An empty required clock, undefined denominator, missing statistic, or
non-finite value fails.

## Comparator novelty

Hash-bind and compare the primary clock against the already-opened:

- CVTR-1 primary;
- CTHD-1 primary;
- CIHM-1 primary;
- deterministic constant-long and random-side daily controls from those
  families when available.

The exact comparator paths and hashes must be frozen in the preregistration.
Over common coverage:

- exact entry-time Jaccard at most 0.45;
- same-entry same-side reproduction at most 0.75;
- absolute signed occupied-exposure Pearson correlation at most 0.60;
- one-source-day tolerant entry Jaccard is report-only because every policy
  shares a CBOE-derived daily availability clock.

Empty required extraction, hash/header drift, or undefined correlation fails
before outcomes.

## Economic and RLLM sequence

No BTC execution source may be opened before source support and novelty pass.
The next stage must separately freeze:

1. exact pre-2024 BTCUSDT OHLC/funding sources and strict MDD accounting;
2. a train-only cheap relational baseline;
3. one small frozen LLM/RLLM model and adapter recipe;
4. reward, optimizer, seed, checkpoint-selection, and abstention policy;
5. 2021 fit / 2022 inner-validation / sealed 2023 selection roles;
6. a post-2023 source extension and later test/eval sequence.

The LLM may advance only if a non-leaky train/inner-validation baseline
demonstrates recoverable relation-token signal. This is a compute gate, not an
authorization to change the source clock or side.

Final strategy qualification remains:

- positive full-calendar absolute return;
- `CAGR / strict MDD >= 3.0`;
- strict MDD at most 15%;
- positive 10 bp/notional/side stress return;
- positive one-day delayed return;
- statistically meaningful clustered evidence;
- positive LONG and SHORT sleeves; and
- no evaluated-split model, threshold, checkpoint, or token selection.

Exact economic thresholds and cluster test are frozen only in the later
economic/RLLM evaluator, after source support passes.

## RLLM relation-token boundary

Allowed tokens:

- fixed side;
- term, tail, and option pressure bucket;
- each surface vote;
- unanimous / split-majority / neutral-supported relation;
- which surface is the minority;
- each surface vote transition from the immediately prior common date;
- prior common-date majority transition;
- calendar gap bucket `1`, `2-3`, or `4+` days;
- source validity; and
- current position state.

Forbidden inputs:

- raw numeric values or ranks;
- observation date, year, month, weekday, timestamp, row number, hash, or
  source identifier;
- BTC price, return, funding, future path, label, PnL, reward, CAGR, MDD, or
  portfolio state beyond current position;
- candidate creation, side reversal, hold change, leverage change, or
  timestamp choice.

Action space:

```text
TRADE_FIXED_SIDE
ABSTAIN
```

No prompt may reveal split identity or historical outcome summaries.

## Failure action

Any source, support, relational-composition, novelty, economic, or RLLM gate
failure retires CXRT-288 unchanged at that stage. A changed formula, state
bucket, vote, latency, hold, source, side, token, or model recipe is a new
candidate and may not reuse sealed outcomes as a holdout.

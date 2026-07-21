# CDLTR-72 cross-domain liquidity-transmission relay mechanism decision

## Decision

The next source-only BTC candidate is **CDLTR-72 — Cross-Domain Liquidity
Transmission Relay, 72-hour hold**.

CDLTR is not another threshold variant of the rejected ON RRP, Cboe volatility,
or Bitcoin-network policies. It tests one ordered interaction:

1. deployable-dollar liquidity and U.S. option-risk term structure enter the
   same directional state;
2. the first subsequently published Bitcoin-network report either confirms
   that state or kills the episode; and
3. only a confirmation creates one BTC candidate clock.

This document opens no source values, feature incidence, BTC market row,
funding value, return, PnL, equity, CAGR, MDD, existing-alpha outcome, or
post-2023 row.

## Evidence and non-pristine boundary

The three component source families are already documented and live-refreshable,
but earlier singleton policies from those families were rejected or failed
support. Their observed development results make CDLTR a **candidate-level new
interaction**, not a globally pristine hypothesis. No component result may be
relabelled as validation for CDLTR, and no component or control may replace a
failed primary.

Only a future shadow period that begins after the final CDLTR policy is frozen
can supply pristine forward evidence. Historical stages remain useful for
falsification and development qualification only.

## Frozen source axes

### 1. New York Fed overnight reverse repo

- panel:
  `data/new_york_fed_overnight_rrp_2018_2023/new_york_fed_overnight_rrp_2018-01-01_2023-12-31.csv.gz`
- SHA-256:
  `49f67ed44b7eb81fd35c17a8209cf14d6a8019d7e9f77fce8c343d1a7fb66b27`
- allowed fields:
  `operation_date,result_available_at_utc,total_amount_accepted_usd,source_complete,quarantine_reason`
- causal time: exact `result_available_at_utc`, already defined as normal
  operation close plus a conservative publication allowance;
- incomplete or quarantined rows are clock evidence only and contribute no
  numeric value.

The RRP vote uses the current normal operation and the operation exactly five
normal-operation slots earlier. All six rows from the current slot through the
fifth prior slot must be complete. A quarantine therefore breaks the baseline;
the implementation may not skip it and bridge to an older value.

```text
rrp_delta_5_operations = accepted_now - accepted_fifth_prior_slot
rrp_vote = LONG  if delta < 0
           SHORT if delta > 0
           NEUTRAL otherwise
```

A declining RRP take is treated only as an upstream deployable-liquidity vote;
it is not independently tradable under CDLTR.

### 2. Cboe volatility term structure

- panel:
  `data/cboe_volatility_term_structure_2018_2023/cboe_vix_term_structure_2018-01-01_2023-12-31.csv.gz`
- SHA-256:
  `6f1b2f7f3a5b1e4d5001d673e6ff54374791879c278248ce27b3d610e4f75dc7`
- allowed fields: `observation_date,VIX9D_close,VIX3M_close`;
- causal time: the prior Cboe source day's close becomes usable only at 09:35
  `America/New_York` on the next date present in the exact three-index Cboe
  intersection.

No missing date is forward-filled. The last pre-2024 observation has no usable
next pre-2024 Cboe date and is boundary quarantine.

```text
cboe_vote = LONG  if VIX9D_close < VIX3M_close
             SHORT if VIX9D_close > VIX3M_close
             NEUTRAL otherwise
```

This is a risk-term vote, not a revival of the rejected CVTR trading clock.

### 3. Bitcoin network breadth

- panel:
  `data/coinmetrics_btc_network_daily_2020_2023.csv.gz`
- SHA-256:
  `97ab2ca9d0c347d85221b51734f98072763370072ca51f1c40e3214191159b42`
- allowed fields:
  `observation_date,available_at,AdrActCnt,TxCnt,TxTfrCnt`;
- causal time: exact `AssetEODCompletionTime` represented by `available_at`,
  interpreted as UTC;
- current-vintage limitation: the file is hash-frozen but is not a complete
  archive of every historical revision. Live promotion requires forward-vintage
  parity or an owned-node replacement with separately proven feature parity.

The restored panel was rebuilt from the fixed Coin Metrics Community API
request and reproduced the existing frozen SHA-256 byte for byte.

For each metric, compare the current observation with the row exactly seven
calendar days earlier. All eight daily observation dates and positive finite
values must be present, and every required row must already be available.

```text
metric_vote = sign(log(value_today / value_7_calendar_days_ago))
network_vote = LONG  if at least two metric votes are positive
               SHORT if at least two metric votes are negative
               NEUTRAL otherwise
```

No BTC price, return, exchange flow, funding, or direction label enters the
network vote.

## Ordered relay clock

All state ages use actual UTC source-availability timestamps, never observation
dates or an assumed business-day calendar.

1. Maintain the most recent non-neutral RRP and Cboe votes.
2. Each vote expires exactly 36 hours after its own availability time. This
   prevents weekend or holiday carry and gives one daily source cycle plus a
   conservative publication-offset allowance.
3. A macro state exists only when both unexpired votes have the same side.
4. A **macro episode onset** occurs when that same-side state changes from
   absent/opposite to LONG or SHORT after either source update.
5. Inspect only the first network report with `available_at` strictly after the
   onset. It must arrive no later than 36 hours after onset.
6. If its network vote equals the macro side, emit one CDLTR candidate. If it
   is neutral, opposite, missing, late, or invalid, the episode dies. No later
   network report may retry that episode.
7. A new episode requires the macro state to become absent/opposite and then
   enter same-side agreement again.

This one-shot order removes a tunable multi-day retry grid and makes the
network publication the causal confirmation clock rather than a stale daily
calendar filter.

## Execution contract

- decision time: confirming network `available_at`;
- entry: `ceil_to_5m(decision_time) + 5 minutes`; an exact five-minute boundary
  still receives the additional five-minute latency bar;
- side: frozen macro/network side;
- exit: entry plus exactly 72 hours;
- exposure: 0.5x notional;
- no stop, take-profit, trailing exit, leverage search, or overlapping entry;
- events are accepted in global chronological order and skipped while a prior
  event remains open;
- an event whose entry or exit crosses a research-window boundary is skipped,
  never truncated or force-closed.

The 5-operation and 7-calendar-day transforms are one-week source units. The
36-hour expiry is one daily source cycle plus timing allowance. The 72-hour
hold is the fixed slow-transmission horizon. None may be searched after source
incidence or outcomes are opened.

## Research windows

- source warm-up only: calendar 2020;
- train clock: `[2021-01-01, 2023-01-01)` UTC;
- selection clock: `[2023-01-01, 2024-01-01)` UTC;
- every source and outcome dated 2024 or later remains closed during this
  source-only stage.

## Source-only admission gates

The preregistration must freeze these minimums before constructing real CDLTR
incidence:

- train: at least 60 events, at least 25 in each calendar year, and at least 12
  in each half-year;
- selection: at least 30 events and at least 12 in each half-year;
- both sides: at least 18 per side in train and 8 per side in selection;
- maximum UTC calendar-month share: 20% in each split;
- maximum UTC weekday share: 35% in each split;
- no source gap, quarantine, stale vote, or boundary event may contribute a
  feature or candidate.

Failure rejects CDLTR-72 before any BTC outcome. No floor, side, span, expiry,
order, or hold may be repaired from the observed support table.

## Frozen source-only falsification clocks

Build and retain these controls without using them to select a replacement:

1. `macro_only`: macro-episode onset, same side;
2. `network_only`: network-vote onset, same side;
3. `reverse_order`: network onset followed by the first macro update;
4. `one_network_report_delay`: primary side executed only on the next valid
   network report;
5. `direction_flip`: exact primary clock, opposite side;
6. `deterministic_random_side`: exact primary clock, SHA-256-fixed side.

The primary must also pass fail-closed timestamp and occupied-exposure novelty
against the frozen ORFR, CVTR, NTB, NWE-7/NWE-8, chain-activity, FLCC, DFFB,
and currently executable live-family clocks. Comparator readers may materialize
only time, side, and interval columns. At minimum, reject on any comparator
with:

```text
decision-date Jaccard > 0.30
fraction of CDLTR dates within +/-1 UTC day > 0.50
absolute signed occupied-exposure Pearson > 0.40
```

An empty required comparator, missing allowlisted field, zero-variance exposure
grid, source-hash drift, or comparator ambiguity fails closed.

## Outcome and RLLM boundary

Only an unchanged source-support and novelty pass may authorize a separately
implemented, tested, committed, and hash-frozen strict evaluator. That later
evaluator must open train first, stop on any economic/statistical failure, and
open selection only after an exact train pass. Full-calendar CAGR must include
idle cash; strict MDD must include the global/pre-entry high-water mark,
in-position adverse path, funding, and all entry/exit/liquidation costs.

Gemma/RLLM is not authorized to create direction, alter the relay, search the
hold, or rescue a failed deterministic candidate. Only after deterministic
train and selection pass may a small Gemma4-class model receive text state
cards and learn a constrained `TRADE`/`ABSTAIN` veto. Any later RL reward must
penalize strict drawdown and turnover and preserve the frozen side, hold, and
causal source clock.

## Next admissible action

Implement and test a preregistration artifact that binds this singleton,
source files/builders/manifests, comparator clocks, column allowlists, support
floors, controls, and stopping rule. Commit it before any real CDLTR feature or
event incidence is computed.

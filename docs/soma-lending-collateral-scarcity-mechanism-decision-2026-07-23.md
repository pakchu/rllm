# SLCS-72 mechanism decision — 2026-07-23

## Decision

Preregister one new-source, candidate-incidence-blind singleton:
**SLCS-72 — SOMA Lending Collateral Scarcity Consensus**.

SLCS combines four weak, independently interpretable observables from daily New
York Fed securities-lending auctions. No one component is treated as a strong
signal. The hypothesis is that their broad agreement identifies a temporary
Treasury-collateral scarcity or relief state that transmits through dollar
funding conditions to BTC.

This document freezes the feature definitions, prior-only normalization,
direction, event clock, hold, controls, support gates, strict economic sequence,
and no-repair rule before any SLCS feature, state, incidence, or BTC outcome is
computed.

## Evidence and contamination boundary

The underlying source has been audited and its row-level values are now
available. Aggregate source counts, date coverage, null patterns, release-time
patterns, and operation/detail reconciliation were opened. No source ratio,
breadth, weighted fee, rank, state, transition, candidate event, or market
outcome was computed.

This is the repository's first NY Fed SOMA securities-lending candidate, but it
is not a pristine broad USD-liquidity research family. Outcomes from earlier
H.4.1, overnight-RRP, SOFR, H.8, and Treasury candidates have been opened. SLCS
uses a distinct daily auction source and exact rule, while every broad-liquidity
interpretation must disclose that prior research.

## Official source and availability

- program and auction description:
  <https://www.newyorkfed.org/markets/domestic-market-operations/monetary-policy-implementation/securities-lending>;
- field definitions and historical result search:
  <https://www.newyorkfed.org/markets/desk-operations/securities-lending>;
- official API:
  <https://markets.newyorkfed.org/static/docs/markets-api.html>.

Frozen source artifacts:

- operation panel:
  `data/new_york_fed_securities_lending_2019_2023/new_york_fed_securities_lending_operations_2019_2023.csv.gz`;
- detail panel:
  `data/new_york_fed_securities_lending_2019_2023/new_york_fed_securities_lending_details_2019_2023.csv.gz`;
- build manifest:
  `data/new_york_fed_securities_lending_2019_2023/build_manifest.json`.

Each operation is unavailable until the later of next UTC midnight and the
recorded New York `lastUpdated`. SLCS may use only details whose exact
`operation_id`, `operation_date`, and `available_at_utc` match the operation
panel. Candidate entry waits one additional complete five-minute bar.

## Four frozen weak-signal components

For operation `t`, sum only its exact CUSIP-detail rows. Every denominator must
be finite and strictly positive. A missing required value makes the entire
operation unavailable; it is never imputed or carried.

### 1. Submitted-demand intensity

```text
demand_intensity[t]
  = sum(par_submitted)
    / sum(actual_available_to_borrow)
```

This measures propositions relative to lendable SOMA inventory. It is not an
acceptance ratio and does not assume that every CUSIP is equally substitutable.

### 2. Accepted-amount-weighted lending fee

```text
weighted_fee[t]
  = sum(par_accepted * weighted_average_rate)
    / sum(par_accepted)
```

Zero-award `N/A` rate rows contribute neither numerator nor denominator.
Nonzero accepted amount with a missing rate is invalid. The New York Fed warns
that this fee is not a repo rate; SLCS uses it only as competitive relative
scarcity pressure inside the lending program.

### 3. Outstanding-loan carry intensity

```text
carry_intensity[t]
  = sum(outstanding_loans)
    / sum(actual_available_to_borrow)
```

This captures loans still outstanding before the current auction relative to
the current available inventory.

### 4. Submitted-demand breadth

```text
demand_breadth[t]
  = count(CUSIP where par_submitted > 0)
    / count(CUSIP where actual_available_to_borrow > 0)
```

This distinguishes broad participation across securities from a large amount
concentrated in very few issues. It does not label individual dealers.

## Strict-prior rank normalization

For each component independently, use exactly the previous 252 complete
operations in source-time order. The current operation is excluded. No calendar
interpolation or expanding fallback is allowed.

Compute an empirical midrank against the 252 prior values:

```text
rank[t] = (count(prior < current) + 0.5 * count(prior == current)) / 252
u[t] = 2 * rank[t] - 1
```

Thus each `u` lies in `[-1, +1]`. Ties are deterministic. The normalization
contains no fitted mean, variance, quantile, threshold grid, or outcome label.

## Frozen scarcity state and direction

```text
consensus_positive = count(u_component > 0) >= 3
consensus_negative = count(u_component < 0) >= 3
score = mean(u_demand, u_fee, u_carry, u_breadth)

HIGH = consensus_positive and score >= +0.50
LOW  = consensus_negative and score <= -0.50
NEUTRAL = otherwise
```

Zero components do not vote. A candidate occurs only when the current state is
`HIGH` or `LOW` and differs from the immediately preceding **complete,
rank-ready operation state**. Missing/unready operations break continuity; the
next complete operation may establish state but cannot trigger.

Direction is fixed:

- `HIGH`: broad/intense/expensive/persistent collateral borrowing pressure,
  **SHORT BTC**;
- `LOW`: broad relief across those dimensions, **LONG BTC**.

The policy does not trade persistence inside an unchanged state.

## Frozen execution

- signal: current operation's conservative `available_at_utc`;
- entry: `ceil_to_5m(signal) + 5 elapsed minutes`, including an exact-grid
  signal;
- exit: exactly 72 elapsed hours / 864 five-minute bars later;
- fixed exposure: 0.5x BTCUSDT perpetual notional;
- global chronological reservation on `[entry, exit)`;
- accept an event only when its entry is at or after the prior accepted exit;
- suppressed events are not queued;
- entry and exit must be contained in the same declared split;
- no stop, take-profit, trailing exit, dynamic size, direction override, or
  current-price filter.

## Frozen windows and source support

- warmup: 2019 source operations;
- train: `[2020-01-01, 2023-01-01)`;
- selection: `[2023-01-01, 2024-01-01)`;
- sealed: 2024 onward.

Before any BTC or funding row is loaded, the accepted primary clock must meet:

- train total at least 60;
- at least 15 events in each of 2020, 2021, and 2022;
- train at least 15 long and 15 short;
- selection total at least 18;
- at least 7 events in each 2023 half;
- selection at least 4 long and 4 short;
- every train and selection quarter active;
- maximum month share at most 15% in train and 20% in selection;
- maximum accepted-entry gap at most 45 elapsed days; and
- exact causal timing, split containment, uniqueness, and non-overlap.

Any failure rejects SLCS-72 before outcomes. Observed incidence cannot lower a
floor, change 252, change 0.50, add persistence trades, or alter the hold.

## Frozen controls and falsifications

Source/component clocks, each using the same transition, latency, 72-hour hold,
split containment, and non-overlap where applicable:

1. `demand_intensity_only`: `u_demand >= +0.50` short and `<= -0.50` long;
2. `weighted_fee_only`: same rule on `u_fee`;
3. `carry_intensity_only`: same rule on `u_carry`;
4. `demand_breadth_only`: same rule on `u_breadth`;
5. `mean_without_consensus`: score threshold without the three-of-four vote;
6. `same_sign_without_magnitude`: three-of-four sign vote without score 0.50;
7. `one_operation_stale`: previous complete rank-ready component vector at the
   current availability;
8. `five_operation_stale`: vector five complete operations old at the current
   availability;
9. `year_component_permutation`: deterministic SHA-256 permutation of each
   component independently within operation year before state construction.

Economic controls on the exact accepted primary clock:

- exact direction flip;
- deterministic SHA-256 random side; and
- constant long and constant short.

No control can replace the primary after outcomes open.

## Novelty and related-source boundary

The source-only stage must compare exact entries, one-day one-to-one overlap,
primary containment, and signed occupied-exposure correlation against frozen
H.4.1 FLCC, overnight-RRP, SOFR, H.8, and current live-portfolio pure clocks.

For every comparator with at least ten entries over 2020–2023:

- exact-entry Jaccard must be at most 0.10;
- SLCS one-day containment must be at most 0.35; and
- absolute signed exposure correlation must be at most 0.35.

Component controls are intentionally related and are not novelty comparators.
They remain mandatory specificity and later economic controls.

## Strict economic sequence

Only a fully passing source-support artifact may authorize a separately tested,
committed, and hash-frozen evaluator. It must open:

1. train 2020–2022;
2. selection 2023 only after an exact unchanged train pass;
3. immutable source extension and 2024 test only after pre-2024 pass;
4. 2025 eval only after 2024 pass; and
5. recent 2026 only after every prior pass.

Train and selection each require positive absolute return, full-calendar
`CAGR / strict MDD >= 3.0`, strict MDD at most 15%, positive 10-bp/side stress
return, sufficient gross edge, cluster-aware significance, positive required
subperiods, and failure of component/stale/permutation/flipped/random/constant
controls. Costs are 6 bp/notional/side base and 10 bp stress; funding is exact;
MDD is global/pre-entry-high-water and intratrade with favorable observations
before adverse observations plus virtual adverse exit cost.

## RLLM boundary

RLLM is disabled until the unchanged deterministic policy passes source,
train, and selection gates. A later small language model may receive only
causal textualized component ranks, state-transition reasons, current position,
time in position, and risk budget. Its actions are limited to
`TRADE_FIXED_SIDE` or `ABSTAIN`. It may not create events, reverse direction,
change leverage/hold, inspect future values, or repair a failed gate.

## Stopping rule

Any provenance, causality, source support, specificity, novelty, train, or
selection failure retires `SLCS-72-NEW-SOURCE` unchanged. Any change to the four
components, 252-operation history, midrank, three-of-four vote, 0.50 threshold,
state transition, side, latency, hold, or support floor requires a new identity
committed before access.

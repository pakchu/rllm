# WCDR-2016 wrapped-collateral dollar-liquidity rotation — mechanism decision

## Decision

The next outcome-blind BTC candidate is **WCDR-2016 — Wrapped-Collateral
Dollar-Liquidity Rotation, 2016 five-minute bars**.

WCDR tests one cross-domain divergence rather than mapping WBTC mint directly
to BTC buying or WBTC burn directly to BTC selling:

- WBTC custody supply contracts while USDC supply expands: a wrapped-collateral
  unwind with dollar liquidity replenishment, mapped to `LONG`;
- WBTC custody supply expands while USDC supply contracts: wrapped-collateral
  expansion against shrinking dollar liquidity, mapped to `SHORT`;
- equal-sign or zero-sign states are ambiguous and create no trade.

The hypothesis is falsifiable. WBTC activity can lag price, USDC mint/burn can
be administrative, and neither contract identifies exchange destination or
economic ownership. The candidate is not production evidence and cannot be
repaired after source incidence or outcomes are opened.

This decision was made without reading a WCDR source state, WCDR event count,
BTC market row, funding value, return, PnL, CAGR, MDD, or post-2023 contract
event. Aggregate source counts and prior rejected stablecoin mechanisms are
research-seen. A separate reviewed proposal computed incidence for a different
six-hour WBTC-turnover/stablecoin-direction rule; that rule and its support are
not WCDR evidence and may not alter WCDR.

## Frozen source axes

### WBTC custody flow

- source:
  `data/wbtc_custody_bridge_flow_2020_2023/wbtc_mint_burn_2020_2023.csv.gz`
- source SHA-256:
  `bfcc6ebc2ded0cd8a57e5cda83a77daafe4de325adf606b23ba43ecf486b3b4e`
- source manifest:
  `results/wbtc_custody_bridge_flow_source_manifest_2026-07-23.json`
- manifest hash:
  `4e4344a7f2841803dc8da625ee1320f79e1821d54cb2366a5464728507b4bcab`
- allowed fields:
  `event,event_sign,amount_raw,actor_address,block_number,transaction_index,semantic_log_index,available_at`

Eligible rows are exactly `asset == "wbtc_eth"` and
`event in {"mint", "burn"}`. Amounts remain exact 8-decimal integers.

### USDC dollar-liquidity flow

- source:
  `data/ethereum_stablecoin_issuance_redemption_2020_2023/ethereum_usdt_usdc_issuance_redemption_2020_2023.csv.gz`
- source SHA-256:
  `70ba3799ba84dc671051623a8d167b1731f043cf84a686b9878a67fcd52e5901`
- source manifest:
  `results/ethereum_stablecoin_issuance_redemption_source_manifest_2026-07-21.json`
- manifest hash:
  `a0c7740db64f7779fade68d76985c629cabe81983bf594e8258cef16a5725a1b`
- allowed fields:
  `asset,event,event_sign,amount_raw,indexed_address_1,block_number,transaction_index,log_index,available_at`

Eligible rows are exactly `asset == "usdc_eth"` and
`event in {"mint", "burn"}`. USDT, black-funds destruction, transfer logs,
API workflow dates, proof-of-reserve balances, and other bridges are excluded.
Amounts remain exact 6-decimal integers.

Both sources become usable only at their canonical block `N+64`
`available_at`. `block_timestamp` is forbidden from feature scheduling.

## Frozen causal state

### Decision grid and operational delay

Evaluate one state at every UTC calendar-day anchor `D = 00:00:00`. The source
cutoff is:

```text
C(D) = D - 6 hours
```

The six-hour delay is an operational ingestion allowance beyond the already
finalized `available_at`. Events with `available_at > C(D)` are invisible.

### WBTC state

Use the half-open 30-calendar-day window:

```text
W(D) = {e: C(D) - 30 days < e.available_at <= C(D)}

wbtc_net_raw(D)   = sum(e.event_sign * integer(e.amount_raw))
wbtc_gross_raw(D) = sum(integer(e.amount_raw))
wbtc_rows(D)      = count(W(D))
wbtc_actors(D)    = count(distinct e.actor_address)
wbtc_top_share(D) = max_actor(sum actor amount_raw) / wbtc_gross_raw(D)
```

A WBTC state is valid only when gross amount is positive, at least three
canonical events and two distinct actors are present, every actor is a valid
nonzero Ethereum address, and `wbtc_top_share <= 0.90`.

### USDC state

Use the half-open 7-calendar-day window:

```text
S(D) = {e: C(D) - 7 days < e.available_at <= C(D)}

usdc_net_raw(D)   = sum(e.event_sign * integer(e.amount_raw))
usdc_gross_raw(D) = sum(integer(e.amount_raw))
usdc_rows(D)      = count(S(D))
```

A USDC state is valid only when gross amount is positive and at least 30
canonical events are present. No amount clipping, quantile threshold,
forward-fill, entity label, exchange-address guess, or full-sample
normalization is allowed.

### Direction

Only opposite nonzero signs create a candidate:

```text
wbtc_net_raw < 0 and usdc_net_raw > 0 -> LONG
wbtc_net_raw > 0 and usdc_net_raw < 0 -> SHORT
otherwise                            -> no candidate
```

Raw net/gross ratios, event-count balances, actor shares, and state ages are
retained as diagnostics and later symbolic RLLM inputs, but they do not add a
threshold or change the deterministic side.

## Frozen execution clock

For a valid state at anchor `D`:

```text
decision_time = D
entry_time    = D + 5 minutes
exit_time     = entry_time + 2016 five-minute bars
```

- exposure: fixed `0.5x` account notional;
- hold: exactly seven elapsed days;
- no stop, take-profit, trailing exit, leverage search, or intraday timing
  choice;
- candidates are processed by `(entry_time, side, source identity hash)`;
- accept only when `entry_time >=` the prior accepted `exit_time`;
- a trade crossing a research-window boundary is skipped, never truncated;
- one position globally, with no pyramiding or same-anchor replacement.

Daily evaluation plus seven-day non-overlap prevents a persistent source state
from manufacturing intraday sample count. Later significance must still
cluster by calendar month because consecutive weekly trades can share source
events.

## Frozen windows and source-support gates

- warm-up only: `[2020-01-01, 2021-01-01)`;
- train clock: `[2021-01-01, 2023-01-01)`;
- selection clock: `[2023-01-01, 2024-01-01)`;
- every source and outcome row at or after `2024-01-01` remains sealed until
  the complete pre-2024 sequence authorizes extension.

Before any BTC outcome is opened, the source-only primary must satisfy every
gate:

1. at least 50 accepted train trades and 20 accepted selection trades;
2. at least 20 trades in each of 2021 and 2022;
3. at least 8 trades in each train half-year and each 2023 half-year;
4. at least 12 trades per side in train and 4 per side in selection;
5. no UTC entry month above 20% in either split;
6. no more than 10 consecutive accepted trades on one side;
7. at least 10 distinct WBTC actors contribute to accepted windows in train
   and at least 5 in selection;
8. exact source hashes, monotone causal clocks, `N+64` confirmation identity,
   half-open lookbacks, split containment, and global non-overlap all pass.

Any failure retires WCDR-2016 without changing a source, window, actor cap,
minimum count, sign, delay, hold, or support floor.

## Frozen controls

Controls are diagnostics and cannot replace a failed primary:

1. `direction_flip`: exact primary clock with both sides reversed;
2. `wbtc_only_contrarian`: use valid WBTC states and side
   `-sign(wbtc_net_raw)`, ignoring USDC sign;
3. `usdc_only_direct`: use valid USDC states and side
   `sign(usdc_net_raw)`, ignoring WBTC sign;
4. `same_sign_direct`: require equal nonzero signs and use the common sign;
5. `stale_7d`: exact primary construction with both source cutoffs delayed an
   additional seven calendar days;
6. `count_sign_consensus`: primary plus amount-net sign equal to event-count
   net sign in both source windows;
7. `year_amount_permutation`: deterministic within-source, within-event,
   within-calendar-year amount permutation before constructing states;
8. `deterministic_random_side`: exact primary clock with a SHA-256-fixed,
   side-count-matched permutation of primary sides.

The source-only support evaluator must materialize these clocks without any
price or return. Controls do not need to satisfy primary support, but their
incidence and overlap must be reported.

## Strict economic sequence

Only a committed source-support pass may authorize a separately implemented,
tested, hash-frozen evaluator.

1. Open train market/funding through 2022 only.
2. Stop if train fails; otherwise open 2023 selection.
3. Stop if selection fails; otherwise build and hash-audit immutable post-2023
   source extensions without reading BTC outcomes.
4. Freeze the extended clock, then open 2024 test, 2025 eval, and 2026 recent
   in that order. A later stage cannot repair or reselect anything.

Each table must report absolute return, full-calendar CAGR including idle cash,
strict MDD, CAGR/strict-MDD, trades, sides, exposure, realized funding, mean
gross/net trade return, and calendar-month clustered significance. Base cost
is 6 bp/notional/side and stress cost is 10 bp/notional/side.

Train and selection each require:

- positive absolute return;
- CAGR/strict-MDD at least `3.0`;
- strict MDD at most `15%`;
- positive stress-cost return;
- one-sided calendar-month cluster sign-flip `p <= 0.10`;
- primary ratio and absolute return greater than both source-component-only
  controls;
- neither direction flip, stale clock, nor random side may satisfy the full
  primary gate.

Later test, eval, and recent windows retain the `3.0` ratio and `15%` strict-MDD
targets. Historical periods have broad repository exposure, so even a complete
pass is a shadow candidate until genuinely forward data accumulates.

## Post-2023 source extension contract

After pre-2024 support and economics pass, extend only the same Ethereum
contract addresses, event topics, receipt/header integrity rules, and `N+64`
availability. The 30-day WBTC window, 7-day USDC window, six-hour delay, actor
cap, minimum event counts, sign mapping, daily anchors, execution, and controls
remain unchanged.

No WBTC Open API date, current dashboard status, cbBTC, tBTC, Tron/Solana WBTC,
USDT, new contract, or address classification may enter WCDR without a new
source/mechanism identity frozen before access.

## RLLM boundary

Gemma/RLLM is authorized only after deterministic train and selection pass. A
compact text state may expose bucketed WBTC/USDC net-gross ratios, event-count
balances, actor concentration, state age, current position, and time to exit.
Allowed actions are `TRADE_FIXED_SIDE` and `ABSTAIN` (or fixed lower size under
a separately frozen policy). The model may not create a clock, reverse side,
change hold, read raw future labels at inference, or rescue deterministic
failure. RL reward must penalize strict drawdown and turnover.

## Next admissible action

Hash-bind this singleton in an immutable preregistration artifact. That stage
may verify file hashes and CSV headers only. Only the committed
preregistration may authorize real WCDR source-state construction.

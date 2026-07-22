# WSCF-72-SOURCE-FAMILY-SEEN mechanism decision — 2026-07-23

## Decision

Freeze one candidate-specific outcome-blind mechanism,
**WSCF-72-SOURCE-FAMILY-SEEN — WBTC / Stablecoin Finalized Confirmation
Relay**.

A finalized WBTC custody-flow batch starts a causal 12-hour observation
interval. The trade exists only if subsequently finalized Ethereum USDC/USDT
issuance-redemption flow first accumulates to the same sign as that WBTC batch.
The interaction is a two-stage event transition, not a rolling liquidity level:

- positive WBTC net mint followed by positive stablecoin cumulative net flow
  maps to `LONG`;
- negative WBTC net burn followed by negative stablecoin cumulative net flow
  maps to `SHORT`;
- no same-sign confirmation within 12 elapsed hours means no candidate.

The 72-hour hold tests whether coordinated expansion or contraction across a
BTC-linked custody boundary and crypto-dollar settlement supply transmits more
slowly than either administrative stream alone.

## Research-boundary disclosure

This is **not a clean first hypothesis**. WBTC and Ethereum stablecoin source
values, aggregate incidence, and prior source-only clocks have already been
seen in this repository. WCDR-2016 and WTSL-168-SOURCE-SEEN were both retired
before their BTC outcomes were opened, and WCDR included a broad
`same_sign_direct` source-only control. WSCF is therefore the third explicit
WBTC/stablecoin source-family candidate.

Before this freeze:

- both source artifacts and sample source rows had been opened;
- WCDR/WTSL and other stablecoin-family source clocks had been opened;
- the exact WSCF atomic-batch, first-passage clock had **not** been derived;
- no WSCF entry was joined to BTC OHLC, funding, return, PnL, CAGR, or MDD;
- no WSCF threshold, direction, delay, hold, or control was selected from a
  market outcome.

Consequently, later source support is a minimum-identifiability and novelty
check, not pristine confirmatory evidence. Candidate-specific outcomes remain
sealed until the exact clock and strict evaluator are separately frozen.

## Source bindings

### WBTC custody flow

- source:
  `data/wbtc_custody_bridge_flow_2020_2023/wbtc_mint_burn_2020_2023.csv.gz`
- SHA-256:
  `bfcc6ebc2ded0cd8a57e5cda83a77daafe4de325adf606b23ba43ecf486b3b4e`
- source manifest:
  `results/wbtc_custody_bridge_flow_source_manifest_2026-07-23.json`
- manifest hash:
  `4e4344a7f2841803dc8da625ee1320f79e1821d54cb2366a5464728507b4bcab`
- eligible rows: exactly `asset == "wbtc_eth"` and
  `event in {"mint", "burn"}`;
- canonical availability: the timestamp of confirmation block `N+64` in
  `available_at`;
- exact amount unit: 8-decimal integer `amount_raw`.

### Ethereum stablecoin issuance-redemption

- source:
  `data/ethereum_stablecoin_issuance_redemption_2020_2023/`
  `ethereum_usdt_usdc_issuance_redemption_2020_2023.csv.gz`
- SHA-256:
  `70ba3799ba84dc671051623a8d167b1731f043cf84a686b9878a67fcd52e5901`
- source manifest:
  `results/ethereum_stablecoin_issuance_redemption_source_manifest_2026-07-21.json`
- manifest hash:
  `a0c7740db64f7779fade68d76985c629cabe81983bf594e8258cef16a5725a1b`
- directional rows: USDC `mint`/`burn` and USDT `issue`/`redeem` only;
- `destroyed_black_funds` is confiscation, not redemption, and contributes
  neither net nor gross primary flow;
- canonical availability: confirmation block `N+64` `available_at`;
- exact amount unit: both tokens use 6-decimal integer `amount_raw`, so their
  directional amounts may be summed without floating-point conversion.

`block_timestamp`, transaction occurrence time, provider receipt time, REST
workflow date, and current dashboard state are forbidden for scheduling.
Official source documentation supports the event semantics and finality
transport, but does not establish the trading alpha:

- https://docs.wbtc.network/how-wbtc-works/mint-burn-mechanism
- https://ethereum.org/developers/docs/apis/json-rpc/
- https://developers.circle.com/stablecoins/usdc-contract-addresses
- https://tether.to/en/supported-protocols/
- https://www.federalreserve.gov/econres/notes/feds-notes/primary-and-secondary-markets-for-stablecoins-20240223.html

## Frozen atomic availability batches

Rows that become knowable at the same exact `available_at` are simultaneous.
No transaction or log ordering inside that information timestamp may create a
fictitious earlier signal.

### WBTC anchor batch

For each exact WBTC `available_at = T_wbtc`, aggregate all eligible rows:

```text
wbtc_net_raw   = sum(event_sign * amount_raw)
wbtc_gross_raw = sum(amount_raw)
wbtc_sign      = sign(wbtc_net_raw)
```

The batch identity is a SHA-256 hash of the sorted canonical row identities
`(block_number, transaction_index, semantic_log_index)`. A zero-net batch is
not an anchor. WBTC gross amount, row count, actor set, and actor concentration
are diagnostics only; no amount, count, or concentration threshold changes the
primary clock.

### Stablecoin availability batch

For each exact stablecoin `available_at`, aggregate all eligible directional
rows after validating asset/event/sign/decimals and canonical identity
`(block_number, transaction_index, log_index)`. `destroyed_black_funds` and any
non-directional event are excluded before net and gross are computed. The
batch identity hashes its sorted eligible row identities. A zero-net batch is
retained because it can change gross diagnostics but cannot confirm a side.

## Frozen primary first-passage clock

For every nonzero WBTC anchor batch, independently inspect stablecoin batches
in the strict half-open/closed interval:

```text
T_wbtc < stablecoin.available_at <= T_wbtc + 12 elapsed hours
```

Start cumulative stablecoin net at zero and update it once per atomic
availability batch in ascending `(available_at, batch_identity)` order. The
first batch for which

```text
sign(cumulative_stablecoin_net_raw) == wbtc_sign
```

is the unique confirmation batch. Its `available_at` is `T_signal`. There is no
absolute amount, imbalance ratio, percentile, actor, or token-share threshold.
A positive pair is `LONG`; a negative pair is `SHORT`. If no batch confirms,
there is no raw candidate.

USDC and USDT attribution, cumulative gross, elapsed confirmation seconds,
WBTC amount/actors, and the count of stablecoin batches are diagnostics only.

## Frozen execution and scheduler

```text
entry_time = ceil_to_5m(T_signal) + 5 elapsed minutes
exit_time  = entry_time + 72 elapsed hours
```

If `T_signal` is already exactly on a five-minute boundary, entry is still the
next bar, never the same bar.

- exposure: fixed `0.5x` account notional;
- reservation interval: `[entry_time, exit_time)`;
- no stop, take-profit, trailing exit, leverage search, or early close;
- build all raw candidates, then sort by
  `(entry_time, T_signal, T_wbtc, wbtc_batch_identity,
  stablecoin_batch_identity, side)`;
- accept only when `entry_time >= prior_accepted_exit`;
- one global position, no pyramiding, no queueing of suppressed candidates;
- an accepted confirmation-batch identity cannot be reused;
- a trade crossing a research split boundary is skipped, never truncated.

## Frozen windows

- source-only warm-up where needed: `[2020-01-01, 2021-01-01)`;
- train: `[2021-01-01, 2023-01-01)`;
- selection: `[2023-01-01, 2024-01-01)`;
- every contract-event value and outcome at or after `2024-01-01T00:00:00Z`
  remains sealed until the complete pre-2024 sequence passes.

Entry and exit must both be contained in the same half-open split.

## Frozen controls

Controls are diagnostic mechanisms and cannot replace a failed primary:

1. `direction_flip`: exact primary entries with both sides reversed;
2. `deterministic_random_side`: exact primary entries with a SHA-256 fixed,
   side-count-matched permutation;
3. `wbtc_only_direct`: signal at each nonzero WBTC batch, side equal to its
   sign, with the same entry latency, hold, and scheduler;
4. `stablecoin_only_12h_grid`: at UTC `00:00` and `12:00`, use the signed net
   of eligible stablecoin batches in `(D-12h, D]`, signal at `D`, and trade its
   nonzero sign with the same execution;
5. `anchored_first_nonzero`: use each WBTC batch only as a timing anchor and
   accept the first subsequent stablecoin cumulative nonzero sign, without
   requiring agreement with WBTC; side is the stablecoin sign;
6. `opposite_confirmation`: first subsequent cumulative stablecoin sign
   opposite to the WBTC sign within 12 hours;
7. `lead_lag_reverse`: at WBTC availability, require the aggregate eligible
   stablecoin net in `(T_wbtc-12h, T_wbtc]` to have the WBTC sign; signal at
   `T_wbtc`;
8. `stale_wbtc_24h` and `stale_wbtc_72h`: shift WBTC availability forward by
   exactly 24 or 72 hours, then apply the unchanged subsequent 12-hour
   confirmation rule;
9. `stablecoin_year_amount_permutation`: deterministically permute
   `amount_raw` within `(asset,event,UTC-year)` while preserving every sign,
   identity, and timestamp;
10. `black_funds_veto`: primary construction, but any
    `destroyed_black_funds` availability in the 12-hour confirmation interval
    vetoes that anchor;
11. `usdc_only_confirmation` and `usdt_only_confirmation`: unchanged primary
    construction using only that token's eligible directional batches.

Full qualification by direction flip, random side, opposite confirmation,
lead-lag reverse, or either stale clock rejects the causal interpretation.
Primary must beat both WBTC-only and stablecoin-only controls on opened train
and selection risk-adjusted performance.

## Source-support gates

Before any WSCF BTC outcome is opened, the exact primary must satisfy all:

- train total at least 50 and selection total at least 20;
- each of 2021 and 2022 at least 20 accepted candidates;
- each train half-year at least 8 and each selection half-year at least 6;
- train at least 12 candidates per side and selection at least 4 per side;
- no UTC entry month above 20% and no UTC entry quarter above 40% of a split;
- no more than 10 consecutive accepted candidates on one side;
- at least 10 distinct contributing WBTC actors in train and 5 in selection;
- no duplicate accepted WBTC or confirmation-batch identity;
- maximum calendar gap between accepted entries at most 90 days;
- exact source hashes, monotone causal batches, `N+64` confirmation identity,
  split containment, and global non-overlap all pass;
- zero BTC market, funding, future-return, or PnL rows are read.

Failure retires WSCF-72-SOURCE-FAMILY-SEEN without threshold, window, delay,
hold, side, token, scheduler, or support-floor repair.

## Frozen novelty checks

Compare only overlapping pre-2024 intervals against hash-bound source-only
clocks:

- WCDR-2016 primary;
- WTSL-168-SOURCE-SEEN primary;
- UGCI-288 primary;
- the sealed AMTR/SQFD/SDDR/UCBR comparator bundle;
- each member of the frozen live-portfolio pure-clock bundle.

For direction-aware clocks, compute same-side and signless metrics separately.
For the timestamp-only sealed comparator bundle, compute signless metrics.
For each comparator view with at least 10 entries, report:

- exact-entry Jaccard;
- WSCF-to-comparator containment within `±12h`;
- comparator-to-WSCF containment within `±12h`.

Hard novelty gates are maximum exact-entry Jaccard `<= 0.10` and maximum
WSCF-to-comparator near containment `<= 0.30` for every eligible comparator
view. Reverse containment is reported but is not a gate because comparator
clock density differs materially. Comparator controls, identities, hashes, and
intervals are frozen before WSCF incidence is built.

## Strict economic sequence and gates

Only a committed source-support pass may authorize a separately implemented,
tested, hash-frozen evaluator:

1. open train BTC market/funding through 2022;
2. stop unless every train gate passes;
3. open 2023 selection only after train pass;
4. stop unless every selection gate passes;
5. extend the unchanged contract/topic/finality policy after 2023;
6. freeze the extended source clock before opening 2024 test, 2025 eval, and
   2026 recent outcomes in order.

Each opened primary split must satisfy:

- positive absolute return over the full calendar split;
- full-calendar CAGR / strict intratrade MDD at least `3.0`;
- strict MDD at most `15%`;
- positive return under 10 bp notional cost per side;
- realized funding included;
- at least 20 trades and at least 4 per side;
- calendar-month clustered sign-flip `p <= 0.10`;
- primary CAGR/MDD strictly above WBTC-only and stablecoin-only controls.

Base cost is 6 bp of notional per side. CAGR includes every idle day. Strict
MDD counts entry gaps and the highest marked-to-market equity reached before or
during every position.

## RLLM boundary

No LLM or RL component may create a WSCF entry, reverse its side, change the
12-hour confirmation interval or 72-hour hold, or optimize these rules before
deterministic train and selection pass. A later compact RLLM may only choose
`TRADE_FIXED_SIDE` or `ABSTAIN` from causal, bucketed diagnostics plus current
position and time-to-exit state. Its reward must penalize strict drawdown and
turnover, and it must beat the frozen deterministic policy on untouched
post-selection windows.

## Stopping rule

Any identity, source, causality, support, novelty, train, or staged-selection
failure retires WSCF-72-SOURCE-FAMILY-SEEN. Any different sign, event grouping,
confirmation window, token set, delay, hold, leverage, control, or gate is a
new hypothesis requiring a new freeze before access.

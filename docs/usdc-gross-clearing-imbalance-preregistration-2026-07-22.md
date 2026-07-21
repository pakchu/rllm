# UGCI-288 USDC gross-clearing imbalance — outcome-blind preregistration

## Decision boundary

Freeze one deterministic candidate, **UGCI-288**, before calculating its real
six-hour packet distribution, accepted event clock, comparator overlap, or any
BTC outcome.  This work unit hashes the promoted Ethereum source and fixes the
policy contract.  The preregistration generator reads zero source rows.  A
preceding outcome-blind schema probe displayed the columns and eight leading
rows, and the promoted source audit had already published aggregate event/year
counts; neither computed a six-hour packet, threshold, accepted incidence, or
comparator overlap.  No BTC OHLC, funding, future return, PnL, CAGR, strict MDD,
or post-2023 contract-event row is opened.

The preceding BPAX semantic candidate failed its frozen Gemma synthetic gate.
UGCI therefore returns to an exact integer ledger transform: no LLM decides an
event, direction, timestamp, threshold, or trade.

## Economic hypothesis

USDC `Mint` and `Burn` logs are grouped by their causally finalized
`available_at` timestamp into fixed UTC six-hour packets.  A packet with an
unusually large **gross two-way clearing amount** can represent a renewal or
contraction of dollar liquidity plumbing.  Its signed net amount is the
falsifiable direction map:

```text
mint_raw = sum(amount_raw where event == "mint")
burn_raw = sum(amount_raw where event == "burn")
gross_raw = mint_raw + burn_raw
net_raw = mint_raw - burn_raw
imbalance_ratio = abs(net_raw) / gross_raw

net_raw > 0  -> LONG BTCUSDT
net_raw < 0  -> SHORT BTCUSDT
```

This is not a claim that every mint reaches an exchange, every burn is customer
redemption, or gross clearing is inherently bullish.  The sign convention and
24-hour response are a single hypothesis that must beat its controls after
costs.

## Source and causal clock

- Source: the promoted Ethereum mainnet USDT/USDC event panel, restricted to
  `asset == "usdc_eth"` and `event in {"mint", "burn"}`.
- Event availability: canonical block `N+64` timestamp in `available_at`; the
  event's own block timestamp is never the decision clock.
- Packet grid: exact UTC intervals `[00:00,06:00)`, `[06:00,12:00)`,
  `[12:00,18:00)`, and `[18:00,24:00)`; zero-event packets remain in the
  historical threshold distribution.
- Source range used for support: 2020 is warm-up; accepted primary events are
  reported only for 2021-2023.  `available_at >= 2024-01-01T00:00:00Z` is
  excluded.
- The fixed source is immutable and independently replayed, but the contract
  logs do not identify economic ownership or exchange destination.

Official source definitions already frozen by the promoted manifest:

- Ethereum JSON-RPC: <https://ethereum.org/developers/docs/apis/json-rpc/>
- Circle USDC contract addresses:
  <https://developers.circle.com/stablecoins/usdc-contract-addresses>
- Circle stablecoin EVM contracts:
  <https://github.com/circlefin/stablecoin-evm>

## Singleton policy

- Prior gross threshold: nearest-rank `q95` over the strictly prior 180-day
  six-hour grid; current packet is excluded and at least 360 prior packets are
  required.
- Material direction: `imbalance_ratio >= 0.60` and `net_raw != 0`.
- Entry: packet end plus one complete five-minute latency bar and then the next
  five-minute open, exactly `packet_end + 10 minutes`.
- Exit: scheduled open after exactly 288 five-minute bars (`24 hours`).
- Exposure: `0.5x`; primary entries are globally non-overlapping.
- There is no threshold, packet width, side, hold, or latency sweep.

Source-only controls are frozen as `no_gross_tail`, `no_imbalance_floor`, and
`stale_6h`.  Later economic controls must also include exact direction flip,
deterministic random side, doubled cost, and an extra one-hour execution delay.

## Source-support and novelty gates

Before any outcome file may be hashed or parsed, the primary clock must satisfy
all of the following:

1. at least 120 accepted events in 2021-2022 combined and at least 50 in 2023;
2. at least 45 events in each of 2021 and 2022, and at least 20 in each 2023
   half;
3. LONG share between 25% and 75% in both the train and 2023 windows;
4. no entry month above 15% of the corresponding window;
5. exact source hashes, complete UTC grid, strictly-prior thresholds, causal
   availability, and non-overlap invariants all pass; and
6. against every frozen comparator, exact-entry Jaccard is at most 0.10 and the
   maximum bidirectional containment within plus/minus six hours is at most
   0.35.

The comparator set is fixed to AMTR-48 `primary` and `cross_minter`, SQFD-6
`primary`/`no_usdt_lag`/`no_participation`, SDDR-12 `primary`, and UCBR-12
`primary`.  Comparator timestamps may be opened only after the primary support
checks pass.  AMTR is compared on `[2021-01-01, 2024-01-01)`; the three Binance
stablecoin clocks are compared on their shared full-signal interval
`[2023-09-01, 2024-01-01)`.  A comparison requires at least ten UGCI and five
comparator entries on its fixed interval; otherwise novelty fails closed.  A
support or novelty failure retires UGCI-288 without repair.

## Why this is not AMTR's promoted control

AMTR-48 matched two individually large, opposite events within 24 hours and
required same-role continuity; its `cross_minter` ablation removed that
continuity.  UGCI does not match event pairs or inspect a role/address.  It
aggregates every finalized mint and burn on a fixed six-hour grid, requires a
gross-tail packet plus a material contemporaneous net imbalance, waits ten
minutes, and holds 24 hours.  AMTR's cross-minter clock is therefore a mandatory
negative comparator, not the UGCI definition or a source of thresholds.

## One-way outcome sequence

Only a committed source-support pass may authorize a separately hash-frozen
strict evaluator.  The evaluator must open 2021-2022 first, then 2023 only after
the train gate passes.  Any later 2024+ source and outcome stays sealed until
the complete pre-2024 sequence passes.  Every economic report must include
absolute return, full-calendar CAGR, strict favorable-before-adverse held-path
MDD, CAGR/strict-MDD, trades, side counts, realized funding, and base/stress
costs.

The repository has broad human exposure to these market years.  Hash freezing
prevents candidate-level retuning; it does not recreate a pristine global
holdout.  Live promotion would still require independent forward shadow data
and destination/ledger parity.

Gemma/RLLM may consume a compact symbolic UGCI state only after deterministic
economics pass.  It may abstain or size within a separately frozen policy, but
it may not create, retime, reverse, or repair UGCI events.

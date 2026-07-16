# NTB-7: Network Topology Broadening preregistration

Status: **frozen before any NTB-7 post-entry return was inspected**.

## Why this axis

NTB-7 uses Bitcoin ledger topology rather than price, funding, premium, OI,
Kimchi, FX, REX, Markov state, wick shape, or aggTrade microstructure.

- Coin Metrics defines `AdrActCnt` as unique addresses active as a recipient or
  originator during the interval. It also warns that addresses are only a proxy
  for users and inherit chain-specific artifacts:
  <https://docs.coinmetrics.io/asset-metrics/addresses/adractcnt>
- `TxCnt` counts ledger transactions:
  <https://docs.coinmetrics.io/asset-metrics/transactions/txcnt>
- `TxTfrCnt` counts positive-value transfers. One transaction can contain many
  transfers, including batched payroll or withdrawals:
  <https://docs.coinmetrics.io/asset-metrics/transactions/txtfrcnt>
- The source was downloaded through Coin Metrics API v4:
  <https://docs.coinmetrics.io/api/v4/>

The hypothesis is deliberately narrower than a generic activity spike. A
positive topology event requires unique-address participation per transfer to
broaden while transfers per transaction contract. This is interpreted as a
shift away from batched/internal fan-out toward broader independent ledger
participation. It is a proxy, not an entity-labelled exchange-flow measure.

## Frozen singleton

Policy `NTB-7`:

1. `fanout = log(TxTfrCnt / TxCnt)`.
2. `breadth = log(AdrActCnt / TxTfrCnt)`.
3. Take seven-observation changes.
4. Standardize each change against the last 180 strictly earlier observations,
   requiring 120 and requiring their `available_at` to precede the candidate.
5. A state is eligible only when:
   - `breadth_z >= 0.5`,
   - `fanout_z <= -0.5`,
   - `breadth_z - fanout_z >= 1.5`, and
   - source publication lag is at most three days.
6. Emit only the first eligible observation after an ineligible observation.
7. Go long at one complete five-minute latency bar after the first five-minute
   open at or after `AssetEODCompletionTime`.
8. Hold exactly 2,016 five-minute bars (seven days), with no overlapping trade.
9. Use 0.5x exposure, 6 bp/notional/side base cost, and 10 bp/notional/side
   stress cost.

There is one candidate. Threshold, side, latency, and hold cannot be repaired
after support or returns are opened.

## Leakage and revision boundary

- The exact signal clock loads only `observation_date`, `available_at`,
  `AdrActCnt`, `TxCnt`, and `TxTfrCnt`.
- `available_at` is Coin Metrics `AssetEODCompletionTime`; no row is usable at
  its observation timestamp.
- Old backfilled rows can seed a reference distribution after publication but
  cannot emit a signal when publication lag exceeds three days.
- Market and funding readers must physically stop before `2024-01-01` during
  selection.
- The downloaded file is a frozen vintage, not a complete point-in-time
  revision database. Live promotion therefore requires forward-vintage parity.

## Support gate before outcomes

The outcome-blind, non-overlapping clock must contain at least:

- 40 train events across 2021-2022,
- 15 in each train year,
- 16 in 2023,
- 6 in each 2023 half,
- no month above 20% of all events.

Failure rejects NTB-7 without opening post-entry outcomes.

## Frozen performance gate

Both 2021-2022 train and 2023 selection must have:

- positive absolute return,
- CAGR / strict MDD at least 3.0,
- strict MDD at most 15%,
- weekly cluster sign-flip `p <= 0.10`,
- mean gross underlying move at least 40 bp,
- positive result at 10 bp/notional/side stress cost.

Both 2023 halves and a one-bar-delayed execution must be positive. Strict MDD
uses the global/pre-entry high-water mark, held favorable-before-adverse OHLC,
funding, and entry/exit/hypothetical-liquidation costs. CAGR uses the full split
wall clock, including idle cash.

## Controls and promotion

Frozen controls are direction flip, breadth-only, fanout-only, seven-day stale
topology, one-bar delayed entry, prior activity-shock, and year-stratified random
clocks. A component-only or stale control passing every primary gate rejects the
claimed joint mechanism.

Only after performance passes will NTB-7 be tested against every already-frozen
promoted/live/shadow sleeve. Promotion requires low clock/PnL overlap and a
positive marginal portfolio contribution. The years 2024, 2025, and 2026 YTD
remain sealed until those pre-2024 gates pass.

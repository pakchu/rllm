# UTXO Fee-Clearing Polarity mechanism decision — 2026-07-20

## Decision

The next standalone BTC candidate will use a new public-ledger observable:
**UTXO Fee-Clearing Polarity (UFCP)**.  UFCP asks whether unusually expensive
confirmed settlement expands or contracts the UTXO set.  This file freezes the
source axis and causal boundary only.  It does not choose a threshold, side
cutoff, hold, leverage, or model, and it opens no event incidence or market
outcome.

The deterministic primitives are restricted to:

- non-coinbase transaction fees;
- transaction input and output counts;
- transaction count and transaction weight; and
- the exact identity `utxo_change = outputs - inputs`.

The provisional economic orientation is two-sided.  Cost-insensitive UTXO
expansion is an urgent fan-out/distribution state and tentatively maps long;
cost-insensitive UTXO contraction is an urgent fan-in/consolidation state and
tentatively maps short.  Ownership is not observable: exchange batching,
wallet maintenance, CoinJoin, inscriptions, and self-transfers can all falsify
that interpretation.  The later preregistration must therefore include
fee-only, topology-only, low-fee, stale, direction-flip, and randomized-clock
controls.  The direction may not be repaired after returns are opened.

## Why this is not a BFC, NWE, BATE, or miner-cadence repair

- BFC-3 used daily aggregate fees relative to issuance plus transactions per
  block and assumed a long continuation.  It was rejected before outcomes
  because its sparse clock failed support.
- NWE-8 combined daily blockspace/address aggregates in a weekly ridge model.
  Its frozen train lost money and its relationship changed sign by regime.
- BATE-288 used block arrival time, block weight, and transaction throughput.
  It did not load fees, inputs, outputs, or UTXO-set change.
- MCR-7 used daily hash-rate recovery and block cadence.

UFCP does not lower any failed threshold or invert any failed policy.  Its new
quantity is the **signed topology of ledger state creation under an explicit
fee burden**.  Neither `inputs`, `outputs`, nor `utxo_change` appears in those
policies.

## Authoritative production source

The production definition is Bitcoin Core `getblockstats`, pinned to one Core
version before live promotion.  The current official RPC documents:

- `totalfee` in satoshis;
- `feerate_percentiles` in sat/vB;
- `ins` and `outs` as input/output counts;
- `utxo_increase` and `utxo_size_inc` as UTXO-set changes; and
- `swtxs` and transaction-weight statistics.

Official references:

- Bitcoin Core 30.0.0 `getblockstats`:
  <https://bitcoincore.org/en/doc/30.0.0/rpc/blockchain/getblockstats/>
- Bitcoin Core 0.17.0 release notes, where `getblockstats` was introduced:
  <https://bitcoincore.org/en/releases/0.17.0/>

An unpruned archival node is required for deterministic historical RPC replay;
the current local WSL disk ceiling does not permit adding a full archival
chain.  Historical research transport will therefore use the public Mempool
block endpoint only for fields that are deterministic from the confirmed
ledger.  A production deployment must replace that transport with an owned
Bitcoin Core node on a suitable volume and prove field-by-field parity before
orders are enabled.

## Bounded source-schema probe

Before this decision, a schema-only probe requested five isolated historical
heights spanning the already frozen 2020-2023 block prefix.  It printed no
feature values, event incidence, price, funding, return, or PnL.  All five
responses exposed:

- `extras.totalFees`;
- `extras.totalInputs` and `extras.totalOutputs`;
- `extras.utxoSetChange`;
- block `tx_count`, `weight`, hashes, height, timestamp, and median time.

For all five bounded rows, the exact source invariant
`utxoSetChange == totalOutputs - totalInputs` held.  The richer endpoint returns
15 blocks per request.  Historical `extras.firstSeen` was null, so the public
archive does **not** support a trustworthy intrablock receipt-time backtest.

Mempool documents its REST surface and self-hosting, and its code is AGPL-3.0:

- REST API: <https://mempool.space/docs/api/rest>
- official repository and self-hosting:
  <https://github.com/mempool/mempool>
- software license:
  <https://github.com/mempool/mempool/blob/master/LICENSE>

The public API output has no separately documented commercial-data licence in
the reviewed official material.  It is a research transport, not the live
source of record.  Pool tags, `expectedFees`, `expectedWeight`, projected
blocks, acceleration annotations, and Mempool-specific fee-range fields are
excluded because they are not consensus-derived historical primitives with a
stable Core parity contract.

## Frozen causal boundary

Because historical block receipt times are absent, UFCP may not use a block
header timestamp as an immediate trade time.  The source-support
preregistration must enforce all of the following before any incidence is
opened:

1. aggregate only completed UTC source days;
2. treat source day `D` as unavailable before `D+2 00:00 UTC`;
3. require at least six hash-linked successor blocks after the final included
   block and require those successor blocks to be present in the frozen source;
4. enter no earlier than one complete five-minute latency bar after the fixed
   publication time; and
5. never use `firstSeen`, pool identity, an unconfirmed mempool state, or a
   post-entry block to create or filter an event.

This delayed calendar contract is deliberately slower than live observation.
It prevents an optimistic immediate-header-time replay, but it is not a
historical archive of actual node receipt timestamps.  Live promotion still
requires forward collection of node receipt time, six-confirmation parity, and
at least 90 shadow days.  If a source-only implementation cannot enforce this
boundary exactly, UFCP is rejected before returns.

## Source alternatives rejected now

- Mempool fee projections and `feeRange`: current mempool state is node-local
  and cannot be reconstructed from the confirmed 2020-2023 chain.
- Mining-pool identity/concentration: pool attribution is tagged metadata, not
  a consensus field, and historical/live classifications can change.
- Binance `bookTicker`: the official REST/WebSocket interface is a current
  best-bid/ask snapshot/stream.  The checked official USD-M daily archive only
  covered 2023-05-16 through 2024-03-30, insufficient for the three-year
  contract.
- A new threshold on BFC/BATE/NWE outputs: that would be a post-failure repair,
  not a new observable.

Official Binance current-snapshot references:

- Spot order-book ticker:
  <https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/market#symbol-order-book-ticker>
- Spot WebSocket streams:
  <https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/ws-streams/~>

## Frozen research sequence

1. Implement a resumable source-only downloader for the exact 2020-2023 block
   prefix.  Persist only deterministic UFCP fields and source-integrity hashes.
2. Reconcile a fixed sample against Bitcoin Core field definitions and reject
   on any input/output/UTXO identity failure or hash-chain gap.
3. Commit one source-support preregistration with exact daily features,
   normalization, direction, support floors, publication lag, controls, and
   stopping rule before opening real incidence.
4. Build and hash-freeze the outcome evaluator before reading any post-entry
   OHLC or funding value.
5. Open 2021-2022 train first.  Only an exact train pass may open 2023;
   2024-2026 remain sealed and sequential.

The branch has broad historical research exposure, so this sequence can make a
candidate-level frozen claim only.  It cannot recreate a globally pristine
human holdout.

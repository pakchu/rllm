# Fee–Endpoint Topology Disagreement mechanism decision — 2026-07-20

## Decision

The next standalone BTC candidate is **FETD-288 — Fee–Endpoint Topology
Disagreement, 24-hour hold**.  It will test whether confirmed fee pressure and
the number of consumed/created UTXO endpoints per unit of block weight move in
opposite directions.

This file freezes the observable axis, tentative direction, source binding,
causal boundary, and staged calendar.  It does not freeze the final transform,
rank threshold, support floor, or evaluator, and it opens no FETD feature,
signal incidence, market value, funding value, return, PnL, CAGR, or MDD.

## Mechanism

For a completed, absolute-height-aligned 72-block packet, the candidate family
is restricted to two confirmed-ledger quantities:

```text
fee_pressure     = total_fees / block_weight
endpoint_density = (total_inputs + total_outputs) / block_weight
```

Inputs consume existing UTXOs and outputs create new transaction endpoints.
The sum is therefore an unsigned measure of confirmed endpoint participation;
dividing by weight asks how much of that topology is being cleared per unit of
scarce blockspace.  It is not an ownership, exchange-flow, or economic-value
measure.

The singleton preregistration will compare strictly causal short-horizon
changes in the two channels and admit only **opposite-direction transport**:

- fee pressure rising while endpoint density falls is tentatively **short**:
  scarce blockspace is repricing upward while broad endpoint clearing thins,
  consistent with concentrated urgency or composition crowding rather than
  broad settlement participation;
- fee pressure falling while endpoint density rises is tentatively **long**:
  more endpoints are clearing per unit weight while fee pressure relaxes,
  consistent with broader, more efficient settlement; and
- same-direction transport is not part of the primary policy.

This story is deliberately falsifiable.  Batching, wallet consolidation,
CoinJoin, inscriptions, script-type migration, exchange operations, and
self-transfers can all move the two aggregates without predicting BTC.  No
owner label or transaction-purpose classifier will be introduced after an
outcome is seen.

## Why this is a distinct ledger observable

- **UFCP-1** used daily fee burden per input/output edge plus the *signed* net
  UTXO polarity `outputs - inputs`.  FETD excludes net UTXO change and instead
  tests an *unsigned endpoint-density versus fee-pressure transport
  disagreement*.  It may not relax UFCP's rejected direction or month gates.
- **BFRT-288** used cross-percentile mined fee-rate transport.  It loaded no
  input or output counts.
- **BATE-288** used transaction and block-weight throughput per elapsed header
  time.  FETD uses no elapsed-time denominator and asks about composition per
  weight, not arrival throughput.
- **BFC-3** used daily total fees relative to issuance and transactions per
  block.  It did not use input/output endpoint density.
- **WCTR-288** used witness-discount composition and transaction fullness.  It
  loaded no fees or input/output topology.
- The rejected miner clock-skew proposal used only header timestamps and MTP.
  FETD introduces deterministic confirmed transaction statistics rather than
  another cadence transform.

The later evaluator must include fee-only, endpoint-only, same-direction,
direction-flip, stale-state, randomized-clock, and delayed-entry controls.  A
component control independently satisfying every primary gate rejects the
claimed disagreement mechanism.

## Frozen source binding

FETD reuses the exact confirmed-ledger source already frozen for UFCP research:

- source artifact:
  `data/bitcoin_utxo_fee_block_stats_2020_2023.csv.gz`;
- source SHA-256:
  `8d5160b59bd104819adb34752f2bd5b01e9c07d7aa108188abcd9e6d323d102f`;
- source manifest:
  `results/bitcoin_utxo_fee_block_stats_source_manifest_2026-07-20.json`;
- source-manifest file SHA-256:
  `ceb096b7bfeaf309729c4cab9f5cd4d2e0d526e261b1e460e9d03cd4f24e8084`;
- heights: `610691..823785`, all with header timestamps before 2024; and
- rows: 213,095 contiguous best-chain blocks.

The source contains block hashes/linkage, `timestamp`, `mediantime`,
`tx_count`, `size`, `weight`, `total_fees`, `total_inputs`, `total_outputs`,
and `utxo_set_change`.  FETD may read only the fields needed for integrity,
availability, fees, weight, inputs, and outputs.  It may not use
`utxo_set_change`, price, funding, premium, OI, liquidation, order book, or a
post-entry field.

Bitcoin Core's `getblockstats` is the authoritative production definition and
documents `totalfee`, `ins`, `outs`, total transaction weight, block hash, and
median time.  The current research transport used the public Mempool block
endpoint and already proved contiguous hashes plus
`utxoSetChange == totalOutputs - totalInputs`.

Official references:

- Bitcoin Core 30.0.0 `getblockstats`:
  <https://bitcoincore.org/en/doc/30.0.0/rpc/blockchain/getblockstats/>
- Mempool REST API:
  <https://mempool.space/docs/api/rest>
- Mempool self-hosted implementation:
  <https://github.com/mempool/mempool>
- Bitcoin header-time rules:
  <https://developer.bitcoin.org/reference/block_chain.html>

Production must use an owned, version-pinned Bitcoin Core node and prove
field-by-field parity.  The public Mempool response has no separately reviewed
bulk-data licence and remains a private research transport.

## Causal availability boundary

Historical header timestamps are miner-reported and are not receipt logs.
Calendar grouping by those timestamps would permit a later-mined, backdated
header to enter an apparently closed historical bucket.  The later
preregistration must therefore:

1. use complete 72-block packets aligned by absolute block height, never by
   header-time calendar membership;
2. require at least six hash-linked successors after the packet end;
3. set synthetic availability from the maximum header timestamp observed from
   packet start through the sixth successor, plus a fixed 48-hour embargo;
4. add one complete five-minute latency bar before next-open entry;
5. reserve events chronologically without score-priority replacement; and
6. keep the complete fixed hold inside its declared split.

Live collection must persist actual local first-seen time and fail closed on a
reorg or stale node.  It must use the later of the historical synthetic
availability and actual verified availability; a faster live clock is not
allowed to create backtest/live asymmetry.

## Contamination boundary

The exact source values were previously parsed by the terminal UFCP-1
source-support run, whose market outcomes remained sealed.  Consequently this
branch cannot claim that the ledger source is globally unseen.  The FETD
formula was selected from schema semantics and prior mechanism definitions,
not from a new raw-value scan or an outcome search.  From this decision onward:

- no FETD incidence may be opened before one exact singleton preregistration
  and implementation are committed;
- no market or funding value may be opened before source-only support passes
  and a strict evaluator is separately hash-frozen; and
- any eventual pass is candidate-level frozen evidence, never a pristine
  global holdout claim.

## Frozen research sequence

1. Commit this mechanism decision without parsing the source artifact.
2. Implement and test one exact FETD singleton preregistration, including
   source-only support floors, direction, hold, controls, and stopping rules.
3. Open FETD source incidence once.  Reject without repair on integrity,
   direction balance, count, or calendar-dispersion failure.
4. Only after support passes, implement and hash-freeze a strict evaluator.
5. Open 2021–2022 train first and stop on failure; only a train pass may open
   calendar-2023 selection.
6. Only a complete pre-2024 pass may authorize a separately frozen 2024–2026
   source extension and sequential recent-year validation.

Every opened performance report must include absolute return, full-calendar
CAGR including idle cash, global/pre-entry-HWM strict MDD, CAGR/strict-MDD,
trade count, both sides, stress cost, delayed entry, and clustered
significance.  The target remains CAGR/strict-MDD at least 3 with strict MDD no
greater than 15% and statistically meaningful trades.

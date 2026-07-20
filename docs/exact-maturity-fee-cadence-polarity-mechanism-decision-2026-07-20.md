# Exact-Maturity Fee-Cadence Polarity mechanism decision — 2026-07-20

## Decision

The next standalone BTC candidate is **EMFC-864 — Exact-Maturity
Fee-Cadence Polarity, 72-hour hold**.  It tests whether the fee component of
coinbase value reaching the exact consensus maturity boundary becomes useful
only when combined with unusually compressed or expanded 100-block cadence.

This decision freezes the observable and falsification boundary only.  It
opens no source incidence, BTC market value, funding value, future return,
PnL, CAGR, or drawdown.  Thresholds, strict-prior normalization, support
floors, and controls must be frozen in a separate preregistration before the
existing block source is read for candidate incidence.

For every canonical block at height `h`, the deterministic primitives are:

1. `matured_fee_component[h] = total_fees[h - 100]`;
2. `maturity_elapsed[h] = mediantime[h] - mediantime[h - 100]`; and
3. `confirmation_height[h] = h + 6`.

The primary clock is block-level at the exact maturity height `h`; it is not
assigned to an origin day or collapsed into a UTC-day aggregate.  The
provisional two-sided orientation is short when both matured fee pressure and
cadence compression are unusually high, and long when both are unusually low.
The hypothesis is that the conjunction describes a delayed settlement
pressure state that is not fully represented by either channel alone.  The
direction and 864 five-minute-bar hold may not be repaired after incidence or
returns are opened.

## What the source does and does not identify

Bitcoin Core defines `COINBASE_MATURITY = 100`, and consensus rejects a
coinbase spend while `spend_height - coin_height < 100`.  A coinbase output
created at height `h` can therefore first be included in a valid block at
height `h + 100`.  Mempool validation may admit that spend while the tip is at
`h + 99`, because it evaluates validity for the next block.

Official references, pinned for this decision to Bitcoin Core v31.1 consensus
source and the 30.0 RPC documentation:

- maturity constant:
  <https://github.com/bitcoin/bitcoin/blob/v31.1/src/consensus/consensus.h>;
- consensus input check:
  <https://github.com/bitcoin/bitcoin/blob/v31.1/src/consensus/tx_verify.cpp>;
- block and mempool validation heights:
  <https://github.com/bitcoin/bitcoin/blob/v31.1/src/validation.cpp>;
- `getblockstats` fields and pruning caveat:
  <https://bitcoincore.org/en/doc/30.0.0/rpc/blockchain/getblockstats/>; and
- Bitcoin Core v31.1 release record:
  <https://bitcoincore.org/en/releases/31.1/>.

The frozen research source contains `total_fees`, not the subsidy, exact
coinbase output value, miner or pool identity, payout policy, ownership, spend
path, exchange destination, or sale.  EMFC therefore measures only the
**fee aggregate associated with an origin block whose coinbase output reaches
the consensus maturity boundary**.  It is not evidence of synchronized miner
selling, realized liquidity release, or exchange inflow.  Those stronger
labels are prohibited.

`getblockstats` exposes `subsidy` and `totalfee` in satoshis, but its historical
calculation depends on retained block/undo data.  A pruned node is suitable for
forward collection only when each new block is processed before pruning and
the derived record is persisted locally; it is not an arbitrary historical
backfill source.  Production promotion must pin one Bitcoin Core version and
prove forward field parity.  No fixed whole-node disk ceiling is inferred
from the configured prune target.

## Existing frozen source

EMFC reuses, without alteration, the already frozen confirmed-ledger prefix:

- path: `data/bitcoin_utxo_fee_block_stats_2020_2023.csv.gz`;
- SHA-256:
  `8d5160b59bd104819adb34752f2bd5b01e9c07d7aa108188abcd9e6d323d102f`;
- heights: `610691..823785` inclusive;
- rows: `213095`;
- source manifest:
  `results/bitcoin_utxo_fee_block_stats_source_manifest_2026-07-20.json`;
- source-manifest file SHA-256:
  `ceb096b7bfeaf309729c4cab9f5cd4d2e0d526e261b1e460e9d03cd4f24e8084`;
- source-manifest canonical hash:
  `98a84b0bd0338300f62eaa047b87498cc5a8d9505a03f6bd1912d1deb9564e8c`.

The file contains a contiguous, hash-linked, pre-2024 canonical prefix and no
market, funding, return, or PnL field.  Its public Mempool transport remains a
private historical research convenience.  Production requires an owned
Bitcoin Core source and forward parity; no raw hosted response is
redistributed.

## Why this is not a miner-sale claim or a failed-alpha repair

- **UFCP** used fee burden and signed UTXO-set creation/contraction.  EMFC does
  not use inputs, outputs, or UTXO change.
- **BATE-288** used six-block transaction/weight throughput divided by elapsed
  header time.  EMFC uses the exact 100-block consensus-maturity lag and no
  transaction-count or block-weight channel.
- **BFRT-288** used fee-rate breadth and transport across transactions.  EMFC
  uses only the aggregate fee component tied to an exact origin height.
- **NWE** combined daily network weak signals in a fitted model.  EMFC is a
  fixed two-channel conjunctive clock with no fitted outcome label.

That distinction is mechanistic, not evidence of independence.  Preliminary
read-only diagnostics indicated that matured daily fees may be highly
correlated with same-day fees and that 100-block elapsed time may be strongly
related to daily block count.  Those diagnostics are not a frozen result.
They motivate fail-fast shadow controls rather than a profitability claim.

## Mandatory source-only falsification

The preregistration must bind all of these controls before opening incidence:

1. matured-fee-only and cadence-only onset clocks;
2. a completed-UTC-day aggregate with the conservative `D+2 00:05 UTC`
   schedule, retained only as a calendar/block-count shadow;
3. a same-height `total_fees[h]` shadow;
4. `h - 99` and `h - 101` pseudo-maturity variants;
5. a seven-day stale-feature clock;
6. an origin-day-shift clock that is evaluated only as a leakage sentinel and
   may never become the primary policy;
7. direction flip, constant-long, and constant-short controls;
8. deterministic random clocks matched by year, month, side, event count, and
   source-activity stratum; and
9. one additional five-minute execution-latency control.

Source-only support must reject the candidate without repair when the primary
clock is too sparse, too dense, materially one-sided, calendar-concentrated,
mechanically pinned to the 72-hour non-overlap boundary, dominated by one
source discontinuity, or indistinguishable from the daily, same-height,
fee-only, cadence-only, stale, or pseudo-maturity shadows under the frozen
novelty rules. Thresholds, lookback, side, hold, onset, and controls may not be
changed after incidence is visible.

## Availability and leakage boundary

Header timestamps are event fields, not archived node receipt times.  For a
candidate maturity height `h`, the primary historical availability is:

```text
raw_available = max(timestamp[h:h+6]) + 2 hours
decision_boundary = ceil(raw_available to a 5-minute UTC boundary)
entry_time = decision_boundary + 5 minutes
```

Thus every signal waits through six hash-linked successors, a conservative
two-hour header-time embargo, and one complete five-minute latency bar.  The
support calendar belongs to `entry_time`, never origin height `h-100`, its UTC
day, or header timestamp `h`.

All strict-prior statistics must use only valid exact-maturity heights strictly
below `h`.  The current block, confirmation blocks, future blocks, full-sample
statistics, outcome-conditioned thresholds, and post-entry source rows are
forbidden from feature normalization.  Live promotion additionally requires
actual local first-seen timestamps, canonical-chain/reorg handling,
six-confirmation parity, and at least 90 shadow days.

## Frozen research sequence

1. Commit this decision without reading candidate incidence.
2. Commit and hash-freeze one source-manifest-only preregistration defining
   exact block-level features, strict-prior statistics, thresholds, controls, support
   floors, novelty comparators, and stopping rules.
3. Run source-only incidence, shadow, dispersion, and novelty checks.  Load no
   market or funding value.
4. Reject without repair if any frozen source gate fails.
5. Only after an exact source pass, build and hash-freeze a strict evaluator
   before reading any post-entry OHLC or funding value.
6. Evaluate 2021-2022 train first, then 2023 selection only after an exact
   train pass.  Keep 2024-2026 sealed and sequential.

The branch has broad prior BTC research exposure.  This protocol can support
only a candidate-level frozen claim, not a globally pristine human holdout.

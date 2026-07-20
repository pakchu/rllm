# Miner clock-skew topology mechanism rejection — 2026-07-20

## Decision

Reject the proposed Bitcoin block-header `timestamp` versus `mediantime`
feature family **before opening the frozen block values, any candidate
incidence, or any market outcome**.  The nearest singleton would have ranked
`timestamp[h] - mediantime[h]` or a short-window dispersion of that quantity,
but it is not a sufficiently independent mechanism from the already tested
BATE and MCR families.

No MTP-slack threshold, side, hold, residual model, or evaluator will be
registered.  This is a mechanism-level rejection, not an unfavorable
performance result.

## Protocol meaning

Bitcoin's header `time` is a miner-reported Unix timestamp.  A valid header
must be strictly newer than the median timestamp of the previous eleven
blocks, while nodes also reject a header too far in the future relative to
their adjusted clock.  Bitcoin Core computes median time past by sorting the
current block index and up to ten predecessors and returning the middle value.
The RPC exposes both `time` and `mediantime`.

Official references:

- Bitcoin block-header timestamp rules:
  <https://developer.bitcoin.org/reference/block_chain.html>
- Bitcoin Core `CBlockIndex::GetMedianTimePast` and the two-hour future-time
  constant:
  <https://github.com/bitcoin/bitcoin/blob/master/src/chain.h>
- Bitcoin Core `getblockheader` fields:
  <https://bitcoincore.org/en/doc/24.0.0/rpc/blockchain/getblockheader/>
- Blockstream Esplora block schema:
  <https://github.com/Blockstream/esplora/blob/master/API.md>

The important consequence is that `mediantime` is a deterministic robust
filter of the same recent miner-reported header timestamps.  It is not a new
receipt-time, propagation, demand, ownership, fee, or order-flow observation.

## Why the mechanism is not independent

### Overlap with BATE

BATE-288 already uses
`timestamp[h] - timestamp[h-6]` as the supply-time denominator for block
weight and transaction throughput.  A level, change, spread, sign pattern, or
dispersion built from `timestamp - mediantime` remains a deterministic
short-window transform of the same header-time path.  Residualizing it against
BATE would not create new information; it would add a fitted representation
of the same primitive.

Relevant repository contracts:

- `docs/block-arrival-throughput-elasticity-mechanism-decision-2026-07-20.md`
- `docs/block-arrival-throughput-elasticity-bate288-support-preregistration-2026-07-20.md`
- `training/build_block_arrival_throughput_elasticity_support.py`

### Overlap with MCR

MCR-7 combines miner hash-rate recovery with block cadence.  MTP slack lacks
MCR's independent hash-rate channel and therefore reduces to an even narrower
miner/cadence observable rather than establishing another economic source.

Relevant repository contracts:

- `docs/miner-cadence-recovery-mcr7-preregistration-2026-07-17.md`
- `training/build_miner_cadence_recovery_support.py`

### Direction is underidentified

A large header-time/MTP gap can arise from slow recent production, miner clock
policy, timestamp non-monotonicity, or behavior near a consensus bound.  The
available fields cannot distinguish those explanations.  None fixes a robust
long or short response without using returns to select the story after the
fact.

Pool attribution would not repair the problem: the reviewed historical pool
labels are mutable tagged metadata rather than a point-in-time consensus
field.  Actual local first-seen timestamps would be a genuinely different
propagation observable, but the frozen historical source does not contain
them.

## Causal and outcome boundary

The existing block-summary artifact was identified by manifest and schema
only.  This decision did not parse or summarize its rows and loaded:

- zero block `timestamp` or `mediantime` values;
- zero candidate features or signal incidences;
- zero market or funding rows;
- zero return, PnL, CAGR, or drawdown values; and
- zero post-2023 source rows.

An independent read-only adversarial review reached the same rejection from
the preregistered definitions and implementations.  It opened no raw source or
market/performance artifact.

## Next admissible axis

The next candidate must add an economic observable, not another transform of
block cadence.  The selected design target is confirmed-ledger
**fee/topology disagreement** using transaction input/output graph density and
confirmed fee pressure.  It must be defined separately from UFCP's signed
daily UTXO polarity, BFRT's fee-percentile transport, and BATE's arrival
throughput, then preregistered before its complete feature incidence is opened.


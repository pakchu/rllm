# Bitcoin coinbase-payout topology source feasibility — 2026-07-21

## Decision

Proceed with a new source-only 2020–2023 panel of the **first coinbase output
script fingerprint** for every confirmed best-chain Bitcoin block.

This is an independent observable axis. Existing ledger research used block
cadence, weight, transaction count, fees, input/output counts, UTXO-set change,
and witness composition. It did not retain or group the script receiving the
first coinbase output. This work opens no BTC market row, funding value, future
return, label, PnL, absolute return, CAGR, strict MDD, candidate direction, or
post-2023 source event.

No alpha is frozen here. In particular, this decision does not claim that a
payout script is a miner, pool, owner, exchange, seller, or economic entity.
It only authorizes an immutable topology source and a later outcome-blind
mechanism review.

## Consensus object and transport field

The first transaction in a Bitcoin block is the coinbase transaction. Its
outputs assign the block subsidy and collected fees to output scripts. Bitcoin
Core `getblock` with verbosity 2 returns every transaction and output, so a
production collector can derive the first output's raw `scriptPubKey` directly
from an owned, version-pinned node.

The historical transport is Mempool's open-source `v1/blocks/:height`
endpoint. At source revision
`e9d6cf8c042f946be53e372bb36530cd7b7851a4`, its backend sets:

```text
coinbaseSignature = coinbaseTx.vout[0].scriptpubkey_asm
coinbaseAddresses = unique address-bearing coinbase outputs
```

The source builder will use only `coinbaseSignature` and the address count/set
as a diagnostic. It will not ingest `pool.id`, `pool.name`, `pool.slug`,
`minerNames`, or a current attribution label. Pool labels are heuristic and
revision-prone; they are not a permissible historical feature.

Bound references:

- Bitcoin Core 30 `getblock`:
  <https://bitcoincore.org/en/doc/30.0.0/rpc/blockchain/getblock/>
- Bitcoin transaction and coinbase format:
  <https://developer.bitcoin.org/reference/transactions.html#coinbase-input-the-input-of-the-first-transaction-in-a-block>
- Mempool REST API:
  <https://mempool.space/docs/api/rest>
- exact Mempool source implementation:
  <https://github.com/mempool/mempool/blob/e9d6cf8c042f946be53e372bb36530cd7b7851a4/backend/src/api/blocks.ts#L310-L314>

## Bounded source-only probe

A deterministic feasibility probe sampled 24 evenly spaced height anchors
from the already hash-bound `610691..823785` source interval. Each Mempool
request returned 15 blocks, yielding 360 unique best-chain blocks.

| Probe item | Result |
|---|---:|
| Sampled blocks | 360 |
| Missing coinbase script/address fields | 0 |
| Distinct first-output script fingerprints | 53 |
| Largest fingerprint share | 15.56% |
| Top-five fingerprint share | 46.94% |
| Distinct full address-set fingerprints | 53 |
| Distinct current pool slugs, diagnostic only | 25 |

The probe establishes enough source breadth to justify a complete builder. It
does not establish a signal, event incidence, temporal stability, source
support, novelty, or profitability. Current pool-slug counts are reported only
to prove that the retained script fingerprint is not being mistaken for an
issuer-provided entity label; no slug or identity may enter the panel.

For block 800,000, the first coinbase output returned by both Mempool and the
independently operated Blockstream Esplora endpoint had identical block
identity, raw output script, ASM, address, and value. The complete builder must
perform a larger deterministic cross-transport sample and fail on any
disagreement.

## Frozen source contract

The builder must:

1. bind the existing complete best-chain reference
   `data/bitcoin_utxo_fee_block_stats_2020_2023.csv.gz` and its SHA-256
   `8d5160b59bd104819adb34752f2bd5b01e9c07d7aa108188abcd9e6d323d102f`;
2. retrieve exactly the same inclusive height range from Mempool's 15-block
   endpoint and require every basic block field and parent link to match;
3. require a non-empty first-output `coinbaseSignature` and at least one
   address-bearing coinbase output;
4. parse only frozen standard script-ASM forms into raw script bytes and hash
   those bytes with SHA-256; an unsupported or malformed form fails closed;
5. retain height/hash/parent/time, script type, script fingerprint,
   address-count, and a sorted address-set fingerprint, but no clear address,
   pool label, ASCII coinbase tag, extranonce, BTC price, or outcome;
6. cross-check a deterministic minimum 64-block sample against Blockstream's
   `block/:hash/txs/0` response, including first-output raw script and the full
   address-bearing output set;
7. materialize causal availability only after six hash-linked successors and
   an additional 48-hour conservative publication allowance;
8. exclude terminal rows that lack all six successors instead of inventing
   confirmation data;
9. write deterministic gzip and a checksum-bound manifest; and
10. record zero market, funding, return, PnL, and post-2023 source access.

Historical API delivery proves neither an SLA nor live parity. Live promotion
requires an owned Bitcoin Core node, canonical-chain/reorg handling, locally
persisted first-seen times, and field parity against this frozen transform.

## Candidate boundary

Only after the source is built and audited may a separate commit freeze a
candidate based on payout-script concentration or transition. That freeze must
choose feature, direction, packet size, prior-only normalization, latency,
hold, source-support floors, controls, and novelty comparators before reading
real temporal event incidence. Pool labels and source identities may not be
used to explain or select a result after the fact.

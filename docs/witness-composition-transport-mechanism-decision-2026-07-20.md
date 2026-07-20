# Witness Composition Transport mechanism decision — 2026-07-20

## Decision

The next standalone BTC candidate is **WCTR-288 — Witness Composition
Transport, 24-hour hold**. It will use paired 12-hour averages of mined-block
serialized size and BIP 141 block weight to estimate whether block payload is
moving toward or away from witness data while blockspace remains meaningfully
utilized.

This file freezes the source axis, algebra, directional hypothesis, and hold
before the complete source response, derived feature values, event incidence,
BTC market rows, funding rows, returns, or PnL are opened. Exact strict-prior
windows, ranks, thresholds, support floors, and controls must be committed
before complete incidence is calculated.

## New observable axis

Mempool publishes paired histories at:

- <https://mempool.space/api/v1/mining/blocks/sizes-weights/4y>

Official implementation at pinned upstream commit
[`e9d6cf8c042f946be53e372bb36530cd7b7851a4`](https://github.com/mempool/mempool/tree/e9d6cf8c042f946be53e372bb36530cd7b7851a4):

- [route registration and response assembly](https://github.com/mempool/mempool/blob/e9d6cf8c042f946be53e372bb36530cd7b7851a4/backend/src/api/mining/mining-routes.ts)
- [size and weight aggregation queries](https://github.com/mempool/mempool/blob/e9d6cf8c042f946be53e372bb36530cd7b7851a4/backend/src/repositories/BlocksRepository.ts)
- [interval-to-12-hour bucket mapping](https://github.com/mempool/mempool/blob/e9d6cf8c042f946be53e372bb36530cd7b7851a4/backend/src/api/mining/mining.ts)
- [BIP 141 weight definition and consensus limit](https://github.com/bitcoin/bips/blob/master/bip-0141.mediawiki)

For non-stale blocks, the implementation separately returns:

```text
sizes:   {avgHeight, timestamp, avgSize}
weights: {avgHeight, timestamp, avgWeight}
```

Both queries group the same block table by `UNIX_TIMESTAMP(blockTimestamp) DIV
43200` and cast averages to integers. WCTR requires a one-to-one match on
bucket identity, average height, and average timestamp; it does not silently
align nearest observations.

A schema/coverage-only probe on 2026-07-20 inspected keys, counts, timestamps,
pairing, and gaps without printing size, weight, or any derived value. Both
arrays contained 2,923 unique paired buckets from 2022-07-20 11:17:27 UTC
through 2026-07-20 05:23:24 UTC. No timestamp gap exceeded 50,982 seconds, so
the public response supplies continuous approximately 12-hour coverage for the
full four-year interval, including the recent year.

## Why this is not BATE or a fee-rate repair

BATE-288 used individual block arrival intervals, block-weight throughput, and
transaction-count throughput. It explicitly treated witness-heavy payload as
a reason not to interpret weight alone. BFRT used the cross-percentile shape of
mined fee rates. Neither candidate used serialized block size or isolated the
payload composition implied by the difference between size and weight.

BIP 141 defines:

```text
weight       = 3 * stripped_size + total_size
total_size   = stripped_size + witness_bytes
witness_bytes = (4 * total_size - weight) / 3
```

The source reports integer-rounded averages over the same bucket. WCTR will
therefore use the bounded aggregate proxy:

```text
witness_share_t = (4 * avgSize_t - avgWeight_t) / (3 * avgSize_t)
fullness_t      = avgWeight_t / 4_000_000
```

The downloader must reject rows unless `avgSize > 0`, `avgWeight > 0`,
`avgWeight <= 4,000,000`, and the implied witness share lies in `[0, 1]` up to
an explicit tolerance for integer-rounded averages. It may not clip an invalid
row.

## Frozen mechanism and direction

The singleton feature family is fixed as:

```text
witness_transport_7d = witness_share_t - witness_share_t-14
witness_impulse_24h  = witness_share_t - witness_share_t-2
```

The later preregistration will require an unusually large absolute seven-day
transport, sign confirmation from the 24-hour impulse, and a strictly-prior
fullness floor. No price, return, volume, fee rate, transaction count, funding,
premium, open interest, liquidation, FX, or order-book field may influence the
source signal.

Direction is frozen before values:

- positive witness transport is tentatively **long**: a larger share of mined
  payload moved into witness data while blocks remained utilized, consistent
  with more weight-efficient settlement/batching capacity; and
- negative witness transport is tentatively **short**: payload moved back
  toward stripped bytes under utilization, consistent with less efficient
  blockspace absorption and greater settlement friction.

This is a falsifiable hypothesis, not a Bitcoin protocol claim. Witness data
may reflect ordinary batching, Lightning operations, multisignature spends,
Taproot, inscriptions, consolidation, or other activity. It does not identify
users, intent, exchange flow, or economic value.

The hold is fixed at 288 five-minute bars. There will be no sign, metric,
source interval, lag, or hold grid.

## Causal availability and edge buckets

The returned timestamp is the average miner-reported block timestamp inside a
12-hour SQL bucket, not a publication timestamp. The exact bucket start is
`floor(timestamp / 43200) * 43200`; its end is start plus 12 hours.

To avoid partial aggregation and historical first-seen assumptions:

1. always discard the first and last response buckets;
2. make a retained bucket unavailable until bucket end plus 48 hours;
3. round availability up to the next five-minute open; and
4. consume one additional complete five-minute latency bar before entry.

Live production must fetch only completed buckets, persist retrieval time and
raw response hash, reject a delayed or revised bucket, and compare newly
observed values with a forward vintage shadow. The historical endpoint is a
frozen current vintage, not an archive of every past revision.

## Frozen research sequence

1. Implement and test one source-only immutable downloader that archives exact
   response bytes, validates paired arrays and 12-hour continuity, drops both
   edges, and emits a deterministic normalized table.
2. Fetch and hash-freeze the exact four-year snapshot without calculating
   WCTR features or source incidence.
3. Commit one machine-readable singleton preregistration with strict-prior
   ranks, support floors, calendar splits, controls, and no-repair gates.
4. Build source-only support. Reject without opening BTC outcomes if coverage,
   both directions, event count, or calendar dispersion fails.
5. If support passes, commit and hash-freeze the strict evaluator before
   loading any market path. Open development train first, then sealed annual
   test/evaluation periods one by one, stopping on the first failed gate.

The eventual evaluator must use next-open execution, full-calendar CAGR,
global/pre-entry-high-water strict MDD over every held five-minute bar, exact
funding, entry/exit costs, virtual adverse exit cost, chronological
non-overlap, and split-contained holds. Every opened period must report
absolute return, CAGR, strict MDD, CAGR/strict-MDD, trades, directions, and
calendar clusters.

The target remains positive absolute return, CAGR/strict-MDD at least 3,
strict MDD no greater than 15%, stress-cost survival, statistically meaningful
trade support, both directions, and positive contained subperiods. This branch
has broad prior BTC exposure, so a pass can establish only a candidate-level
frozen sequence, not a pristine global holdout.

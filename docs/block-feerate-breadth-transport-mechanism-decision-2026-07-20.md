# Block Fee-Rate Breadth Transport mechanism decision — 2026-07-20

## Decision

The next standalone BTC candidate is **BFRT-288 — Block Fee-Rate Breadth
Transport, 24-hour hold**. It will use the cross-percentile distribution of fee
rates in already-mined Bitcoin blocks to distinguish a broad blockspace
repricing from a move confined to a small urgent tail.

This file freezes the source axis, tentative economic direction, causal
boundary, and staged calendar before any fee-rate value, derived feature,
signal incidence, BTC execution row, funding row, return, PnL, CAGR, or MDD is
printed or calculated. A bounded source probe parsed only response keys, row
count, timestamp/height boundaries, and cadence.

## New observable axis

The Mempool Open Source Project exposes a mined-block endpoint that returns
average fee-rate percentiles for the requested trailing interval:

- REST reference:
  <https://mempool.space/docs/api/rest>;
- frozen research endpoint:
  <https://mempool.space/api/v1/mining/blocks/fee-rates/3y>;
- upstream repository:
  <https://github.com/mempool/mempool>.

At upstream commit
[`e9d6cf8c042f946be53e372bb36530cd7b7851a4`](https://github.com/mempool/mempool/commit/e9d6cf8c042f946be53e372bb36530cd7b7851a4):

- the three-year route groups non-stale mined blocks into fixed 43,200-second
  Unix buckets;
- each row contains average height, average block timestamp, and averaged
  `fee_span` coordinates labelled `avgFee_0`, `avgFee_10`, `avgFee_25`,
  `avgFee_50`, `avgFee_75`, `avgFee_90`, and `avgFee_100`;
- the inner five coordinates correspond to the 10th, 25th, 50th, 75th, and
  90th fee-rate percentiles retained by the backend.

Pinned implementation references:

- [12-hour interval mapping](https://github.com/mempool/mempool/blob/e9d6cf8c042f946be53e372bb36530cd7b7851a4/backend/src/api/mining/mining.ts#L667-L680);
- [non-stale block aggregation and output fields](https://github.com/mempool/mempool/blob/e9d6cf8c042f946be53e372bb36530cd7b7851a4/backend/src/repositories/BlocksRepository.ts#L834-L860);
- [block fee-span construction](https://github.com/mempool/mempool/blob/e9d6cf8c042f946be53e372bb36530cd7b7851a4/backend/src/api/blocks.ts#L279-L299);
- [effective-fee percentile implementation](https://github.com/mempool/mempool/blob/e9d6cf8c042f946be53e372bb36530cd7b7851a4/backend/src/api/common.ts#L1040-L1100).

Bitcoin Core independently documents block fee-rate statistics at the 10th,
25th, 50th, 75th, and 90th **weight-unit** percentiles via `getblockstats`:
<https://bitcoincore.org/en/doc/30.0.0/rpc/blockchain/getblockstats/>. That RPC
is a semantic comparator, not an assumed byte-for-byte replacement: the pinned
Mempool backend may use effective transaction fee rates and its own aggregation
logic when full transaction data are available.

The primary BFRT feature family is restricted to the five inner percentiles.
The provider-labelled 0 and 100 coordinates are retained only for ordering and
source-quality checks until a later preregistration explicitly says otherwise.
No mempool projection, recommended fee, pool tag, block-audit score, price, or
unconfirmed transaction enters this source.

## Mechanism

For a closed 12-hour bucket, transform the five inner fee-rate coordinates with
`log1p`. A later singleton policy will combine only strictly prior changes in:

1. **location transport** — common movement of the central fee-rate surface;
2. **transport coherence** — whether the five percentile changes share one
   sign rather than cancelling; and
3. **tail divergence** — whether the upper and lower percentile gaps widen
   asymmetrically instead of the whole distribution moving together.

The tentative direction is frozen:

- coherent positive transport with limited tail divergence is **long**: many
  parts of the mined fee distribution repriced upward together, consistent
  with broad confirmed blockspace demand;
- coherent negative transport with limited tail divergence is **short**: the
  distribution repriced downward together, consistent with broad demand
  withdrawal;
- a tail-only move is not directional evidence and must be filtered or exposed
  as a control, not assigned a side after outcomes are known.

This interpretation is falsifiable. Fee rates can be driven by batching,
ordinals, consolidations, fee-estimation behavior, CPFP packages, or miner
selection policy without forecasting BTC price.

## Why this is not a repair of a failed family

- BFC-3 used **daily total fees relative to issuance** plus transactions per
  block; it never used the within-block fee-rate distribution.
- NWE-8 combined daily aggregate network levels in a weekly supervised ridge;
  it did not model cross-percentile transport.
- BATE-288 used block-time, weight, and transaction throughput and loaded no
  fee values.
- UFCP-1 used total fee burden and signed input/output topology; it was rejected
  on source support before outcomes and cannot be threshold-repaired.
- Premium, funding, OI, Cboe, Coinbase, order-book, liquidation, and Trollbox
  families do not enter BFRT's clock.

BFRT is therefore a new confirmed-ledger distribution observable, not a sign
flip or threshold relaxation of `BFC-3`, `NWE-8`, `BATE-288`, or `UFCP-1`.

## Bounded source probe and coverage

On 2026-07-20, a schema/coverage-only request returned:

- 2,193 rows and 320,868 raw JSON bytes;
- first average timestamp `2023-07-20T10:20:58Z` at average height 799,514;
- last average timestamp `2026-07-20T04:34:15Z` at average height 958,823;
- strictly increasing unique timestamps and average heights; and
- the nine expected fields: `avgHeight`, `timestamp`, and seven fee-rate
  coordinates.

No fee-rate coordinate was printed, summarized, ranked, or joined to a market
outcome during this probe. The rolling endpoint currently spans exactly three
years, including the most recent year. Its lower and upper edge buckets can be
partial and will be dropped.

## Frozen causal and vintage boundary

The source builder must:

1. archive the exact raw response bytes, HTTP retrieval time, response headers,
   upstream commit, and SHA-256 before normalization;
2. derive the bucket ID as `floor(timestamp / 43,200)` and reject duplicates,
   unordered rows, non-finite/negative coordinates, or violated percentile
   ordering;
3. discard the first and last response buckets as potentially partial;
4. treat a retained bucket as unavailable until **48 hours after its fixed
   bucket end**, then add one complete five-minute execution-latency bar;
5. never forward-fill a missing bucket or use a later retrieval to backdate a
   live signal; and
6. persist no BTC price, funding, premium, OI, return, PnL, or market outcome in
   the source artifact.

The public response is a current rolling aggregate, not a historical vintage
archive. A production candidate must forward-store every retrieved closed
bucket and prove at least 90 shadow days of schema/value stability. Live
promotion requires either the pinned public route with fail-closed freshness
monitoring or a self-hosted pinned Mempool backend; Bitcoin Core percentile
parity cannot be assumed without an explicit comparison.

The Mempool code is AGPL-3.0. The reviewed official material does not provide a
separate bulk-data licence for the public response, so this snapshot remains a
private research transport until operational/legal review.

## Frozen staged calendar

- warm-up only: source start through `2023-10-31T23:59:59Z`;
- train: `2023-11-01T00:00:00Z` through `2024-12-31T23:59:59Z`;
- test: calendar 2025;
- eval: `2026-01-01T00:00:00Z` through the frozen source end.

Train is the only window that may fit any learned coefficient or threshold.
The later preregistration should prefer one fixed, label-free percentile rule;
if a learned decoder is used, its family, regularization, warm-up, refit clock,
and abstention rule must be frozen before any BTC outcome is opened. Test and
eval are report-only and may not select, rerank, invert, or repair the policy.

## Frozen research sequence

1. Commit this decision before persisting the full source response.
2. Implement and test a source-only downloader; archive and hash-freeze one
   exact response without deriving feature incidence.
3. Commit exactly one BFRT policy, support gate, controls, and stopping rule
   before opening full feature incidence.
4. Reject without outcomes if direction balance, calendar dispersion, or event
   support fails.
5. Only after support passes, commit and hash-freeze a strict evaluator before
   loading any BTC OHLC or funding value.
6. Open train first, then test, then eval, stopping permanently at the first
   failed gate.

Every performance report must include absolute return, full-calendar CAGR,
global/pre-entry-HWM strict MDD, CAGR/strict-MDD, trade count, both sides,
stress cost, delayed entry, and clustered significance. The project target
remains CAGR/strict-MDD at least 3 with strict MDD no greater than 15% and
statistically meaningful trades. Because the branch has broad prior BTC
exposure, any pass is candidate-level frozen evidence, never a pristine global
holdout.

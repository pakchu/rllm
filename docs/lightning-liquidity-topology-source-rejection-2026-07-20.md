# Lightning liquidity topology source rejection — 2026-07-20

## Decision

Reject the public mempool.space Lightning statistics history as the source for
the next BTC alpha candidate. Do not preregister a Lightning signal, persist a
source snapshot, inspect source values, calculate event incidence, or open any
market outcome from this feed.

The economic axis remains distinct and potentially useful, but this public
history cannot support the repository's causal, recent, and live-reproducible
research contract. A future Lightning candidate requires a different provider
or an independently collected forward archive.

## Proposed axis

The source was considered for a **Lightning Liquidity Topology Elasticity**
mechanism: divergence among public channel count, aggregate public channel
capacity, and node reachability composition. This is distinct from existing
repository work:

- NTB-7 uses on-chain active addresses, transactions, and transfers;
- ARCR uses the stock of funded on-chain addresses and active-address flow;
- miner-security candidates use hash rate, issuance, fees, and blocks; and
- premium, funding, open-interest, FX, and price-action families do not observe
  the Lightning graph.

Repository search found no existing tracked use of Lightning statistics.

## Official implementation evidence

At pinned upstream commit
[`e9d6cf8c042f946be53e372bb36530cd7b7851a4`](https://github.com/mempool/mempool/tree/e9d6cf8c042f946be53e372bb36530cd7b7851a4),
the official mempool implementation:

- registers `GET /api/v1/lightning/statistics/:interval` and the `latest`
  route in
  [`general.routes.ts`](https://github.com/mempool/mempool/blob/e9d6cf8c042f946be53e372bb36530cd7b7851a4/backend/src/api/explorer/general.routes.ts);
- accepts `3y` and `4y` intervals in
  [`common.ts`](https://github.com/mempool/mempool/blob/e9d6cf8c042f946be53e372bb36530cd7b7851a4/backend/src/api/common.ts);
- returns `added`, `channel_count`, `total_capacity`, `tor_nodes`,
  `clearnet_nodes`, `unannounced_nodes`, and `clearnet_tor_nodes`, ordered by
  `added DESC`, in
  [`statistics.api.ts`](https://github.com/mempool/mempool/blob/e9d6cf8c042f946be53e372bb36530cd7b7851a4/backend/src/api/explorer/statistics.api.ts); and
- repeatedly upserts a graph snapshot under a UTC-midnight key in
  [`stats-updater.service.ts`](https://github.com/mempool/mempool/blob/e9d6cf8c042f946be53e372bb36530cd7b7851a4/backend/src/tasks/lightning/stats-updater.service.ts)
  and
  [`stats-importer.ts`](https://github.com/mempool/mempool/blob/e9d6cf8c042f946be53e372bb36530cd7b7851a4/backend/src/tasks/lightning/sync-tasks/stats-importer.ts).

That implementation also shows why the timestamp is not a publication-time
vintage. It is a day key, while the current day's record is overwritten from
successive graph observations. Historical topology files may separately seed
missing rows. The public response does not expose retrieval time, first
publication time, revision time, or a revision identifier.

## Bounded schema and coverage probe

On 2026-07-20 UTC, a bounded probe called the official public endpoints:

- <https://mempool.space/api/v1/lightning/statistics/3y>
- <https://mempool.space/api/v1/lightning/statistics/4y>
- <https://mempool.space/api/v1/lightning/statistics/latest>

The probe inspected only HTTP metadata, object keys, row counts, timestamps,
ordering, uniqueness, and timestamp gaps. It did not print or persist channel,
capacity, node, feature, signal, market, funding, return, or PnL values.

Observed source metadata:

| Check | Result |
|---|---:|
| 3-year rows | 535 |
| 3-year timestamp range | 2023-07-23 through 2026-05-21 UTC |
| 4-year rows | 871 |
| 4-year timestamp range | 2022-08-29 through 2026-05-22 UTC |
| latest endpoint timestamp | 2026-05-22 UTC |
| age of latest row at probe date | 59 calendar days |
| expected daily rows within the observed 4-year bounds | 1,363 |
| missing rows within those bounds | 492 (36.10%) |

The 4-year timestamp sequence contained five non-daily jumps:

| Previous row | Next row | Calendar separation |
|---|---|---:|
| 2023-01-27 | 2023-02-03 | 7 days |
| 2023-07-15 | 2023-07-23 | 8 days |
| 2023-07-23 | 2023-07-27 | 4 days |
| 2023-08-25 | 2024-12-14 | 477 days |
| 2025-05-27 | 2025-06-17 | 21 days |

Separate 3-year and 4-year requests made seconds apart also disagreed on the
newest timestamp by one day. This is additional evidence that a path label is
not an immutable vintage contract.

## Failed source gates

The source fails before any alpha rule is defined:

1. **Recent/live parity:** the latest row is 59 days stale, so a signal cannot
   be reproduced from current live observations.
2. **Coverage:** a 477-day hole removes most of the intended 2024 development
   period, and smaller gaps remain elsewhere.
3. **Point-in-time provenance:** the API exposes only the observation day, not
   when each historical value first became knowable or whether it was revised.
4. **Immutable reproducibility:** current-day upserts and historical imports
   can alter the current public vintage without a row-level revision clock.

Forward-filling would turn an outage into a fabricated constant state and
would not restore either point-in-time provenance or live parity. Lowering the
coverage requirement after seeing the gaps would be a source-gate repair.
Neither is allowed.

## Stop condition

This Mempool Lightning source is permanently rejected for historical alpha
selection in this branch. No performance statistic exists for the proposed
mechanism. Work proceeds to a different source axis; Lightning may return only
after a provider demonstrates continuous recent coverage and a defensible
publication/revision clock, or after this repository accumulates its own
forward snapshots long enough for a prospective test.

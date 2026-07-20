# Mempool block-template audit source feasibility rejection — 2026-07-20

## Decision

Reject Mempool's historical block-template match rate as the next BTC alpha
source. The observable is genuinely new in this repository, but the public
history does not contain a continuous three-year interval. No policy,
threshold, side, hold, source snapshot, market outcome, return, funding, or PnL
was opened.

## Proposed observable

Mempool audits compare transactions in a mined block with a template assembled
from the observer's public mempool. A falling `match_rate` can therefore proxy
for private order flow, acceleration, propagation disagreement, or transaction
selection that the public template did not explain. Repository search found no
prior alpha using this endpoint or metric.

Official implementation at pinned upstream commit
[`e9d6cf8c042f946be53e372bb36530cd7b7851a4`](https://github.com/mempool/mempool/tree/e9d6cf8c042f946be53e372bb36530cd7b7851a4):

- [route registration and response](https://github.com/mempool/mempool/blob/e9d6cf8c042f946be53e372bb36530cd7b7851a4/backend/src/api/mining/mining-routes.ts)
- [historical audit query](https://github.com/mempool/mempool/blob/e9d6cf8c042f946be53e372bb36530cd7b7851a4/backend/src/repositories/BlocksAuditsRepository.ts)
- [public 4-year endpoint](https://mempool.space/api/v1/mining/blocks/predictions/4y)

The implementation returns `[time, height, match_rate]` and downsamples a
three- or four-year request into 12-hour buckets. The metric is observer
specific: mismatch does not by itself prove censorship, miner intent, or a
complete private-order-flow quantity.

## Bounded coverage probe

On 2026-07-20, a schema/clock-only probe inspected response size, tuple shape,
timestamps, heights, ordering, uniqueness, and time gaps. It did not print or
persist any `match_rate` value.

| Check | Result |
|---|---:|
| 4-year response rows | 1,679 |
| first timestamp | 2023-06-23 10:34:31 UTC |
| next timestamp | 2024-08-08 00:00:07 UTC |
| first gap | 411.56 days |
| rows in 2023 | 1 |
| rows in 2024 | 547 |
| rows in 2025 | 730 |
| rows in 2026 YTD | 401 |
| continuous usable history | approximately 2024-08-08 onward |

The separate 3-year endpoint began only on 2024-04-02 and contained another
large discontinuity before the same recent block of observations. The public
`X-total-count` describes all stored audits, not continuous historical response
coverage.

## Stop condition

The source fails the user's three-year validation requirement before feature
design. Filling the 411-day hole, lowering the required history, or combining
unmatched vintages would be a post-probe source repair. This axis may return
after the continuous archive itself exceeds three years, no earlier than
2027-08, and only with a forward vintage-parity check.

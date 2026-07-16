# BFC-3 support rejection

Decision: **rejected before any post-entry return was loaded**.

BFC-3 was preregistered as a singleton in commit `104f8fd`. Its support clock
loads only Coin Metrics `FeeTotNtv`, `IssTotNtv`, `BlkCnt`, `TxCnt`, and
availability timestamps. It reports zero market/funding rows loaded and keeps
2024, 2025, and 2026 YTD sealed.

## Outcome-blind clock

| Window | Non-overlapping events | Frozen minimum | Result |
|---|---:|---:|---|
| 2021-2022 train | 27 | 35 | fail |
| 2021 train | 3 | 14 | fail |
| 2022 train | 24 | 14 | pass |
| 2023 selection | 15 | 14 | pass |
| 2023 H1 | 13 | 5 | pass |
| 2023 H2 | 2 | 5 | fail |

The largest month contains 11.90% of events, below the frozen 20% maximum.
However, the clock is structurally concentrated in 2022 and early 2023. It
contains only three fresh 2021 events and two 2023-H2 events, so it cannot
support a stable multi-regime inference.

## Integrity anchors

- Preregistration manifest hash:
  `7f99268452456e83a59eb10b546ea4c4f084f6f598079298ab65e46d0a56c12b`
- Support result hash:
  `1719c78bcf5054d2acd3cfea1734bbd95bee280695486fa5fbb2af9d5cc13fb5`
- Support JSON SHA-256:
  `0d00b40b3a513c66e595b44b5e9efdbbbfed4661d722e4910c30ee8f8e18813b`
- Clock CSV SHA-256:
  `edda7bb8ae8a1de4e51a3b86e98d533748e73d203125a3ded1a487e9a0e93632`
- Clock frame hash:
  `1bf7602c5e63f437337f31782b8fa4ddd3a2c525c7a77c7a8e9e857be288d7d6`

## Decision boundary

No absolute return, CAGR, strict MDD, direction flip, or control PnL was
calculated. Lowering the z-score/composite thresholds or changing the hold
after observing support would violate the frozen singleton contract.

The raw fee-share and transaction-density features remain admissible weak
features for a future properly nested model. This exact sparse event policy is
not eligible for alpha or portfolio promotion.

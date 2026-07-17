# CIHM-1 evaluator freeze — 2026-07-18

## Status

**SEALED; no outcome was simulated.**

The evaluator opened the preregistration, source-only support manifest, and
frozen clock ledger.  During freeze it parsed **0 BTC OHLC rows**, **0 funding
rows**, and ran **0 simulations**.

## Immutable evaluation contract

- Engine: strict 5-minute Binance UM futures evaluator already used by the
  repository's causal external-source audits.
- Leverage: 0.5x.
- Base cost: 6 bp/notional/side.
- Stress cost: 10 bp/notional/side.
- Funding: exact `entry <= mark < exit` interval.
- CAGR: entire wall-clock split, including idle cash.
- Strict MDD: global/pre-entry HWM plus held-path OHLC, funding, entry, exit,
  and hypothetical-liquidation costs.
- Statistical test: two-sided weekly-cluster sign flip, 20,000 deterministic
  draws with seed 20,260,717 when exact enumeration is too large.

Stage 1 physically parses `[2021-01-01, 2023-01-01)`.  The parser stops before
the 2023 boundary.  Stage 2 cannot reach its loader unless the stored Stage-1
artifact is a passing result, has the same evaluator identity, and replays
byte-for-byte under the unchanged evaluator.

The physically contained primary schedule has 151 Stage-1 trades.  One of the
152 source-supported clocks enters during 2022 but exits during 2023, so it is
correctly excluded from the bounded Stage-1 simulation.  The minimum remains
150.  Sealed 2023 contains 65 complete primary trades; its minimum is 60.

## Frozen controls

- institutional-gap only;
- VIX-call-pressure only;
- index-share only;
- level composite;
- direction flip;
- one-release delay;
- seven-release placebo.

No control can replace the primary.  No parameter remains mutable.

## Identities

- Evaluator source SHA-256:
  `b02b68acf1f2a57e9a55a57e76380e3984c68d49f1b872de7e3608058235e9e5`
- Freeze manifest hash:
  `adeebd3c552789dec754e1fdd0f6e697c9fe1d0f0e83265e360422f3b7197112`
- Freeze JSON SHA-256:
  `3e20f73e6023bdbd5174ce86e117e0545b6316df0a39223c446afaf41d5cd6c1`

The evaluator source must not change after this point.  Any new idea requires a
new candidate ID and a new preregistration, not a repair of CIHM-1.

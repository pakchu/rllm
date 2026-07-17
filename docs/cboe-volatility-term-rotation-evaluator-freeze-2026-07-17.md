# CVTR-1 strict evaluator freeze — 2026-07-17

The strict evaluator is frozen before opening any BTC outcome.

## Immutable execution contract

- Stage 1 physically parses only `[2021-01-01, 2023-01-01)`.
- Calendar 2023 cannot be loaded unless the stored Stage-1 report passes and
  exactly replays under the same evaluator SHA-256.
- 2024+ remains sealed after Stage 2.
- Exposure is 0.5x; costs are 6 bp/notional/side and 10 bp stress.
- Binance funding is applied exactly for `entry <= timestamp < exit`.
- CAGR uses the full calendar window, including idle cash.
- Strict MDD uses the global pre-entry high-water mark and intratrade
  favorable-before-adverse OHLC with hypothetical liquidation cost.
- Weekly-cluster significance is deterministic, two-sided sign-flip inference.

## Frozen schedule support

| Clock | Stage-1 trades | Long | Short | sealed-2023 trades |
|---|---:|---:|---:|---:|
| primary | 281 | 127 | 154 | 101 |
| front slope | 285 | 117 | 168 | 120 |
| broad slope | 290 | 139 | 151 | 152 |
| VIX level | 301 | 172 | 129 | 199 |
| direction flip | 281 | 154 | 127 | 101 |
| one-release delay | 281 | 127 | 154 | 100 |
| deterministic random side | 281 | 135 | 146 | 101 |
| constant long | 281 | 281 | 0 | 101 |

Freeze-time execution OHLC rows parsed: `0`. Funding rows parsed: `0`.
Simulation executed: `false`. Mutable parameters: none.

## Integrity anchors

- evaluator SHA-256:
  `1bb47f6d704c2f977e44e378bf57acf4d4f6ab6455346e7b720149132f2f1f0e`
- evaluator-freeze manifest hash:
  `b27ff7b86817be1a2fb24497b194630fae25239d7636b5c406f4d1e1ceaa69f3`
- evaluator-freeze JSON SHA-256:
  `d9979b82fcf88b54ca51bd67dd0848dd700a8a153bccb7052c44ce13d785d3fe`
- strict engine SHA-256:
  `e309f5217f033d57d2eadfec936843e736ce287f5c47f957c0ac6f0c71879c23`

Any evaluator edit invalidates the freeze and blocks Stage 1.

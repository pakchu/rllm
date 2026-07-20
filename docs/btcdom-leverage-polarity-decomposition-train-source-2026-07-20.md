# DLPD-12 2022 train execution source — 2026-07-20

The evaluator was already immutable before this source was opened. No strategy
return, trade PnL, CAGR, or drawdown was calculated while preparing it.

## Exact physical window

- Window: `[2022-01-01T00:00:00Z, 2023-01-01T00:00:00Z)`
- BTCUSDT 5-minute OHLC: 105,120 rows, exact continuous grid
- BTCUSDT funding settlements: 1,095 rows, exact 8-hour grid
- Maximum funding timestamp offset: 31 ms
- Exit-boundary row required: no
- Post-stage numeric rows parsed: 0

The parent `.gz` files were authenticated only by comparing their compressed
bytes with precommitted SHA-256 values. The stage files were then deterministically
reconstructed and are rechecked against those frozen parents before evaluation.
2023 strategy outcomes remain unopened.

## Frozen identities

- source manifest hash:
  `d688bc802b60c6e3667ab4fcb2e92540c3c7cf064b79f44bfa2a666824811b0a`
- source JSON SHA-256:
  `a47beeb9822c4319101a378e305bf0c28770dc191338467e0ff2ad309dfbd209`
- market slice SHA-256:
  `7e1aab436a96c83680be45047f4dd36a62fa72cb9b8f3991d32e16d4fe1a4be3`
- funding slice SHA-256:
  `2c3eb607ae343201ee2d12b29fd07fb25dad4baa059340eef821155a6c3ea2c8`
- evaluator freeze manifest:
  `c55b8b23f1ec45821ca670fab6f5811826b37d839c01d1e44d2a0ca1b30a31fd`

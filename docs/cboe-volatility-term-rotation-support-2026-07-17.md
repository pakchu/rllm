# CVTR-1 source-only support freeze — 2026-07-17

The support builder replayed the frozen Cboe panel and preregistration without
opening BTC bars, returns, funding, labels, or existing-alpha PnL.

## Primary clock support

| Window | Events | Long | Short | Max month share |
|---|---:|---:|---:|---:|
| 2021 | 148 | 109 | 39 | 12.16% |
| 2022 | 133 | 18 | 115 | 14.29% |
| Stage 1 | 281 | 127 | 154 | 6.76% |
| 2023 H1 | 55 | 50 | 5 | 27.27% |
| 2023 H2 | 46 | 28 | 18 | 23.91% |
| sealed 2023 | 101 | 78 | 23 | 14.85% |

The preregistered concentration gate applies to the full Stage-1 and full-2023
windows, not to individual half-years. Every source-only support gate passes.

## Causal controls

The frozen ledger contains the primary clock plus:

- front slope only,
- broad slope only,
- VIX level only, and
- an exact one-Cboe-release delay.

Direction flip, deterministic random side, and constant-long controls are
derived from the immutable primary entry/exit clock inside the evaluator.

All clocks use strict-prior ranks, next-source-session entry, no missing-date
forward fill, and globally non-overlapping holds. Stage 1 is now eligible to be
opened; calendar 2023 BTC outcomes remain sealed until Stage 1 passes.

Integrity anchors:

- support manifest hash:
  `9ab934110ce15059f7c584f1a993a631f26d81fc6621b28b74a4a7c762d23958`
- support JSON SHA-256:
  `ce89b3fb831f0283d9e19537d50086329b28b0f7bfe474baa074f4bcd23531d8`
- five-clock ledger SHA-256:
  `47f4ca447daa2b03a0827ad243ed1107eb34a37e5d7bab18ecd3c4331736959d`

# DTAC-8 source-only preregistration — 2026-07-19

## Status

DTAC-8 passed source-only support, direction-balance, future-incidence
feasibility, and novelty gates. **No BTC execution price, funding, return,
excursion, PnL, equity, CAGR, or MDD was opened.** This permits an evaluator
freeze; it is not profitability evidence.

## Frozen mechanism

- Sources: completed-hour normalized taker flow and premium-index open/close for
  ADA, BNB, DOGE, ETH, SOL, and XRP USD-M perpetuals.
- Tail calibration: positive and negative magnitudes use separate strictly-prior
  2160-hour rolling quantiles, each requiring 360 same-sign observations.
- Long vote: negative-flow tail plus positive premium-impulse tail on the same
  symbol. Short vote is the exact sign mirror.
- Consensus: at least 2 matching votes and at most
  one opposite vote.
- Selected tails: flow q80, premium
  q60.
- Clock: directional-state onset/polarity change, entry +5m, fixed 8h hold,
  one position.

The selected cell maximized consensus count, then flow-tail strength, then
premium-tail strength among cells passing 2023 source-incidence gates. A naïve
shared unsigned threshold prototype was rejected source-only for direction bias;
it opened no BTC outcome and is not an alternate policy.

## Source-only incidence

| Stage | Events | Long | Short | Max month share |
|---|---:|---:|---:|---:|
| train 2023 | 143 | 84 | 59 | 0.175 |
| test 2024 | 190 | 120 | 70 | 0.142 |
| eval 2025 | 247 | 148 | 99 | 0.113 |
| final 2026H1 | 115 | 54 | 61 | 0.270 |

## Clock novelty

- FCIR-12: exact Jaccard 0.0065, ±2h max near-share 0.1336
- SQFD-6: exact Jaccard 0.0071, ±2h max near-share 0.1107
- OPDR-24: exact Jaccard 0.0054, ±2h max near-share 0.1200
- PCBR-12: exact Jaccard 0.0011, ±2h max near-share 0.0813
- PSR-30/6: exact Jaccard 0.0007, ±2h max near-share 0.0787
- TGR-12: exact Jaccard 0.0086, ±2h max near-share 0.1446

## Sequential outcome rule

The strict evaluator and every source-only control must be committed before
2023 BTC execution outcomes are opened. Train must pass all frozen economic,
significance, half-stability, stress, and mechanism-margin gates. Failure
retires the exact policy and keeps 2024+ sealed; no threshold, side, hold, or
control repair is allowed.

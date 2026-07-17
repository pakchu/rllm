# CVTR-1 rejected at sealed Stage 1 — 2026-07-17

Decision: **reject without repair; calendar-2023 BTC outcomes remain sealed**.

## Primary result

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Mean gross move | p(two-sided) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage 1 (2021-2022) | -11.44% | -5.90% | 39.51% | -0.15 | 281 | +13.18 bp | 0.9156 |
| 2021 | -20.55% | -20.56% | 32.58% | -0.63 | 147 | -3.88 bp | 0.5732 |
| 2022 | +12.68% | +12.69% | 22.09% | 0.57 | 133 | +33.62 bp | 0.5104 |
| 10 bp stress | -20.86% | -11.05% | 42.42% | -0.26 | 281 | +13.18 bp | 0.6961 |

The candidate passed trade-count and side-support gates only. It failed return,
CAGR/MDD, MDD, significance, gross-edge, stress-cost, subperiod-stability, and
mechanism-margin gates.

## Controls

| Clock | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| front slope only | -5.97% | -3.03% | 34.28% | -0.09 | 285 |
| broad slope only | -7.07% | -3.60% | 34.74% | -0.10 | 290 |
| VIX level only | -24.21% | -12.95% | 38.37% | -0.34 | 301 |
| direction flip | -31.13% | -17.02% | 44.76% | -0.38 | 281 |
| one-release delay | -17.12% | -8.97% | 42.16% | -0.21 | 281 |
| deterministic random side | -32.47% | -17.83% | 42.89% | -0.42 | 281 |
| constant long | -30.76% | -16.80% | 49.73% | -0.34 | 281 |

## Root cause

The Cboe term surface identified broad 2022 risk pressure better than 2021, but
did not produce a stable next-session BTC edge. Mean signed underlying movement
was only `13.18 bp` across 281 trades—too close to the 12 bp/notional round-trip
cost before funding and compounding. The primary combination was also worse than
either slope component, so the joint-surface mechanism was not supported.

This is not repaired by inversion: the exact direction flip loses more. It is
also not generic BTC long beta: constant-long on the same clock loses. The
failure is regime dependence plus insufficient per-trade edge, not a single
cost or direction bug.

No threshold, rank window, side, clock, hold, leverage, or control is changed.
No 2023 OHLC or funding row was parsed.

## Integrity

- physically parsed market rows: `210,240`, ending `2022-12-31 23:55 UTC`
- parser stopped before the first 2023 row: `true`
- physically parsed funding rows: `2,190`
- evaluator SHA-256:
  `1bb47f6d704c2f977e44e378bf57acf4d4f6ab6455346e7b720149132f2f1f0e`
- Stage-1 manifest hash:
  `9f5a5f42d4686c04566b2a1916bfe7959b3e0359e6bd9db3b464ae70a0cfd120`
- Stage-1 JSON SHA-256:
  `7afe18e3d50d1bd06e2a93cd5838c1a979f5654d717da9e3a07fb13cc4ae6ba3`

The Stage-2 command fails closed with `CVTR-1 Stage1 failed; 2023 remains
sealed` before calling the execution loader.

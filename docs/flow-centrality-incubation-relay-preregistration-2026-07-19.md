# FCIR-12 source-only preregistration — 2026-07-19

## Status

`FCIR-12` passed source-only support and novelty gates. **No BTC price,
funding, return, excursion, PnL, equity, CAGR, or MDD was opened.** This is
permission to freeze an evaluator, not evidence of profitability.

## Frozen mechanism

- Source: normalized completed-hour taker flow for six USD-M alts.
- Directed edge: lag-one correlation advantage, estimated from the prior
  720 target hours with at least 672 observations; newest target is `t-1`.
- Central flow: current flow weighted by the strictly-prior outgoing net-lead
  network.
- Crowd gate: absolute equal-weight current flow is at or below its strictly
  prior median from a rolling 90-day window, activated after at least 720
  prior valid observations.
- Strength gate: absolute central flow is at or above its strictly-prior
  q75 from the same rolling
  90-day/minimum-720-prior-observation contract.
- Network breadth: effective names at least
  `3.0`.
- Side: sign of central flow.
- Clock: false-to-true onset, entry `+5m`, fixed `12h` hold, one position.
- Funding: interior exact-time events are symmetric; exact entry/exit credits
  are dropped while debits are retained, and every settlement mark is visited.

The selected cell was chosen only from 2023 source incidence by maximizing
mechanism strength among support-passing cells. Later source incidence was
opened only after selection and did not alter the cell.

## Source-only incidence

| Stage | Events | Long | Short | Max month share |
|---|---:|---:|---:|---:|
| train 2023 | 62 | 26 | 36 | 0.194 |
| test 2024 | 90 | 46 | 44 | 0.178 |
| eval 2025 | 61 | 32 | 29 | 0.230 |
| final 2026H1 | 34 | 16 | 18 | 0.353 |

## Clock novelty

- CLD-72: exact Jaccard `0.0063`, ±6h max near-share `0.0667`
- SQFD-6: exact Jaccard `0.0138`, ±6h max near-share `0.3333`
- OPDR-24: exact Jaccard `0.0060`, ±6h max near-share `0.1301`
- PCBR-12: exact Jaccard `0.0000`, ±6h max near-share `0.1659`
- PSR-30/6: exact Jaccard `0.0000`, ±6h max near-share `0.1498`

## Why the earlier strict-disagreement variant was dropped

Requiring at least four of six raw flow signs to oppose central flow produced
only 14 non-overlapping 2023 events. It was retired before outcomes. FCIR-12
tests a different and more coherent incubation claim: influential flow is
strong while aggregate crowd flow is still quiet, not necessarily opposite.

## Sequential outcome rule

The evaluator and all controls must be committed before 2023 BTC outcomes are
opened. Train must pass every frozen economic, significance, stability,
stress, and mechanism-margin gate. Failure retires the exact policy without
opening 2024 or repairing from controls.

# Inventory purge/reclaim alpha — pre-2024 freeze (2026-07-15)

## Decision

One candidate was frozen before any 2024+ outcome was opened. It is not promoted until the frozen replay is complete.

The mechanism is a 4-hour directional price tail plus contracting delayed OI, followed by a 1-hour price/taker-flow reclaim. Long trades pass unchanged. Short trades additionally require the 7-day smart-vs-retail positioning state to support the short direction.

| split | absolute return | CAGR | strict MDD | CAGR/MDD | trades | L/S |
|---|---:|---:|---:|---:|---:|---:|
| fit | 25.82% | 10.94% | 7.57% | 1.44 | 172 | 132/40 |
| fit_2020q4 | 3.46% | 17.24% | 3.58% | 4.81 | 17 | 11/6 |
| fit_2021 | 17.85% | 17.86% | 7.57% | 2.36 | 111 | 78/33 |
| fit_2022 | 3.19% | 3.20% | 5.03% | 0.64 | 44 | 43/1 |
| select_2023 | 11.40% | 11.41% | 2.78% | 4.10 | 29 | 22/7 |
| select_2023_h1 | 5.62% | 11.66% | 2.76% | 4.23 | 17 | 12/5 |
| select_2023_h2 | 5.47% | 11.16% | 1.81% | 6.18 | 12 | 10/2 |

## Leakage and multiplicity controls

- Binance positioning metrics are backward-asof joined and delayed by one complete 5-minute source bar.
- Fit thresholds use 2020-10-15 through 2022-12-31 only; 2023 ranks the bounded family.
- The base non-overlapping schedule is created before the positioning gate, so the gate cannot add or reschedule a trade.
- Realized funding, 6 bp/notional-side costs, full-window CAGR, next-open entry, and favorable-then-adverse strict MDD are applied.
- 3,760 base policies and 1,248 context policies were examined. This multiplicity is material; 2023 is development confirmation, not pristine OOS.
- The short gate is statistically thin (7 shorts in 2023 and 2 in 2023 H2); it is a falsifiable candidate rule, not established short alpha.
- The manifest and activation hashes were written with `oos_opened=false`. 2024+ must be replayed unchanged.

## Artifacts

- Manifest: `results/inventory_purge_reclaim_manifest_2026-07-15.json`
- Manifest SHA-256: `b9c84a439ce3563e45e206171ccac9a559044afafa40d1a6228bab0fe1d6ae2a`
- Search: `training/search_inventory_purge_reclaim_alpha.py`

# Pullback–Inventory Shadow Handoff selection (2026-07-15)

## Verdict

**PRE_OOS_CANDIDATE_FROZEN**. This single preregistered composition passes the pre-2024 target without opening post-2023 rows.

The mechanism is not a return blend. Pullback-squeeze and inventory purge/reclaim each advance an independent non-overlapping virtual position clock. The shared 0.5x account accepts the earliest virtual trade that does not overlap its current position. A skipped virtual trade does not release or reschedule that component's clock.

## Frozen performance at 6 bp/notional/side

| window | absolute return | CAGR | strict MDD | CAGR/MDD | trades |
|---|---:|---:|---:|---:|---:|
| fit | 145.15% | 49.98% | 11.20% | 4.46 | 214 |
| fit_2020q4 | 18.97% | 125.59% | 6.37% | 19.72 | 22 |
| fit_2021 | 57.31% | 57.36% | 7.96% | 7.20 | 123 |
| fit_2022 | 28.21% | 28.23% | 11.20% | 2.52 | 68 |
| select_2023 | 24.77% | 24.79% | 4.86% | 5.10 | 42 |
| select_2023_h1 | 14.25% | 30.84% | 4.86% | 6.34 | 23 |
| select_2023_h2 | 9.21% | 19.11% | 3.08% | 6.20 | 19 |
| pre_2024 | 205.87% | 41.64% | 11.20% | 3.72 | 256 |

At 10 bp/notional/side, the primary ratios remain fit **3.79**, 2023 **4.67**, and full pre-2024 **3.18**.

Two accounting-boundary sensitivities are immaterial: excluding funding exactly at entry leaves full pre-2024 CAGR/MDD at **3.718**; charging a hypothetical exit cost at every adverse mark leaves it at **3.709**.

## Why the interaction works

- Pullback-squeeze supplies sparse, high-payoff long convexity over a 48-hour horizon.
- Inventory purge/reclaim supplies denser 24-hour mean-reversion/reclaim opportunities and a small short sleeve.
- Chronological handoff removes overlapping risk instead of averaging predictions. In pre-2024 it accepted 87 pullback and 169 inventory trades while rejecting 37 overlaps.
- Both ablations are weaker: pullback-only pre-2024 CAGR/MDD 2.88; inventory-only 1.46.

## Statistical stress

A deterministic circular moving-block bootstrap (5000 samples, 5-trade blocks) gives P(positive full-period return) **1.000** and P(CAGR/strict-MDD >= 3) **0.756**. The ratio 5/50/95 percentiles are 1.93 / 4.08 / 7.11.

## Integrity and caveats

- Sources are physically truncated at `2024-01-01`; entry is next 5-minute open; costs are charged on both sides; realized funding is included.
- Strict MDD retains the global/pre-entry high-water mark and applies each position's favorable envelope before its adverse envelope.
- All component and shared schedules are split-contained; no exit may cross an evaluated boundary.
- The standalone component families have already been viewed on later data in prior research. Therefore a later 2024+ replay is implementation-OOS relative to this script, but not an epistemically pristine first-ever holdout. It must be reported as a contamination-aware replay, not proof of untouched generalization.
- 2022 is positive but its standalone annual CAGR/MDD is 2.52; robustness is created at the multi-year mechanism level, not every calendar year independently.

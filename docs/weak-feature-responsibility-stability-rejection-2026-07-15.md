# Weak-feature responsibility stability rejection — 2026-07-15

**Rejected; 2024+ remains sealed.** The apparent alpha was a seed-sensitive random-forest selection hit.

Metric: absolute return / CAGR / strict MDD / CAGR-to-strict-MDD / trades.

## Audit protocol

- Reproduce the original 160-tree, seed-715 candidate, then rerun seeds 7/71/715/2026/71515 at 160 and 2,000 trees.
- Average the five 2,000-tree prediction vectors as a 10,000-tree convergence ensemble.
- Promotion requires at least three large-forest seed passes and an ensemble pass; neither occurred.
- Target is 48h net trade return; anchors use a 144-bar cooldown and are fit only on 2020-07-01..2022-12-31.
- Funding leg: 48h cap, 4% take, no stop. Premium leg: 12h cap, no take, 3% stop.
- Decisions occur on `:00`; current market bar is excluded; nested barrier and market-braid diagnostics are shifted one 5m row.
- Market/funding/premium/OI/spot-premium inputs are physically truncated before 2024; `oos_opened=false`.

## Original selection hit

| Window | Result |
|---|---:|
| train | 110.31% / 34.59% / 7.75% / 4.46 / 125 |
| select_2023 | 9.08% / 9.08% / 2.94% / 3.09 / 15 |
| select_2023_h1 | 7.15% / 14.95% / 2.94% / 5.09 / 10 |
| select_2023_h2 | 1.80% / 3.60% / 2.88% / 1.25 / 5 |
| pre_2024 | 129.40% / 26.76% / 7.75% / 3.45 / 140 |

## Stability result

- 160-tree seeds passing: **1/5**.
- 2,000-tree seeds passing: **0/5**.
- 10,000-tree mean ensemble passing: **False**.
- Decision: **reject**.

| Large-forest run | pre-2024 | 2023 | 2023 H2 |
|---|---:|---:|---:|
| seed_7_2000 | 114.41% / 24.34% / 9.40% / 2.59 / 140 | 5.68% / 5.68% / 2.94% / 1.93 / 17 | -0.14% / -0.28% / 2.60% / -0.11 / 4 |
| seed_71_2000 | 114.30% / 24.32% / 8.23% / 2.96 / 139 | 4.14% / 4.15% / 3.30% / 1.26 / 17 | -0.14% / -0.28% / 2.60% / -0.11 / 4 |
| seed_715_2000 | 114.49% / 24.35% / 9.69% / 2.51 / 139 | 5.68% / 5.68% / 2.94% / 1.93 / 17 | -0.14% / -0.28% / 2.60% / -0.11 / 4 |
| seed_2026_2000 | 108.81% / 23.40% / 9.69% / 2.41 / 140 | 4.14% / 4.15% / 3.30% / 1.26 / 17 | -0.14% / -0.28% / 2.60% / -0.11 / 4 |
| seed_71515_2000 | 108.72% / 23.38% / 9.40% / 2.49 / 138 | 5.94% / 5.94% / 2.94% / 2.02 / 17 | -0.35% / -0.69% / 2.60% / -0.27 / 3 |
| mean_5x2000 | 115.43% / 24.50% / 8.23% / 2.98 / 140 | 6.16% / 6.16% / 2.94% / 2.10 / 18 | -0.14% / -0.28% / 2.60% / -0.11 / 4 |

Audit hash: `cc30fdf7b8f01b471c7896afd7f475e7fd327395c21cf98ffec40d8a7bbb0c99`

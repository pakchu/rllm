# Delayed conditional-pullback eval freeze — 2026-07-15

**Frozen after 2024 test and before opening 2025+.**

- Test-audit commit: `a6eb32d849fe1e09a2cd86bedb4b0aeb2758b6ac`.
- Manifest hash: `2cc5d9f5837188372b3b9239c04f2873e7a43389aa6d0a385f0b100f028839bc`.
- Fixed one-hour information delay; five deterministic 2,000-tree forests.

Metric: absolute return / CAGR / strict MDD / CAGR-to-strict-MDD / trades.

| Window | Frozen result |
|---|---:|
| train | 105.65% / 33.39% / 8.02% / 4.16 / 126 |
| select_2023 | 11.99% / 12.00% / 3.12% / 3.85 / 26 |
| pre_2024 | 130.32% / 26.90% / 8.02% / 3.35 / 152 |
| test_2024 | 16.31% / 16.28% / 4.62% / 3.52 / 27 |

## Frozen eval gate

| Window | Minimum ratio | Maximum MDD | Minimum trades |
|---|---:|---:|---:|
| eval_2025 | 3.0 | 15.0% | 12 |
| holdout_2026h1 | 3.0 | 15.0% | 6 |
| eval_all | 3.0 | 15.0% | 18 |

Every eval window must also have positive absolute return. Parameters may not change after this freeze.

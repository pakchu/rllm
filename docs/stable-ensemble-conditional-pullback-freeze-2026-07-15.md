# Stable ensemble conditional-pullback freeze — 2026-07-15

**Frozen before opening any 2024+ rows.**

Metric: absolute return / CAGR / strict MDD / CAGR-to-strict-MDD / trades.

- Selection commit: `f9f60dbe19587b26658a9147caab34565bf39d0f`.
- Audit commit: `7d791489e60d5446baeaf5ee9a48b11f55c14047`.
- Manifest hash: `ebf5e4602ac1cfd18d4c98a8955839f88df0ad358ded0d37ae911cf0c4aa20be`.
- Five deterministic 2,000-tree forests; source-specific q=.30/.50 score calibration.
- In compressed 28-day ranges, funding events require a completed-daily deep pullback; premium events keep only their source score gate.

## Frozen selection evidence

| Window | Result |
|---|---:|
| train | 104.16% / 33.01% / 8.15% / 4.05 / 128 |
| select_2023 | 11.09% / 11.10% / 3.12% / 3.56 / 19 |
| select_2023_h1 | 8.90% / 18.78% / 3.12% / 6.02 / 13 |
| select_2023_h2 | 2.01% / 4.03% / 2.30% / 1.75 / 6 |
| pre_2024 | 126.80% / 26.35% / 8.15% / 3.23 / 147 |

## Frozen OOS gate

| Window | Minimum ratio | Maximum MDD | Minimum trades |
|---|---:|---:|---:|
| test_2024 | 3.0 | 15.0% | 12 |
| eval_2025 | 3.0 | 15.0% | 12 |
| holdout_2026h1 | 3.0 | 15.0% | 6 |
| oos_2024_2025 | 3.0 | 15.0% | 30 |
| oos_all | 3.0 | 15.0% | 36 |

All windows must also have positive absolute return. No threshold, feature, exit, or gate may change after this freeze.

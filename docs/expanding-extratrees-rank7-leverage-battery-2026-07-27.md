# ExtraTrees rank-7 preregistered leverage battery — 2026-07-27

Verdict: **TARGET_HIT_PROTOCOL_ISOLATED**

This is a protocol-isolated sizing audit, not globally pristine discovery OOS. The leverage was chosen using 2023–2024 only. The frozen alpha policy, annual refits, features, thresholds, exits, direction, and trade clocks did not change.

## Pre-2025 fixed grid

| Leverage | 2023 | 2024 | Combined | 10bp combined abs | Pass |
| ---: | --- | --- | --- | ---: | :---: |
| 0.50x | abs 12.86%, CAGR 12.87%, strict MDD 3.15%, CAGR/MDD 4.09, trades 19 | abs 16.40%, CAGR 16.36%, strict MDD 3.49%, CAGR/MDD 4.68, trades 22 | abs 31.37%, CAGR 14.61%, strict MDD 3.49%, CAGR/MDD 4.18, trades 41 | 29.23% | yes |
| 0.75x | abs 19.81%, CAGR 19.83%, strict MDD 4.67%, CAGR/MDD 4.25, trades 19 | abs 25.42%, CAGR 25.36%, strict MDD 5.15%, CAGR/MDD 4.92, trades 22 | abs 50.26%, CAGR 22.57%, strict MDD 5.15%, CAGR/MDD 4.38, trades 41 | 46.61% | yes |
| 1.00x | abs 27.12%, CAGR 27.15%, strict MDD 6.15%, CAGR/MDD 4.41, trades 19 | abs 35.02%, CAGR 34.94%, strict MDD 6.76%, CAGR/MDD 5.17, trades 22 | abs 71.65%, CAGR 30.99%, strict MDD 6.76%, CAGR/MDD 4.58, trades 41 | 66.11% | yes |
| 1.25x | abs 34.82%, CAGR 34.84%, strict MDD 7.61%, CAGR/MDD 4.58, trades 19 | abs 45.25%, CAGR 45.14%, strict MDD 8.32%, CAGR/MDD 5.42, trades 22 | abs 95.82%, CAGR 39.90%, strict MDD 8.32%, CAGR/MDD 4.80, trades 41 | 87.95% | yes |
| 1.50x | abs 42.90%, CAGR 42.94%, strict MDD 9.03%, CAGR/MDD 4.76, trades 19 | abs 56.12%, CAGR 55.98%, strict MDD 9.83%, CAGR/MDD 5.69, trades 22 | abs 123.11%, CAGR 49.33%, strict MDD 9.83%, CAGR/MDD 5.02, trades 41 | 112.38% | yes |

Selected leverage: `1.5`

## Fixed report-only result

| Period | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | 10bp stress abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2025 | 55.59% | 55.64% | 13.83% | 4.02 | 21 | 51.72% |
| 2026h1 | 22.87% | 64.02% | 12.47% | 5.13 | 12 | 21.11% |
| future | 91.17% | 58.06% | 13.83% | 4.20 | 33 | 83.74% |
| all | 326.51% | 52.88% | 13.83% | 3.82 | 74 | 290.23% |

- robustness pass: `True`
- user target hit: `True`
- future repair/reselection: `False`

## Integrity

- selected-position hash: `8ffbd55f07ceda0e82c270fe4b370fffba44bb3fcfc807368c4385d2ba97f531`
- every leverage cell preserved the exact frozen trade clocks;
- full-calendar CAGR includes idle periods;
- absolute return is reported for every evaluated period;
- hardened strict MDD and conservative funding boundaries are unchanged.

Result hash: `9eb4f1a850a164667222e7f43474232110631e594875c9949aac21dc6caab148`

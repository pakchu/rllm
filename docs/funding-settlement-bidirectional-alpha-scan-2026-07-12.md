# Post-funding-settlement alpha scan

Date: 2026-07-12

## Protocol

- Entry candidates occur only after an actual Binance funding settlement record.
- The current settled funding rate and only completed prior BTC/premium/flow bars are used.
- Entry is delayed by one 5-minute bar.
- Thresholds are frozen on Train (`2020-2023`); Test 2024 ranks candidates; 2025/2026 are report-only.
- Cost is 6 bp/side and strict MDD includes intraposition adverse excursion.
- Tested combinations: 680.

This pass measures **post-settlement price alpha**. It does not credit hypothetical funding carry. A position held across a later funding timestamp would require a separate exchange cashflow ledger before production promotion.

## Result

No candidate passed the alpha-pool or live-grade validator.

Best Test-2024 candidate: extreme negative/positive basis with high preceding 8-hour volume, hold 24 bars.

| split | absolute return | CAGR | strict MDD | CAGR/MDD | trades | long/short |
|---|---:|---:|---:|---:|---:|---:|
| Train | -4.48% | -1.14% | 12.45% | -0.09 | 176 | 112 / 64 |
| Test 2024 | +3.87% | 3.87% | 1.61% | 2.40 | 34 | 22 / 12 |
| Eval 2025 | -3.08% | -3.09% | 4.56% | -0.68 | 25 | 25 / 0 |
| 2026 YTD | -0.22% | -0.54% | 1.57% | -0.34 | 14 | 14 / 0 |

## Interpretation

The direction distribution collapsed after 2024 and the best 2024 score did not reach the required ratio of 3. Extreme basis at settlement is therefore not a stable standalone BTC direction alpha under this rule family.

The result also shows why funding must not be treated as a generic directional predictor. Its stronger use is likely carry accounting, crowding context, or a veto/size input combined with an independently valid setup.

## Artifacts

- Script: `training/search_funding_settlement_bidirectional_alpha.py`
- Result: `results/funding_settlement_bidirectional_alpha_scan_2026-07-12.json`

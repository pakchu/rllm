# Pre-2025 REX ML exit overlay (2026-07-13)

## Verdict

Fixed stop/take-profit exits did not make the ML critic pass every later year.
The strongest later-window overlay improved combined return and ratio, but 2025
remained below the target.  No live candidate was promoted.

## Protocol

- Base policies: frozen pre-2025 taker/range ML Top-10.
- Exit grid: price stops 0/0.5/1/1.5/2/3% and take profits
  0/1/2/3/4/6%.
- Overlay selection: 2024 full/H1/H2 only.
- Same-bar stop and take touched: stop fills first.
- Manifest hash:
  `832b67a7032b5e41ee65ca0623647d48aea249031ca341580a83624e629dcee7`
- Full-window CAGR, 0.5x, 6 bp per side, corrected strict MDD.

## Best later-window overlay

ExtraTrees TAKE q30, both sides, no stop, 4% take profit:

| Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | p approx |
|---|---:|---:|---:|---:|---:|---:|
| 2024 Selection | +9.68% | 9.66% | 6.63% | 1.46 | 44 | 0.142 |
| 2025 Eval | +5.34% | 5.34% | 2.55% | 2.09 | 13 | 0.088 |
| 2026 Holdout | +7.62% | 19.31% | 1.74% | 11.13 | 13 | 0.006 |
| 2025-2026 Combined | +13.37% | 9.27% | 2.55% | 3.63 | 26 | 0.002 |

## Decision

The overlay improves the already strong combined window, but does not solve the
sparse 2025 ratio.  The statistically broader candidate remains the raw
Top-10 taker/range gate: 54 OOS trades over 2025-2026, ratio 3.12, approximate
`p=0.021`.  Promote that raw gate as a research alpha candidate and retain the
ML + TP4 variant as a higher-conviction, lower-count overlay candidate.

## Artifacts

- Search: `training/search_rex_pre2025_exit_overlay.py`
- Tests: `tests/test_search_rex_pre2025_exit_overlay.py`
- Manifest: `results/rex_pre2025_taker_range_exit_overlay_top10_manifest_2026-07-13.json`
- Result: `results/rex_pre2025_taker_range_exit_overlay_scan_2026-07-13.json`

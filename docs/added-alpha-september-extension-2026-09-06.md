# Additional-alpha September extension — 2026-09-06

## Scope

Fixed 80/20, 60/40 and 50/50 macro/OI weights plus standalone controls. No reoptimization and no live changes.
Common window: June 1 through **September 5 00:00 UTC (09:00 KST)**. September-only means September 1–4, not a complete September.

## Data repair

Database BTCUSDT 5m OI ends August 3 13:35 UTC. The existing official Binance daily-metrics downloader recovered May 1–September 4: 36,573 of 36,576 expected snapshots. September 5 archive returned 404.
Archive observations are delayed five minutes before use; query inputs are physically capped at the frozen as-of. Historical database and live data were not modified. OI feature availability in the evaluated window: 0.9999638310185185.
This source/timing change is applied to the full replay, including earlier periods, so the result is not assumed identical to the prior database-OI replay. Original publication/arrival-time parity remains unproven.

## Fixed-weight results, 6bp/side plus realized funding

| Window | Allocation macro/OI | Return | MDD | Net entry episodes | Orders including liquidation | 10bp return |
|---|---|---:|---:|---:|---:|---:|
| common_to_september | june_selected_80_20 | 5.86% | 3.26% | 36 | 810 | 4.94% |
| common_to_september | retrospective_60_40 | 6.07% | 2.53% | 36 | 810 | 5.19% |
| common_to_september | fixed_50_50 | 6.17% | 2.37% | 36 | 810 | 5.31% |
| july_to_september | june_selected_80_20 | 2.24% | 3.26% | 24 | 671 | 1.57% |
| july_to_september | retrospective_60_40 | 1.33% | 2.53% | 24 | 671 | 0.79% |
| july_to_september | fixed_50_50 | 0.87% | 2.17% | 24 | 671 | 0.40% |
| extension_since_aug4 | june_selected_80_20 | 3.70% | 3.26% | 13 | 373 | 3.28% |
| extension_since_aug4 | retrospective_60_40 | 2.69% | 2.53% | 13 | 373 | 2.35% |
| extension_since_aug4 | fixed_50_50 | 2.18% | 2.17% | 13 | 373 | 1.89% |
| september_only | june_selected_80_20 | -2.00% | 3.26% | 3 | 76 | -2.10% |
| september_only | retrospective_60_40 | -1.58% | 2.53% | 3 | 76 | -1.68% |
| september_only | fixed_50_50 | -1.38% | 2.17% | 3 | 76 | -1.47% |

## Interpretation

The June-selected 80/20 subsequent July-to-September report is now positive (+2.24% base, +1.57% stress), unlike the earlier cutoff. This does not erase its July/early-August loss.
The August 4 onward recovery is mainly macro: macro alone +4.72%, OI alone -0.33%. September 1–4 is negative for every fixed macro/OI blend. Do not interpret the extended aggregate gain as continuing September strength.
Nine OI trades occurred across the complete common window, including one new September 4 trade; no August OI trades in this delayed-archive replay.

## Accounting / limitations

Same-symbol units are netted before trading costs and funding. Ordinary events resize only the active sleeve; the documented portfolio net-risk override can resize all sleeves at an event. Risk is not continuously capped intrabar.
Each reported subwindow restarts from cash and initializes known target states. Returns are not chained segment contributions. Entry episodes count aggregate-net activations, not independent sleeve trades.
All periods are exposed research; no pristine OOS or optimal-weight claim. Short-window annualization is not a reliable forecast. Legacy Gross9 integration remains outside this three-sleeve extension.

## Artifacts

`research/added_alpha_september/{design,source_download,report}.json`
`training/evaluate_added_alpha_september.py`
`tests/test_evaluate_added_alpha_september.py`

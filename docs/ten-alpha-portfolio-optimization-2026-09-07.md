# Ten-alpha portfolio optimization — 2026-09-07

## Scope and outcome

Included all previous eight sleeves plus the standalone dollar-rally short and failed-rebound short.544 unique weights:512 seeded capacity-bounded samples plus32 predeclared G9/G9+macro short-addition cells.
2024 selects by10bp Calmar, then return;2025/full2026H1/recent reports reuse fixed weights. No subsequent-period reranking.
**The dollar-rally short is selected at1.0x in rank1; failed-rebound is0.0x.** Rank1 removes several original sleeves, including macro, and fails to preserve later-period performance. It is not authorized for replacement/live use.

## Frozen rank1 weights (notional/equity coefficients)

Rank7 1.25; Markov long1.75; OI pullback long0.5; dollar-rally short1.0. All other weights0. G9 legacy coefficients use the previously documented0.5x base-sleeve conversion.

## Base-cost results: return / MDD (6bp/side + realized funding)

| Period | Existing G9+macro1 | Frozen optimized rank1 | Existing + dollar short0.5 | Existing + both shorts0.5 each |
|---|---:|---:|---:|---:|
| 2024 | 262.92% / 19.03% | 576.00% / 14.01% | 331.48% / 17.50% | 354.89% / 15.73% |
| 2025 | 208.93% / 15.24% | 200.75% / 18.96% | 217.47% / 15.24% | 223.38% / 15.56% |
| 2026H1 | 100.20% / 18.83% | 97.72% / 19.01% | 114.89% / 18.36% | 135.07% / 16.49% |
| recent | 16.31% / 16.22% | 12.29% / 18.61% | 18.03% / 15.82% | 23.77% / 16.49% |
| since_july | 7.13% / 5.94% | -0.61% / 8.43% | 5.07% / 5.99% | 3.64% / 5.99% |
| september_only | -2.69% / 3.89% | -0.17% / 0.58% | -2.69% / 3.89% | -2.88% / 4.02% |

## Stress returns (10bp/side)

| Period | Existing G9+macro1 | Frozen optimized rank1 | Existing + dollar short0.5 | Existing + both shorts0.5 each |
|---|---:|---:|---:|---:|
| 2024 | 214.84% | 494.30% | 265.49% | 273.52% |
| 2025 | 175.72% | 172.32% | 178.06% | 170.74% |
| 2026H1 | 80.86% | 81.83% | 91.72% | 104.16% |
| recent | 12.14% | 9.77% | 13.34% | 17.64% |
| since_july | 5.13% | -1.80% | 2.86% | 1.13% |
| september_only | -3.01% | -0.21% | -3.01% | -3.24% |

## Interpretation

Inclusion in optimization is not the same as validated adoption. The frozen winner has strong2024 fit but2025 MDD18.96% versus parent15.24%, and July onward return-0.61% versus parent+7.13%. Do not replace the parent with the rank1 weights.
The anchored dollar-short0.5 cell improves returns and generally preserves or lowers MDD across the annual/H1 and overlapping June-Sept reports. But July onward returns fall from7.13% to5.07%. It is a research comparison cell, not a new OOS winner.
Both shorts0.5 each improve base-cost aggregate returns, but2025 stress return170.74% is below parent175.72%; failed-rebound is not an unconditional addition.

## Accounting and source checks

10-sleeve contexts preserve the first six core arrays byte-exact; zero-added baseline and both new short standalone metrics match prior exact results.2026H1 is Jan1–June30; recent is June1–Sep5 00:00UTC, overlapping H1. September-only is Sep1–4.
New OI-long historical construction explicitly force-closes at report boundaries (2024 count65, rather than the prior contained-clock64). Other source conventions are preserved and documented in the context report. This boundary change is not silently claimed as byte-identical prior8-sleeve execution.
Signed units offset before open-event costs/funding. Net4.5 cap is enforced after fees at open/active rebalances; it is not a continuous intrabar cap. Barrier risk uses conservative subset ordering and absorbing ruin. Rank7 coefficient limit1.5 is preserved; new sleeves/shorts capped at1.0.
No fee-ratio, frequency, or cross-sleeve overlap rejection. Same-sleeve source schedules remain nonoverlapping. No live files were modified.

## Verification and provenance

Final-context replay completed under frozen hashes. A provisional review smoke happened while context metadata was being finalized; it was quarantined before root acceptance. All numerical reports from the final replay match that provisional smoke, but only the final-source report is accepted.
Independent scoped optimizer review found no blockers. Root tests include context schema/clock parity, ten-sleeve grid, recent short mapping and reviewed ledger behavior.

## Artifacts

`research/ten_alpha_context/report.json` and three NPZ files.
`research/ten_alpha_optimization/report.json`, `selection_freeze.json`, `shadow_config.json`.
`configs/shadow/ten_alpha_portfolio_research_2026-09-07.json` — disabled research weight records, runtime integration not authorized.
`training/optimize_ten_alpha_portfolio.py` — bounded optimizer and fixed-period replay.

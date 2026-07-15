# Cross-collateral near-pressure frozen OOS

The 104-cell 2023 search, q0.985 threshold, 288-bar hold, and event clock were hash-frozen before this evaluator joined the outcome-blind 2024+ book panel to execution prices.

## Performance

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |
|---|---:|---:|---:|---:|---:|---:|
| test_2024 | -6.2609% | -6.2485% | 30.2510% | -0.2066 | 254 | 128/126 |
| eval_2025 | -5.7065% | -5.7103% | 20.3323% | -0.2808 | 276 | 125/151 |
| holdout_2026h1 | -16.1206% | -34.4538% | 19.8891% | -1.7323 | 119 | 55/64 |
| future_2025_2026h1 | -20.0807% | -14.6457% | 31.0992% | -0.4709 | 395 | 180/215 |
| oos_2024_2026h1 | -25.0844% | -11.2604% | 32.9901% | -0.3413 | 649 | 308/341 |
| all_2023_2026h1 | 20.6122% | 5.6381% | 32.9901% | 0.1709 | 887 | 425/462 |

## Independence from frozen rank-7

| Window | Exact entry Jaccard | Candidate entries within 6h | Position Jaccard | Daily PnL Pearson | Spearman | Pass |
|---|---:|---:|---:|---:|---:|---|
| test_2024 | 0.0000 | 0.0394 | 0.0493 | 0.0131 | 0.0065 | True |
| eval_2025 | 0.0000 | 0.0362 | 0.0664 | 0.0415 | -0.0292 | True |
| holdout_2026h1 | 0.0000 | 0.0672 | 0.1183 | -0.2677 | -0.1777 | True |
| future_2025_2026h1 | 0.0000 | 0.0456 | 0.0821 | -0.0735 | -0.0785 | True |
| oos_2024_2026h1 | 0.0000 | 0.0431 | 0.0692 | -0.0406 | -0.0452 | True |
| all_2023_2026h1 | 0.0000 | 0.0395 | 0.0637 | -0.0154 | -0.0175 | True |

## Integrity

- Feature inputs contain only checksum-verified USD-M/COIN-M bookDepth paths.
- Every robust baseline excludes the current bar; missing source bars fail closed.
- Entry is next-open, cost is 6 bp/side, realized funding is included, and strict MDD includes intratrade extremes.
- CAGR uses the full calendar window, including idle periods.
- Rank-7 frozen hashes/stats and the candidate's complete 2023 schedule replay before OOS is accepted.

## Verdict

The event clock passed every independence gate, but the frozen pressure-sign directional policy failed OOS and is rejected. Retain only the event clock for a separately frozen causal direction model.

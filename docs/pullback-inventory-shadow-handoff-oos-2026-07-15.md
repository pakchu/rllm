# Pullback–Inventory Shadow Handoff OOS replay (2026-07-15)

## Verdict

**REJECTED_OOS**.

| window | absolute return | CAGR | strict MDD | CAGR/MDD | trades |
|---|---:|---:|---:|---:|---:|
| test_2024 | 2.94% | 2.93% | 9.30% | 0.32 | 65 |
| eval_2025 | 5.96% | 5.96% | 4.44% | 1.34 | 47 |
| holdout_2026 | 9.16% | 23.45% | 4.99% | 4.70 | 30 |
| oos_2024_2026 | 19.27% | 7.56% | 9.30% | 0.81 | 143 |

At 10 bp/notional/side, full 2024–2026H1 CAGR/MDD is **0.49**.

## Interpretation

- The evaluator validated frozen manifest `3d8e92d1302ba9f79a4e7011d3addb0a2bd2d8a4c7dc5deb38dfd8a9f74f6333` and reconstructed every pre-2024 schedule hash before loading future rows.
- The result uses next-bar execution, 6 bp/notional/side, realized funding, split-contained exits, and strict favorable-before-adverse MDD.
- This is an implementation-OOS replay for the newly frozen handoff. It is not epistemically pristine because both component families had already been viewed on later periods before the handoff hypothesis was formed.
- A live-grade promotion requires both 2024 and 2025 independently to have positive return, CAGR/strict-MDD >= 3, strict MDD <= 15%, and at least 20 trades, plus full-period ratio >= 3 with at least 50 trades.

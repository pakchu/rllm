# CCBS-12 2023 development evaluation — 2026-07-17

2023 is outcome-blind development, not pristine OOS; 2024 remains sealed.
This result uses the committed CCBS evaluator, fractional derivative quantities,
6 bp base / 10 bp stress costs, full-calendar CAGR, and global favorable-before-
adverse strict MDD. COIN-M BTC collateral remains outside this research ledger.

| Cost | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 6 bp | -5.4898% | -5.4934% | 6.7271% | -0.8166 | 58 |
| 10 bp | -9.7807% | -9.7871% | 10.9269% | -0.8957 | 58 |

Disposition: **REJECT_2023_KEEP_2024_SEALED**.

Failed gates: `['absolute_return_positive', 'cagr_to_strict_mdd_at_least_3', 'h1_absolute_return_positive', 'h2_absolute_return_positive', 'um_rich_branch_positive', 'cm_rich_branch_positive', 'ten_bp_stress_absolute_return_positive', 'pre_cost_pnl_exceeds_transaction_cost', 'monthly_signflip_pvalue_at_most_10pct']`.

2024 may be opened only after every development and subsequent PnL-
orthogonality gate passes. Even a pass is not live-ready until the BTC-
collateral ledger and forward shadow are complete.

Report content hash: `1d41d55f29dec340f207d81600ee8d8f3a595c641d952756079ecf571d423f9a`

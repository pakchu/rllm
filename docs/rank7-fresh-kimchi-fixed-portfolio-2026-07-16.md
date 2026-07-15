# Frozen rank-7 + Fresh Kimchi fixed portfolio audit

Exactly one allocation is evaluated: 75% frozen annual rank-7 and 25% Fresh Kimchi/FX, held in fixed subaccounts with no rebalancing or weight grid.

The strict portfolio path combines both sleeves at the same five-minute BTC open/low/high/close point, includes entry/exit costs and realized funding, and applies the execution engine's stop-first barrier result. This avoids the impossible assumption that a simultaneous long and short both receive their own adverse BTC price.

| Window | Portfolio abs return | CAGR | Sync strict MDD | CAGR/MDD | Rank7 CAGR/MDD | Delta ratio | Trades (R7/Kimchi) |
|---|---:|---:|---:|---:|---:|---:|---:|
| selection_2024 | 14.9511% | 14.9183% | 2.2856% | 6.5271 | 5.4648 | +1.0623 | 22/30 |
| eval_2025 | 15.2518% | 15.2630% | 3.0216% | 5.0513 | 4.3864 | +0.6649 | 21/17 |
| holdout_2026h1 | 7.8748% | 19.9791% | 3.5167% | 5.6811 | 4.3008 | +1.3803 | 12/28 |
| future_2025_2026h1 | 24.3089% | 16.6176% | 3.5393% | 4.6951 | 3.9534 | +0.7417 | 33/45 |
| oos_2024_2026h1 | 42.9189% | 15.9184% | 3.5683% | 4.4611 | 3.8925 | +0.5686 | 55/75 |

## Interpretation

The fixed 25% Fresh Kimchi sleeve improves synchronized risk efficiency in 2025-2026H1: strict MDD falls and CAGR/MDD rises, while raw CAGR is slightly diluted. Treat it as a risk-budget diversification shadow candidate, not a return-maximizing replacement for rank-7. Do not enable live until a new untouched period confirms the marginal gain.

This is a diagnostic marginal-value audit, not pristine OOS evidence: the Fresh Kimchi gate used 2024 for selection and 2025/2026 had already been viewed. No threshold or weight was changed after replay.

Canonical component strict MDD from the original execution evaluator and the synchronized portfolio MDD are both retained in the JSON. Promotion still requires an untouched forward window.

# Combined price-action + macro sparse scan — 2026-06-25

## Purpose

Combine the robust momentum/macro sparse pool with newly integrated price-action extreme features to test whether a broader but still interpretable event pool improves statistical reliability.

## Scan

Report:

`results/sparse_setup_combined_pa_macro_2026-06-25/report.json`

Feature families:

- macro: DXY, kimchi premium, USDKRW, higher timeframe market features
- wave: momentum, CVD, flow
- price action: `pa__pa_ext_144/288/576_*`

Notable discovered candidate:

`mkt__kimchi_premium_change high & pa__pa_ext_576_to_max_high_pct low`, h=144, q=0.05

Strict candidate summary:

- positive folds: `7/7`
- ratio3/MDD15 folds: `7/7`
- total trades: `30`
- median CAGR: `67.65%`
- median strict MDD: `5.45%`
- worst fold CAGR: `27.24%`

This is promising but too low-frequency by itself.

## Walk-forward selector

Report:

`results/sparse_setup_combined_pa_macro_2026-06-25/walkforward_selector.json`

Final continuous replay:

- CAGR: `21.17%`
- strict MDD: `13.53%`
- CAGR/MDD: `1.56`
- trades: `267`
- approximate p-value: `0.0033`
- required trades for 80% power: `244`
- observed trades: `267`

Fold detail:

| fold | trades | CAGR | strict MDD | CAGR/MDD |
| --- | ---: | ---: | ---: | ---: |
| 2023H1 | 21 | 24.54% | 6.70% | 3.66 |
| 2023H2 | 2 | inflated | 2.07% | not meaningful |
| 2024H1 | 81 | 55.01% | 9.01% | 6.11 |
| 2024H2 | 55 | 40.93% | 9.22% | 4.44 |
| 2025H1 | 48 | 41.15% | 8.26% | 4.98 |
| 2025H2 | 30 | -15.86% | 9.36% | -1.69 |
| 2026H1 | 30 | 56.19% | 11.38% | 4.94 |

## Interpretation

This is the strongest statistically powered result so far in this branch: the trade count and p-value are credible, and most folds exceed the target ratio. The remaining blocker is concentrated in 2025H2. That means the next improvement should not be another broad gate sweep; it should specifically detect the 2025H2-like regime and disable or alter the combined pool there.

Current status against target:

- strict MDD <= 15: pass
- statistically meaningful trades: pass
- CAGR/MDD >= 3: fail, due to 2025H2
- CAGR 50%: fail on full period, but several folds exceed it

Next step: build a pre-fold / in-fold regime diagnostic for the selected candidates and test whether 2025H2 can be identified without looking at 2025H2 returns.

## Candidate-limit / ensemble-size partial sweep

A follow-up sweep started over candidate limits and ensemble sizes, but the naive loop was stopped after 18 completed configs because it recomputed the same feature frame for every config. Partial output:

`results/sparse_setup_combined_pa_macro_2026-06-25/selector_sweep/partial_summary.json`

Best completed partial variant:

- candidate limit: `5` or `8`
- ensemble size: `1`
- trades: `170`
- CAGR: `13.77%`
- strict MDD: `9.28%`
- CAGR/MDD: `1.48`
- p-value: `0.0038`

This is below the original combined selector ratio `1.56`. Reducing candidate count improves MDD but gives up too much CAGR. The bottleneck remains regime-specific, not simple over-expansion.

Engineering note: repeated selector sweeps should cache feature/event construction. Current selector CLI is correct for one-off replay but inefficient for grid search.

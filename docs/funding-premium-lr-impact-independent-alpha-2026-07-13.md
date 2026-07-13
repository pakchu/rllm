# Funding/premium squeeze + liquidity-impact alpha (2026-07-13)

## Verdict

A new non-REX alpha-pool candidate was found by applying an independently sourced liquidity-recovery feature to the fixed funding/premium squeeze alpha.

The frozen candidate is manifest rank 8 inside a predeclared Top-10 family. It was identified as the only Top-10 member that generalized, so the later windows validate a 10-candidate family rather than one pristine preregistered rule. It clears CAGR/strict-MDD 3.0 in 2024, 2025, 2026 YTD and the combined 2024-2026 window under corrected high-water strict MDD.

## Rule

Long when either condition is true:

1. Funding component with liquidity-impact gate:
   - `funding_rate <= -0.0000167`
   - `trend_96 >= 0.007485218212390219`
   - `-0.20030301257467914 <= lr_impact_72 <= 0.24664964484849766`
2. Premium component, unchanged:
   - `premium_index_change <= -0.00023471`
   - `htf_1d_return_4 >= 0.0940403008961932`

`lr_impact_72` is the completed 72-bar log price displacement divided by absolute 72-bar signed taker-flow share plus `1e-4`. The central gate removes funding setups where price displacement is extreme relative to observed taker flow.

Execution:

- signal from completed 5m bar
- enter next 5m open
- fixed hold 576 bars (48h)
- stride 12, one non-overlapping position
- long only, 0.5x
- fee 5bp + slippage 1bp per side
- no TP/SL

## Selection integrity

- Base rule and execution were fixed before the independent-gate search.
- Gate quantiles were fitted on 2020-2022.
- Gate variants were ranked on 2023 and both 2023 half-years.
- Market rows from 2024 onward were physically excluded before the Top-10 manifest was written.
- 2024, 2025 and 2026 were replay-only for the frozen family, but they were inspected to identify rank 8 as the sole Top-10 generalizer. The effective OOS multiplicity is therefore 10.
- Manifest hash: `65cf6eabc836bf2d9ca8dcd2b1b5ef88114c98c2b05935aa3ae2045340ecdfd5`.
- A second full run reproduced the manifest hash, selected rows, baseline and all qualifier statistics exactly.

The broader research program and the base funding/premium union have previously inspected later periods, so this is not pristine global OOS proof. It remains a shadow candidate.

## Statistics

| Window | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | Approx. p |
|---|---:|---:|---:|---:|---:|---:|
| Fit 2020-2022 | +55.62% | 15.88% | 20.06% | 0.79 | 144 | 0.1204 |
| Select 2023 | +24.30% | 24.32% | 10.20% | 2.38 | 25 | 0.0391 |
| Test 2024 | +24.48% | 24.43% | 5.67% | 4.31 | 26 | 0.0027 |
| Eval 2025 | +22.44% | 22.46% | 6.56% | 3.42 | 21 | 0.0061 |
| Holdout 2026 to Jun 02 | +10.80% | 27.95% | 6.82% | 4.10 | 22 | 0.0482 |
| 2024-2026 combined | +68.89% | 24.21% | 7.99% | 3.03 | 69 | 0.0000059 |

The combined approximate p-value remains about `0.000059` after a conservative Bonferroni correction for the frozen Top-10 family. Annual p-values should not be treated as family-wise significant after that correction.

## Gate contribution versus live-safe baseline

The baseline below also requires current funding/premium availability, avoiding stale auxiliary bars.

| Window | Baseline return / ratio | Candidate return / ratio |
|---|---:|---:|
| 2024 | +25.48% / 3.85 | +24.48% / 4.31 |
| 2025 | +12.19% / 1.86 | +22.44% / 3.42 |
| 2026 YTD | +9.20% / 2.03 | +10.80% / 4.10 |
| Combined | +53.73% / 1.68 | +68.89% / 3.03 |

The gate removes 13 of 82 combined baseline trades while increasing absolute return by 15.16 percentage points and reducing strict MDD from 11.58% to 7.99%.

## Quarter stability

- Positive-return quarters: 7/10
- Flat/no-trade quarters: 2/10
- Negative-return quarters: 1/10 (`2025Q4`, -0.32%)
- 2026Q1: +6.30%, ratio 4.13, 13 trades
- 2026Q2 through Jun 02: +2.70%, ratio 5.17, 8 trades

Quarter samples remain small; annual and combined statistics are the primary evidence.

## Independence and caveats

- Candidate construction uses no REX feature, event, prediction, model or REX-derived dataset.
- Exact signal-date Jaccard against the reference REX q75 stream is 0.0 in 2024, 2025 and 2026.
- The new dependency is the separately promoted `liquidity_recovery_efficiency_features_20260712` family.
- `lr_impact_72` is not mathematically orthogonal to momentum: its Spearman correlation with `trend_96` is about 0.71 pre-2024 because both contain price displacement. Its incremental information is the taker-flow denominator and central-impact state. Therefore “independent” here means a distinct non-REX alpha-feature family and causal data transformation, not zero correlation.
- Fresh shadow/live-forward data is still required before capital deployment.

## Artifacts

- `training/search_funding_premium_independent_gate_alpha.py`
- `results/funding_premium_independent_gate_top10_manifest_2026-07-13.json`
- `results/funding_premium_independent_gate_alpha_scan_2026-07-13.json`
- `configs/live/funding_premium_lr_impact_central_research_candidate.json`

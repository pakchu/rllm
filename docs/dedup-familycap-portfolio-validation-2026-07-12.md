# Deduplicated and family-capped portfolio validation

Date: 2026-07-12

## Changes enforced

- Exact return/adverse-excursion duplicates are hashed across every split and collapsed to one canonical sleeve.
- 19 duplicate groups / 71 redundant aliases are removed from allocation choices.
- Maximum gross allocation per sleeve family is 2.0.
- Nonzero weight remains at least 0.25 in 0.05 increments.
- Train strict MDD cap is 40%; each OOS strict MDD cap is 20%.
- Each OOS split must have positive return and CAGR/strict-MDD at least 3 for the train-sane passing set.
- Cost remains 6 bp/side and strict MDD includes adverse excursion.

## Result

- Evaluated canonical portfolios: 2,602
- Passing train/OOS risk gate in saved top set: 33
- The previous gross 3.85 candidate remains rank 1 unchanged.

| split | absolute return | CAGR | strict MDD | CAGR/MDD | trades |
|---|---:|---:|---:|---:|---:|
| Train | +523.60% | 73.21% | 31.90% | 2.30 | 818 |
| Test 2024 | +66.94% | 66.76% | 13.88% | 4.81 | 172 |
| Eval 2025 | +61.20% | 61.25% | 10.01% | 6.12 | 109 |
| 2026 YTD | +24.89% | 70.00% | 7.27% | 9.63 | 65 |

Weights:

- `oi_upbit_ratio288_low`: 0.65
- `new_long_minimal_funding_premium`: 1.75
- `cand_rex_veto_7`: 1.45
- total gross: 3.85

## Interpretation

The candidate's reported performance is not caused by exact duplicate aliases or excessive allocation to one named family. The three sleeves also passed the prior path-overlap audit as distinct realised paths.

This improves structural confidence, but it does not repair candidate-provenance contamination. The 2025/2026 values remain research diagnostics because those periods influenced the broader alpha discovery process. The strategy is suitable for frozen shadow/forward validation, not for being labelled pristine final OOS.

## Artifacts

- Optimizer: `training/portfolio_opt_all_discovered_alpha_gross10.py`
- Tests: `tests/test_portfolio_all_discovered_dedup.py`
- Result: `results/portfolio_all_discovered_dedup_familycap2_trainmdd40_oosmdd20_2026-07-12.json`
- Full report: `docs/portfolio-all-discovered-dedup-familycap2-trainmdd40-oosmdd20-2026-07-12.md`

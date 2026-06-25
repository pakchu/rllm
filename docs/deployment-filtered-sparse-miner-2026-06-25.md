# Deployment-filtered sparse miner attempt (2026-06-25)

## Purpose

The previous sparse setup miner ranked candidates mostly by fold median behavior. That let candidates with attractive per-fold medians survive even when their continuous live-style deployment curve had poor drawdown or weak CAGR/MDD.

This pass adds deployment-style scoring directly to `rolling_sparse_setup_miner.py`:

- each candidate is replayed across all discovery folds as one continuous signal stream;
- `strict_summary` now includes:
  - `deployment_cagr_pct`
  - `deployment_strict_mdd_pct`
  - `deployment_cagr_to_strict_mdd`
  - `deployment_trades`
  - `deployment_p_value_mean_ret_approx`
- optional discovery filters:
  - `--min-deployment-ratio`
  - `--max-deployment-mdd-pct`
  - `--min-deployment-trades`

The strict replay stage also now skips folds where active train samples are below `min_fold_events`, avoiding undefined side selection from empty means.

## Run

Split:

- train discovery: `2020H1` through `2024H2`
- test selection: `2025H1`, `2025H2`
- eval holdout: `2026H1`

Discovery filter:

```text
--min-deployment-ratio 1.0
--max-deployment-mdd-pct 15.0
--min-deployment-trades 50
```

Artifact:

- `results/sparse_setup_tte_2020train_combined_pa_macro_2026-06-25/train_discovery_deployment_filtered_report_v2.json`
- `results/sparse_setup_tte_2020train_combined_pa_macro_2026-06-25/validator_deployment_filtered_v2.json`

Only 2 candidates survived the train deployment filter.

## Surviving train candidates

1. `wave__mom_144 high & pa__pa_ext_144_to_max_high_pct low`, h=72, q=0.1
   - train deployment CAGR `23.45%`
   - strict MDD `13.56%`
   - ratio `1.73`
   - trades `92`
   - p-value approx `0.0027`

2. `mkt__dxy_momentum low & wave__mom_48 low`, h=72, q=0.05
   - train deployment CAGR `15.08%`
   - strict MDD `14.72%`
   - ratio `1.02`
   - trades `79`
   - p-value approx `0.0104`

## Train/test/eval result

Best config selected by test score only: `candidate_limit=2`, `max_ensemble_size=2`.

| Period | CAGR | Strict MDD | CAGR/MDD | Trades | p-value approx | Power gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train 2020-2024 | 5.21% | 13.80% | 0.38 | 73 | 0.2107 | 293 |
| Test 2025 | -1.89% | 9.32% | -0.20 | 25 | 0.8935 | 10915 |
| Eval 2026H1 | 41.85% | 1.16% | 35.94 | 3 | 0.3291 | 22 |
| All | 4.06% | 13.80% | 0.29 | 101 | 0.2527 | 505 |

## Interpretation

The deployment filter correctly removes most overfit candidates, but the remaining pool is too sparse and weak. Eval looks numerically high only because it has 3 trades over a short February-only window, so it is not meaningful.

This confirms that the current sparse PA+macro two-predicate family is not enough. The next productive direction is not selector tuning; it is adding a different label/feature layer that can identify 2025-like failure regimes before trading.

Recommended next step:

- Build a regime-aware event label dataset where sparse setup candidates are features, not direct trades.
- Add explicit 2025-failure regime descriptors: trend/chop, volatility compression/expansion, dollar/DXY impulse, kimchi premium shock, and distance-to-rolling-extreme context.
- Train a lightweight ranker/classifier to abstain from sparse setups in regimes matching 2025 failures, using train-only labels and test/eval TTE.

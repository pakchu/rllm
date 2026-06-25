# Train/Test/Eval validation for combined PA + macro sparse setups (2026-06-25)

## Purpose

Validate whether the combined price-action + macro sparse setup strategy remains valid when candidate discovery, config selection, and final evaluation are separated.

This is a stricter check than the earlier full-fold walk-forward scan because the eval period is not allowed to influence candidate discovery or selector configuration.

## Split

- Train candidate discovery: `2023H1`, `2023H2`, `2024H1`, `2024H2`
- Test config selection: `2025H1`, `2025H2`
- Untouched eval: `2026H1` through `2026-06-01`

Artifacts:

- Train discovery: `results/sparse_setup_tte_combined_pa_macro_2026-06-25/train_discovery_report.json`
- Validator: `results/sparse_setup_tte_combined_pa_macro_2026-06-25/validator.json`

## Leakage controls

The validator records these guards in `validator.json`:

- candidate discovery report is expected to be train-only
- selector config is chosen by test score only
- eval is not used for config selection
- fold thresholds and side fitting occur before each fold start
- eval selector history may include only past train/test history, never future eval bars

A miner scoring fix was required for train-only discovery: `_score_event_folds` no longer requires at least five usable folds when the train split has fewer folds. It now requires `min_positive_folds` bounded by available folds.

## Selected config

Chosen by test score only:

```json
{
  "candidate_limit": 8,
  "max_ensemble_size": 2,
  "test_score": 27.35687720346518
}
```

## Results

| Period | Dates | CAGR | Strict MDD | CAGR/MDD | Trades | p-value approx | Power gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 2023-01-12 → 2024-12-20 | 31.92% | 10.70% | 2.98 | 68 | 0.0025 | 0 |
| Test | 2025-01-08 → 2025-12-01 | 21.50% | 8.64% | 2.49 | 32 | 0.0980 | 60 |
| Eval | 2026-01-02 → 2026-05-21 | 3.23% | 11.80% | 0.27 | 17 | 0.8610 | 4332 |
| All | 2023-01-12 → 2026-05-21 | 24.09% | 11.80% | 2.04 | 117 | 0.0012 | 0 |

## Interpretation

This strict train/test/eval run does **not** validate the strategy for live deployment.

The earlier strong 2026H1 result from the combined full-fold scan was likely selection bias: eval-era folds were part of candidate discovery/ranking, so the process could indirectly choose setups that happened to fit 2026. When candidates are discovered only on 2023-2024 and config is selected only on 2025, the untouched 2026H1 eval drops to CAGR/MDD `0.27` with only 17 trades and no statistical support.

## Next direction

The main failure is not execution mechanics but candidate robustness. The train-only pool is too small and too sparse for stable future generalization.

Recommended next run:

1. Expand train discovery history, e.g. `2020-2024` train, `2025` test, `2026H1` eval.
2. Enforce stronger candidate stability constraints:
   - higher total train trades
   - minimum trades per train fold
   - reject candidates with isolated one-fold dominance
   - require positive performance across multiple market regimes
3. Keep eval completely untouched until one final report.


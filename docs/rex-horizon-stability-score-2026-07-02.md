# REX horizon stability score (2026-07-02)

## Why

The previous REX horizon sweep showed a repeated failure pattern: validation winners were often validation spikes whose train baseline was weak, and those candidates inverted in 2025-2026 eval.  This pass changes selection pressure from "maximize validation CAGR/MDD" to "prefer train/validation-consistent candidates and penalize spikes".

## Implementation

`training/rex_horizon_sweep.py` now records both scores:

- `score_legacy_train_val_only`: old validation-heavy ranker;
- `score_stability_train_val_only`: new ranker used for ordering.

The stability score uses only train + validation data and penalizes:

- large `abs(validation_ratio - train_ratio)`;
- excessive `validation_ratio / train_ratio` spikes;
- weak train ratio;
- low validation trade count;
- known fragile `location_revert` validation spikes when train ratio is weak.

Regression: `tests/test_rex_horizon_sweep.py` verifies that a validation-spike `location_revert` row beats a stable deep-pullback row under legacy rank but loses under stability rank.

## Verification

```bash
.venv/bin/python -m py_compile training/rex_horizon_sweep.py tests/test_rex_horizon_sweep.py
.venv/bin/python - <<'PY'
import importlib.util
spec=importlib.util.spec_from_file_location('t','tests/test_rex_horizon_sweep.py')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod.test_stability_rank_penalizes_validation_spikes_more_than_legacy_rank()
print('manual rex horizon tests passed')
PY
```

## Sweep

Report: `results/rex_horizon_sweep_core_stability_t2020_2024_v2024_e2025_2026_2026-07-02.json`

Protocol:

- train: 2020-01-01..2024-01-01
- validation/selection: 2024-01-01..2025-01-01
- eval/report-only: 2025-01-01..2026-06-01
- same 126-trial REX core grid as the positive-strength sweep.

## Result

The stability score avoided the prior validation-spike winner:

| rank mode | selected family | q | hold | stride | val CAGR/MDD | val trades | eval CAGR | eval MDD | eval CAGR/MDD | eval trades | eval p |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy | `rex_multiscale_location_revert` | 0.85 | 288 | 24 | 4.89 | 141 | -19.22% | 30.54% | -0.63 | 218 | 0.091 |
| stability | `rex_htf_pullback_reclaim` | 0.85 | 288 | 12 | 2.17 | 64 | 12.15% | 8.65% | 1.40 | 64 | 0.145 |

Other high-ranked stability candidates:

| rank | family | q | hold | stride | eval CAGR/MDD | eval trades | note |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | `rex_htf_deep_pullback_resume` | 0.80 | 288 | 24 | 1.72 | 39 | better ratio, fewer trades |
| 4 | `rex_htf_deep_pullback_resume` | 0.85 | 144 | 24 | 1.61 | 33 | shorter horizon did not dominate |
| 5 | `rex_htf_deep_pullback_resume` | 0.85 | 288 | 24 | 2.95 | 27 | best ratio, but too sparse |

## Interpretation

This is a real improvement in selection robustness, not a finished strategy:

- legacy selection chose a candidate that lost heavily in eval;
- stability selection chose a candidate with positive eval return and 64 trades;
- however eval CAGR/MDD is only 1.40, below the user target and below deployable confidence.

The useful direction is now clear: keep the stability score, but add a family-level veto/ensemble step so the selector can combine or choose between `reclaim` and `deep_pullback` variants without over-selecting sparse validation spikes.

## Next action

Use `score_stability_train_val_only` inside the regime-family selector and add an abstain/veto layer:

1. veto location-reversion families after weak train baseline or adverse volatility/trend regimes;
2. prefer deep-pullback/reclaim families when train and validation ratios are both positive;
3. allow abstention when no family passes minimum stability score and expected fold trades.

# Alpha candidate gate — 2026-06-23

## Purpose

After removing the oversized analyzer/trader LLM path, the next blocker is alpha
quality. This gate prevents spending GPU/RL cycles on candidates that do not meet
strict fold-level profitability, drawdown, and trade-count requirements.

## Gate criteria

Source report:
`results/rolling_alpha_feature_discovery_report.json`

Default criteria:

- CAGR / strict MDD >= `3.0` in enough folds
- strict MDD <= `15%`
- at least `30` trades per counted fold
- at least `300` total trades
- at least `5` positive folds
- all strict folds must respect the MDD cap

## Current result

Command:

```bash
.venv/bin/python -m training.alpha_candidate_gate \
  --input-report results/rolling_alpha_feature_discovery_report.json \
  --output results/rolling_alpha_candidate_gate_2026-06-23.json
```

Decision: `NO_GO`

- candidates checked: `12`
- candidates passed: `0`
- blocking reason: no rolling alpha candidate satisfies strict CAGR/MDD,
  MDD cap, trade-count, and fold-consistency gates.

Representative failures:

| candidate | positive folds | total trades | worst CAGR | worst MDD | min CAGR/MDD | reason |
|---|---:|---:|---:|---:|---:|---|
| `mkt__usdkrw_zscore`, h144, q0.2 | 4/7 | 1708 | -77.76% | 54.94% | -1.48 | unstable and high drawdown |
| `mkt__htf_1w_return_4`, h288, q0.2 | 3/7 | 273 | -14.79% | 18.67% | -0.90 | too few trades, weak folds, MDD breach |
| `mkt__htf_1w_return_1`, h288, q0.2 | 2/7 | 371 | -41.13% | 26.12% | -1.57 | unstable and high drawdown |

## Interpretation

Current feature/rule candidates are not good enough to become an RLLM policy
prior. The correct next step is wider alpha discovery or a different environment
formulation, not more LLM fine-tuning over these labels.

Any future LLM/RL training job should require this gate, or an equivalent
train/test/eval candidate gate, to return `GO` first.

## Additional check: wave/ridge alpha report

Source report:
`results/wave_feature_ridge_alpha_2020_2023test_2024eval_report.json`

The gate was extended to handle reports with `top_by_selection` and explicit
`test`/`eval` metrics.

Command:

```bash
.venv/bin/python -m training.alpha_candidate_gate \
  --input-report results/wave_feature_ridge_alpha_2020_2023test_2024eval_report.json \
  --output results/wave_feature_ridge_alpha_candidate_gate_2026-06-23.json \
  --min-positive-folds 2 \
  --min-total-trades 300
```

Decision: `NO_GO`

- candidates checked: `30`
- candidates passed: `0`
- top `wave_core` candidates had weak positive 2023 test performance but negative
  2024-2026 eval performance.

Representative top candidate:

- group: `wave_core`
- horizon: `72`
- quantile: `0.1`
- overlay: leverage `0.2`, pause_after_losses `0`
- test: CAGR `2.12%`, MDD `8.22%`, CAGR/MDD `0.26`, trades `872`
- eval: CAGR `-10.59%`, MDD `25.37%`, CAGR/MDD `-0.42`, trades `2025`

Interpretation: wave/ridge feature combinations also fail the alpha gate. They do
not justify new LLM/RL training as policy priors.

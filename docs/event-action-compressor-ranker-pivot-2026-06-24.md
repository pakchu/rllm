# Event-action compressor/ranker pivot — 2026-06-24

## Why pivot

Direct Gemma4 final-action selection failed in three forms:

1. Binary TAKE/SKIP value SFT lost money on 2026 eval.
2. Pairwise A/B ranking collapsed into position/token priors.
3. Neutral Q-code utility SFT was less collapsed, but first-40 candidate selection barely improved over the first-candidate baseline and missed the Q4 high-utility tail.

The next viable RLLM structure is therefore not "LLM directly chooses trade". It is:

```text
past-only market/action context
  -> single LLM compressor emits leakage-safe symbolic state tokens
  -> transparent ranker/regressor selects candidate action
  -> strict backtest / walk-forward validation
```

This keeps the LLM where it is strongest: compressing mixed text/numeric context into structured regime language. It keeps final execution in a testable ranker.

## New data surface

Script: `training/event_action_compressor_ranker_data.py`

Input: existing event-action rows with prompt-visible past-only state and candidate action.

Output schema:

- `feature_snapshot`: 20 numeric features from prompt-visible state plus action strength/hold/side fields.
- `state_tokens`: 21 deterministic compressor tokens including trend, side-aligned trend, DXY, USDKRW, kimchi, HTF, flow, action family, side, hold bucket.
- `llm_compressor_prompt`: token compression prompt for single-LLM SFT.
- `llm_compressor_target`: deterministic JSON token target for SFT bootstrap.
- `reward`: label-only future utility metadata for ranker training/evaluation.

Leakage guard: no target code or future outcome is copied into `feature_snapshot`, `state_tokens`, or compressor prompt.

## Converted data sizes

| Split | Rows | Signals | Numeric features | Tokens | Reward positive frac | Reward mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train pre-2026 | 116,880 | 5,844 | 20 | 21 | 0.2979 | -0.00892 |
| eval 2026 | 11,940 | 597 | 20 | 21 | 0.3028 | -0.00846 |

Generated files:

- `data/event_action_compressor_ranker_train_pre2026_2026-06-24.jsonl`
- `data/event_action_compressor_ranker_eval2026_2026-06-24.jsonl`
- `results/event_action_compressor_ranker_train_pre2026_summary_2026-06-24.json`
- `results/event_action_compressor_ranker_eval2026_summary_2026-06-24.json`

## First transparent ranker baseline

Command used `training.event_candidate_ridge_ranker` with:

- train: pre-2026 compressor/ranker rows
- validation selection: 2024-01-01 through 2025-12-31
- eval: 2026 rows
- market: `data/2020-01-01_2026-06-01_btcusdt_futures_5m.csv.gz`
- ridge alpha: 100
- quantiles: 0.70, 0.80, 0.85, 0.90, 0.95
- selected on validation only

Report: `results/event_action_compressor_ridge_ranker_eval2026_report_2026-06-24.json`

### Validation-selected policy

Best validation config: q=0.95, full_margin=0.0.

| Split | Trades | CAGR | Strict MDD | CAGR/MDD | Mean trade | p approx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation 2024-2025 | 112 | 6.70% | 7.29% | 0.92 | 0.119% | 0.141 |
| eval 2026 | 20 | -1.92% | 3.17% | -0.60 | -0.038% | 0.725 |

## Conclusion

The new schema works and is leakage-safe, but the first ridge ranker does not generalize to 2026 and has too few eval trades. This is still useful: it shows the failure is now visible in a cheap, auditable ranker loop rather than hidden inside LLM label priors.

Next experiments should focus on:

1. Train Gemma4 as a compressor over `llm_compressor_prompt -> llm_compressor_target`, then compare deterministic tokens vs model-emitted tokens.
2. Replace ridge with pairwise/listwise ranker trained per-signal, not global utility regression.
3. Use rolling/continuous train windows so 2026 sees recent 2025 regimes without eval leakage.
4. Add stricter statistical gates: eval/test trade count thresholds and no selection on eval utility.

## IC-weighted ranker follow-up

Report: `results/event_action_compressor_ic_ranker_eval2026_report_2026-06-24.json`

Configuration used the same compressor/ranker train/eval split and selected quantile/full-margin on 2024-2025 validation only. IC feature selection used `min_abs_ic=0.005`, `min_sign_consistency=0.5`.

| Split | Trades | CAGR | Strict MDD | CAGR/MDD | Mean trade | p approx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation 2024-2025 | 114 | -0.41% | 7.63% | -0.05 | -0.0046% | 0.945 |
| eval 2026 | 18 | -7.56% | 3.84% | -1.97 | -0.1765% | 0.176 |

IC weighting is worse than ridge. Some feature ICs are visible, but they do not compose into a robust executable policy.

Updated next step: implement a schema-native per-signal pairwise/listwise ranker. The target should be candidate ordering within each signal, not global utility regression or independent IC weighting.

## Pairwise ranker follow-up

Script: `training/event_candidate_pairwise_ranker.py`

Report: `results/event_action_compressor_pairwise_ranker_eval2026_report_2026-06-24.json`

This ranker directly trains within-signal winner-over-loser pairs, then uses validation-only threshold selection.

| Split | Trades | CAGR | Strict MDD | CAGR/MDD | Mean trade | p approx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation 2024-2025 | 131 | -4.39% | 29.53% | -0.15 | -0.046% | 0.803 |
| eval 2026 | 23 | -26.14% | 20.99% | -1.25 | -0.504% | 0.340 |

Pairwise ranking is worse than ridge on this feature/candidate pool. The failure is now unlikely to be just the loss objective. Current evidence points to the candidate pool and/or prompt-visible features not carrying stable 2026 edge, or the edge being regime-conditional and inverted across periods.

Updated next step: audit feature/reward drift and candidate-family/side/hold performance by year before adding more model capacity.

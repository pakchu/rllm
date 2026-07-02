# Event candidate pool positive-strength audit (2026-07-02)

## Why this pass

The recent RLLM/Gemma4 surface was no longer blocked by label plumbing alone.  The bigger failure was that the candidate pool was sparse and unstable: validation-selected families often failed in the next chronological holdout.  While rechecking the price-action/REX pool, I found an additional candidate-generation flaw: families whose strength quantile collapsed to `0.0` could emit zero-strength rows, effectively trading almost every stride rather than only true setup events.

## Code change

`training/event_candidate_pool_probe.py` now:

- fits family thresholds only on positive finite train strengths;
- emits a candidate only when `strength > max(0, threshold)`;
- therefore prevents zero-strength placeholder rows from becoming trades.

Regression coverage: `tests/test_event_candidate_pool_probe.py` covers both above-threshold filtering and the collapsed-zero-threshold case.

## Verification

Commands:

```bash
.venv/bin/python - <<'PY'
import importlib.util
spec=importlib.util.spec_from_file_location('t','tests/test_event_candidate_pool_probe.py')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod.test_candidate_rows_require_positive_above_threshold_strength()
mod.test_candidate_rows_do_not_trade_zero_strength_when_threshold_collapses_to_zero()
print('manual tests passed')
PY
.venv/bin/python -m py_compile training/event_candidate_pool_probe.py tests/test_event_candidate_pool_probe.py
```

Result: both manual tests passed and py_compile passed.  `pytest` is not installed in the current Python/venv, so the tests were invoked directly.

## Focused probe results after the fix

Candidate families: REX price-action + macro/kimchi/flow/candle/compression families.  Data: `data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz`.  Execution: 5m bars, `hold_bars=288`, `stride_bars=24`, `quantile=0.80`, leverage 0.5, strict bar MDD.

### Split A: train 2020-2023, validation 2023, eval 2024-01-01..2026-06-01

Report: `results/event_candidate_pool_probe_pa_rex_macro_stability_fixed_t2020_2023_v2023_e2024_2026_2026-07-02.json`

Selected by train-positive + validation min-trade rule:

| family | split | CAGR | strict MDD | CAGR/MDD | trades | p-value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rex_htf_context_pullback_resume` | validation 2023 | 18.07% | 7.98% | 2.26 | 72 | 0.128 |
| `rex_htf_context_pullback_resume` | eval 2024-2026H1 | 5.86% | 9.43% | 0.62 | 112 | 0.350 |

Interpretation: positive but not strong enough; validation edge decays sharply over the longer eval.

### Split B: train 2020-2024, validation 2024, eval 2025-01-01..2026-06-01

Report: `results/event_candidate_pool_probe_pa_rex_macro_stability_fixed_t2020_2024_v2024_e2025_2026_2026-07-02.json`

Selected by train-positive + validation min-trade rule:

| family | split | CAGR | strict MDD | CAGR/MDD | trades | p-value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rex_multiscale_location_revert` | validation 2024 | 30.60% | 9.83% | 3.11 | 154 | 0.115 |
| `rex_multiscale_location_revert` | eval 2025-2026H1 | -14.65% | 27.33% | -0.54 | 240 | 0.269 |

Interpretation: validation-selected REX location reversion flips negative in the next holdout.  This is not deployable and is strong evidence of regime instability.

## Conclusion

The zero-strength bug was a real source of misleading candidate-pool optimism.  After fixing it, the remaining issue is still structural: REX/price-action families can look excellent in a single validation year but are not stable across the next chronological eval.  The next useful step is not more direct Gemma4 SFT on the same targets; it is a regime-conditional family selector or a richer candidate source that can learn when each weak family is valid.

## Next step

Build a fold-safe regime-conditioned family selector:

1. Treat each candidate family as an expert.
2. Use only pre-fold train/validation history to estimate which expert works under current causal regime descriptors.
3. Evaluate on at least 6-month target folds without using target outcomes for selection.
4. Export the selected family/state cards as the LLM-facing representation only after the selector shows non-degenerate OOS stability.

# Wavefull regenerated symbolic ridge audit (2026-06-27)

## Why this was run

The prior PAE mfc100 symbolic ridge run looked strong on 2025 validation but failed on 2026 Jan-May holdout. A token-drift audit showed the old 2026 verifier dataset had stale macro context: every 2026 row encoded `dollar_pressure=dxy_neutral` and `kimchi_context=kimchi_neutral`, despite the wavefull market cache containing non-neutral 2026 external features.

## Data correction

Regenerated verifier rows from `data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz` and re-applied causal PAE tokens:

- `data/event_action_verifier_text_v3k8_2024_wavefull_regen_pae_2026-06-27.jsonl`: 46,848 rows
- `data/event_action_verifier_text_v3k8_2025_wavefull_regen_pae_2026-06-27.jsonl`: 46,720 rows
- `data/event_action_verifier_text_v3k8_2026_jan_may_wavefull_regen_pae_2026-06-27.jsonl`: 19,104 rows
- `data/event_action_verifier_text_v3k8_train_2020_2024_wavefull_regen_pae_2026-06-27.jsonl`: 233,856 rows

Macro-token drift old vs regenerated 2026:

- old 2026: `dxy_neutral=100%`, `kimchi_neutral=100%`
- regenerated 2026: `dxy_neutral=38.19%`, `kimchi_neutral=39.20%`, with high/low/premium/discount buckets restored.
- artifact: `results/token_drift_old_vs_wavefull_regen_2026jm_2026-06-27.json`

## Validation results

### Train 2024 -> validation 2025 -> holdout 2026 Jan-May

Report: `results/symbolic_ridge_recent_pae_mfc100_wavefull_regen_2024_2025_2026jm_strict_2026-06-27/report.json`

- selected: `target=net_return`, `alpha=10000`, `threshold=0`, `min_gap=0`
- validation: CAGR `52.18%`, strict MDD `9.76%`, ratio `5.35`, trades `353`
- holdout: CAGR `-17.79%`, strict MDD `21.20%`, ratio `-0.84`, trades `142`

Conclusion: restoring macro tokens did not rescue 2026. The earlier failure was not only a stale-macro-data issue.

### Train 2020-2024 -> validation 2025 -> holdout 2026 Jan-May

Report: `results/symbolic_ridge_recent_pae_mfc100_wavefull_regen_train2020_2024_val2025_eval2026jm_strict_2026-06-27/report.json`

- no config passed strict validation.
- best-ranked failed validation: `target=tail_risk`, `alpha=1000`, `threshold=-0.003`, `min_gap=0`
- validation: CAGR `-10.93%`, strict MDD `14.66%`, ratio `-0.75`, trades `397`
- holdout: abstained due to validation failure.

Conclusion: adding more historical training data exposes that the 2024-only fit was likely a narrow 2025 overfit, not durable alpha.

## Current diagnosis

1. The macro-token data bug is real and fixed in regenerated datasets, but it was not the main alpha source.
2. The strongest validation profile still depends on validation-era action/family luck and collapses in 2026.
3. Longer training does not improve robustness; it removes the apparent validation edge. This argues against more gate tuning and toward changing the alpha representation/labels.

## Next direction

Do not proceed with Gemma SFT on this label surface yet. The current symbolic labels are not stable enough. Next useful work is to redesign the target/action representation around causal price-action continuation/reversal episodes with purged rolling validation before any LLM fine-tune.

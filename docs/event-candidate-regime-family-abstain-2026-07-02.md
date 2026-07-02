# Regime-family selector with abstention (2026-07-02)

## Why

The first regime-family selector improved diagnosis but still failed because it selected `rex_multiscale_location_revert` into 2025H1, causing a major drawdown.  The new rule adds history-trend penalties and abstention so that a family whose previous fold scores are deteriorating is not automatically carried forward.

## Code changes

- `training/event_candidate_regime_family_selector.py`
  - adds `--min-selection-score`;
  - scores family history with regime-distance-weighted previous fold scores;
  - penalizes negative recent history, large recent score declines, and declining `location_revert` histories;
  - abstains when the best pre-fold score is below the threshold.
- `training/event_candidate_pool_probe.py`
  - candidate row generation now uses cached date strings and array positions instead of repeated pandas `iloc`, avoiding `attrs` deepcopy overhead and making selector reruns fast.
- `tests/test_event_candidate_regime_family_selector.py`
  - verifies declining `location_revert` is ranked below steadier `pullback_reclaim`.

## Verification

```bash
.venv/bin/python -m py_compile \
  training/event_candidate_pool_probe.py \
  training/event_candidate_regime_family_selector.py \
  training/rex_horizon_sweep.py \
  tests/test_event_candidate_regime_family_selector.py
```

Manual direct tests were run for:

- `tests/test_event_candidate_pool_probe.py`
- `tests/test_rex_horizon_sweep.py`
- `tests/test_event_candidate_regime_family_selector.py`

## Walk-forward result

Report: `results/event_candidate_regime_family_selector_rex_core_abstain_6m_2023_2026h1_2026-07-02.json`

Protocol:

- train start: 2020-01-01
- target folds: 6-month folds from 2023-01-01 to 2026-06-01
- family pool: REX core families
- fixed candidate params: `hold_bars=288`, `stride_bars=24`, `quantile=0.80`
- selection uses only pre-fold prior or prior fold outcomes; target fold outcome remains report-only.

Final stitched replay:

| selector | CAGR | strict MDD | CAGR/MDD | trades | p-value |
| --- | ---: | ---: | ---: | ---: | ---: |
| prior selector | -0.28% | 19.35% | -0.01 | 250 | 0.954 |
| decline-penalty + abstain | 4.75% | 14.76% | 0.32 | 217 | 0.366 |

Important fold change:

| fold | old selected family | old fold result | new selected family | abstained? | new selected diagnostic |
| --- | --- | ---: | --- | --- | ---: |
| 2025H1 | `rex_multiscale_location_revert` | -25.41% CAGR / 14.05% MDD | `rex_htf_pullback_reclaim` | yes | +11.53% CAGR / 6.64% MDD |

The 2025H1 diagnostic was positive for the new family, but it was not traded because the pre-fold selection score was below `0.75`.  This confirms the abstention mechanism is conservative and did reduce realized drawdown.

## Interpretation

This is progress but still not a deployable strategy:

- MDD is now under the user's 15% threshold on this 2023-2026H1 stitched replay;
- CAGR is far too low, so CAGR/MDD remains only 0.32;
- early 2023 losses and undertrading remain the main blockers;
- the strongest evidence is structural: family-veto/abstention works and should be combined with broader candidate generation, not direct LLM trade picking yet.

## Next action

Use the selector output to build compact LLM state-card records:

- family candidates: `pullback_reclaim`, `deep_pullback`, `context_pullback`, `abstain`;
- include pre-fold history trend, regime distance, expected trade count, and veto flags;
- train/evaluate the LLM as a reasoning/ranking layer over these cards, not over raw numeric OHLC dumps.

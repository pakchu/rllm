# Validation reliability token augmentation (2026-06-23)

## Purpose

After the side-failure audit showed 2026 side-map collapse, we tested whether a past-only reliability token helps the RLLM event-context ranker learn when the side map is normal, decaying, or inverse-candidate.

## Implementation

Added `training/augment_context_with_validation_reliability.py`.

The script reads prior rolling month scores from `results/rolling_event_context_preference_summary_2026-06-23.json` and injects coarse text buckets into each context row:

- `side_map_reliability=unknown_pre_roll`
- `side_map_reliability=reliable_normal`
- `side_map_reliability=weak_or_decaying`
- `side_map_reliability=inverse_candidate`

It also injects:

- `prior_validation_health=unknown|positive|nonpositive|severe_decay`

Leakage guard:

- no target-month returns are read by the augmenter;
- raw numeric validation scores are not inserted into prompts;
- rolling scores are treated as prior-to-target-month state.

## Generated h288 augmented context

Command output:

- `data/llm_context_regime_events_valrel_h288_2026-06-23.jsonl`
- `results/llm_context_regime_events_valrel_h288_summary_2026-06-23.json`

Distribution:

| Token | Rows |
| --- | ---: |
| unknown_pre_roll | 5,844 |
| reliable_normal | 2,316 |
| weak_or_decaying | 608 |
| inverse_candidate | 598 |

The `inverse_candidate` bucket covers 2026-01 through 2026-05.

## Ranker result

Re-ran `training.rolling_event_context_preference_ranker` on the augmented context.

| Variant | Trades | CAGR | Strict MDD | CAGR/MDD | p-value |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline h288 pairwise | 488 | 8.82% | 22.06% | 0.40 | 0.367 |
| validation-reliability token h288 | 488 | 8.82% | 22.06% | 0.40 | 0.367 |

## Decision

This was a useful structural test but a no-op in performance.

Interpretation:

- Merely adding a text token does not make the current linear pairwise ranker learn side-map inversion.
- The token is likely too late/too sparse for the existing action space: the model still chooses among `WAIT/LONG/SHORT` with the same learned side mapping.
- To exploit this signal, the action space or training target must expose side-map reliability explicitly.

Next direction:

1. Replace plain `LONG/SHORT` choice with reliability-aware candidates, e.g. `NORMAL_LONG`, `NORMAL_SHORT`, `INVERT_LONG`, `INVERT_SHORT`, `WAIT`.
2. Or train a separate causal side-map head that outputs `normal|inverse|unreliable`, then compose it with the base side ranker.
3. Keep eval untouched; select any reliability/action-space transform on 2024-2025 before replaying 2026.

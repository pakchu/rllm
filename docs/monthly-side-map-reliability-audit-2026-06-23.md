# Monthly side-map reliability audit (2026-06-23)

## Purpose

The previous clean tests showed that simple inversion and score-geometry transforms are insufficient. This audit labels each month directly as:

- `normal`: original predicted side beats invert/block;
- `inverse`: flipped side beats pass/block;
- `unreliable`: no-trade/block beats both pass and invert.

This is audit/training-label design only. Same-month outcomes are used for labels, so the labels are not a live selector.

## Implementation

Added:

- `training/monthly_side_map_reliability_audit.py`
- `tests/test_monthly_side_map_reliability_audit.py`

For each month, the script strict-replays:

1. `pass`: original generated prediction rows;
2. `invert`: original trade sides flipped;
3. `block`: all predictions converted to no-trade.

Default execution overlay for this audit uses TP `3%`, matching the strongest prior diagnostic overlay.

## h288 result

Input:

- `results/rolling_event_context_preference_predictions_2026-06-23.jsonl`

Output:

- `results/monthly_side_map_reliability_h288_tp3_2026-06-23.json`

Label counts over 2024-01 through 2026-05:

| Label | Months |
| --- | ---: |
| normal | 15 |
| inverse | 7 |
| unreliable | 7 |

2026 labels:

| Month | Label | Pass CAGR | Invert CAGR |
| --- | --- | ---: | ---: |
| 2026-01 | unreliable | -26.85% | -5.86% |
| 2026-02 | inverse | -75.17% | 379.81% |
| 2026-03 | unreliable | -17.46% | -9.74% |
| 2026-04 | normal | 14.17% | -16.02% |
| 2026-05 | unreliable | -14.49% | -9.29% |

## Prior validation score is insufficient

Joined with `results/rolling_event_context_preference_summary_2026-06-23.json`:

| Prior score bucket | normal | inverse | unreliable | Total |
| --- | ---: | ---: | ---: | ---: |
| `>= 0.5` | 13 | 4 | 2 | 19 |
| `< 0.5` and `>= -500` | 1 | 2 | 2 | 5 |
| `< -500` | 1 | 1 | 3 | 5 |

This explains the previous failures:

- severe validation decay is not a clean inversion signal;
- most 2026 severe-decay months should be blocked, but 2026-02 is strongly inverse and 2026-04 is normal;
- a month-level side-map classifier needs more state than prior validation score and ranker score gap.

## Decision

The current base ranker likely has weak entry timing but unstable side mapping. The next useful RLLM direction is to train/evaluate a separate side-map reliability head or prompt target with labels `normal|inverse|unreliable`, using richer causal monthly state.

Do not claim monthly labels as deployable without a rolling classifier that predicts them from prior-only features.

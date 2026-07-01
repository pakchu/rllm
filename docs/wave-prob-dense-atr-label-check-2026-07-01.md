# Dense wave-probability ATR label check (2026-07-01)

## Purpose

The fixed-hold dense labels produced a q0.90 token subset that looked good in simplified reward space but failed strict ATR/cooldown replay. This reruns the dense label surface with per-candidate ATR trailing-stop labels so the label target is closer to live execution.

## Data

Same cached probability source and inclusion thresholds as the fixed-hold dense surface:

- LONG when `teacher_probability_long >= 0.54`
- SHORT when `teacher_probability_long <= 0.46`
- hold 12 bars, entry delay 3
- per-candidate ATR trailing stop: multiplier 3.75, period 45

Outputs:

- `data/wave_prob_dense_atr_take_skip_train_2024h2_2025.jsonl`
- `data/wave_prob_dense_atr_take_skip_eval_2026_jan_may.jsonl`
- `data/wave_prob_dense_atr_take_skip_summary_2026-07-01.json`
- `results/wave_prob_dense_atr_take_skip_token_rule_2026-07-01.json`

| split | rows | A/take labels | B/skip labels | LONG | SHORT | mean ATR reward | positive rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| train 2024H2-2025 | 8,582 | 3,725 | 4,857 | 6,217 | 2,365 | -0.0937% | 43.40% |
| eval 2026 Jan-May | 1,121 | 515 | 606 | 668 | 453 | -0.1114% | 45.94% |

## Train-only token rule result

| train-score q | eval selected | CAGR | MDD | CAGR/MDD | mean trade | p-value |
|---:|---:|---:|---:|---:|---:|---:|
| 0.70 | 270 | -30.54% | 24.83% | -1.23 | -0.041% | 0.492 |
| 0.80 | 179 | -30.55% | 24.83% | -1.23 | -0.061% | 0.477 |
| 0.90 | 104 | -35.55% | 23.47% | -1.51 | -0.130% | 0.337 |
| 0.00 | 1,121 | -95.74% | 72.49% | -1.32 | -0.111% | 0.000009 |

## Decision

ATR-aligned dense labels remove the apparent fixed-hold edge. This is useful because it identifies the true failure: the cached wave probability teacher is not producing a robust recent edge under live-style exits. The LLM cannot rescue this surface unless the upstream candidate generator changes.

Next work should stop trying to tune the LLM on this wave probability teacher and instead either:

1. generate a different candidate pool with demonstrable non-LLM fold edge first, or
2. use the LLM only as a meta-risk explainer after a non-LLM alpha source passes chronological strict replay.

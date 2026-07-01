# Wave state margin split validation (2026-07-01)

## Purpose

The base Gemma4 high-margin subset looked mildly positive when inspected over the whole 2024H2-2026 eval file. This check prevents threshold leakage by selecting the confidence margin on a chronological test slice only, then reporting the later holdout separately.

## Protocol

- Options: `data/wave_state_top5_take_skip_option_eval_2024h2_2026.jsonl`
- Predictions: `results/wave_state_top5_take_skip_option_gemma4_base_predictions.jsonl`
- Test: date `<= 2025-12-31 23:59:59`
- Eval: date `> 2025-12-31 23:59:59`
- Candidate margins: `0, 0.25, 0.5, 1.0, 1.5, 2.0`
- Margin selection uses test only; eval is not used for selection.

## Result

| margin | test trades | test compound | test CAGR | test MDD | test CAGR/MDD | test mean | test p | eval trades | eval compound |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 45 | -8.93% | -6.19% | 17.58% | -0.35 | -0.200% | 0.276 | 3 | 21.38% |
| 0.5 | 58 | -17.87% | -12.59% | 25.20% | -0.50 | -0.332% | 0.026 | 4 | 21.57% |
| 1.5 | 35 | -5.82% | -4.01% | 13.18% | -0.30 | -0.164% | 0.419 | 3 | 21.38% |
| 0.25 | 66 | -20.19% | -14.28% | 26.23% | -0.54 | -0.334% | 0.027 | 6 | 21.94% |
| 0.0 | 70 | -20.27% | -14.34% | 26.37% | -0.54 | -0.316% | 0.030 | 7 | 24.17% |
| 2.0 | 27 | -0.56% | -0.41% | 8.16% | -0.05 | -0.014% | 0.949 | 3 | 21.38% |

The test-selected best margin by the configured score is `1.0`, but all tested margins are negative on the actual test slice. The apparent 2026 eval gains are unusable because the eval split contains only 12 rows total and selected trade counts are 3-7.

## Decision

Reject the base Gemma4 high-margin subset as a validated edge. The full-file high-margin result was not a leak-safe selection result. Current wave top5 eval coverage is too sparse after 2025 to prove live-readiness.

Next work should increase candidate density / chronological coverage before more SFT. If we keep the LLM, use it as a state/rationale scorer over many more candidates, then let a fold validator calibrate thresholds.

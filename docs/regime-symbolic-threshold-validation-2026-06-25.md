# Regime symbolic policy threshold validation (2026-06-25)

## Goal
Re-test the LLM-style symbolic/action-token policy with a no-leak threshold-selection protocol after some 2026-only threshold sweeps appeared to meet the headline objective.

## Data
- History/eval source: `data/event_action_neutral_code_train_pre2026_2026-06-24.jsonl`, `data/event_action_neutral_code_eval2026_2026-06-24.jsonl`
- Split for threshold validation:
  - train/history: rows before 2025-01-01 (`87680` rows)
  - validation: 2025-01-01 through 2025-12-31 (`29200` rows)
  - final eval candidate: 2026-01-01 through 2026-05-31

## Initial 2026-only sweep warning
The following looked good when threshold was swept directly on 2026:

| Target | Threshold | 2026 CAGR | 2026 strict MDD | Trades | p-value |
|---|---:|---:|---:|---:|---:|
| tail_risk | -0.0134 | 56.66% | 13.02% | 180 | 0.201 |
| distributional_safety | -0.0125 | 52.47% | 13.98% | 180 | 0.228 |

This cannot be claimed as success because the threshold was chosen after seeing 2026 eval performance.

## No-leak validation results on 2025
Same targets/thresholds selected from the 2026-looking sweep failed on the 2025 validation split:

| Target | Threshold | 2025 CAGR | 2025 strict MDD | Trades | p-value |
|---|---:|---:|---:|---:|---:|
| tail_risk | -0.0154 | -51.39% | 58.29% | 538 | 0.0051 |
| tail_risk | -0.0134 | -43.14% | 47.29% | 375 | 0.0045 |
| tail_risk | -0.0108 | -26.63% | 32.15% | 151 | 0.0220 |
| distributional_safety | -0.0139 | -56.20% | 61.90% | 550 | 0.0016 |
| distributional_safety | -0.0125 | -42.36% | 46.28% | 394 | 0.0072 |
| distributional_safety | -0.0101 | -34.36% | 41.09% | 211 | 0.0048 |
| risk_adjusted | -0.0104 | -59.52% | 63.15% | 636 | 0.0011 |
| risk_adjusted | -0.0094 | -44.96% | 50.03% | 450 | 0.0109 |

## Side-flip diagnostic on 2025
Flipping selected sides did not rescue the validation split. The best inverted diagnostic among these was still negative/flat:

- `tail_risk -0.0108` inverted: CAGR -0.72%, strict MDD 13.30%, 151 trades, p=0.992
- `distributional_safety -0.0101` inverted: CAGR -2.11%, strict MDD 16.16%, 211 trades, p=0.941

## Conclusion
The 2026-only symbolic threshold success is an eval-selection artifact. The LLM-style symbolic surface remains useful for structured representation, but this thresholded ridge expert policy is not a validated alpha source.

## Next direction
Move from static thresholded expert policies to a truly validation-driven monthly selector, or discard thresholding and train a calibrated abstention/risk model on rolling windows. Any future result must report: validation-selected parameters, untouched eval performance, monthly breakdown, p-value, and strict MDD.

# Event candidate A/B/C option-choice baseline — 2026-06-29

## Purpose

The long JSON DPO preference format failed because response likelihood was dominated by formatting/length priors.
This experiment compresses the decision surface to one option token:

- `A` = LONG trade
- `B` = SHORT trade
- `C` = NO_TRADE

The prompt still contains only signal-time price-action, micro, external, and HTF context. The target is label-only
from realized utility.

## Data

Generator: `training/export_event_candidate_option_choice.py`

Outputs:

- Train: `data/event_candidate_option_choice_wavefull_ext_micro_c72_s2_train_2026-06-29.jsonl`
- Eval: `data/event_candidate_option_choice_wavefull_ext_micro_c72_s2_eval_2026-06-29.jsonl`
- Summary: `results/event_candidate_option_choice_wavefull_ext_micro_c72_s2_2026-06-29/summary.json`

Stats:

| Split | Rows | A/LONG | B/SHORT | C/NO_TRADE | Mean prompt chars |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 6,890 | 2,565 | 2,253 | 2,072 | 1,753 |
| Eval | 1,912 | 672 | 644 | 596 | 1,747 |

## Base Gemma 4 E4B option-logprob baseline

Evaluator: `training/eval_option_choice_logprob.py`

Balanced eval256, seed 42:

- Accuracy: 34.77% (89/256)
- Targets: A 86, B 85, C 85
- Predictions: A 161, B 95, C 0
- Accuracy by target:
  - A: 63.95%
  - B: 40.00%
  - C: 0.00%

## Interpretation

The A/B/C surface is much better than long JSON DPO for measuring the model directly, but the base model has a
strong trade-option prior and never chooses NO_TRADE. This is a clear, actionable bias to train against with SFT.

## Next step

Run a small option-token SFT, then re-run the same balanced eval256 option-logprob benchmark. Promotion requires
material improvement over 34.77% and nonzero `C` predictions before any trading backtest.

# Event Candidate Option Choice Eval Fix — 2026-06-30

## Context

The first A/B/C option-choice evaluator scored each candidate by appending `A`, `B`, or `C` to the chat prompt and summing token logprobs over the suffix span. That implementation was fragile because the evaluated text was re-tokenized with truncation and the suffix span could collapse onto shared prompt tokens rather than the intended answer option.

## Fix

`training/eval_option_choice_logprob.py` now scores the next-token logprob after the chat prompt directly:

1. Render the user prompt with the chat template and generation prompt.
2. Run the model once per prompt batch.
3. Take logits at the final non-padding prompt token.
4. Compare log-probabilities for the single-token option IDs for `A`, `B`, and `C`.

This makes the evaluator match the option-choice task: “given this prompt, which answer token would the model emit next?”

## Corrected eval256 results

Dataset: `data/event_candidate_option_choice_wavefull_ext_micro_c72_s2_eval_2026-06-29.jsonl`
Sampling: balanced 256 rows, seed 42.
Model alias: `gemma4` → `google/gemma-4-E4B-it`.

| model | accuracy | correct | pred A/B/C | A target acc | B target acc | C target acc |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| base | 0.3828125 | 98/256 | 123 / 73 / 60 | 0.5348837 | 0.3058824 | 0.3058824 |
| sft_s64 | 0.3671875 | 94/256 | 112 / 76 / 68 | 0.4767442 | 0.2941176 | 0.3294118 |
| sft_s256 | 0.3750000 | 96/256 | 118 / 82 / 56 | 0.5232558 | 0.3176471 | 0.2823529 |

Reports:

- `results/event_candidate_option_choice_wavefull_ext_micro_c72_s2_2026-06-29/base_eval256_corrected_report.json`
- `results/event_candidate_option_choice_wavefull_ext_micro_c72_s2_2026-06-29/sft_s64_eval256_corrected_report.json`
- `results/event_candidate_option_choice_wavefull_ext_micro_c72_s2_2026-06-29/sft_s256_eval256_corrected_report.json`

## Interpretation

The corrected evaluator removes the earlier degenerate all-A scoring artifact. However, the current SFT adapters still underperform the base model on balanced eval256. The current A/B/C supervised target surface should not be promoted to backtest policy without a stronger training objective or a cleaner target construction.

## Next direction

- Do not use the old non-corrected reports for model selection.
- Treat current s64/s256 adapters as rejected unless a later full-eval/backtest contradicts this small eval256 result.
- Prefer a smaller, less ambiguous target surface or preference/ranking objective that preserves relative candidate utility without forcing a noisy single hard class.

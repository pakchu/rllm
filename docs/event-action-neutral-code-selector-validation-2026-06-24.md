# Event-action neutral Q-code selector validation — 2026-06-24

## Goal

Validate whether the Gemma4 E4B neutral-code SFT can select better action candidates inside each signal, without using future utility at selection time.

## Inputs

- Model: `google/gemma-4-E4B-it`
- Adapter: `checkpoints/event_action_neutral_code_gemma4_e4b_sft32_2026-06-24`
- Eval candidate source: `data/event_action_neutral_code_eval2026_2026-06-24.jsonl`
- Train-only centering source: `results/event_action_neutral_code_gemma4_sft32_train_bal256_audit_2026-06-24.json`
- Clean validation subset: first 40 eval-2026 signals, 800 action candidates, `2026-01-01 02:55:00` through `2026-01-10 20:55:00`.

## Operational note

A single long `audit_label_priors` process developed severe slowdown after several hundred rows while holding ~32GB VRAM. The audit tool now supports checkpoint/resume, but the clean first-40 validation was produced by scoring eight independent 100-row batched chunks and combining them to avoid mixed scorer semantics.

Do not use the rejected KV-cache scoring path for Gemma4 label scoring. A 10-row equivalence check showed large Q2/Q4 score deltas versus the full-sequence batched scorer, changing predictions from Q2 to Q4.

## Label audit summary: first 40 signals / 800 candidates

Clean combined report: `results/event_action_neutral_code_gemma4_sft32_eval_first40signals_clean_audit_2026-06-24.json`

| Metric | Value |
| --- | ---: |
| rows scored | 800 |
| target accuracy | 0.4725 |
| dominant label | Q2 |
| prediction counts | Q1=214, Q2=540, Q3=45, Q4=1 |
| target counts | Q1=294, Q2=398, Q3=95, Q4=13 |
| mean score spread | 0.3663 |

The apparent 47% target accuracy is mostly Q1/Q2 behavior. The model almost never predicts Q4, so it is not reliably identifying the high-utility tail.

## Selector diagnostics

Selectors consume only model label scores. Oracle utility is diagnostic-only.

| Selector | Centering | Signals | Selected mean utility | First-candidate mean | Oracle mean | Selected - first | Oracle gap | Positive frac |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| expected_rank | raw | 40 | -0.005054 | -0.004749 | 0.005696 | -0.000305 | 0.010750 | 0.350 |
| expected_rank | train-centered | 40 | -0.005220 | -0.004749 | 0.005696 | -0.000471 | 0.010916 | 0.325 |
| q4_minus_q1 | raw | 40 | -0.004003 | -0.004749 | 0.005696 | +0.000746 | 0.009699 | 0.375 |
| q4_minus_q1 | train-centered | 40 | -0.004003 | -0.004749 | 0.005696 | +0.000746 | 0.009699 | 0.375 |
| q4_minus_q2 | raw | 40 | -0.008710 | -0.004749 | 0.005696 | -0.003961 | 0.014406 | 0.125 |
| q4_minus_q2 | train-centered | 40 | -0.008710 | -0.004749 | 0.005696 | -0.003961 | 0.014406 | 0.125 |

## Conclusion

The Q-code SFT is not a usable final action selector. The first-20 q4-minus-q1 improvement did not survive a modest expansion to 40 signals. The model retains label-prior structure, misses Q4 tail candidates, and leaves most oracle utility unextracted.

Next architecture should stop asking the LLM to directly choose the final trade action. Use the LLM as a feature compressor / regime narrator over leakage-safe price-action, macro, kimchi-premium, DXY, and multi-timeframe context, then train a small transparent ranker/regressor for action selection with strict train/test/eval boundaries.

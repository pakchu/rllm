# REX event reasoning Gemma2 SFT POC (2026-07-12)

## What ran
Two short LoRA SFT POCs on the symbolic REX event policy dataset.

Base model:
- `google/gemma-2-2b-it`

Hardware observed:
- RTX 5090 32GB

Training configs:
- samples: 384 balanced (`LONG`/`SHORT`/`NO_TRADE` each 128)
- max steps: 32
- max seq length: 1536
- LoRA: r=16, alpha=32, dropout=0.05
- batch size: 1, grad accum: 8

## Runs
### Full target JSON run
Checkpoint dir, not committed as weights:
- `checkpoints/rex_event_reasoning_gemma2_2b_lora_s32_20260712`

Training:
- runtime: 104.5 sec
- train loss: 1.144

Candidate-logprob action eval, mean normalization:

| split | samples | accuracy | behavior |
|---|---:|---:|---|
| test 2025 | 104 | 49.04% | predicts all `NO_TRADE` |
| eval 2026H1 | 81 | 48.15% | predicts all `NO_TRADE` |

### Action-only target run
Checkpoint dir, not committed as weights:
- `checkpoints/rex_event_reasoning_action_only_gemma2_2b_lora_s32_20260712`

Training:
- runtime: 101.8 sec
- train loss: 0.8697

Candidate-logprob eval:

| mode | split | samples | accuracy | behavior |
|---|---|---:|---:|---|
| mean | test 2025 | 104 | 49.04% | predicts all `NO_TRADE` |
| mean | eval 2026H1 | 81 | 48.15% | predicts all `NO_TRADE` |
| sum | test 2025 | 104 | 25.96% | predicts only `LONG/SHORT`, misses `NO_TRADE` |
| sum | eval 2026H1 | 81 | 27.16% | predicts only `LONG/SHORT`, misses `NO_TRADE` |
| generation | eval 2026H1 | 81 | 20.99% | noisy/biased; test generation parser failed on malformed multi-JSON output |

## Interpretation
This POC did not produce a usable policy. It exposed a concrete RLLM failure mode:

1. Candidate-logprob scoring is highly sensitive to completion length/normalization.
   - mean => `NO_TRADE` collapse
   - sum => trade-only collapse
2. Short 32-step SFT learns target format but not robust OOS side/no-trade decision boundaries.
3. Symbolic prompt data is not trivially solved by token frequency or short SFT.

## Next correction
Do not continue by simply increasing steps. First fix the action scoring formulation:

- Train/evaluate as **multiple-choice single-token labels**, e.g. `A_LONG`, `B_SHORT`, `C_SKIP`, to remove JSON length bias.
- Or score equal-length canonical completions such as `ACTION_LONG`, `ACTION_SHORT`, `ACTION_SKIP`.
- Add explicit calibration/temperature or class-prior correction on train only.
- Only after evaluator is unbiased should longer SFT/DPO be meaningful.

## Artifacts
Committed:
- summaries/eval reports
- action-only SFT JSONL splits

Not committed:
- LoRA adapter weights (~113MB each checkpoint dir)

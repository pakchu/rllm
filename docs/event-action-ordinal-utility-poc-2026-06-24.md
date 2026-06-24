# Event-action ordinal utility POC — 2026-06-24

## Purpose

After two failed LLM target shapes:

1. binary `TAKE/SKIP` candidate value SFT failed full 2026 strict backtest;
2. pairwise `A/B` ranking SFT learned a position/token prior;

this POC tested candidate-wise ordinal labels: `AVOID`, `LOW`, `MID`, `HIGH`.

The goal was to avoid pair-position bias and turn future utility into a coarse qualitative target that should be more LLM-friendly than numeric regression.

## Data

Built from `event_action_value_*_2026-06-24.jsonl` using:

- `AVOID`: utility < -0.01
- `LOW`: otherwise below MID
- `MID`: utility >= 0.004
- `HIGH`: utility >= 0.012 and MAE <= 0.018

Dataset sizes:

| split | rows | signals | AVOID | LOW | MID | HIGH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train pre-2026 | 116,880 | 5,844 | 40,738 | 55,677 | 12,764 | 7,701 |
| eval 2026 | 11,940 | 597 | 4,190 | 5,700 | 1,401 | 649 |

SFT used balanced sampling:

- samples: 4,096
- label counts: 1,024 each
- model: `google/gemma-4-E4B-it`
- LoRA r=16 alpha=32 dropout=0.05
- steps: 32
- runtime: 188.9s
- train loss: 1.087

## Evaluation bottleneck

Four-label full-sequence batch scoring with batch size 8 was too slow and VRAM-saturated. It was interrupted after more than 13 minutes without reaching the first 512-row progress checkpoint.

Stable smoke setting:

- batch size: 1
- train balanced 256 rows
- runtime: 118s

## Result

Balanced train256 raw logprob evaluation:

| metric | value |
| --- | ---: |
| accuracy | 25.0% |
| mean absolute rank error | 1.5 |
| prediction AVOID | 256 / 256 |
| prediction LOW/MID/HIGH | 0 / 256 |

Label logprob means on train balanced 256:

| label | mean logprob |
| --- | ---: |
| AVOID | -5.10 |
| LOW | -8.91 |
| MID | -9.08 |
| HIGH | -10.05 |

This means the model's label prior dominates the learned signal. AVOID is about 3.8 to 5.0 mean-logprob points above the other labels.

Train-only label mean centering reduced collapse but did not create useful skill:

| metric | value |
| --- | ---: |
| centered train256 accuracy | 27.7% |
| centered mean absolute rank error | 1.105 |
| centered predictions | AVOID 76 / LOW 39 / MID 69 / HIGH 72 |

## Conclusion

Ordinal labels avoid pair-position bias structurally, but current SFT/logprob scoring still fails because token/label prior overwhelms weak feature signal. This is not a usable trading alpha path as-is.

The repeated pattern across targets is now clear:

- direct action JSON SFT: selector/bias failure;
- binary value SFT: full 2026 strict backtest failure;
- pairwise A/B SFT: position prior failure;
- ordinal utility SFT: label prior failure.

## Next direction

Do not continue scaling SFT steps blindly. The LLM should not be the final scorer yet.

More defensible next architecture:

1. use LLM only as a **feature compressor** producing structured qualitative tags from past-only context;
2. feed those tags plus numeric price-action features into a small transparent ranker/regressor;
3. evaluate ranker with frozen train/test/eval splits and strict backtest;
4. only after a non-LLM teacher shows stable alpha, distill explanations or policy constraints back into the LLM.

If staying LLM-first, the next experiment must use labels with calibrated neutral token priors, e.g. single-token synthetic labels with measured base priors, not semantically loaded words like `AVOID` and `HIGH`.

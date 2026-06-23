# Event side-rationale base-prior subtraction (2026-06-23)

## Purpose

The rationale-rich Gemma DPO16 POC produced a positive strict replay:

- 80 trades;
- CAGR 20.36%;
- strict MDD 10.43%;
- ratio 1.95.

However, the model predicted almost all `INVERSE`. This follow-up tests whether that result is learned adapter signal or simply base-model / response-template prior.

## Code change

`training/eval_event_side_rationale_preference.py` now supports:

- `--prior-json`: a base/prior eval JSON containing raw candidate scores by row index;
- `--prior-weight`: multiplier for prior score subtraction.

Adjusted score:

```text
adjusted_score(candidate) = adapter_score(candidate) - prior_weight * base_score(candidate)
```

The output now keeps:

- `raw_scores`: adapter/base raw scores for the current run;
- `prior_scores`: subtracted prior scores when provided;
- `scores`: adjusted scores used for prediction.

## Base prior result

Base Gemma4 E4B without LoRA already strongly prefers the `INVERSE` rationale template.

| Scoring | Prediction distribution | Accuracy |
| --- | ---: | ---: |
| base mean | 191 inverse / 0 normal | 51.83% |
| base sum | 188 inverse / 3 normal | 50.26% |

This means the previous near-all-inverse DPO output is not reliable evidence of learned event discrimination.

## Prior-adjusted DPO result

After subtracting base prior scores from DPO adapter scores:

| Scoring | Prediction distribution | Accuracy |
| --- | ---: | ---: |
| adjusted mean | 101 normal / 90 inverse | 52.36% |
| adjusted sum | 102 normal / 89 inverse | 51.83% |

The distribution becomes balanced, but trading performance collapses.

## Strict replay

Both adjusted mean and adjusted sum produced the same strict replay metrics:

| Method | Trades | CAGR | Strict MDD | CAGR / strict MDD | Mean trade ret | p approx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prior-adjusted mean | 78 | -31.99% | 23.48% | -1.36 | -0.189% | 0.229 |
| prior-adjusted sum | 78 | -31.99% | 23.48% | -1.36 | -0.189% | 0.229 |
| unadjusted rationale DPO16 | 80 | 20.36% | 10.43% | 1.95 | 0.101% | 0.473 |

## Decision

No-go for this rationale DPO checkpoint.

The positive unadjusted result was mostly a constant/base-prior `INVERSE` strategy. Once base-template prior is removed, the adapter's residual signal is not profitable.

## Implication

This is a useful failure because it isolates the core issue:

- LLM scoring can be dominated by response-template priors even when the candidate responses contain causal rationale text.
- Positive replay from an unadjusted logprob candidate scorer must be treated as suspicious until base-prior subtraction is run.

## Next direction

Before any more Gemma training:

1. create length- and wording-symmetric candidate responses;
2. keep base-prior subtraction mandatory in evaluation;
3. use score-spread abstention on adjusted scores rather than forcing every event to normal/inverse;
4. compare adjusted-score abstention against monthly history-majority and token_signature baselines.

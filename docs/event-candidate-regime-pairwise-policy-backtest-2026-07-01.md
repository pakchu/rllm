# Regime Pairwise Option Policy Backtest — 2026-07-01

## Bug fixed before interpretation

The first pairwise prediction backtest was invalid because random eval1024 predictions were matched against the first rows of the full pair JSONL. `training/eval_option_choice_logprob.py` now preserves sampled row metadata (`row_index`, `candidates`, `utility_gap`), and `training/backtest_pairwise_option_oracle.py` uses prediction rows directly when selector=`prediction`.

## Oracle upper bound

Using the true pair target on the compact eval pairs is an oracle, not a deployable result. It confirms that the pair labels can in principle map to a profitable candidate policy:

- events: 1,915
- trades: 370
- CAGR: 29,726.87%
- strict MDD: 10.22%
- mean trade return: 2.207%
- p≈0

This is expected to be huge because the target uses future reward. It only proves the label space is economically meaningful.

## Gemma base compact prediction backtest

Re-generated compact random1024 predictions with metadata:

- predictions: `results/event_candidate_regime_pairwise_option_compact_paext_rex_2026-06-30/base_eval1024_random_predictions.jsonl`
- report: `results/event_candidate_regime_pairwise_option_compact_paext_rex_2026-06-30/base_eval1024_random_report.json`

Classification:

- accuracy: 0.515625
- pred A/B: 598 / 426
- target A/B: 542 / 482

Valid prediction backtest by absolute A/B margin threshold:

| margin quantile | events | trades | CAGR | strict MDD | ratio | mean trade | p approx |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.80 | 192 | 144 | 31.21 | 18.79 | 1.66 | 0.296% | 0.1569 |
| 0.85 | 149 | 117 | 11.58 | 18.64 | 0.62 | 0.157% | 0.4788 |
| 0.90 | 102 | 89 | 2.89 | 19.97 | 0.14 | 0.076% | 0.7824 |
| 0.95 | 52 | 48 | -0.72 | 17.56 | -0.04 | 0.002% | 0.9951 |

## Interpretation

The compact pairwise surface is now the best LLM-shaped path found so far:

- It uses LLM-relative comparison rather than global numeric thresholding.
- It has a real, but weak, deployable base-model signal at q0.80.
- It still misses the target: CAGR/MDD is 1.66, strict MDD is above 15%, and p-value is not strong enough.

## Decision

Proceed only with compact pairwise if the next step directly improves ranking quality and risk filtering. Do not train large/long. A small SFT or DPO PoC is justified because the oracle confirms label usefulness, but promotion requires prior-preserving backtest improvement beyond q0.80 base.

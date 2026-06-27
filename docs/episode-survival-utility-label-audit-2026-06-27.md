# Episode survival utility-label audit (2026-06-27)

## Purpose

The binary survival SFT v1 was weak as a deployable ranker. This audit tests a stricter train-fitted utility label: keep the same causal prompts, but relabel `TRADE` only when the target path utility is in the train top-utility region and still passes net/MAE/MFE constraints.

## Relabel command

```bash
.venv/bin/python -m training.relabel_episode_survival_utility \
  --train-jsonl data/episode_survival_sft_v1_natural_2026-06-27/episode_survival_train.jsonl.gz \
  --test-jsonl data/episode_survival_sft_v1_natural_2026-06-27/episode_survival_test.jsonl.gz \
  --eval-jsonl data/episode_survival_sft_v1_natural_2026-06-27/episode_survival_eval.jsonl.gz \
  --output-dir data/episode_survival_utility_v2_2026-06-27 \
  --utility-quantile 0.70 \
  --min-net-pct 0.25 \
  --max-mae-pct 2.0 \
  --min-mfe-to-mae 1.25
```

Train-fitted utility threshold: `0.15815%`.

| Split | Rows | TRADE | NO_TRADE | TRADE mean utility |
| --- | ---: | ---: | ---: | ---: |
| Train | 76,956 | 19,904 | 57,052 | 1.145% |
| Test | 41,274 | 11,572 | 29,702 | 0.945% |
| Eval | 8,304 | 2,422 | 5,882 | 0.907% |

## Logistic baseline on v2

Output: `results/episode_survival_utility_baseline_v2_2026-06-27/report.json`

| Split | Precision | Recall | Predicted TRADE |
| --- | ---: | ---: | ---: |
| Train | 0.291 | 0.660 | 45,125 |
| Test | 0.334 | 0.629 | 21,827 |
| Eval | 0.315 | 0.609 | 4,686 |

Strict backtest of accepted candidates:

| Split | CAGR | Strict MDD | Ratio | Trades | p-value |
| --- | ---: | ---: | ---: | ---: | ---: |
| Test | 0.45% | 30.85% | 0.01 | 491 | 0.867 |
| Eval | -18.37% | 16.40% | -1.12 | 104 | 0.609 |

## Decision

1. Utility relabeling alone does not fix the problem.
2. The current prompt feature set is under-informative for selecting high-utility survivors: event name + single-bar setup quality + macro z-scores are insufficient.
3. Do not spend GPU time on Gemma SFT against v1/v2 expecting trading gains.
4. Next dataset needs richer causal history/context:
   - multi-bar path shape before entry;
   - regime age and drawdown/recovery phase;
   - recent realized volatility and compression duration;
   - prior MAE-like proxy features that do not use future bars;
   - candidate competition/ranking at the same timestamp instead of independent binary rows.
5. The next RLLM step should be a pairwise/ranking dataset per timestamp, not another independent binary classifier.

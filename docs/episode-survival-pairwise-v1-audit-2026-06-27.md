# Episode survival pairwise v1 audit (2026-06-27)

## Purpose

Independent binary survival labels were weak and threshold-sensitive. This pass changes the RLLM framing to same-timestamp candidate ranking:

> Given two candidate trades at the same signal timestamp, choose the one with higher future path-risk-adjusted utility.

The prompt contains only causal history/setup/context. Chosen/rejected labels use future path utility for offline preference training only.

## Export command

```bash
.venv/bin/python -m training.export_episode_survival_pairwise_data \
  --train-jsonl data/episode_survival_sft_v1_natural_2026-06-27/episode_survival_train.jsonl.gz \
  --test-jsonl data/episode_survival_sft_v1_natural_2026-06-27/episode_survival_test.jsonl.gz \
  --eval-jsonl data/episode_survival_sft_v1_natural_2026-06-27/episode_survival_eval.jsonl.gz \
  --market-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz \
  --output-dir data/episode_survival_pairwise_v1_2026-06-27 \
  --min-utility-gap-pct 0.35 \
  --max-pairs-per-signal 3 \
  --max-rows-per-split 50000
```

## Dataset

Directory size: 6.3MB.

| Split | Rows | Choice A | Choice B | Chosen LONG | Chosen SHORT | Mean utility gap | Median utility gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 31,010 | 15,489 | 15,521 | 17,604 | 13,406 | 1.398% | 0.988% |
| Test | 15,447 | 7,754 | 7,693 | 8,765 | 6,682 | 1.122% | 0.874% |
| Eval | 3,257 | 1,655 | 1,602 | 1,759 | 1,498 | 1.141% | 0.916% |

Prompt additions versus binary SFT:

- candidate A/B setup and macro context;
- causal history at 12/48/144/576 bars:
  - return;
  - realized volatility;
  - range position;
  - drawdown from recent high.

## Pairwise logistic baseline

A no-dependency numpy logistic baseline parses only prompt fields and predicts A/B.

Commands:

```bash
.venv/bin/python -m training.eval_episode_survival_pairwise_baseline \
  --train-jsonl data/episode_survival_pairwise_v1_2026-06-27/episode_survival_pairwise_train.jsonl.gz \
  --eval-jsonl data/episode_survival_pairwise_v1_2026-06-27/episode_survival_pairwise_test.jsonl.gz \
  --output results/episode_survival_pairwise_baseline_train_to_test_2026-06-27/report.json \
  --epochs 350 --lr 0.08 --l2 0.001

.venv/bin/python -m training.eval_episode_survival_pairwise_baseline \
  --train-jsonl data/episode_survival_pairwise_v1_2026-06-27/episode_survival_pairwise_train.jsonl.gz \
  --eval-jsonl data/episode_survival_pairwise_v1_2026-06-27/episode_survival_pairwise_eval.jsonl.gz \
  --output results/episode_survival_pairwise_baseline_train_to_eval_2026-06-27/report.json \
  --epochs 350 --lr 0.08 --l2 0.001
```

Results:

| Eval split | Rows | Accuracy | Pred A rate | Target A rate |
| --- | ---: | ---: | ---: | ---: |
| Test 2024-2025 | 15,447 | 54.77% | 47.04% | 50.20% |
| Eval 2026-06 | 3,257 | 51.73% | 47.80% | 50.81% |

## Decision

1. Pairwise framing is better aligned with RLLM than binary threshold classification, but current features still carry only weak signal.
2. The task is at least learnable above random on 2024-2025 test, but transfer to 2026 is marginal.
3. Do not start full Gemma training yet unless treating it strictly as a PoC; expected trading lift is limited.
4. Next data iteration should strengthen causal history descriptors before SFT:
   - volatility compression duration;
   - trend/regime age;
   - prior realized adverse excursion proxies;
   - higher timeframe alignment;
   - candidate competition summary beyond pair A/B raw fields.

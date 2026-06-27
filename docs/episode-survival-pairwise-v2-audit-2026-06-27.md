# Episode survival pairwise v2 audit (2026-06-27)

## Purpose

Pairwise v1 was only weakly learnable. This pass keeps labels and pair construction fixed, but enriches the causal prompt descriptors before any Gemma training.

## Descriptor changes

Added to `causal_history`:

- range bps at 12/48/144/576 bars;
- volatility compression ratios:
  - `vol12_to_vol144`;
  - `vol48_to_vol576`;
- trend alignment score and stack label;
- SMA side/age for 48 and 144 bars;
- prior realized adverse-risk proxies from completed bars:
  - `prior_long_mae_proxy_12/48/144`;
  - `prior_short_mae_proxy_12/48/144`;
  - `tail_risk_max_12/48/144`.

Added `competition_context`:

- same side;
- same event type;
- horizon difference;
- A-B differences for risk, range, close quality, and wick fraction.

All additions are causal and computed from bars at or before the signal timestamp.

## Export/eval commands

```bash
.venv/bin/python -m training.export_episode_survival_pairwise_data \
  --train-jsonl data/episode_survival_sft_v1_natural_2026-06-27/episode_survival_train.jsonl.gz \
  --test-jsonl data/episode_survival_sft_v1_natural_2026-06-27/episode_survival_test.jsonl.gz \
  --eval-jsonl data/episode_survival_sft_v1_natural_2026-06-27/episode_survival_eval.jsonl.gz \
  --market-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz \
  --output-dir data/episode_survival_pairwise_v2_2026-06-27 \
  --min-utility-gap-pct 0.35 \
  --max-pairs-per-signal 3 \
  --max-rows-per-split 50000

.venv/bin/python -m training.eval_episode_survival_pairwise_baseline \
  --train-jsonl data/episode_survival_pairwise_v2_2026-06-27/episode_survival_pairwise_train.jsonl.gz \
  --eval-jsonl data/episode_survival_pairwise_v2_2026-06-27/episode_survival_pairwise_test.jsonl.gz \
  --output results/episode_survival_pairwise_v2_baseline_train_to_test_2026-06-27/report.json \
  --epochs 350 --lr 0.08 --l2 0.001

.venv/bin/python -m training.eval_episode_survival_pairwise_baseline \
  --train-jsonl data/episode_survival_pairwise_v2_2026-06-27/episode_survival_pairwise_train.jsonl.gz \
  --eval-jsonl data/episode_survival_pairwise_v2_2026-06-27/episode_survival_pairwise_eval.jsonl.gz \
  --output results/episode_survival_pairwise_v2_baseline_train_to_eval_2026-06-27/report.json \
  --epochs 350 --lr 0.08 --l2 0.001
```

## Dataset

Pair counts are unchanged from v1 because only descriptors changed:

| Split | Rows | Mean utility gap | Median utility gap |
| --- | ---: | ---: | ---: |
| Train | 31,010 | 1.398% | 0.988% |
| Test | 15,447 | 1.122% | 0.874% |
| Eval | 3,257 | 1.141% | 0.916% |

## Baseline comparison

| Version | Test accuracy | Eval accuracy | Features expanded |
| --- | ---: | ---: | ---: |
| Pairwise v1 | 54.77% | 51.73% | 137 |
| Pairwise v2 | 54.93% | 52.35% | 172 |

## Decision

1. Richer causal descriptors improve transfer slightly, but not enough for a trading edge by themselves.
2. The descriptor direction is valid; pairwise v2 is a better Gemma PoC dataset than binary survival v1/v2.
3. Expected Gemma gain should be treated as a PoC unless model accuracy clearly exceeds the logistic baseline on natural eval.
4. Next step can be a small Gemma LoRA/DPO-style PoC on pairwise v2, with strict stop criteria:
   - eval pairwise accuracy must beat 52.35%;
   - accepted/ranked candidate backtest must be checked before any RL stage.

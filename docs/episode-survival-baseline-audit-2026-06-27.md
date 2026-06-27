# Episode survival baseline audit (2026-06-27)

## Purpose

Before Gemma SFT, this audit checks whether the survival-filter SFT prompt fields are learnable without an LLM. A simple numpy logistic classifier parses only causal prompt fields and predicts `TRADE` vs `NO_TRADE`.

The threshold is selected on test only, then applied to 2026 eval. Accepted candidates are also strict-backtested by taking the highest-probability candidate per signal timestamp.

## Balanced train/test SFT baseline

Command:

```bash
.venv/bin/python -m training.eval_episode_survival_baseline \
  --train-jsonl data/episode_survival_sft_v1_2026-06-27/episode_survival_train.jsonl.gz \
  --test-jsonl data/episode_survival_sft_v1_2026-06-27/episode_survival_test.jsonl.gz \
  --eval-jsonl data/episode_survival_sft_v1_2026-06-27/episode_survival_eval.jsonl.gz \
  --market-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz \
  --output results/episode_survival_baseline_v1_2026-06-27/report.json \
  --epochs 350 --lr 0.06 --l2 0.001 \
  --threshold-metric utility \
  --min-test-predictions 80
```

Classification:

| Split | Precision | Recall | Predicted TRADE |
| --- | ---: | ---: | ---: |
| Train balanced | 0.546 | 0.576 | 10,552 |
| Test balanced | 0.573 | 0.535 | 9,322 |
| Eval natural | 0.333 | 0.524 | 3,971 |

Strict backtest of accepted candidates:

| Split | CAGR | Strict MDD | Ratio | Trades | p-value |
| --- | ---: | ---: | ---: | ---: | ---: |
| Test balanced | 48.21% | 17.72% | 2.72 | 528 | 0.0027 |
| Eval natural | -1.77% | 15.47% | -0.11 | 102 | 0.986 |

Balanced test overstates deployability because it does not match natural base rates.

## Natural split regeneration

To avoid threshold calibration against a 50/50 artificial test split, a natural dataset was exported:

```bash
.venv/bin/python -m training.export_episode_survival_sft_data \
  --input-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz \
  --output-dir data/episode_survival_sft_v1_natural_2026-06-27 \
  --train-start 2020-01-01 --train-end '2023-12-31 23:59:59' \
  --test-start 2024-01-01 --test-end '2025-12-31 23:59:59' \
  --eval-start 2026-01-01 --eval-end 2026-06-01 \
  --windows 576,2016,4032 \
  --horizons 72,144,288 \
  --event-types failed_breakdown_long,low_sweep_reclaim,reclaim_mid_from_below,failed_mid_loss_long,downtrend_pullback_reject,failed_mid_reclaim_short \
  --min-trade-net-pct 0.25 \
  --max-trade-mae-pct 2.0 \
  --min-mfe-to-mae 1.25 \
  --mae-penalty 0.2 \
  --max-negative-per-positive 999 \
  --max-rows-per-split 100000
```

Natural labels:

| Split | Rows | TRADE | NO_TRADE |
| --- | ---: | ---: | ---: |
| Train 2020-2023 | 76,956 | 20,890 | 56,066 |
| Test 2024-2025 | 41,274 | 11,976 | 29,298 |
| Eval 2026-06 | 8,304 | 2,520 | 5,784 |

## Natural train/test baseline

Output: `results/episode_survival_baseline_v1_natural_2026-06-27/report.json`

| Split | Precision | Recall | Predicted TRADE |
| --- | ---: | ---: | ---: |
| Train natural | 0.302 | 0.744 | 51,501 |
| Test natural | 0.339 | 0.700 | 24,764 |
| Eval natural | 0.330 | 0.696 | 5,320 |

Strict backtest:

| Split | CAGR | Strict MDD | Ratio | Trades | p-value |
| --- | ---: | ---: | ---: | ---: | ---: |
| Test natural | 9.69% | 25.66% | 0.38 | 502 | 0.412 |
| Eval natural | -3.44% | 16.39% | -0.21 | 105 | 0.977 |

## Balanced train + natural validation baseline

Output: `results/episode_survival_baseline_v1_baltrain_naturalval_2026-06-27/report.json`

| Split | Precision | Recall | Predicted TRADE |
| --- | ---: | ---: | ---: |
| Train balanced | 0.541 | 0.640 | 11,835 |
| Test natural | 0.345 | 0.594 | 20,637 |
| Eval natural | 0.335 | 0.590 | 4,441 |

Strict backtest:

| Split | CAGR | Strict MDD | Ratio | Trades | p-value |
| --- | ---: | ---: | ---: | ---: | ---: |
| Test natural | 2.31% | 28.62% | 0.08 | 485 | 0.761 |
| Eval natural | -8.26% | 15.67% | -0.53 | 102 | 0.858 |

## Decision

1. The current survival SFT v1 prompt is weakly learnable as a classifier but not useful enough as a profit/risk ranker.
2. Balanced validation is misleading for threshold choice. Future thresholding must use natural validation distribution.
3. Logistic baseline eval precision stays near 0.33, only slightly above natural eval base rate (~0.30). This is not enough to justify Gemma SFT as-is.
4. Before Gemma training, labels/features should be revised:
   - use utility quantile/ranking target instead of binary survival only;
   - include richer causal regime/history summaries, not only setup bar and macro z-scores;
   - evaluate by natural validation backtest, not balanced classification accuracy.
5. Gemma SFT can still be run as a PoC, but expected outcome is limited unless the dataset is improved.

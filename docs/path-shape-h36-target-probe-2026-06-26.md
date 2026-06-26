# Path-shape h36/t0.5/s0.6 target probe — 2026-06-26

## Objective

Test whether shorter path-shape labels (`max_hold=36`, `take_profit=0.5%`, `stop=0.6%`) make the text-token trader learnable under a strict train/val/eval protocol.

This is not a production model selection report. The target-echo runs are future-label upper bounds only.

## Data / split

Input SFT files:

- `data/economic_path_shape_trader_sft_h36_t0p5_s0p6_train_pa_micro_aug.jsonl`
- `data/economic_path_shape_trader_sft_h36_t0p5_s0p6_val_pa_micro_aug.jsonl`
- `data/economic_path_shape_trader_sft_h36_t0p5_s0p6_oos_pa_micro_aug.jsonl`

Market bars:

- `data/2023-01-01_2026-02-28_d2a88c0700504d6a5e15bc3839ad84b6.csv.gz`

Rows / labels:

| split | rows | LONG | SHORT | NO_TRADE |
|---|---:|---:|---:|---:|
| train | 2370 | 804 | 770 | 796 |
| val | 552 | 173 | 164 | 215 |
| eval | 535 | 156 | 190 | 189 |

## Upper bound sanity check: target echo

Target echo writes the label target directly into predictions. It verifies that the label definition itself encodes profitable future information, not that the model can learn it.

| split | CAGR | strict MDD | CAGR/MDD | trades | mean trade |
|---|---:|---:|---:|---:|---:|
| train | 1638.76% | 1.14% | 1440.44 | 1574 | 0.3927% |
| val | 1336.17% | 0.75% | 1778.55 | 337 | 0.3969% |
| eval | 1471.10% | 0.94% | 1569.63 | 346 | 0.3887% |

Interpretation: the label has a huge oracle upper bound. The blocker is learnability/stability from past-only text features, not the target payoff mechanics.

## Past-only token policy result

Artifact: `results/path_shape_token_policy_tte_h36_t0p5_s0p6_pa_micro_aug/report.json`

Val-selected raw token policy selected `side_mode=invert`, `prob_threshold=0.34`, `margin_threshold=0.0`.

| split | CAGR | strict MDD | CAGR/MDD | trades | mean trade | p approx |
|---|---:|---:|---:|---:|---:|---:|
| val | -36.11% | 21.88% | -1.65 | 293 | -0.0754% | 0.0076 |
| eval | -57.66% | 35.35% | -1.63 | 336 | n/a | n/a |

Interpretation: direct token classification is not merely weak; it is loss-making out of sample.

## Val-mined veto results

Artifacts:

- `results/path_shape_val_token_veto_h36_t0p5_s0p6_pa_micro_aug_exact_min16/report.json`
- `results/path_shape_val_token_veto_h36_t0p5_s0p6_pa_micro_aug_exact_min24/report.json`
- `results/path_shape_val_token_veto_h36_t0p5_s0p6_pa_micro_aug_semantic_fast_min16/report.json`
- `results/path_shape_val_token_veto_h36_t0p5_s0p6_pa_micro_aug_semantic_fast_min24/report.json`

| veto mode | min token trades | val CAGR/MDD | val trades | eval CAGR | eval strict MDD | eval CAGR/MDD | eval trades | eval mean trade | eval p approx |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exact | 16 | 8.07 | 64 | -4.23% | 5.37% | -0.79 | 51 | -0.0403% | 0.533 |
| exact | 24 | 9.34 | 105 | -11.92% | 6.47% | -1.84 | 92 | -0.0661% | 0.182 |
| semantic-fast | 16 | 2.18 | 119 | -24.58% | 14.16% | -1.74 | 113 | -0.1204% | 0.0085 |
| semantic-fast | 24 | 6.21 | 78 | -5.93% | 4.87% | -1.22 | 54 | -0.0541% | 0.391 |

Interpretation: validation vetoes overfit. Semantic folding reduces token specificity but does not make the edge durable.

## Token edge stability diagnostic

New diagnostic script:

```bash
PYTHONPATH=. .venv/bin/python training/path_shape_token_edge_stability.py \
  --train-jsonl data/economic_path_shape_trader_sft_h36_t0p5_s0p6_train_pa_micro_aug.jsonl \
  --val-jsonl data/economic_path_shape_trader_sft_h36_t0p5_s0p6_val_pa_micro_aug.jsonl \
  --eval-jsonl data/economic_path_shape_trader_sft_h36_t0p5_s0p6_oos_pa_micro_aug.jsonl \
  --output results/path_shape_token_edge_stability_h36_t0p5_s0p6_pa_micro_aug_semantic.json \
  --unit-mode semantic --exclude-regex '^recent=' --min-count 24 --top-n 30
```

Semantic unit agreement:

| pair | common tokens | nonzero common | same sign | opposite sign |
|---|---:|---:|---:|---:|
| train-val | 286 | 257 | 64.98% | 35.02% |
| train-eval | 284 | 269 | 29.00% | 71.00% |
| val-eval | 280 | 244 | 34.84% | 65.16% |

Exact token agreement:

| pair | common tokens | nonzero common | same sign | opposite sign |
|---|---:|---:|---:|---:|
| train-val | 554 | 502 | 60.36% | 39.64% |
| train-eval | 557 | 531 | 30.70% | 69.30% |
| val-eval | 542 | 481 | 38.88% | 61.12% |

Interpretation: the current PA/micro text tokens carry regime-local directionality. They partially align in train→val but mostly invert in eval. This explains why val-selected gates/vetoes look good and then fail.

## Decision

Do not spend Gemma/Gemma4 SFT cycles on this h36 token representation as-is. The bottleneck is not LLM capacity; it is unstable, regime-flipping supervision from current past-only tokens.

Next viable direction:

1. Keep the strict train/val/eval discipline.
2. Replace side-label imitation with regime-conditioned payoff/ranking labels: tokens should describe setup quality and expected path risk, not directly memorize LONG/SHORT labels.
3. Add explicit regime-change/context tokens before LLM SFT: rolling max/min location, distance-to-range edges, breakout/rejection state, higher-timeframe compression/expansion, and cross-market context from `../wave_trading` where available.
4. Require a cheap non-LLM/token baseline to show eval-positive edge before any expensive Gemma fine-tuning.

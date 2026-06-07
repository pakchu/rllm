# Stable trader Gemma4 SFT POC (2026-06-07)

## Purpose
Test whether Gemma4 can imitate the fold-stable trader labels exported from the stable baseline:
`h144 / target 1.8% / stop 1.5% / teacher_only min_n20 score>=0.0005`.

## Training
- Model alias: `gemma4-e4b` → `google/gemma-4-E4B-it`.
- Dataset: `data/stable_trader_policy_h144_t1p8_s1p5_train.jsonl`.
- Samples: 512 balanced.
- LoRA: r=16, alpha=32, dropout=0.05.
- Steps: 16.
- Runtime: 112.9s.
- Final train loss: 0.5297.
- Late token accuracy reached ~0.95 on training batches.
- Local adapter: `checkpoints/stable_trader_gemma4_e4b_h144_t1p8_s1p5_step16` (~745MB, not committed).

## Evaluation limitation
The first evaluator used sample-by-sample generation and was too slow for full 552+535 row evaluation. Full eval was stopped and POC evaluation used 128 sequential samples per split.

## Val128 action metrics
- Action accuracy: 37.50%.
- Risk accuracy: 16.41%.
- Exact accuracy: 8.59%.
- Target trades: 27/128.
- Predicted trades: 85/128.
- Side accuracy when target trade: 40.74%.

Val128 strict backtest:
- Trades: 69.
- Return: +4.15%.
- CAGR: +42.40%.
- Strict MDD: 3.99%.
- CAGR/MDD: 10.64.
- Mean trade: +0.0617%.
- p-value: 0.497.

## Eval128 action metrics
- Action accuracy: 61.72%.
- Risk accuracy: 7.81%.
- Exact accuracy: 4.69%.
- Target trades: 15/128.
- Predicted trades: 50/128.
- Side accuracy when target trade: 40.00%.

Eval128 strict backtest:
- Trades: 39.
- Return: -2.10%.
- CAGR: -16.86%.
- Strict MDD: 3.91%.
- CAGR/MDD: -4.31.
- Mean trade: -0.0527%.
- p-value: 0.581.

## Interpretation
Gemma4 did learn the output format and some action structure, but it over-trades relative to the stable baseline and fails on eval. The attractive Val128 result is not reliable because Eval128 immediately reverses.

## Decision
Reject this step16 checkpoint as a trading policy. Keep it as proof that the stable trader data is trainable. The next improvement should focus on:
1. balanced/no-trade-preserving sampling or loss weighting,
2. candidate log-prob classification instead of free generation,
3. batched evaluator so full val/eval can be run quickly,
4. checkpoint selection by full val backtest before a single fixed eval run.

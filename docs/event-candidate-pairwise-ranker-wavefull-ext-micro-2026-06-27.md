# Event candidate pairwise ranker baseline — wavefull ext+micro — 2026-06-27

## Purpose

After the focused Gemma categorical-label policy failed, this checks whether the existing same-signal pairwise
ranking structure can recover edge from the richer price-action + external + micro-path candidate data.

## Inputs

- Train candidates: `results/event_candidate_ranking_wavefull_ext_micro_c72_s2_train.jsonl`
- Eval candidates: `results/event_candidate_ranking_wavefull_ext_micro_c72_s2_eval.jsonl`
- Market: `data/2020-01-01_2026-06-01_btcusdt_futures_5m.csv.gz`
- Train period: 2020-01-01..2024-12-31, 13,780 candidate rows
- Eval period: 2025-01-01..2026-05-30, 3,824 candidate rows
- Validation window inside train: 2024-01-01..2024-12-31
- Feature surface: 94 numeric + 183 categorical, expanded to 374 features

## Pairwise ranker result

The ranker fit pairwise winner-loser comparisons within each signal and selected score quantile/full-margin on
2024 validation only. All validation candidates were weak/negative; the selected setting was simply the least bad:

- Selected: `q=0.90`, `full_margin=0.0`
- Validation: 102 trades, CAGR -19.25%, strict MDD 28.33%, CAGR/MDD -0.68, p≈0.173
- Eval: 110 trades, CAGR 3.03%, strict MDD 11.55%, CAGR/MDD 0.26, p≈0.707

## Direction inversion audit

The selected eval predictions were side-inverted as a diagnostic:

- Original eval: CAGR 3.03%, strict MDD 11.55%, CAGR/MDD 0.26
- Side-inverted eval: CAGR -11.45%, strict MDD 23.73%, CAGR/MDD -0.48

This does not look like a simple global side-inversion bug. The issue is that the selected ranker is weak and the
validation window is already negative.

## Decision

Do not promote this pairwise linear ranker. It is useful as evidence that same-signal relative ranking is the right
shape of problem, but the current linear ranker does not have enough stable edge. For RLLM, the better next step is
not more scalar gate tuning; it is exporting same-signal listwise prompts where the model must choose among
`LONG`, `SHORT`, and `NO_TRADE` using price-action context and realized utility labels for training only.

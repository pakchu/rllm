# Regime-Matched Pairwise Option Surface — 2026-06-30

## Motivation

The binary `EDGE/NO_EDGE` surface still made the LLM act like a global numeric threshold classifier. That mismatches LLM strengths. This experiment reframes the task as relative setup comparison:

- compare two candidates from the same month, side, hold, and family
- randomize whether the better setup is placed at A or B
- answer exactly one option token: `A` or `B`
- keep future reward out of the prompt; use it only to build pair labels

## Data

Source split:

- train candidates: `results/event_action_compressor_ranker_paext_rex_train_2022_2024.jsonl`
- eval candidates: `results/event_action_compressor_ranker_paext_rex_eval_2025_2026.jsonl`

Pairing rule:

- group key: `(month, side, hold_bars, family)`
- sort by realized utility inside each group
- pair top-vs-bottom, next-vs-next, up to 12 pairs/group
- require utility gap >= 0.006
- winner position randomized with seed 45

## Surfaces tested

| surface | train pairs | eval pairs | prompt chars mean | base acc eval1024 | pred A/B | mean chosen utility |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| verbose | 25,020 | 11,689 | 3,874 | 0.4873 | 345 / 679 | -0.00997 |
| compact | 25,020 | 11,689 | 2,490 | 0.5156 | 598 / 426 | -0.00835 |
| ultra_compact | 25,020 | 11,689 | 1,054 | 0.4648 | 486 / 538 | -0.01020 |

Reports:

- `results/event_candidate_regime_pairwise_option_paext_rex_2026-06-30/base_eval1024_random_report.json`
- `results/event_candidate_regime_pairwise_option_compact_paext_rex_2026-06-30/base_eval1024_random_report.json`
- `results/event_candidate_regime_pairwise_option_ultracompact_paext_rex_2026-06-30/base_eval1024_random_report.json`

## Interpretation

Compact relative tokens are the best prompt shape so far: they improve base accuracy from 48.7% to 51.6% and reduce the worst position bias. However, base model chosen utility remains negative and high-confidence margins do not reliably select profitable candidates.

This suggests:

1. LLM-friendly relative comparison helps.
2. Removing too much context hurts (`ultra_compact`).
3. Base Gemma is not enough; if continuing, train on compact pairwise option rows rather than verbose/ultra.
4. Before SFT, verify the pairwise oracle can be converted into profitable candidate selection; otherwise optimizing pairwise accuracy may not matter.

## Decision

Keep the compact surface as the current LLM pairwise surface. Do not train until an oracle conversion/backtest confirms that pair labels can drive a profitable candidate ranking pipeline.

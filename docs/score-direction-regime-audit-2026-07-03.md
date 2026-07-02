# Score-direction regime audit (2026-07-03)

## Why this exists

The clean pairwise family-choice dataset did not give the LLM a stable learnable
signal.  Simple prompt-visible rules already showed a split flip: higher
pre-fold score was weakly useful in 2025 but strongly wrong in 2026H1.  This
experiment decomposes the problem into a fold-level intermediate target:

- `HIGH_SCORE_WINS`: the clean target family was above the fold's pre-fold score median.
- `LOW_SCORE_WINS`: the clean target family was below that median.
- `ABSTAIN`: the clean target is abstain or absent from pre-fold options.

The goal is not to trade this classifier directly.  It is a leak-guarded probe
for whether market context can tell the LLM when pre-fold score rankings should
be trusted or inverted.

## Dataset construction

Script: `training/build_score_direction_regime_dataset.py`

Inputs:

- Selector report:
  `results/event_candidate_regime_family_selector_rex_core_abstain_scoreboard_1m_2023_2026h1_2026-07-02.json`
- Market features:
  `data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz`

Leakage guards:

- Market snapshots use only rows with `date < fold_start`.
- Target-fold diagnostics are used only for labels, not prompt features.
- Threshold audit fits rules on train only; test/eval are diagnostic.

Generated rows:

| split | rows | labels |
| --- | ---: | --- |
| train 2023-2024 | 24 | ABSTAIN 8, HIGH 13, LOW 3 |
| test 2025 | 12 | ABSTAIN 2, HIGH 10 |
| eval 2026H1 | 5 | ABSTAIN 1, HIGH 1, LOW 3 |

Each row includes a JSON prompt with pre-fold market-regime features and a
pre-fold family-scoreboard summary, plus a completion of the form
`{"direction_regime": "..."}`.

## Threshold audit

Script: `training/audit_score_direction_regime_thresholds.py`
Report: `results/score_direction_regime_threshold_audit_2026-07-03.json`

Top train/test-selected rule:

- `rex_8640_range_width_pct_last < 0.3400495 => HIGH_SCORE_WINS`
- train binary: 15/16 = 93.75%
- test binary: 10/10 = 100%
- eval binary: 1/4 = 25%

Several other rules show the same pattern: strong train/test fit because 2025 is
almost entirely `HIGH_SCORE_WINS`, then failure on 2026H1 where `LOW_SCORE_WINS`
dominates.

## Interpretation

This is evidence of a missing-regime / distribution-shift problem, not an LLM
capacity problem.  Training an LLM on 24 train rows with only 3 train LOW
examples and zero 2025 test LOW examples would mostly teach the model a brittle
HIGH prior.  The next useful step is to expand fold labels back before 2023
while preserving final splits, e.g. train 2020-2024, test 2025, eval 2026H1.

Do not select parameters on 2026H1.  Use it only as final eval until a new final
holdout is explicitly established.

## Expanded 2021-2026 label pass

After the initial audit, I regenerated the monthly selector from 2021-01-01 so
train has more historical regimes without touching the 2026H1 final eval for
selection:

- selector report:
  `results/event_candidate_regime_family_selector_rex_core_abstain_scoreboard_1m_2021_2026h1_2026-07-03.json`
- train 2021-2024: 48 rows, ABSTAIN 23, HIGH 20, LOW 5
- test 2025: 12 rows, ABSTAIN 3, HIGH 9, LOW 0
- eval 2026H1: 5 rows, ABSTAIN 3, HIGH 1, LOW 1

The extra history improved LOW examples from 3 to 5, but this is still too small
for reliable fine-tuning.  The train-only threshold audit still mostly learns a
HIGH prior because the 2025 test split has zero LOW examples; top rules show
train 92%, test 100%, eval 50% on only two binary eval rows.

Implementation note: the JSONL now keeps `target`/`completion` as JSON strings
for compatibility with `training/train_text_sft.py`, while preserving parsed
`label` for audits.

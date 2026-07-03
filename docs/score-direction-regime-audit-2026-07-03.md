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

## Gemma 4 E4B SFT probe

I ran one Gemma 4 E4B LoRA SFT probe on the expanded score-direction rows:

- adapter: `checkpoints/score_direction_regime_gemma4_sft_s16_len3072_2026-07-03`
- train stream: 96 balanced-oversampled rows (32 HIGH / 32 LOW / 32 ABSTAIN)
- max sequence length: 3072 tokens. A previous 2048-token run produced zero loss
  because the completion was truncated off the end of the prompt; the real
  prompt+completion length is about 2.7k tokens.
- training: 16 steps, effective batch 4, train loss fell from ~0.78 to ~0.07-0.15.

Generation evaluation:

| split | accuracy | confusion summary |
| --- | ---: | --- |
| test 2025 | 2/12 = 16.7% | mostly predicts LOW; misses HIGH-heavy 2025 |
| eval 2026H1 | 4/5 = 80.0% | gets LOW + ABSTAIN, misses the lone HIGH |

Candidate-logprob evaluation is worse because the adapter heavily ranks
`LOW_SCORE_WINS`: test 0/12, eval 1/5.  So generation and logprob disagree, and
the adapter is not usable as a selector yet.

Conclusion: the LLM can learn the output format only when the target tokens are
not truncated, but class-balanced oversampling over only 5 train LOW examples
creates a LOW-biased generator.  The immediate next improvement should not be
more steps; it should be a better prompt/target formulation and more real LOW
regime labels, or a binary high-vs-not-high router with separate abstain logic.

## Binary compact prompt probe

I then removed the mixed abstain class and built a smaller binary SFT surface:

- builder: `training/build_score_direction_binary_sft.py`
- target: `{"trust_score_rank": "HIGH"}` vs `{"trust_score_rank": "LOW"}`
- prompt: qualitative market buckets + pre-fold scoreboard, not the full 114-feature numeric JSON
- train 2021-2024: 25 binary rows (HIGH 20 / LOW 5)
- test 2025: 9 binary rows (HIGH 9 / LOW 0)
- eval 2026H1: 2 binary rows (HIGH 1 / LOW 1)

Gemma 4 E4B LoRA probes:

| adapter | train stream | test 2025 | eval 2026H1 | read |
| --- | --- | ---: | ---: | --- |
| `score_direction_binary_gemma4_sft_s16_imbal_2026-07-03` | natural 20/5 class prior | 7/9 = 77.8% | 1/2 = 50.0% | best so far; keeps HIGH prior but misses LOW eval |
| `score_direction_binary_gemma4_sft_s16_bal80_2026-07-03` | oversampled 40/40 | 5/9 = 55.6% | 1/2 = 50.0% | oversampling adds false LOW calls |

The compact binary prompt is a clear improvement over the three-class numeric
prompt on the HIGH-heavy 2025 test split, and it trains about 9x faster because
prompts are ~1.7k chars instead of ~4.5k.  It still does not solve the real
problem: there are only five real LOW train examples and no LOW examples in the
2025 test split, so LOW recall cannot be validated without either older labels,
additional assets/timeframes, or a different final holdout design.

## Label expansion with all candidate families

The core REX-only selector did not produce enough balanced LOW labels.  I ran a
wider selector without `--family-include`, keeping the same chronological split
and using 2026H1 only as eval:

- selector report:
  `results/event_candidate_regime_family_selector_allfamilies_scoreboard_1m_2021_2026h1_2026-07-03.json`
- train 2021-2024: ABSTAIN 28, HIGH 11, LOW 9
- test 2025: ABSTAIN 7, HIGH 2, LOW 3
- eval 2026H1: ABSTAIN 2, HIGH 3, LOW 0
- binary train rows: HIGH 11 / LOW 9
- binary test rows: HIGH 2 / LOW 3
- binary eval rows: HIGH 3 only

Gemma 4 E4B binary compact adapter:

- adapter: `checkpoints/score_direction_binary_allfamilies_gemma4_sft_s24_2026-07-03`
- train stream: natural 20 binary rows, no oversampling
- training: 24 steps, loss down to ~0.04-0.18 near the end
- test 2025 generation: 4/5 = 80.0% (HIGH 2/2, LOW 2/3)
- eval 2026H1 generation: 2/3 = 66.7% (all target HIGH, one false LOW)

This is the first LLM probe that has both train and test LOW examples and does
not collapse to a single side.  Caveat: the all-family selector itself has poor
trading PnL, so this is not yet a trading strategy.  It is evidence that the
LLM-shaped regime classifier becomes learnable when the candidate pool is broad
enough to create balanced direction labels.  Next work should transfer this
regime-router idea back into a profitable candidate subset instead of using the
all-family selected trades directly.

## Router-to-PnL bridge

I added `training/backtest_score_direction_router.py` to connect the LLM regime
router back to actual candidate events.  For each fold it takes the selector's
pre-fold scoreboard and chooses:

- `HIGH`: highest finite pre-fold score among visible options
- `LOW`: lowest finite pre-fold score among visible options

Then it recomputes that family's fold events from the original market data and
runs the same strict-MDD simulator.  This uses external route labels/predictions
only; target-fold diagnostics are not read by the backtester.

All-family 2025 results:

| route source | trades | CAGR | strict MDD | CAGR/MDD | note |
| --- | ---: | ---: | ---: | ---: | --- |
| always HIGH baseline | 258 | -32.4% | 35.6% | -0.91 | the raw all-family scoreboard is bad |
| oracle HIGH/LOW target | 114 | 2.5% | 12.0% | 0.21 | direction label improves but not enough |
| Gemma S24 prediction | 112 | 4.7% | 12.0% | 0.39 | slightly above oracle due one different fold |

All-family 2026H1 eval:

| route source | trades | CAGR | strict MDD | CAGR/MDD |
| --- | ---: | ---: | ---: | ---: |
| oracle HIGH/LOW target | 60 | -36.4% | 17.0% | -2.14 |
| Gemma S24 prediction | 61 | -33.1% | 15.3% | -2.16 |

Interpretation: the LLM route is now connected to real PnL and can reduce the
2025 damage relative to always-high, but the widened all-family pool is not a
profitable trading surface.  The useful finding is structural: use LLM as a
regime router over a candidate pool, but the pool itself must be filtered to
families with positive out-of-sample expectancy before routing.

## Rejected pool: orderflow + vol-compression only

I tested a narrower pool suggested by the 2025 all-family router selections:
`orderflow_*` plus `vol_compression_breakout`.

- selector report:
  `results/event_candidate_regime_family_selector_flow_vol_scoreboard_1m_2021_2026h1_2026-07-03.json`
- final stitched selector result: CAGR -14.7%, strict MDD 47.8%, 342 trades

This pool is not worth LLM routing yet.  It contains many high-turnover negative
expectancy folds and worsens the base trading surface.  Keep the all-family
label expansion as a learning probe, but do not promote this orderflow/vol pool
as a candidate trading strategy.

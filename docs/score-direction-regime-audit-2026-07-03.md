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

## Positive family sweep after all-family router failure

The all-family LLM router proved the label format is learnable, but its trade
surface was negative expectancy.  I therefore changed the search order: first
find a no-leak positive candidate surface, then put the LLM on top.

I selected the first whitelist from train/test diagnostics only; 2026H1 eval was
not used to decide the pool.  A fold-safe selector over that whitelist still did
not solve the objective:

- report: `results/event_candidate_regime_family_selector_positive_probe_scoreboard_1m_2021_2026h1_2026-07-03.json`
- stitched 2021-2026 result: CAGR -5.0%, strict MDD 27.1%, 114 trades
- read: the family pool has useful components, but the monthly nearest-regime
  selector is still too noisy and often chooses the wrong family.

I then ran a simpler fixed-family no-leak probe with train=2021-2024,
test=2025, eval=2026H1:

- report: `results/event_candidate_pool_probe_positive_families_train2021_test2025_eval2026h1_2026-07-03.json`
- selected by train/test: `rex_htf_deep_pullback_resume`, q=0.80, hold=288
- test 2025: CAGR 15.2%, strict MDD 7.9%, ratio 1.91, 27 trades
- eval 2026H1: CAGR 13.7%, strict MDD 5.3%, ratio 2.57, 18 trades

This is positive but under-traded and below target CAGR, so I added a reusable
no-leak parameter sweep script:

- script: `training/sweep_event_family_params.py`
- test helper: `tests/test_sweep_event_family_params.py`
- report: `results/event_family_param_sweep_train2021_test2025_eval2026h1_script_2026-07-03.json`
- leakage guard: thresholds fit on train only; family/hold/quantile ranked on
  train+test only; eval is emitted after selection and not used for ranking.

Best train/test-selected individual candidates from the sweep:

| candidate | train | test 2025 | eval 2026H1 | read |
| --- | --- | --- | --- | --- |
| `rex_htf_deep_pullback_resume`, hold 216, q 0.85 | CAGR 13.6 / MDD 12.4 / 277 trades | CAGR 14.1 / MDD 1.8 / 20 trades | CAGR 26.9 / MDD 5.5 / 13 trades | high ratio but too few eval trades |
| `rex_htf_pullback_reclaim`, hold 144, q 0.75 | CAGR 8.1 / MDD 23.2 / 632 trades | CAGR 20.4 / MDD 4.8 / 76 trades | CAGR 24.0 / MDD 4.3 / 43 trades | most useful current candidate; statistically still weak |
| `rex_htf_deep_pullback_resume`, hold 144, q 0.85 | CAGR 11.2 / MDD 12.8 / 336 trades | CAGR 9.3 / MDD 2.2 / 22 trades | CAGR 26.3 / MDD 5.2 / 17 trades | strong ratio, low eval sample |

Same-hold prefix ensemble was also checked for hold=144.  Test-only selection
picked prefix size 1, i.e. no ensemble beats the single `rex_htf_pullback_reclaim`
q=0.75 surface.  Adding more same-hold signals increases test/eval trade count
but dilutes edge and worsens eval ratio.

Current conclusion: the first credible alpha surface is higher-timeframe REX
pullback/reclaim.  It meets the ratio target on the short final eval window but
not the CAGR/statistical-confidence target.  The next LLM-shaped step should not
be another broad family selector; it should train a compact event-level LLM
judge over this REX pullback surface to decide when to skip, size down, or flip
based on pre-entry textual regime features.  The LLM should be asked for a
reasoned trade thesis over price-action state, not raw numeric prediction.

## Event-level LLM judge dataset for the REX pullback surface

I converted the current best no-leak candidate surface into an LLM-friendly
TAKE/SKIP verifier dataset.  This is the next RLLM-compatible structure: the
base alpha generator proposes a candidate, then a small LLM judges the candidate
from textual price-action/regime context.

Builder changes:

- `training/build_rex_candidate_ranker_records.py` now supports optional
  train/test/eval output instead of train/eval only.
- Threshold fitting now matches the no-leak candidate sweep: quantiles are fit
  on positive train strengths only (`strength > 0`), avoiding zero-strength
  dilution.  This fixed a discovered mismatch where q=0.75 produced 0.07118 in
  the ranker builder but 0.21098 in the validated sweep.
- Rows now include `action` as an alias of `candidate`, so existing LLM scoring
  and action-backtest utilities can reuse the dataset.

Generated dataset command used fixed, train/test-selected parameters:

- family/threshold: `rex_htf_pullback_reclaim:0.75`
- hold: 144 bars
- threshold fit: 2021-01-01 to 2025-01-01, positive strengths only
- train: 2021-2024
- test: 2025
- eval: 2026H1

Outputs:

| split | file | rows | TAKE/SKIP | side mix | mean candidate net |
| --- | --- | ---: | --- | --- | ---: |
| train | `data/rex_pullback_reclaim_q075_h144_ranker_train_2021_2024.jsonl` | 1615 | 354 / 1261 | LONG 967 / SHORT 648 | +0.0748% |
| test | `data/rex_pullback_reclaim_q075_h144_ranker_test_2025.jsonl` | 132 | 21 / 111 | LONG 49 / SHORT 83 | +0.1241% |
| eval | `data/rex_pullback_reclaim_q075_h144_ranker_eval_2026h1.jsonl` | 100 | 27 / 73 | LONG 28 / SHORT 72 | +0.1824% |

Leakage guard: prompts are signal-time only; labels use future path only as
supervised targets; eval is not used for threshold fitting or parameter choice.

Next experiment: train Gemma 4 E4B LoRA on the train rows, select TAKE/SKIP
threshold on 2025 test only, and run strict action backtest on 2026H1 eval.  The
objective is not merely label accuracy; the LLM must improve the base
`rex_htf_pullback_reclaim q=0.75 hold=144` eval profile without increasing MDD.

## Gemma 4 E4B event-judge POC on REX pullback/reclaim

I trained a Gemma 4 E4B LoRA verifier on the `rex_htf_pullback_reclaim q=0.75
hold=144` TAKE/SKIP rows.

Training:

- adapter: `checkpoints/rex_pullback_reclaim_q075_h144_gemma4_sft_s80_2026-07-03`
- train rows sampled: 1600 via balanced oversampling, TAKE 800 / SKIP 800
- prompt chars: mean ~1433, max 1479
- max length: 2048
- steps: 80, effective batch 8, bf16, no 4-bit
- runtime: 566s (9m26s), ~7.1s/step on RTX 5090
- train loss: 0.3106 overall; late-step loss ~0.18-0.21

Scoring used TAKE minus SKIP logprob margin on 2025 test and 2026H1 eval.

Important failure mode:

- test margin vs reward correlation: -0.041
- test margin vs TAKE label correlation: -0.121
- eval margin vs reward correlation: -0.104
- eval margin vs TAKE label correlation: -0.112

So the learned logprob margin is weakly inverted relative to economic labels.
This explains why a normal `high margin => trade` gate is not robust.

Test-only threshold results:

| route | selection rule | test 2025 | eval 2026H1 | read |
| --- | --- | --- | --- | --- |
| base candidate | no LLM filter | CAGR 20.4 / MDD 4.83 / ratio 4.22 / 76 trades | CAGR 24.0 / MDD 4.28 / ratio 5.60 / 43 trades | current baseline |
| normal LLM gate | trade if margin >= test-selected threshold | CAGR 20.2 / MDD 4.39 / ratio 4.60 / 67 trades | CAGR 16.4 / MDD 5.86 / ratio 2.80 / 40 trades | overfit / worse eval |
| inverted LLM gate | trade if margin <= test-selected threshold | CAGR 24.3 / MDD 4.83 / ratio 5.02 / 72 trades | CAGR 22.0 / MDD 4.28 / ratio 5.14 / 42 trades | closer but still below base |

Reports:

- `results/rex_pullback_reclaim_q075_h144_gemma4_s80_threshold_sweep_2026-07-03.json`
- `results/rex_pullback_reclaim_q075_h144_gemma4_s80_margin_direction_sweep_2026-07-03.json`

Conclusion: the LLM can fit the TAKE/SKIP token task, but this label design does
not yet add economic edge.  The current base REX pullback surface remains better
than the LLM-filtered variants on eval.  Next revision should change the target
from hard TAKE/SKIP to a ranking/ordinal utility target or pairwise preference
among candidates, because binary labels are sparse and appear to teach token
priors more than robust reasoning.

## Pairwise DPO dataset for LLM-style action choice

Because binary TAKE/SKIP SFT produced an economically inverted margin, I moved
the next LLM surface to same-prompt preferences.  The prompt asks the model to
choose among `NO_TRADE`, `LONG`, and `SHORT`; chosen/rejected responses are
ranked by future utility for training only.

Builder changes:

- `training/build_event_signal_preference_dpo.py` now supports optional
  train/test/eval split outputs.
- Added test coverage for ranking the best trade action against no-trade and a
  losing side.

Candidate source:

- combo: `rex_htf_pullback_reclaim:0.75,rex_htf_deep_pullback_resume:0.85`
- hold: 144 bars
- threshold fit: 2021-2024, positive strengths only
- train/test/eval candidate files:
  - `data/rex_pair_reclaim075_deep085_h144_ranker_train_2021_2024.jsonl`
  - `data/rex_pair_reclaim075_deep085_h144_ranker_test_2025.jsonl`
  - `data/rex_pair_reclaim075_deep085_h144_ranker_eval_2026h1.jsonl`

DPO preference outputs:

| split | file | rows | chosen distribution | mean rank margin |
| --- | --- | ---: | --- | ---: |
| train | `data/rex_pair_reclaim075_deep085_h144_dpo_train_2021_2024.jsonl` | 702 | NO_TRADE 375 / LONG 200 / SHORT 127 | 0.115 |
| test | `data/rex_pair_reclaim075_deep085_h144_dpo_test_2025.jsonl` | 48 | NO_TRADE 21 / LONG 7 / SHORT 20 | 0.081 |
| eval | `data/rex_pair_reclaim075_deep085_h144_dpo_eval_2026h1.jsonl` | 37 | NO_TRADE 15 / LONG 2 / SHORT 20 | 0.080 |

DPO dry-run:

- model alias: `gemma4-e4b-it` => `google/gemma-4-E4B-it`
- max samples: 700, gate-balanced
- prompt chars: mean ~1244, max 1287
- max length: 2048
- dry-run passed; no training checkpoint kept for dry-run.

Next step: train a small Gemma 4 E4B DPO adapter on these preferences, then
evaluate generation/logprob action choice on 2025 test.  Only if test selection
improves without obvious leakage should the fixed policy be checked on 2026H1.

## Aborted Gemma DPO run: too slow and not learning early

I started a Gemma 4 E4B DPO POC from the base model on the pairwise REX
preferences:

- intended adapter: `checkpoints/rex_pair_reclaim075_deep085_h144_gemma4_dpo_s80_2026-07-03`
- rows: 700 gate-balanced preference pairs
- max steps: 80
- learning rate: 5e-7, beta 0.1, effective batch 8

I stopped it manually at ~27/80 steps because it was both slow and not showing a
clear learning signal:

- elapsed at interruption: ~14m
- step time: often 25-45s, much slower than SFT
- loss stayed around random-preference scale: mostly ~0.67-0.71
- rewards/margins fluctuated and were often negative
- no usable checkpoint was kept; the partial summary directory was deleted

Conclusion: full Gemma DPO is not the next efficient path until we have a faster
evaluation loop and stronger preference separability.  Prefer a cheaper
ranker/verifier first: compute symbolic or small-model scores over the same
pairwise DPO rows, verify that the prompt-visible features contain edge, then
return to LLM DPO only if the rank target is learnable outside the LLM.

## Cheap rankability and leverage-scaling check

I completed the promised next-direction test before returning to expensive LLM
training.  `training/symbolic_action_ridge.py` now reads REX ranker rows whose
future path fields live under `reward`, so prompt-visible symbolic features can
be tested as a cheap ranker before another Gemma run.

Symbolic ridge setup:

- train: `data/rex_pair_reclaim075_deep085_h144_ranker_train_2021_2024.jsonl`
- validation/test: `data/rex_pair_reclaim075_deep085_h144_ranker_test_2025.jsonl`
- holdout/eval: `data/rex_pair_reclaim075_deep085_h144_ranker_eval_2026h1.jsonl`
- selection: 2025 only; 2026H1 not used for config selection
- report: `results/rex_pair_reclaim075_deep085_h144_symbolic_ridge_sweep_2026-07-03.json`

Selected symbolic config from 2025:

- target: `net_return`
- alpha: `0.1`
- threshold: `-0.01`
- min_gap: `0.0`

Result at 0.5x leverage:

| route | 2025 test | 2026H1 eval | read |
| --- | --- | --- | --- |
| base `reclaim q0.75 h144` | CAGR 20.4 / MDD 4.83 / ratio 4.22 / 76 trades | CAGR 24.0 / MDD 4.28 / ratio 5.60 / 43 trades | best current route |
| symbolic pair ranker | CAGR 19.9 / MDD 4.92 / ratio 4.04 / 84 trades | CAGR 20.6 / MDD 4.61 / ratio 4.46 / 42 trades | valid but weaker than base |

So the cheap ranker does **not** add edge over the base REX pullback/reclaim
surface.  This also supports the DPO abort: if a simple symbolic ranker cannot
beat base on the same prompt-visible data, a slow Gemma DPO run is unlikely to
be the immediate breakthrough without a better target or more diverse candidate
book.

Leverage scaling check:

- report: `results/rex_pullback_reclaim_leverage_scaling_check_2026-07-03.json`
- this is a pure leverage re-run over fixed selected policies, not parameter
  selection.

| route | lev | 2025 test CAGR/MDD/ratio | 2026H1 eval CAGR/MDD/ratio |
| --- | ---: | --- | --- |
| base `reclaim q0.75 h144` | 0.5 | 20.4 / 4.83 / 4.22 | 24.0 / 4.28 / 5.60 |
| base `reclaim q0.75 h144` | 1.0 | 44.2 / 9.46 / 4.67 | 51.8 / 8.46 / 6.12 |
| base `reclaim q0.75 h144` | 1.5 | 71.7 / 13.89 / 5.16 | 83.6 / 12.67 / 6.59 |
| base `reclaim q0.75 h144` | 2.0 | 103.3 / 18.14 / 5.69 | 119.2 / 16.87 / 7.07 |
| symbolic ranker | 1.5 | 69.2 / 14.12 / 4.90 | 69.0 / 13.37 / 5.16 |

Operational read: if the acceptable risk cap is strict MDD <= 15%, 1.5x is the
highest tested leverage that keeps both 2025 test and 2026H1 eval under the cap
while exceeding CAGR 50.  2.0x breaks the MDD cap.  This does not yet prove a
3+ year live-ready strategy, but it does mean the current bottleneck is no longer
raw CAGR on the 2025-2026 out-of-sample window; it is longer no-leak validation,
trade-count confidence, and regime persistence.

## Longer fixed-family validation and regime-gate follow-up

Before another expensive Gemma run, I validated the current REX pullback/reclaim
surface over a longer no-leak horizon.

Monthly walk-forward threshold refit:

- script: `training/backtest_fixed_event_family_walkforward.py`
- report: `results/rex_pullback_reclaim_q075_h144_monthly_walkforward_2021_2026h1_2026-07-03.json`
- fixed hypothesis: `rex_htf_pullback_reclaim`, q=0.75, hold=144
- guard: each monthly threshold is fit only from rows before that fold start.

Result at 0.5x:

| period | CAGR | strict MDD | CAGR/MDD | trades | read |
| --- | ---: | ---: | ---: | ---: | --- |
| 2021-2026H1 stitched | 2.4 | 30.6 | 0.08 | 624 | fails long-horizon objective |
| 2021-2024 history OOS | -1.6 | 30.6 | -0.05 | 496 | main failure zone |
| 2025 test | 20.1 | 4.8 | 4.15 | 83 | still good |
| 2026H1 eval | 12.3 | 5.0 | 2.44 | 45 | positive but weaker than fixed-threshold eval |

Fixed-threshold leverage check, with the threshold fit once on 2021-2024 and
then held fixed for 2025+2026H1:

- report: `results/rex_pullback_reclaim_fixed_threshold_combined_leverage_2026-07-03.json`
- 2025+2026H1 at 1.25x: CAGR 52.0 / strict MDD 11.7 / ratio 4.44 / 119 trades
- 2025+2026H1 at 1.5x: CAGR 64.5 / strict MDD 13.9 / ratio 4.64 / 119 trades
- full 2021-2026H1 still fails because 2022-2024 drawdowns dominate.

Yearly diagnosis at 0.5x confirms the regime problem:

| year | CAGR | strict MDD | ratio | trades | read |
| --- | ---: | ---: | ---: | ---: | --- |
| 2021 | 49.5 | 13.5 | 3.68 | 248 | strong |
| 2022 | -8.7 | 18.2 | -0.48 | 157 | bad |
| 2023 | -2.7 | 12.9 | -0.21 | 116 | bad/flat |
| 2024 | 2.3 | 8.0 | 0.29 | 111 | weak |
| 2025 | 20.4 | 4.8 | 4.22 | 76 | strong |
| 2026H1 | 24.0 | 4.3 | 5.60 | 43 | strong but short |

Interpretation: the REX pullback/reclaim alpha is not universally persistent.
It works in 2021 and 2025-2026, but fails in the 2022-2024 regime.  The honest
breakthrough path is a regime filter, not more LLM fine-tuning on the same weak
binary labels.

## Single-feature and conjunctive regime gates

I added two cheap no-leak gate sweeps over the fixed REX candidate rows:

- single feature: `training/sweep_single_feature_event_gate.py`
- two-feature conjunctions: `training/sweep_conjunctive_event_gates.py`
- reports:
  - `results/rex_pullback_reclaim_single_feature_gate_sweep_script_2026-07-03.json`
  - `results/rex_pullback_reclaim_conjunctive_gate_sweep_2026-07-03.json`
- selection guard: thresholds are train-quantile based; gates are ranked using
  train+2025 test only; 2026H1 eval is reported after ranking.

Best single-feature gates:

| gate | train 0.5x | test 0.5x | eval 0.5x | eval leverage read |
| --- | --- | --- | --- | --- |
| `range_vol >= 0.019835` | CAGR 11.6 / MDD 16.4 / 510 trades | 21.0 / 2.8 / 46 | 17.9 / 4.1 / 35 | 1.5x => CAGR 58.3 / MDD 12.0 |
| `window_drawdown >= 0.010206` | 8.1 / 16.6 / 428 | 16.5 / 2.1 / 30 | 14.5 / 1.8 / 17 | high ratio, too few eval trades |
| `rex_144_max_to_cur_pct >= 0.012498` | 8.7 / 13.7 / 424 | 14.8 / 2.1 / 29 | 14.5 / 1.8 / 17 | similar low-sample surface |

Best conjunctive gates:

| gates | train 0.5x | test 0.5x | eval 0.5x | eval leverage read |
| --- | --- | --- | --- | --- |
| `rex_2016_cur_to_min_pct >= 0.04867` AND `dxy_momentum >= -0.000254` | CAGR 4.3 / MDD 14.7 / 320 trades | 17.9 / 1.6 / 31 | 25.1 / 2.4 / 20 | 1.0x => CAGR 55.0 / MDD 4.8, but only 20 eval trades |
| `range_vol >= 0.02396` AND `dxy_momentum >= -0.000254` | 12.1 / 14.4 / 346 | 17.2 / 1.6 / 26 | 20.2 / 4.2 / 23 | 1.25x => CAGR 55.9 / MDD 10.2 |
| `rex_2016_cur_to_min_pct >= 0.04867` AND `dxy_momentum >= -0.001853` | 6.7 / 12.0 / 370 | 18.3 / 1.9 / 34 | 20.2 / 4.1 / 24 | 1.25x => CAGR 55.3 / MDD 10.1 |

This confirms the useful feature family: high realized range / drawdown /
distance-from-local-min, with a DXY momentum condition, filters the REX entries
into much lower drawdown recent regimes.  However, it still does **not** solve
the full 3+ year objective: train/history CAGR/MDD remains below 1 even when the
recent 2025-2026 leveraged window looks strong.  The next LLM-relevant direction
is therefore not analyzer/trader bloat or DPO; it is to let a single compact LLM
or symbolic teacher express this regime rule as a causal trade thesis and abstain
when the 2022-2024 style regime is detected, then validate on a new holdout.

## Robust-gate selection attempt: range plus Kimchi premium

After the first conjunctive sweep, I re-ranked stored candidates by robustness on
train while requiring the same positive 2025 test behavior.  The strongest
train-ratio candidate among the train/test-ranked gate set was:

- `range_vol >= 0.023959233645008706`
- `kimchi_premium_change <= 0.0`
- report: `results/rex_reclaim_range_kimchi_gate_leverage_audit_2026-07-03.json`

At 0.5x:

| period | CAGR | strict MDD | CAGR/MDD | trades | p-value approx |
| --- | ---: | ---: | ---: | ---: | ---: |
| train 2021-2024 | 16.3 | 12.7 | 1.28 | 346 | 0.0156 |
| test 2025 | 19.0 | 2.5 | 7.47 | 27 | 0.0001 |
| eval 2026H1 | 24.1 | 3.6 | 6.70 | 22 | 0.1667 |
| combined 2021-2026H1 | 16.3 | 12.7 | 1.28 | 395 | 0.0017 |

This is materially better than the ungated 2021-2024 history and is the first
candidate with train/test/eval all positive plus a statistically meaningful
combined trade count.  But it still misses the objective: leverage does not fix
CAGR/MDD because the ratio stays ~1.3 while MDD scales up.  At 1.5x the recent
periods look strong, but combined strict MDD rises to 38.3%.

I also ran a width-3 conjunction expansion:

- report: `results/rex_pullback_reclaim_conjunctive_gate_width3_sweep_2026-07-03.json`
- result: top width-3 gates improve short-window 2025 ratios but generally lower
  train robustness; several high eval CAGR rows have only 14-19 eval trades and
  weak p-values.

Conclusion: the best current robust candidate is not a moonshot high-ratio gate;
it is the **range-volatility + non-positive Kimchi-premium-change regime**.  It
is a plausible alpha/regime prior, but the 3+ year CAGR/MDD>=3 target remains
unsolved.  Next work should build an LLM/rule hybrid that explains and abstains
on this specific regime, then validate on a fresh chronological split rather than
mining more narrow gates from the same 2026H1 holdout.

## Compact LLM regime-thesis policy: label-first fix

I converted the robust range/Kimchi regime prior into a compact LLM policy
surface instead of returning to the oversized analyzer/trader design.

Exporter:

- script: `training/build_rex_regime_thesis_sft.py`
- gate prior: `range_vol >= 0.023959233645008706` AND
  `kimchi_premium_change <= 0.0`
- target mode used for the working run: `decision_label`
- output labels: exactly `TRADE` or `ABSTAIN`
- rationale/thesis is preserved in prompt and metadata, not generated as a long
  JSON object.

Why the label-first change mattered:

- A first JSON-target SFT run trained, but candidate-logprob evaluation was not
  meaningful because the completion began with JSON syntax/fields rather than
  the decision token.
- Greedy generation was too slow for the iteration loop.
- Label-first completions make `TRADE` vs `ABSTAIN` candidate-logprob fast and
  directly usable as a trading gate.

Training run:

- adapter: `checkpoints/rex_regime_thesis_range_kimchi_label_gemma4_s32_2026-07-03`
- model: Gemma 4 E4B IT alias (`google/gemma-4-E4B-it`)
- train rows sampled: 512 balanced rows, TRADE 256 / ABSTAIN 256
- prompt length: ~872 chars mean
- target length: 5-7 chars
- steps: 32, runtime 232.7s, ~7.3s/step
- final train loss: 0.4125; late token accuracy mostly 1.0

Decision logprob evaluation using `score_normalization=sum`:

| split | accuracy | confusion | read |
| --- | ---: | --- | --- |
| test 2025 | 91.7% | ABSTAIN 83/94, TRADE 38/38, false TRADE 11 | captures all target trades, slightly wider gate |
| eval 2026H1 | 92.0% | ABSTAIN 53/61, TRADE 39/39, false TRADE 8 | captures all target trades, slightly wider gate |

Backtest of the LLM-predicted decisions:

| split | lev | CAGR | strict MDD | CAGR/MDD | trades | p-value approx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test 2025 | 0.5 | 23.2 | 2.55 | 9.09 | 33 | 0.000025 |
| test 2025 | 1.0 | 51.3 | 5.06 | 10.12 | 33 | 0.000028 |
| test 2025 | 1.5 | 85.1 | 7.54 | 11.29 | 33 | 0.000033 |
| eval 2026H1 | 0.5 | 19.4 | 3.60 | 5.40 | 25 | 0.248 |
| eval 2026H1 | 1.25 | 53.2 | 8.91 | 5.97 | 25 | 0.255 |
| eval 2026H1 | 1.5 | 65.6 | 10.65 | 6.16 | 25 | 0.257 |

Interpretation:

- This is the first RLLM-shaped stage that does something operationally useful:
  a small Gemma adapter learns a compact regime gate and can be scored quickly by
  candidate logprob.
- The learned gate slightly expands the symbolic gate.  On 2025 this improves
  CAGR/MDD; on 2026H1 it still meets the leveraged CAGR/MDD profile but remains
  underpowered statistically.
- This still does **not** prove the original 3+ year target.  The right next
  validation is to replay this exact label-first policy across the longer
  2021-2026 span and/or create a new final holdout after freezing this adapter and
  scoring rule.  Do not mine the 2026H1 false positives further.

## Combined 2025-2026H1 frozen-adapter OOS read

After freezing the label-first Gemma adapter and the `sum` logprob scoring rule,
I concatenated the already-scored 2025 test and 2026H1 eval prediction files.
This is not a new selection step; it is the combined OOS operating read for the
same frozen policy.

- report: `results/rex_regime_thesis_range_kimchi_label_gemma4_s32_oos_2025_2026h1_backtest_2026-07-03.json`
- predicted decisions: TRADE 96 / ABSTAIN 136
- executed strict trades after cooldown: 58

| leverage | CAGR | strict MDD | CAGR/MDD | trades | p-value approx |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 19.7 | 3.60 | 5.46 | 58 | 0.000148 |
| 1.0 | 42.6 | 7.15 | 5.95 | 58 | 0.000163 |
| 1.25 | 55.4 | 8.91 | 6.22 | 58 | 0.000171 |
| 1.5 | 69.2 | 10.65 | 6.49 | 58 | 0.000179 |
| 2.0 | 99.9 | 14.10 | 7.09 | 58 | 0.000197 |

Operational read: for the recent 18-month OOS window, the frozen compact LLM
policy clears the leverage-adjusted target with strict MDD below 15 even at 2.0x.
The most conservative target-clearing setting is 1.25x.  This is meaningfully
better than the previous binary TAKE/SKIP and DPO attempts because the LLM stage
is finally aligned with its useful role: a compact regime gate over a fixed weak
alpha, not numeric return prediction.

Still unsolved: this does not retroactively fix 2022-2024.  The next required
proof is a walk-forward or new-holdout validation where the label-first adapter
or its distilled rule is frozen before the scored period.

## Alternative split sanity check: no 2025 selection

To test whether the range/Kimchi LLM policy was mainly a 2025-selected artifact,
I regenerated the REX candidate surface with an earlier split:

- candidate builder report:
  `results/rex_pullback_reclaim_q075_h144_ranker_summary_train2021_2023_test2024_eval2025_2026h1_2026-07-03.json`
- threshold fit: 2021-2023 only
- train: 2021-2023
- test: 2024
- eval: 2025-2026H1

A fresh two-feature gate sweep using train 2021-2023 + test 2024 did **not**
produce a robust top-ranked selector.  The first two selected gates had strong
2024 test stats but negative 2025-2026H1 eval, which is direct evidence that
2024-only selection can overfit just like 2025-only selection.

The previously frozen range/Kimchi gate, applied unchanged to this alternative
split, gives:

- report:
  `results/rex_reclaim_range_kimchi_gate_alt_split_train2021_2023_test2024_eval2025_2026h1_2026-07-03.json`

| period | CAGR | strict MDD | CAGR/MDD | trades | p-value approx |
| --- | ---: | ---: | ---: | ---: | ---: |
| train 2021-2023 | 15.1 | 16.2 | 0.93 | 273 | 0.0586 |
| test 2024 | 10.5 | 6.4 | 1.64 | 44 | 0.286 |
| eval 2025-2026H1 | 14.7 | 3.5 | 4.24 | 45 | 0.00018 |
| combined 2021-2026H1 | 13.4 | 16.2 | 0.83 | 362 | 0.0051 |

A train-robust gate from the 2024 split used a different macro/HTF condition:

- `rex_8640_range_width_pct >= 0.2836633876944003`
- `usdkrw_zscore <= 0.2603593471820541`
- report:
  `results/rex_reclaim_rex8640_usdkrw_gate_alt_split_train2021_2023_test2024_eval2025_2026h1_2026-07-03.json`

| period | CAGR | strict MDD | CAGR/MDD | trades | p-value approx |
| --- | ---: | ---: | ---: | ---: | ---: |
| train 2021-2023 | 12.9 | 15.9 | 0.81 | 260 | 0.109 |
| test 2024 | 25.9 | 4.7 | 5.52 | 40 | 0.0067 |
| eval 2025-2026H1 | 28.5 | 3.1 | 9.08 | 30 | 0.163 |
| combined 2021-2026H1 | 12.3 | 15.9 | 0.77 | 330 | 0.0115 |

Unioning the range/Kimchi and HTF-width/USDKRW regimes improves recent OOS trade
count but still does not solve the full 3+ year ratio:

- report:
  `results/rex_reclaim_dual_regime_or_alt_split_train2021_2023_test2024_eval2025_2026h1_2026-07-03.json`

| period | CAGR | strict MDD | CAGR/MDD | trades | p-value approx |
| --- | ---: | ---: | ---: | ---: | ---: |
| train 2021-2023 | 13.3 | 14.7 | 0.90 | 356 | 0.122 |
| test 2024 | 21.6 | 5.8 | 3.74 | 62 | 0.058 |
| eval 2025-2026H1 | 18.8 | 3.7 | 5.05 | 57 | 0.00032 |
| combined 2021-2026H1 | 15.1 | 14.7 | 1.03 | 475 | 0.0048 |

Interpretation: the recent OOS edge is not purely a 2025 selection artifact; it
also appears when 2025-2026H1 is held out and 2024 is the test split.  But the
original 3+ year CAGR/MDD>=3 target remains unsatisfied because 2021-2023 train
still carries too much drawdown relative to CAGR.  The next honest step is not
more threshold mining; it is a regime-level risk overlay or abstention model that
specifically reduces early-history drawdowns while preserving the recent OOS
edge.

## Online risk-overlay attempt

I then tested whether the remaining multi-year failure is mostly a risk-management
problem rather than an alpha problem.  The tested overlay uses only completed
prior trades or current trade OHLC stops; it does not inspect future outcomes
when deciding whether to allow the next entry.

Inputs:

- fixed dual-regime predictions:
  - `results/rex_dual_regime_train_2021_2023_predictions_2026-07-03.jsonl`
  - `results/rex_dual_regime_test_2024_predictions_2026-07-03.jsonl`
  - `results/rex_dual_regime_eval_2025_2026h1_predictions_2026-07-03.jsonl`
- exporter: `training/export_dual_regime_predictions.py`
- train-only overlay sweep: `training/sweep_online_risk_overlay.py`
- sweep report:
  `results/rex_dual_regime_online_risk_overlay_sweep_train2021_2023_test2024_eval2025_2026h1_2026-07-03.json`

Best train-selected online pause overlay:

- pause after 2 consecutive losses
- pause 288 bars
- no rolling/monthly loss stop

At 0.5x:

| period | CAGR | strict MDD | CAGR/MDD | trades | skipped overlay | p-value approx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train 2021-2023 | 12.6 | 11.7 | 1.08 | 309 | 183 | 0.106 |
| test 2024 | 10.7 | 10.5 | 1.02 | 56 | 24 | 0.217 |
| eval 2025-2026H1 | 16.1 | 3.7 | 4.32 | 53 | 9 | 0.0010 |
| all 2021-2026H1 | 12.8 | 11.7 | 1.10 | 418 | 216 | 0.0082 |

This improves full-period MDD, but not enough: leverage scales MDD faster than
CAGR and the full-period ratio stays near 1.1.

I also ran a partial stop/take-profit/ATR grid on train.  The best completed
simple exit was take-profit at 4% with no stop/ATR:

- partial summary:
  `results/rex_dual_regime_stop_grid_partial_summary_2026-07-03.json`

TP 4% replay:

| period | lev | CAGR | strict MDD | CAGR/MDD | trades | p-value approx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test 2024 | 0.5 | 18.1 | 5.8 | 3.13 | 62 | 0.059 |
| eval 2025-2026H1 | 0.5 | 18.3 | 3.7 | 4.92 | 57 | 0.0003 |
| eval 2025-2026H1 | 1.5 | 50.9 | 10.5 | 4.85 | 57 | 0.0006 |
| all 2021-2026H1 | 0.5 | 17.3 | 14.7 | 1.18 | 477 | 0.0017 |

Interpretation: simple online pause and TP overlays reduce some drawdown and keep
recent OOS attractive, but they do **not** solve the 3+ year target.  The
multi-year miss is therefore not just a missing stop-loss.  The next step needs a
higher-level regime classifier that prevents entire bad historical regimes, or a
portfolio diversification source outside this single REX BTC surface.

## Month-level bad-regime abstention attempt

After simple trade-level overlays failed, I tested a higher-level abstention
surface: decide once per calendar month whether to disable the fixed dual-regime
REX policy.  Each candidate month rule uses only the first candidate row's
signal-time feature snapshot for that month, then blocks all trades in matching
months.

Implementation:

- script: `training/sweep_month_feature_abstain_gate.py`
- test helper: `tests/test_sweep_month_feature_abstain_gate.py`
- sweep report:
  `results/rex_dual_regime_month_feature_abstain_sweep_train2021_2023_test2024_eval2025_2026h1_2026-07-03.json`
- selection guard: month rules are ranked on train 2021-2023 only; test 2024 and
  eval 2025-2026H1 are replayed after selection.

Best train-selected month abstention rule:

- block month when first candidate's `usdkrw_zscore <= -1.1786030781205512`
- blocked months in full 2021-2026H1 replay: 2021-02, 2021-04, 2022-02,
  2022-09, 2024-09, 2025-04

At 0.5x by split:

| period | CAGR | strict MDD | CAGR/MDD | trades | p-value approx |
| --- | ---: | ---: | ---: | ---: | ---: |
| train 2021-2023 | 15.7 | 13.3 | 1.18 | 317 | 0.056 |
| test 2024 | 17.5 | 5.8 | 3.02 | 61 | 0.073 |
| eval 2025-2026H1 | 18.4 | 3.7 | 4.95 | 55 | 0.0003 |

Full 2021-2026H1 replay:

| leverage | CAGR | strict MDD | CAGR/MDD | trades | p-value approx |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 16.4 | 13.3 | 1.23 | 433 | 0.0015 |
| 1.0 | 33.0 | 25.1 | 1.31 | 433 | 0.0017 |
| 1.25 | 41.3 | 30.6 | 1.35 | 433 | 0.0019 |
| 1.5 | 49.4 | 36.1 | 1.37 | 433 | 0.0020 |
| 2.0 | 64.8 | 46.8 | 1.38 | 433 | 0.0024 |

Interpretation: month-level abstention is better than trade-level pause/stop for
this surface, and it preserves the 2024/2025-2026 held-out edge.  But it still
misses the original 3+ year CAGR/MDD>=3 objective by a wide margin.  This is now
strong evidence that a single BTC REX policy surface is not enough.  Next work
should widen the portfolio: additional assets, independent non-REX alpha families,
or capital allocation across uncorrelated strategies.  Continuing to mine
single-feature abstention rules on the same BTC REX surface is unlikely to be the
breakthrough.

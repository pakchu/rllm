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

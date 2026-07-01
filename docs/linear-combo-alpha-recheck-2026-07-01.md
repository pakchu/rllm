# Linear-combo alpha recheck — 2026-07-01

## Objective
Re-check the strongest non-LLM alpha surfaces after rejecting the wave-probability teacher.  The goal was not to tune on 2026, but to find a leak-safe base edge that an LLM/RL layer could later explain, veto, or size.

## Protocol
- Market input: `data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz`.
- Chronological fit: `2020-01-01` through `2024-06-30 23:59:59`.
- Selection/test: `2024-07-01` through `2025-12-31 23:59:59`.
- Untouched eval: `2026-01-01` through `2026-06-01`.
- Strict execution: entry at next bar, actual OHLC bar-by-bar replay, costs, non-overlapping holds, intrabar adverse excursion included in strict MDD.
- Leak guard: linear model and quantile thresholds fit only on train; overlay parameters selected only on test; eval never used for selection.

## DXY/kimchi regime recheck
Report: `results/alpha_candidate_gate_dxy_kimchi_recheck_2026-07-01.json`

Result: **NO_GO**.

Best robust-looking rows stayed positive, but none met CAGR/strict-MDD >= 3 on both test and eval:
- `kimchi_premium_change` under low `dxy_zscore`, horizon 288:
  - test: CAGR 21.69%, strict MDD 17.04%, ratio 1.27, 326 trades.
  - eval: CAGR 20.38%, strict MDD 10.47%, ratio 1.95, 89 trades.
- `kimchi_premium_zscore` under low `dxy_momentum`, horizon 288:
  - test: CAGR 24.35%, strict MDD 14.60%, ratio 1.67, 331 trades.
  - eval: CAGR 10.79%, strict MDD 11.14%, ratio 0.97, 91 trades.

Interpretation: the old DXY-low × kimchi result was a real weak effect, but not strong enough by the current stricter 2026 split.

## Linear combo recheck
Report: `results/alpha_candidate_gate_linear_combo_recheck_2026-07-01.json`

Result: **NO_GO**, but it found better base surfaces than wave-teacher labels.

Notable candidates:
- `market_derivatives`, horizon 576, q0.20:
  - test: CAGR 32.16%, strict MDD 16.88%, ratio 1.90, 273 trades.
  - eval: CAGR 30.58%, strict MDD 14.24%, ratio 2.15, 74 trades.
- `kimchi_plus_range`, horizon 576, q0.15:
  - test: CAGR 24.70%, strict MDD 14.78%, ratio 1.67, 271 trades.
  - eval: CAGR 66.54%, strict MDD 11.64%, ratio 5.72, 74 trades.
- `external`, horizon 288, q0.05:
  - test: CAGR 17.18%, strict MDD 14.58%, ratio 1.18, 269 trades.
  - eval: CAGR 28.93%, strict MDD 9.55%, ratio 3.03, 89 trades.

Interpretation: simple linear combinations are more promising than recent LLM/wave probability surfaces, but still fail the all-fold ratio gate.  The eval trade count also remains too low for high confidence.

## Overlay audit
New exporter: `training/export_linear_combo_rule_predictions.py`

This converts a frozen train-fit linear combo rule into live-style JSONL predictions, so the same `online_risk_overlay_backtest` can audit test-selected execution overlays.

### External h288 q0.05
Report: `results/linear_combo_external_h288_q005_overlay_sweep_2026-07-01.json`

Selected on test: pause after 3 consecutive losses, no TP/SL/ATR/monthly stop.
- test: CAGR 17.98%, strict MDD 12.77%, ratio 1.41, 262 trades.
- eval: CAGR 36.82%, strict MDD 9.07%, ratio 4.06, 86 trades.

This is directionally interesting but still **not acceptable** because selection/test ratio is only 1.41 and eval has 86 trades.

### Market-derivatives h576 q0.20
Report: `results/linear_combo_market_deriv_h576_q020_overlay_sweep_2026-07-01.json`

Selected on test: 1% stop / 2% take-profit.
- test: CAGR 36.18%, strict MDD 12.40%, ratio 2.92, 529 trades.
- eval: CAGR -46.86%, strict MDD 24.63%, ratio -1.90, 145 trades.

This is an explicit overfit warning: risk-overlay optimization can create attractive test metrics that collapse on eval.

## Decision
Do **not** promote any of these to live trading yet.

The best next RLLM direction is not two-LLM analyzer/trader or wave-teacher imitation.  It should be a single Gemma-family text policy used as a conservative meta-controller over weak but real candidate alphas:
1. Input compact symbolic state cards from the better alpha surfaces (`external`, `kimchi_plus_range`, `market_derivatives`).
2. Output only `TAKE / SKIP / SIZE_BUCKET`, not raw side discovery.
3. Train/validate on test-like windows with purged chronological splits.
4. Freeze all gates before final eval.
5. Reject if test+eval both fail ratio/trade-count gates.

The main lesson: **LLM should not invent alpha from noisy prices; it should compress multi-feature context and veto/sizing decisions around a separately verified weak edge.**

## Meta-controller SFT surface
New builder: `training/build_linear_alpha_meta_sft.py`

This converts frozen alpha predictions into single-LLM SFT rows.  Prompts contain only signal-time context and explicitly prohibit side invention.  Targets use future realized trade outcomes only as supervised labels:
- `TAKE/FULL` if realized return is at least +0.35%.
- `TAKE/SMALL` if realized return is positive but below +0.35%.
- `SKIP/NONE` otherwise or if the alpha did not trigger.

Generated smoke datasets for `external h288 q0.05`:
- `data/linear_alpha_external_h288_q005_meta_sft_test_2024h2_2025.jsonl`
  - rows: 16,453
  - target decisions: SKIP 9,276 / TAKE 7,177
  - size buckets: FULL 5,125 / SMALL 2,052 / NONE 9,276
- `data/linear_alpha_external_h288_q005_meta_sft_eval_2026_jan_may.jsonl`
  - rows: 7,190
  - target decisions: SKIP 3,588 / TAKE 3,602
  - size buckets: FULL 2,522 / SMALL 1,080 / NONE 3,588

Important caveat: these labels are per-candidate realized outcomes, not non-overlapping portfolio outcomes.  They are suitable for a Gemma meta-controller POC, but final selection must still be audited through `online_risk_overlay_backtest` with frozen predictions.

## Compact conservative meta-controller variant
Builder update: `training/build_linear_alpha_meta_sft.py` now supports:
- `--target-schema decision_size` to remove free-form `risk_reason` from targets.
- `--prompt-style conservative` to explicitly make `SKIP/NONE` the default unless signal-time evidence is strong.

Generated compact split summaries:
- train 2024H1: 4,430 rows; SKIP 2,732 / TAKE 1,698; FULL 1,245 / SMALL 453 / NONE 2,732.
- test 2024H2-2025: 16,453 rows; SKIP 9,276 / TAKE 7,177; FULL 5,125 / SMALL 2,052 / NONE 9,276.
- eval 2026 Jan-May: 7,190 rows; SKIP 3,588 / TAKE 3,602; FULL 2,522 / SMALL 1,080 / NONE 3,588.

This variant is intended to reduce generation truncation and align with candidate-logprob scoring labels: `SKIP/NONE`, `TAKE/SMALL`, `TAKE/FULL`.

## Gemma meta-controller smoke comparison
Adapters tested on 64 balanced `2024H2-2025` rows with candidate-logprob scoring:

| Dataset / prompt | Train sample | Steps | Norm | Decision acc | Size acc | Pred distribution | Finding |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| risk_reason/default | 512 | 16 | mean | 62.5% | 26.6% | SKIP 8 / SMALL 53 / FULL 3 | Over-predicts TAKE/SMALL. |
| compact/conservative | 512 | 16 | mean | 54.7% | 53.1% | SKIP 61 / SMALL 2 / FULL 1 | Over-corrects to SKIP. |
| compact/conservative size-balanced | 768 | 24 | mean | 34.4% | 34.4% | SKIP 63 / SMALL 1 / FULL 0 | Still SKIP-collapsed. |
| compact/default size-balanced | 768 | 24 | mean | 76.6% | 42.2% | SKIP 6 / SMALL 58 / FULL 0 | Best decision accuracy, but cannot learn FULL. |
| compact/default size-balanced | 768 | 24 | sum | 73.4% | 40.6% | SKIP 10 / SMALL 54 / FULL 0 | Similar, slightly more SKIP. |
| compact/default size-balanced | 768 | 24 | first_token | 32.8% | 32.8% | SKIP 64 | Not usable. |

Interpretation:
- Removing `risk_reason` stabilized training loss substantially.
- Conservative wording is too strong and collapses to SKIP.
- Size-bucket balancing works at sampler level, but Gemma still prefers SMALL over FULL under constrained scoring.
- Next POC should simplify the action space to binary `TAKE` vs `SKIP`, then handle size outside the LLM using calibrated score/volatility rules.

## Binary TAKE/SKIP POC
Because size buckets kept collapsing to either SMALL or SKIP, the next POC simplified Gemma output to binary `{"decision":"TAKE|SKIP"}` and moved sizing outside the LLM.

Binary adapter:
- train data: `data/linear_alpha_external_h288_q005_meta_sft_train_2024h1_binary.jsonl`
- test data: `data/linear_alpha_external_h288_q005_meta_sft_test_2024h2_2025_binary.jsonl`
- adapter: `checkpoints/linear_alpha_meta_binary_gemma4_sft_s512_step16_2026-07-01` (deleted after documenting failure)
- training: 512 balanced TAKE/SKIP rows, 16 steps, train loss 0.9285.

128-row balanced test logprob result:
- target: SKIP 64 / TAKE 64
- prediction: SKIP 17 / TAKE 111
- decision accuracy: 57.0%
- confusion: TP 60, FP 51, TN 13, FN 4

Margin threshold audit on `TAKE_score - SKIP_score`:
- best checked accuracy: 59.4% at threshold -0.5 / -0.25, but still FP 52 and TAKE 116/128.
- high thresholds reduce false positives but miss nearly all true TAKE labels.

Decision: binary simplification improves output form but not separability.  Current LLM prompt/state surface does not reliably distinguish TAKE from SKIP for this weak alpha.  The retained adapter is `compact_default` only for reproduction; failed smoke checkpoints were deleted to keep disk usage low.

## Binary meta-state separability diagnostic
New diagnostic: `training/linear_alpha_meta_feature_diagnostic.py`

Purpose: before spending more GPU time on Gemma, test whether the current text/numeric state card contains enough signal for a simple train-only classifier to separate `TAKE` from `SKIP` without future leakage.

Run configuration:
- train: `data/linear_alpha_external_h288_q005_meta_sft_train_2024h1_binary.jsonl`
- test: `data/linear_alpha_external_h288_q005_meta_sft_test_2024h2_2025_binary.jsonl`
- eval: `data/linear_alpha_external_h288_q005_meta_sft_eval_2026_jan_may_binary.jsonl`
- output: `results/linear_alpha_meta_feature_diagnostic_binary_2026-07-01.json`
- model: simple numpy logistic baseline, train split only, 256 prompt-derived features, 800 steps, L2 0.02.

Results:

| Split | Rows | Accuracy | Majority | Beats majority | Balanced recall | TAKE recall | SKIP recall | Pred distribution |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| train 2024H1 | 4,430 | 73.3% | 61.7% | yes | 69.4% | 52.8% | 86.1% | SKIP 3,154 / TAKE 1,276 |
| test 2024H2-2025 | 16,453 | 55.6% | 56.4% | no | 53.3% | 34.5% | 72.0% | SKIP 11,374 / TAKE 5,079 |
| eval 2026 Jan-May | 7,190 | 53.3% | 50.1% | weak yes | 53.4% | 33.8% | 72.9% | SKIP 5,002 / TAKE 2,188 |

Top train-only univariate clues were weak but consistent with price-action/risk context being more useful than raw LLM phrasing:
- `range_vol`: corr +0.223, effect +0.459 std.
- `window_drawdown`: corr +0.088.
- `dxy_momentum`: corr -0.087.
- `usdkrw_zscore`: corr -0.076.
- `dxy_zscore`: corr -0.069.
- rolling-extrema context (`rex_*`) and range position also appear, but weakly.

Decision:
- Current meta-controller state has only weak out-of-sample separability.  Test does not beat majority and eval beats majority only marginally.
- More Gemma SFT steps on this same surface are not justified; the problem is not only model capacity.
- Next work should redesign the state/label around price-action path quality: rolling extrema distance, range/volatility regime, side-conditioned adverse excursion, MFE/MAE, and strict path-risk labels.  The LLM should make a compact veto/position-quality decision over genuinely informative state, not memorize noisy realized-return labels.

## Path-quality label/state redesign POC
Builder update: `training/build_linear_alpha_meta_sft.py` now supports `--label-mode path_quality` and adds side-conditioned price-action context to the prompt state:
- side-adjusted trend/range/RSI/BB/taker/rolling-extrema/higher-timeframe features.
- rolling-extrema room-to-adverse/favorable extreme proxies.
- future path labels record realized return, max favorable excursion, max adverse excursion, and MFE/MAE ratio in metadata only; prompts still use signal-time features only.

Path-quality binary datasets for `external h288 q0.05` were generated with:
- `TAKE/FULL`: realized return >= 0.35%, max adverse <= 0.35%, MFE >= 0.55%, MFE/MAE >= 1.25.
- `TAKE/SMALL`: realized return > 0%, max adverse <= 0.70%, MFE >= 0.20%, MFE/MAE >= 1.0.
- otherwise `SKIP`.

Label distributions:

| Split | Rows | SKIP | TAKE | FULL | SMALL | max adverse mean / p90 | max favorable mean / p90 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| train 2024H1 | 4,430 | 3,248 | 1,182 | 630 | 552 | 1.42% / 3.24% | 1.03% / 2.23% |
| test 2024H2-2025 | 16,453 | 11,232 | 5,221 | 2,630 | 2,591 | 0.98% / 2.11% | 0.97% / 2.17% |
| eval 2026 Jan-May | 7,190 | 4,687 | 2,503 | 1,377 | 1,126 | 0.98% / 2.15% | 0.97% / 2.55% |

CPU separability diagnostic on this redesigned state/label surface (`results/linear_alpha_meta_feature_diagnostic_pathq_binary_2026-07-01.json`):

| Split | Accuracy | Majority | Beats majority | Balanced recall | TAKE recall | SKIP recall |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| train 2024H1 | 91.8% | 73.3% | yes | 88.3% | 80.7% | 95.8% |
| test 2024H2-2025 | 63.1% | 68.3% | no | 55.0% | 32.8% | 77.1% |
| eval 2026 Jan-May | 59.4% | 65.2% | no | 57.0% | 48.7% | 65.2% |

Interpretation:
- The new labels are more risk-realistic and side-conditioned features reveal plausible clues (`side_htf_*`, `side_rex_*`, `side_trend_96`, `range_vol`).
- However train/test drift is still severe: train is highly separable, while test/eval fail majority accuracy.  This is a better diagnostic surface, but not yet a deployable LLM target.
- Next iteration should reduce overfit by using rolling/monthly train calibration or feature family stability filters before Gemma fine-tuning.

## Trade-only chronological feature stability audit
New diagnostic: `training/linear_alpha_meta_stability_diagnostic.py`

Purpose: identify prompt features whose TAKE/SKIP relation keeps the same sign across chronological regimes before using them in another Gemma run.  This avoids overfeeding the LLM unstable features that fit 2024H1 but invert later.

Trade-only half-year audit:
- inputs: path-quality binary train/test/eval SFT rows.
- periods: 2024H1, 2024H2, 2025H1, 2025H2, 2026H1.
- rows after `--trade-only`: 25,768.
- output: `results/linear_alpha_meta_stability_pathq_binary_trade_only_halfyear_2026-07-01.json`.

Most stable non-trivial clues are very weak:
- `bucket:rex_2016_range_pos=neg_s`: same negative sign in all five periods, min abs corr 0.077, mean abs corr 0.111.
- `num:trend_96`: same positive sign, min abs corr 0.021, mean abs corr 0.042.
- `bucket:usdkrw_zscore=pos_m`: same negative sign, min abs corr 0.018, mean abs corr 0.048.
- `bucket:kimchi_premium_zscore=zero` / `bucket:usdkrw_zscore=zero`: same positive sign, min abs corr 0.017, mean abs corr 0.052.
- `tok:dxy_bucket=high`: same negative sign, but unstable magnitude, min abs corr 0.010.

Decision:
- Stable alpha exists only as a weak-feature bundle; no single feature is strong enough.
- Next prompt should be feature-family constrained: emphasize stable rolling-extrema/range/trend + macro availability/bucket context, and reduce unstable raw numeric dumps.
- Another Gemma SFT should be gated by this audit and evaluated on trade-only candidate decisions before portfolio backtest.

## Stable price-action prompt Gemma smoke
Builder update: `--prompt-style stable_pa` filters the prompt down to stable price-action/macro families instead of dumping all numeric state.  The compact prompt keeps rolling-extrema/range/volatility/trend/macro bucket context and side-adjusted room/range features.

CPU diagnostic on the stable prompt (`results/linear_alpha_meta_feature_diagnostic_pathq_stablepa_binary_2026-07-01.json`):
- feature count: 159 vs 384 in the full path-quality prompt.
- train: 82.98% accuracy, balanced recall 72.73%.
- test: 64.47% accuracy vs 68.27% majority, balanced recall 53.43%.
- eval: 64.73% accuracy vs 65.19% majority, balanced recall 58.44%.

Interpretation: pruning reduced train overfit and improved eval accuracy materially, but still does not beat the majority baseline.  It is a better LLM prompt shape, not yet a valid alpha.

Gemma-4-E4B smoke:
- train data: `data/linear_alpha_external_h288_q005_meta_sft_train_2024h1_pathq_stablepa_binary.jsonl`
- model: `gemma4-e4b` (`google/gemma-4-E4B-it`)
- sample/steps: 512 balanced rows, 16 steps, LoRA r8/alpha16, runtime 112.4s, final train loss 0.1686.
- test eval: 128 balanced rows, candidate-logprob binary scoring.
- result: decision accuracy 50.0%; predicted TAKE for all 128 rows.
- margin threshold audit: best checked accuracy 59.4% at threshold 0.25, but still FP 52 / TN 12 and no FN; threshold 0.5 flips to all SKIP.

Decision: even with a cleaner stable prompt and path-quality labels, Gemma still learns label priors/margins rather than robust trade vetoes in this 16-step POC.  The failed checkpoint was deleted.  Next work should either use stronger preference/ranking formulation or a rolling calibration layer before another adapter run.

## Preference/ranking formulation start
New builder: `training/build_linear_alpha_meta_preference.py`

Rationale: binary SFT repeatedly learned label priors or brittle logprob margins.  The next formulation turns the same stable no-leak prompt into DPO-style `chosen`/`rejected` completions, ranking the desired `TAKE` or `SKIP` answer above its opposite.  This better matches the LLM strength: comparative judgment over a compact state card.

Trade-only DPO datasets from `stable_pa + path_quality` SFT rows:

| Split | Pairs | Chosen SKIP | Chosen TAKE | Skipped non-trade |
| --- | ---: | ---: | ---: | ---: |
| train 2024H1 | 3,958 | 2,776 | 1,182 | 472 |
| test 2024H2-2025 | 14,972 | 9,751 | 5,221 | 1,481 |
| eval 2026 Jan-May | 6,838 | 4,335 | 2,503 | 352 |

DPO dry-run:
- model alias: `gemma4-e4b` -> `google/gemma-4-E4B-it`.
- sample: 128 balanced rows.
- prompt length mean: 1,645.9 chars.
- chosen balance: SKIP 64 / TAKE 64.
- output: `checkpoints/linear_alpha_meta_pref_pathq_stablepa_binary_dpo_dryrun_2026-07-01/dpo_summary.json`.

Code note: `training/train_text_dpo.py` now buckets `{"decision": ...}` completions correctly for balanced sampling summaries.

Next validation: run a small Gemma DPO adapter and score `TAKE` vs `SKIP` on balanced test rows.  Promote only if it stops all-TAKE/all-SKIP collapse and beats the SFT smoke baseline.

## Gemma DPO preference smoke result
Small DPO adapter:
- train: `data/linear_alpha_external_h288_q005_meta_pref_train_2024h1_pathq_stablepa_binary.jsonl`
- model: `gemma4-e4b` (`google/gemma-4-E4B-it`)
- sample/steps: 512 balanced preference pairs, 16 steps, LoRA r8/alpha16, beta 0.1, lr 5e-6.
- runtime: 151.8s.
- train loss: 0.6938, reward margins unstable around zero.

Balanced candidate-logprob test result:
- test 128 rows: accuracy 60.2%, pred SKIP 13 / TAKE 115.
- threshold audit on `TAKE_score - SKIP_score`:
  - threshold 0.25: 65.6% accuracy, TP 59 / FP 39 / TN 25 / FN 5.
  - threshold 0.50: 66.4% accuracy, TP 47 / FP 26 / TN 38 / FN 17.

Balanced eval result:
- eval 256 rows: accuracy 53.9%, pred SKIP 24 / TAKE 232.
- threshold transferred from test does not hold:
  - threshold 0.25: 55.5% accuracy, TP 107 / FP 93 / TN 35 / FN 21.
  - threshold 0.50: 51.6% accuracy, TP 76 / FP 72 / TN 56 / FN 52.

Decision:
- DPO is better than binary SFT on test because it no longer predicts TAKE for every row, but the improvement does not transfer to eval.
- This confirms the current issue is still regime/label instability, not merely output formatting or SFT vs DPO objective.
- Failed DPO and dry-run checkpoints were deleted to keep disk usage bounded.
- Next direction should be rolling calibration / regime-conditioned preference data, not a larger global adapter over 2024H1 only.

## Rolling continuous calibration preflight
New diagnostic: `training/linear_alpha_meta_walkforward_diagnostic.py`

Purpose: test the user's continuous-learning idea cheaply before launching rolling Gemma jobs.  For each half-year period, a logistic model is fit only on past periods using prompt-derived stable price-action features, then evaluated on the next period.  Feature space, standardization, weights, and threshold calibration use past rows only.

Trade-only half-year walk-forward result (`results/linear_alpha_meta_walkforward_pathq_stablepa_trade_only_halfyear_2026-07-01.json`):

| Eval period | Train periods | Eval rows | Fixed acc | Majority | Balanced recall | TAKE recall | SKIP recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024H2 | 2024H1 | 4,903 | 62.4% | 65.3% | 52.8% | 21.6% | 84.0% |
| 2025H1 | 2024H1-2024H2 | 6,927 | 62.4% | 67.7% | 50.8% | 18.0% | 83.5% |
| 2025H2 | 2024H1-2025H1 | 3,142 | 51.8% | 59.1% | 46.0% | 14.6% | 77.5% |
| 2026H1 | 2024H1-2025H2 | 6,838 | 61.5% | 63.4% | 51.4% | 13.8% | 89.0% |

Aggregate over evaluated periods:
- fixed threshold accuracy: 60.6% vs 64.6% majority.
- balanced recall: 50.7%.
- TAKE recall: 16.9%, SKIP recall: 84.5%.
- train-calibrated thresholds increased TAKE attempts but did not beat majority or improve balanced recall consistently.

Decision:
- Continuous/rolling refit alone does not solve the current alpha surface.  It mostly learns a conservative SKIP-biased classifier and misses profitable TAKE labels.
- This weakens the case for rolling Gemma adapters on the same labels/features.
- Next productive direction is to change candidate construction and labels: make the model compare higher-quality candidate alternatives or use stronger entry candidates, rather than only vetoing a weak external linear alpha.

## Multi-candidate pairwise ranking pivot
New builder: `training/build_linear_alpha_candidate_pairwise.py`

Rationale: vetoing one weak alpha failed.  The next attempt expands the candidate pool and asks Gemma to compare two simultaneous frozen-alpha trade candidates.  This is closer to an LLM strength: comparative ranking over symbolic alternatives.

Candidate pool exported from three frozen linear combo rules fit only on `2023-01-01` to `2024-06-30`:
- `external h288 q0.05`
- `market_derivatives h576 q0.20`
- `kimchi_plus_range h576 q0.15`

Pairwise dataset construction:
- At each timestamp, keep candidate pairs where at least two rules fire.
- Prompt includes only signal-time descriptors: source, side, hold bars, alpha score.
- Target chooses A/B using future path utility only for offline label: `return + 0.25*MFE - 0.75*MAE`.
- Pairs with utility gap below 0.15% are skipped.

Dataset sizes:

| Split | Rows | Choice A | Choice B | Always-A random baseline |
| --- | ---: | ---: | ---: | ---: |
| train 2024H1 | 14,488 | 8,092 | 6,396 | not used |
| test 2024H2-2025 | 41,891 | 22,590 | 19,301 | 55.65% on random 2,000 |
| eval 2026 Jan-May | 10,045 | 5,512 | 4,533 | 55.75% on random 2,000 |

Gemma-4-E4B pairwise SFT smoke:
- train: 512 balanced A/B rows, 16 steps, LoRA r8/alpha16, runtime 115.6s, train loss 1.003.
- first-1000 eval initially looked high, but that was row-order/choice-position bias: always-A got 93.6% on the same first eval rows.
- random 2,000 test: Gemma 54.3%, always-A 55.65%.
- random 2,000 eval: Gemma 54.0%, always-A 55.75%.

Decision:
- Multi-candidate ranking is a better problem formulation, but the current prompt is too sparse and the model mostly learns positional bias/noise.
- Failed pairwise SFT and dry-run checkpoints were deleted.
- Next attempt should randomize A/B order at data-build time more carefully across all splits, include stable state context around the timestamp, and evaluate with random/balanced sampling only.  No sequential first-N metric should be trusted for pairwise rows.

## Pairwise ranking v2: randomized order + state context
Builder update: `training/build_linear_alpha_candidate_pairwise.py` now defaults to:
- randomizing A/B order with a seed, removing first-N positional leakage.
- adding compact stable state context: trend/range/drawdown, DXY/kimchi/USDKRW, rolling-extrema position, and higher-timeframe returns.

V2 dataset sizes:

| Split | Rows | Choice A | Choice B | Always-A random baseline |
| --- | ---: | ---: | ---: | ---: |
| train 2024H1 | 14,488 | 7,218 | 7,270 | not used |
| test 2024H2-2025 | 41,891 | 21,027 | 20,864 | 49.45% on random 2,000 |
| eval 2026 Jan-May | 10,045 | 5,024 | 5,021 | 52.90% on random 2,000 |

Gemma-4-E4B v2 pairwise SFT smoke:
- train: 512 balanced A/B rows, 16 steps, LoRA r8/alpha16, runtime 113.4s, train loss 1.043.
- random 2,000 test: 49.35%, pred A 1,438 / B 562.
- random 2,000 eval: 52.30%, pred A 1,640 / B 360.

Decision:
- Randomization fixed the misleading positional baseline, and state context is included, but Gemma still does not learn robust A/B ranking from 512-row smoke.  It remains A-biased and does not beat the random-sample baseline.
- Failed v2 checkpoints were deleted.
- The current bottleneck is likely that labels are too path-outcome/noisy for the sparse candidate prompt, and the candidate pool itself may not expose stable discriminative text patterns.  Next direction should add explicit path-quality bins from past-only analogs or use a non-LLM candidate selector as teacher before asking Gemma to imitate/compress it.

## Past-only pairwise teacher diagnostic
New diagnostic: `training/linear_alpha_candidate_pairwise_teacher.py`

Purpose: before asking Gemma to infer noisy future path labels directly, test whether a simple past-only analog teacher can learn candidate-family/context win rates and select A/B for future periods.  The teacher uses only prior pairwise rows and simple context buckets from the no-leak prompt.

Walk-forward half-year result on randomized state-context pairwise v2 rows:

| Eval period | Train rows | Eval rows | Teacher acc | Always-A | Always-B | Pred A/B |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2024H2 | 14,488 | 12,916 | 50.0% | 49.4% | 50.6% | 6,566 / 6,350 |
| 2025H1 | 27,404 | 13,025 | 53.2% | 50.8% | 49.2% | 6,439 / 6,586 |
| 2025H2 | 40,429 | 15,950 | 57.5% | 50.3% | 49.7% | 8,044 / 7,906 |
| 2026H1 | 56,379 | 10,045 | 54.9% | 50.0% | 50.0% | 5,045 / 5,000 |
| aggregate | - | 51,936 | 54.0% | 50.2% | 49.8% | 26,094 / 25,842 |

Decision:
- A cheap past-only teacher outperforms positional baselines without A/B collapse, unlike Gemma direct SFT.
- This is the first relatively stable signal in the pairwise branch, but still weak.
- Next useful LLM role is not direct future-label prediction; it is distilling/compressing a stronger teacher or teacher+context rationale once the teacher is strengthened and connected to portfolio selection.

## Pairwise teacher portfolio backtest
New exporter: `training/apply_pairwise_teacher_to_candidates.py`

Purpose: convert the past-only pairwise teacher into live-style prediction rows by selecting one candidate per timestamp using only previous-period teacher stats, then audit the result with strict bar-by-bar backtest.

Base teacher-selected predictions:
- output rows: 37,508 from 2024H2 through 2026H1.
- period selections: 2024H2 9,489; 2025H1 9,486; 2025H2 11,308; 2026H1 7,225.
- strong long bias remains: 2026H1 selected LONG 6,714 / SHORT 511.

Strict backtest at 0.5 leverage, max hold 576:

| Period | CAGR | Strict MDD | CAGR/MDD | Trades | Mean trade | p-value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024H2-2026H1 | 4.55% | 27.28% | 0.17 | 477 | 0.026% | 0.658 |
| test 2024H2-2025 | 6.45% | 17.00% | 0.38 | 380 | 0.033% | 0.621 |
| eval 2026H1 | -1.77% | 19.21% | -0.09 | 97 | -0.001% | 0.994 |

Vote-margin sweep on eval 2026H1:

| Pair margin | Selected rows | Trades | CAGR | Strict MDD | CAGR/MDD | Mean trade | p-value |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.005 | 6,963 | 96 | -13.07% | 20.66% | -0.63 | -0.053% | 0.653 |
| 0.010 | 6,963 | 96 | -13.07% | 20.66% | -0.63 | -0.053% | 0.653 |
| 0.020 | 5,199 | 91 | 3.94% | 18.74% | 0.21 | 0.024% | 0.845 |
| 0.030 | 4,073 | 49 | 6.23% | 8.01% | 0.78 | 0.054% | 0.726 |

Decision:
- The weak pairwise teacher improves classification above baseline but does not translate into a tradable portfolio edge.
- Higher margin reduces MDD but also cuts trades and remains statistically insignificant.
- This closes the current linear-combo candidate-selection branch as non-promotable.
- The useful lesson is architectural: LLM should not learn from noisy realized path labels directly; we need a stronger teacher/candidate source before distillation.

## Deductive symbolic selector audit
New selector: `training/linear_alpha_deductive_candidate_selector.py`

Purpose: shift from numeric classification to explicit LLM-style deduction.  The selector converts signal-time candidate/state data into symbolic premises and applies transparent rules:
- multi-timeframe trend alignment,
- range/extrema location,
- volatility and drawdown risk,
- macro/kimchi pressure,
- small candidate-source priors.

No future labels are used by the selector.  It emits live-style predictions plus a compact deduction JSON containing premises and conclusion.

Direct deductive rule eval on 2026H1 failed:

| Min score | Eval rows | Trades | CAGR | Strict MDD | CAGR/MDD | Mean trade | p-value |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 3,996 | 74 | -15.24% | 10.26% | -1.48 | -0.087% | 0.382 |
| 1.0 | 3,230 | 68 | -18.86% | 9.95% | -1.89 | -0.119% | 0.303 |
| 1.5 | 1,497 | 50 | -23.64% | 10.63% | -2.22 | -0.210% | 0.164 |
| 2.0 | 531 | 37 | -24.32% | 11.44% | -2.13 | -0.294% | 0.131 |

Inverted-side audit on 2026H1 looked tempting but did not transfer:

| Min score | Eval CAGR/MDD | Eval trades | Test CAGR/MDD | Test trades | Interpretation |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1.0 | 1.17 | 68 | -0.65 | 266 | not stable |
| 1.5 | 2.10 | 50 | -0.65 | 214 | not stable |
| 2.0 | 3.93 | 37 | -0.62 | 170 | eval-only mirage; test significantly negative |

Decision:
- The user insight is correct: LLM should be used for explicit premise/rule/conclusion reasoning, not raw numeric classification.
- But hand-written deductive rules are not yet valid alpha.  The direct rule is contra-profitable; the inverted rule is eval-only and fails badly on test.
- Next direction should be rule discovery and rule validation: let LLM propose symbolic rules, but only accept rules that pass chronological walk-forward stability and strict backtest.  LLM is a hypothesis generator/reasoner; walk-forward tests are the judge.

## Symbolic rule discovery v1
New scanner: `training/symbolic_candidate_rule_discovery.py`

Purpose: instead of hand-writing deductive rules, generate explicit symbolic hypotheses of the form:
- candidate/source/side premise,
- state bucket premise,
- action `follow` or `invert`.

Protocol:
- generate rule supports from train only,
- rank rules on test only,
- report eval untouched.

Run result:
- examples: train 24,379; test 71,150; eval 17,100.
- candidate symbolic rules: 1,698 evaluated.
- many top test rules looked excellent by isolated future-return labels, but most had tiny eval support or failed eval.

Most interesting test-ranked rule:
- rule: `id_side=market_derivatives|h576|original|SHORT` + `range_vol=high`
- action: `invert` (take LONG instead of candidate SHORT)
- offline isolated-label metrics:
  - train n=238, mean +0.484%, win 58.8%.
  - test n=295, mean +1.539%, win 99.7%.
  - eval n=52, mean +1.248%, win 76.9%.

Strict portfolio audit of that rule showed the offline label is misleading under actual bar-by-bar/cooldown execution:
- train: CAGR 7.77%, MDD 6.20%, ratio 1.25, only 5 trades.
- test: CAGR 8.08%, MDD 11.90%, ratio 0.68, 10 trades.
- eval: CAGR -82.47%, MDD 11.90%, ratio -6.93, 4 trades.
- all: CAGR 4.68%, MDD 11.90%, ratio 0.39, 19 trades.

Decision:
- Symbolic rule discovery is the right bridge for LLM deductive reasoning, but the scoring objective must be strict-backtest-aware, not isolated future-return labels.
- Next scanner should evaluate candidate rules by constructing prediction rows and running strict overlay per split, then rank on test CAGR/MDD/trade-count and report eval.

## Strict-backtest-aware symbolic rule scan
New scanner: `training/symbolic_rule_strict_backtest_scan.py`

Purpose: the first symbolic scanner ranked rules by isolated future-return labels and then failed under real portfolio mechanics.  This scanner instead materializes live-style prediction rows for every candidate rule and runs the same strict bar-by-bar online overlay on train/test/eval.  Rule generation uses train support, ranking uses test only, and eval remains untouched until final reporting.

Run configuration:
- inputs: pairwise state v2 train 2024H1, test 2024H2-2025, eval 2026 Jan-May.
- market data: `data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz`.
- rule budget: 160 strict-scanned rules from train-supported symbolic predicates.
- minimum support: train 50 candidate rows, test 100 candidate rows, test 20 executed trades.
- execution model: 0.5 leverage, 576-bar max hold, 1-bar entry delay, fee/slippage included.

Top test-ranked rule:
- rule: `kimchi_premium_zscore=mid` + `side=SHORT`.
- action: `invert`.
- support: train 7,177 candidates, test 26,868 candidates, eval 4,736 executed candidate rows before overlay/cooldown.

| Split | CAGR | Strict MDD | CAGR/MDD | Trades | Mean trade p-value |
| --- | ---: | ---: | ---: | ---: | ---: |
| train 2024H1 | 54.66% | 10.99% | 4.97 | 90 | 0.174 |
| test 2024H2-2025 | 23.09% | 18.49% | 1.25 | 258 | 0.154 |
| eval 2026 Jan-May | -9.94% | 20.60% | -0.48 | 67 | 0.840 |

Other high-ranked rules showed the same pattern: acceptable-looking train/test pockets but negative 2026 eval transfer.  Example `dxy_zscore=mid` + `side=SHORT`, action `invert`, produced test CAGR 14.62% / MDD 21.42% / ratio 0.68, then eval CAGR -27.60% / MDD 23.51% / ratio -1.17.

Decision:
- The strict scanner is the correct validation surface for LLM-generated symbolic rules.
- No rule from this first grammar is promotable: top rules fail eval, exceed the strict MDD target, or lack statistical significance.
- The failure is useful: the previous strong-looking symbolic result was mostly objective mismatch, not alpha.  Future LLM work should propose richer price-action/regime rules, but promotion must stay strict-backtest-first with chronological train/test/eval separation.

## Strict symbolic scan v2: three-premise grammar and test prefilter
Scanner updates:
- `--max-rule-terms 3` permits anchor + two state predicates so deductive LLM-style rules can express conjunctions.
- `--prefilter-mode test_return` uses test-only isolated return as triage before expensive strict backtests; final ranking still uses strict test overlay.
- duplicate predicate aliases are removed by actual test prediction signature before strict overlay.

Completed v2 run before duplicate removal:
- generated rules: 3,661.
- strict-scanned rules: 16.
- top rule: `htf_1w_return_4=mid` + `id_side=market_derivatives|h576|original|LONG` + `kimchi_premium_zscore=mid`, action `follow`.

| Split | CAGR | Strict MDD | CAGR/MDD | Trades | Mean trade p-value |
| --- | ---: | ---: | ---: | ---: | ---: |
| train 2024H1 | 6.06% | 8.81% | 0.69 | 34 | 0.764 |
| test 2024H2-2025 | 13.52% | 10.48% | 1.29 | 88 | 0.249 |
| eval 2026 Jan-May | -11.50% | 19.34% | -0.59 | 30 | 0.645 |

Duplicate-removal smoke run:
- strict-scanned unique rules: 3.
- top rule remained the same and still failed eval.
- one apparent eval blow-up had only one eval trade, so it is non-promotable regardless of CAGR arithmetic.

Decision:
- Three-premise grammar finds more interpretable pockets, but still no stable alpha.
- The current recurring failure mode is regime transfer: selected 2024H2-2025 rules lose edge in 2026.
- Next scans should use deduplicated candidates and broader candidate budgets, but promotion requires positive eval with adequate trade count, strict MDD <= 15, and statistical support.

## Strict symbolic scan v3: deduplicated broader search
Run: `results/symbolic_rule_strict_backtest_scan_v3_dedupe_terms3_prefilter_2026-07-01.json`

Protocol:
- generated rules: 6,389.
- test-prefiltered candidates: 12,008.
- unique strict candidates: 24.
- strict-scanned candidates: 24.
- eval was not used for prefiltering or ranking.

Best test-ranked rule:
- rule: `id=market_derivatives|h576|original` + `rex_8640_range_pos=high` + `side_range_vol=mid`.
- action: `follow`.

| Split | CAGR | Strict MDD | CAGR/MDD | Trades | Mean trade | p-value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train 2024H1 | 1.39% | 11.74% | 0.12 | 32 | 0.043% | 0.914 |
| test 2024H2-2025 | 21.77% | 7.09% | 3.07 | 64 | 0.390% | 0.034 |
| eval 2026 Jan-May | 16.89% | 2.96% | 5.72 | 15 | 0.373% | 0.174 |

Decision:
- This is the first strict-symbolic candidate with positive test and positive untouched eval plus strict MDD below 15.
- It is not promotable yet because eval has only 15 trades and p-value is not significant.
- This is a useful lead: the premise is interpretable and price-action oriented (`market_derivatives`, long-horizon range high, side-adjusted volatility mid), but it needs longer-period validation and more trade count before being treated as alpha.
- No candidate in this run met the full promotion gate: test positive, eval positive, eval strict MDD <= 15, and eval trades >= 20.

## Rejected: event pair strict scan on preference-pair data
A temporary event-pair strict scanner was tested against:
- `event_candidate_regime_pairwise_option_compact_paext_rex_train_2022_2024_2026-06-30.jsonl`
- `event_candidate_regime_pairwise_option_compact_paext_rex_eval_2025_2026_2026-06-30.jsonl`

Smoke result looked unrealistically strong, for example `drawdown_state=medium`, action `invert`, showed train/test/eval all highly profitable.  This is invalid as live-trading evidence.

Root cause from `training/export_event_candidate_regime_pairwise_option.py`:
- rows are grouped by month/side/hold/family,
- sorted by future `_utility`,
- the exporter pairs high-utility winners against low-utility losers,
- `leakage_guard.target_uses_future_reward_for_training_only` is true.

Decision:
- These pairwise option files are valid for preference/fine-tuning tasks where reward is a label only.
- They are not a live candidate stream and must not be materialized into backtest trades.
- The temporary scanner was removed instead of committed to avoid future misuse.
- Longer-period validation must use a live-style candidate stream where every candidate would have existed at signal time without future utility selection.

## Live-style binary-edge symbolic scanner
New scanner: `training/binary_edge_symbolic_rule_strict_backtest_scan.py`

Purpose: validate symbolic LLM-style rules on a longer candidate stream without using future-utility-selected pair data.  Inputs are candidate-level binary-edge rows where every candidate row exists independently; future reward is label-only and is not part of the prompt.

Chronological protocol:
- train: 2022-2024 candidate rows.
- test: 2025 candidate rows.
- eval: 2026 Jan-May candidate rows.
- rule generation: train support only.
- candidate triage: test-only cheap prefilter.
- final selection: strict test backtest.
- final report: untouched eval.

Smoke run:
- inputs: `event_candidate_binary_edge_paext_rex_train_2022_2024_2026-06-30.jsonl`, `event_candidate_binary_edge_paext_rex_eval_2025_2026_2026-06-30.jsonl`.
- strict-scanned unique rules: 3.
- best smoke rule: `side_trend_96=strong_up`, action `invert`.

| Split | CAGR | Strict MDD | CAGR/MDD | Trades | p-value |
| --- | ---: | ---: | ---: | ---: | ---: |
| train 2022-2024 | -6.63% | 28.15% | -0.24 | 438 | 0.378 |
| test 2025 | 7.34% | 7.76% | 0.95 | 101 | 0.371 |
| eval 2026 Jan-May | -18.52% | 9.76% | -1.90 | 43 | 0.154 |

Decision:
- This is a valid longer-period live-style validation surface.
- Smoke did not find a promotable rule, which is expected and much more believable than the rejected future-utility-selected event-pair scan.
- Next step is a broader binary-edge scan with more unique strict candidates.

## Binary-edge scanner runtime cap and capped smoke
The broad binary-edge scan with 24 strict candidates was stopped after 32 minutes because candidate prefilter/signature work was too expensive.  The scanner now supports:
- `--max-generated-rules` to cap support-ranked symbolic rules before exact prefilter scoring.
- `--max-prefilter-candidates` to cap expensive signature deduplication.

Capped smoke run:
- `--max-generated-rules 400`
- `--max-prefilter-candidates 60`
- `--max-rules 3`
- train 2022-2024, test 2025, eval 2026 Jan-May.

Best capped rule:
- rule: `hold=432` + `rex_2016_cur_to_min_pct=pos_large`, action `invert`.
- test: CAGR 13.48%, strict MDD 10.54%, ratio 1.28, 204 trades, p=0.423.
- eval: CAGR -13.23%, strict MDD 15.96%, ratio -0.83, 84 trades, p=0.714.

Decision:
- No promotable binary-edge symbolic rule in the capped smoke.
- The longer live-style candidate stream is useful, but the first broad direction again shows 2025-only pockets that fail 2026.
- Runtime caps are necessary for iterative exploration; larger scans should be staged in bounded batches, not one huge run.

## Binary-edge capped scan v2
Run: `results/binary_edge_symbolic_rule_strict_scan_v2_capped_2022_2026_2026-07-01.json`

Protocol:
- train 2022-2024, test 2025, eval 2026 Jan-May.
- generated rules capped at 1,200.
- test-prefiltered candidates: 2,400.
- dedupe candidate cap: 120.
- unique strict candidates: 6.

Result:
- No promotable rule.
- All top strict-tested candidates were negative on test and/or eval despite high cheap prefilter scores.

Top strict-ranked candidate:
- rule: `rex_144_range_width_pct=pos_large` + `rex_8640_cur_to_min_pct=pos_large`, action `invert`.
- train: CAGR -37.09%, strict MDD 77.74%, ratio -0.48, 2,021 trades.
- test: CAGR -17.50%, strict MDD 26.57%, ratio -0.66, 655 trades.
- eval: CAGR -28.29%, strict MDD 14.72%, ratio -1.92, 273 trades.

Decision:
- The binary-edge live-style surface is valid, but the current cheap prefilter is not aligned with strict overlay execution.
- The failure mode is now clearer: isolated candidate reward and strict executed portfolio return diverge, especially for broad predicates and inverted actions.
- Next step should add strict-aligned prefilter constraints or staged strict scoring that rejects candidates with negative train/test overlay before spending more batches.

## Binary-edge staged strict train gate
Scanner update: `training/binary_edge_symbolic_rule_strict_backtest_scan.py`

Problem: cheap prefilter ranked isolated candidate rewards, but strict overlay showed the selected broad predicates were negative.  The scanner now supports an optional staged train strict gate:
- run strict train overlay first,
- reject candidates before test/eval if train strict evidence is weak,
- configurable thresholds: train trades, CAGR, strict MDD, CAGR/MDD, mean-return p-value, and effect size.

Strict-gate smoke:
- generated rules: 400.
- prefiltered candidates: 800.
- dedupe candidates: 60.
- unique candidates checked by train strict gate: 3.
- strict-scanned test/eval candidates: 0.

Rejected examples:
- `hold=432` + `rex_144_range_width_pct=pos_large`, action `invert`: train CAGR 18.14%, MDD 36.60%, ratio 0.50, p=0.167, effect=0.056.
- `hold=432` + `rex_2016_cur_to_min_pct=pos_large`, action `invert`: train CAGR 6.01%, MDD 22.71%, ratio 0.26, p=0.532, effect=0.025.
- `hold=432` + `rex_8640_cur_to_min_pct=pos_large`, action `invert`: train CAGR 4.46%, MDD 47.61%, ratio 0.09, p=0.605, effect=0.021.

Decision:
- The staged strict gate correctly blocks weak broad predicates before eval exposure.
- This should be used for subsequent binary-edge batches; otherwise isolated reward prefilter repeatedly wastes strict test/eval on non-robust candidates.

## Binary-edge staged scan v3
Run: `results/binary_edge_symbolic_rule_strict_scan_v3_stage_2022_2026_2026-07-01.json`

Protocol:
- generated rules: 3,000.
- prefiltered candidates: 5,996.
- dedupe candidates: 240.
- unique train-gated candidates: 12.
- strict-scanned test/eval candidates: 0.

All top candidates were rejected by train strict gate.  The common failure was broad `invert` rules over large range-width predicates.  Examples:
- `rex_144_range_width_pct=pos_large` + `rex_8640_range_width_pct=pos_large`, action `invert`: train CAGR -41.93%, MDD 82.49%, ratio -0.51, p=0.000.
- `rex_2016_range_width_pct=pos_large` + `rex_8640_range_width_pct=pos_large`, action `invert`: train CAGR -38.81%, MDD 79.55%, ratio -0.49, p=0.000.
- `rex_144_range_width_pct=pos_large` + `rex_8640_cur_to_min_pct=pos_large`, action `invert`: train CAGR -37.09%, MDD 77.74%, ratio -0.48, p=0.000.

Decision:
- The staged gate is working: it prevented obviously bad train-strict candidates from consuming eval.
- The current cheap prefilter is dominated by `invert` candidates that are contra-profitable in train strict execution.
- Next search should separate actions and run follow-only / invert-only batches rather than letting broad invert candidates crowd out the candidate queue.

## Binary-edge action-separated smoke
Scanner update:
- added `--actions follow,invert|follow|invert` so broad invert predicates cannot crowd out follow-only searches.

Follow-only strict-gated smoke:
- generated rules: 400.
- prefiltered candidates: 400.
- unique train-gated candidates: 3.
- strict-scanned test/eval candidates: 0.

Rejected follow-only examples:
- `hold=72` + `rex_576_range_pos=pos_mid`, action `follow`: train CAGR -10.97%, MDD 42.92%, ratio -0.26, p=0.301.
- `drawdown_state=low` + `hold=72`, action `follow`: train CAGR -21.74%, MDD 55.32%, ratio -0.39, p=0.019.
- `hold=72` + `rex_144_range_pos=pos_mid`, action `follow`: train CAGR -31.96%, MDD 69.69%, ratio -0.46, p=0.001.

Decision:
- Action separation works mechanically, but the tested follow-only broad predicates are also not profitable in train strict execution.
- Next step should diagnose the binary-edge candidate pool base profitability by family/hold/side before spending more search on combinations.

## Binary-edge base strict diagnostic
New diagnostic: `training/binary_edge_base_strict_diagnostic.py`

Purpose: before adding more symbolic conjunctions, evaluate whether the live-style binary-edge candidate pool has any base tradable slice by singleton `family`, `hold`, `side`, `id`, and `id_side` rules.

Run: `results/binary_edge_base_strict_diagnostic_v1_2022_2026_2026-07-01.json`
- train 2022-2024, test 2025, eval 2026 Jan-May.
- base features evaluated: 133 features x follow/invert = 266 strict backtests.

Top test-ranked slices were not robust:
- `id_side=macro_kimchi_divergence|h288|SHORT`, follow: train CAGR -26.40%, MDD 68.85%; test CAGR 17.92%, MDD 13.14%; eval CAGR -15.53%, MDD 18.94%.
- `id_side=kimchi_extreme_fade|h432|LONG`, invert: train CAGR -17.85%, MDD 59.07%; test CAGR 19.67%, MDD 15.30%; eval CAGR 33.74%, MDD 7.23%, 58 trades, but invalidated by negative train.
- `id_side=kimchi_extreme_fade|h288|LONG`, invert: train CAGR -20.57%, MDD 54.99%; test CAGR 15.29%, MDD 14.79%; eval CAGR -13.34%, MDD 15.78%.

Decision:
- The binary-edge candidate pool has repeated 2025 pockets but lacks stable positive base slices across 2022-2024 train.
- The apparent eval-positive `kimchi_extreme_fade|h432|LONG` inverted slice is not promotable because it loses badly in train.
- Further search should either change the candidate generation/reward design or require train-positive base slices before LLM/rule distillation.

## Binary-edge base diagnostic v2: train/stable rankings
Diagnostic update:
- `training/binary_edge_base_strict_diagnostic.py` now stores `top_by_train` and `top_stable` in addition to `top_by_test`.

Run: `results/binary_edge_base_strict_diagnostic_v2_ranked_2022_2026_2026-07-01.json`
- base features: 133.
- strict backtests: 266.

Best train-ranked slice:
- `id=orderflow_follow|h144`, action `invert`.
- train positive, but test/eval fail; not stable.

Best stable-ranked slice:
- `id_side=orderflow_fade|h144|LONG`, action `follow`.
- train: CAGR 4.85%, MDD 10.44%, ratio 0.46, 433 trades.
- test: CAGR -0.71%, MDD 12.40%, ratio -0.06, 185 trades.
- eval: CAGR 0.94%, MDD 6.55%, ratio 0.14, 62 trades.

Decision:
- Even the best stable singleton slice is economically too weak and fails the target by a wide margin.
- The current binary-edge candidate pool lacks a base strict alpha.  Further LLM rule distillation on this pool is unlikely to reach CAGR/MDD >= 3 without changing candidate generation, reward shaping, or execution/risk overlay.

## Event action target oracle ceiling
New diagnostic: `training/event_action_target_strict_backtest.py`

Purpose: test whether the event-action candidate book contains profitable actions if an oracle target selector chooses the best future-utility-labeled action.  This is not live tradable because targets use future utility labels, but it measures candidate-book ceiling.

Run filters:
- `min_rank_utility=0.003`
- `min_mfe_to_mae=0.8`
- `allowed_confidence=MID,HIGH`
- strict overlay with fees/slippage and 0.5 leverage.

Results:
| Split | Rows | CAGR | Strict MDD | CAGR/MDD | Trades | p-value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train 2022-2024 | 2,715 | 3738.36% | 5.81% | 643.07 | 684 | 0.0000 |
| test 2025 | 901 | 2029.95% | 3.42% | 593.54 | 232 | 0.0000 |
| eval 2026 Jan-May | 364 | 2433.44% | 2.56% | 950.66 | 89 | 0.0000 |

Decision:
- The candidate book has a very high oracle ceiling, unlike the binary-edge symbolic pool.
- The next structure should focus on learning/verifying target-action selection from past-only prompts, not mining the previous binary-edge pool.
- Treat this as a ceiling only; live validation still requires a no-leak selector trained only on prior periods.

## Event action verifier oracle ceiling
New diagnostic: `training/event_action_verifier_target_strict_backtest.py`

Purpose: measure the post-ranker verifier surface.  Rows contain exact executable actions and ALLOW/BLOCK labels derived from future audit.  This is not live tradable, but it shows whether a verifier that could learn ALLOW decisions would have enough ceiling.

Inputs:
- train: `event_action_verifier_text_v3k8_train_2020_2024_wavefull_regen_pae_2026-06-27.jsonl`, filtered to 2022-2024.
- test: `event_action_verifier_text_v3k8_2025_wavefull_regen_pae_2026-06-27.jsonl`.
- eval: `event_action_verifier_text_v3k8_2026_jan_may_wavefull_regen_pae_2026-06-27.jsonl`.

Oracle ALLOW strict results:
| Split | Rows | CAGR | Strict MDD | CAGR/MDD | Trades | p-value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train 2022-2024 | 1,190 | 1630.08% | 3.74% | 435.52 | 411 | 0.0000 |
| test 2025 | 362 | 1010.92% | 5.21% | 194.14 | 125 | 0.0000 |
| eval 2026 Jan-May | 147 | 1256.69% | 2.92% | 430.77 | 53 | 0.0000 |

Decision:
- The verifier surface has a strong oracle ceiling and is much more promising than the binary-edge symbolic pool.
- The right structure is a no-leak learned verifier/post-ranker: use LLM reasoning over categorical past-only prompt + exact action, then strict backtest accepted actions.
- Next step should be a no-leak baseline verifier, then LLM distillation/fine-tuning only if the baseline can recover part of the oracle without leakage.

## 2026-07-01 verifier-token structure/alpha baseline

Purpose: test whether the LLM-friendly verifier prompt already contains usable no-leak categorical alpha before spending GPU time on Gemma fine-tuning.  The baseline trains only on train labels, then post-ranks exact candidate actions per signal timestamp and selects thresholds by 2025 test strict backtest.

Implementation: `training/event_action_verifier_token_baseline.py`.

Key structural fixes added:
- Parse the actual prompt alpha surface (`Regime tokens`, `Candidate book tokens`, `Selected action tokens`, and `state_tokens`) instead of only action family/side/horizon.
- Add low-cardinality family/side × regime interactions so weak alphas can combine conditionally.
- For each signal timestamp, score all candidate actions and emit only the highest-scored candidate; this matches live post-ranker semantics and avoids first-row file-order bias.
- Store compact scan reports by stripping bulky executed-trade arrays from result JSON.

Strict no-leak protocol:
- Fit token reliability on 2020-2024 train rows only.
- Choose threshold by 2025 test score.
- Report 2026 Jan-May eval untouched.
- Backtest uses entry delay 1 bar and strict MDD including intrabar adverse excursion.

Results:

| run | scoring | selected threshold | train | test | eval | verdict |
| --- | --- | ---: | --- | --- | --- | --- |
| v3 prompt-token mean, first accepted row | mean | 0.09 | 5.51% CAGR / 55.55% MDD / 1505 trades | -5.81% / 14.75% / 301 trades | -26.86% / 17.45% / 132 trades | Reject: prompt tokens alone did not generalize and file-order action selection was structurally wrong. |
| v4 prompt-token best-action | mean | 0.105 | 16.83% CAGR / 32.10% MDD / 231 trades | 0.86% / 10.08% / 35 trades | 18.89% / 8.35% / 14 trades | Weak lead: structure improved, but test/eval trade counts are too low and p-values are weak. |
| v4 prompt-token best-action | max | 0.09 | 27.08% CAGR / 55.44% MDD / 1427 trades | 1.44% / 18.28% / 270 trades | -14.76% / 19.85% / 114 trades | Reject: statistically broader but eval fails. |

Readout:
- The verifier/candidate-book ceiling remains strong, but train-only token reliability is far too weak to recover it robustly.
- The important structural gain is not the token score itself; it is the live-compatible `score all exact actions -> pick best action -> threshold gate` verifier shape.
- Next useful move is a richer learned verifier over these same no-leak symbolic tokens (e.g. online logistic/FTRL or Gemma distillation), not more one-token gates.

## 2026-07-01 sparse linear verifier baseline

Purpose: test whether multiple weak symbolic verifier features can combine into a generalizable no-leak verifier before moving to Gemma distillation.

Implementation: `training/event_action_verifier_linear_baseline.py`.

Structure:
- Reuses the same no-leak prompt/action/state tokens as the token verifier.
- Trains an online sparse logistic model on 2020-2024 only.
- Uses live-compatible post-ranking: score all exact actions per signal, choose the highest-scored action, then threshold.
- Chooses threshold on 2025 test and reports 2026 Jan-May eval untouched.

Run: `results/event_action_verifier_linear_baseline_v1_2026-07-01.json`

| selected threshold | train | test | eval | verdict |
| ---: | --- | --- | --- | --- |
| 0.20 | 22.77% CAGR / 40.59% MDD / 1007 trades / p=0.049 | 0.53% / 18.58% / 204 trades / p=0.906 | -33.68% / 22.32% / 84 trades / p=0.184 | Reject: train signal does not generalize. |
| 0.25 | 27.42% / 32.63% / 982 trades / p=0.025 | -2.62% / 15.29% / 199 trades / p=0.953 | -47.03% / 26.58% / 82 trades / p=0.033 negative | Reject. |
| 0.65 | 9.25% / 31.90% / 236 trades / p=0.155 | -3.59% / 15.61% / 58 trades / p=0.789 | -16.08% / 9.55% / 19 trades / p=0.297 | Reject. |

Readout:
- Sparse linear composition improved in-sample CAGR but exposed clear overfit/regime drift.
- This strongly suggests the current event-action prompt tokens are not enough as a standalone alpha source, even though the oracle labels have a high ceiling.
- Next structural/alpha move should change the candidate generator/feature surface, not merely the verifier optimizer: add price-action location/rolling-extrema semantics into candidate families and prompt tokens, then re-run the same verifier protocol.

## 2026-07-01 rolling-extrema alpha surface expansion

Purpose: improve both alpha and structure by adding rolling max/min price-action location tokens to the verifier rows, then replaying the same no-leak verifier protocol.

Data augmentation:
- Script used: `training.augment_event_candidate_rolling_extrema`.
- Inputs: existing verifier train/test/eval rows with PAE tokens.
- Outputs: `*_rex_2026-07-01.jsonl` generated under `data/` (ignored artifact).
- Match rate: 100% for train/test/eval.
- Added feature surface: rolling extrema windows `36,72,144,288,576,2016,4032,8640`; token windows `72,144,288,576,2016,4032,8640`.
- Leakage guard: backward-asof market join; features use candles at or before signal timestamp; reward fields unchanged.

### REX token verifier

Run: `results/event_action_verifier_token_baseline_v5_rex_mean_2026-07-01.json`

| selected threshold | train | test | eval | verdict |
| ---: | --- | --- | --- | --- |
| 0.115 | 21.76% CAGR / 23.84% MDD / 186 trades / p=0.0038 | 1.40% / 5.92% / 31 trades / p=0.803 | 29.69% / 3.86% / 12 trades / p=0.056 | Weak lead: both test/eval positive and MDD controlled, but trade counts are too low for promotion. |
| 0.110 | 23.56% / 33.37% / 271 trades / p=0.0062 | 0.25% / 10.00% / 46 trades / p=0.941 | 17.20% / 7.19% / 14 trades / p=0.376 | Weak lead but statistically thin. |
| 0.090 | 18.41% / 58.52% / 975 trades / p=0.103 | -1.46% / 17.65% / 178 trades / p=0.985 | -18.89% / 16.71% / 78 trades / p=0.490 | Reject. |

### REX sparse linear verifier

Run: `results/event_action_verifier_linear_baseline_v2_rex_2026-07-01.json`

| selected threshold | train | test | eval | verdict |
| ---: | --- | --- | --- | --- |
| 0.70 | 18.76% CAGR / 19.31% MDD / 170 trades / p=0.00022 | -5.14% / 11.68% / 42 trades / p=0.576 | 22.68% / 6.36% / 20 trades / p=0.193 | Reject: threshold chosen by test is still negative. |
| 0.40 | 21.69% / 36.00% / 506 trades / p=0.022 | -6.62% / 16.35% / 137 trades / p=0.710 | -16.48% / 14.07% / 56 trades / p=0.595 | Reject. |
| 0.30 | 29.87% / 35.89% / 660 trades / p=0.0077 | -6.12% / 14.07% / 166 trades / p=0.741 | 3.21% / 14.66% / 69 trades / p=0.868 | Reject. |

Readout:
- Rolling max/min location is useful: adding REX creates the first verifier baseline where selected test and eval are both positive with low MDD, but the trade count is too small.
- Over-parameterized sparse linear composition overfits badly, even with REX. This argues against heavy numeric/classifier optimization as the next move.
- Best next structure: keep conservative REX/token-style selection, widen statistically by generating more high-quality candidate opportunities and using LLM-style deductive filters (rule consistency, contradiction, side/horizon sanity), not by stronger gates alone.

## 2026-07-01 REX candidate-family expansion

Purpose: move REX/location alpha earlier into candidate generation instead of relying on a late verifier gate.

Implementation: `training/event_candidate_pool_probe.py` now adds these past-only candidate families:
- `rex_multiscale_extreme_fade`
- `rex_extreme_breakout_follow`
- `rex_compression_breakout`
- `rex_compression_fakeout`
- `rex_htf_pullback_resume`
- `rex_multiscale_location_revert`

Also fixed family ranking so 0-trade `Infinity` ratios no longer dominate `top_val` or fallback selection.

Protocol:
- Train threshold: 2020-2024 only.
- Validation/family selection: 2025 only, after train-positive/trade-count filter.
- Eval: 2026 Jan-May/Jun boundary only, not used for selection.
- Candidate backtest is non-overlapping via `next_allowed=exit_pos`; stride controls candidate timing density, not overlapping execution.

Key fixed-family readout for `rex_htf_pullback_resume`:

| hold / q / stride | train | 2025 val | 2026 eval | readout |
| --- | --- | --- | --- | --- |
| h288 q0.85 s24 | 20.90% CAGR / 20.11% MDD / 573 trades / p=0.026 | 7.33% / 6.53% / 63 / p=0.581 | 14.71% / 6.72% / 33 / p=0.629 | Stable direction, thin stats. |
| h288 q0.80 s24 | 11.95% / 33.53% / 730 / p=0.174 | 2.69% / 14.30% / 111 / p=0.805 | 23.37% / 7.46% / 46 / p=0.486 | Best eval ratio/trade balance; val weak but positive. |
| h288 q0.75 s12 | 6.02% / 44.96% / 994 / p=0.423 | 12.49% / 11.96% / 165 / p=0.408 | 7.81% / 7.93% / 61 / p=0.758 | Wider opportunity, lower ratio. |
| h288 q0.85 s12 | 17.13% / 30.85% / 645 / p=0.063 | 7.53% / 7.55% / 69 / p=0.558 | 21.95% / 7.93% / 37 / p=0.515 | Repeated positive, still thin. |
| h144 / h432 variants | mixed | mixed | mixed | Not stable enough. |

Rejected families:
- `rex_multiscale_extreme_fade`: negative train/val/eval.
- `rex_multiscale_location_revert`: negative train/val/eval.
- `rex_compression_*`: inconsistent; eval or train fails.
- `macro_kimchi_divergence` and `vol_compression_breakout`: strong 2025 validation but negative train/eval, likely regime overfit.

Readout:
- `rex_htf_pullback_resume` is now the most credible candidate-family lead: it repeats positive on train/val/eval under several h288 settings and includes both long/short actions.
- It is not statistically sufficient yet: p-values remain weak and eval trades are mostly 33-61.
- Next step: regenerate verifier/action-book rows after adding REX candidate families, then test whether the conservative verifier can pick a broader, cleaner subset from a better candidate book.

## 2026-07-01 REX family-selection → verifier structure

After adding REX candidate families, v4rex verifier rows were regenerated:
- train 2020-2024: 233,856 rows, ALLOW 17,478, allow-rate 7.47%.
- test 2025: 46,720 rows, ALLOW 2,424, allow-rate 5.19%.
- eval 2026 Jan-May: 19,104 rows, ALLOW 1,065, allow-rate 5.57%.

The oracle ceiling stayed extremely high and roughly unchanged versus the prior verifier split:
- train: 2893.12% CAGR / 11.71% MDD / 787 trades.
- test: 1010.92% / 5.21% / 125 trades.
- eval: 1256.69% / 2.92% / 53 trades.

Distribution check:
- REX families entered the candidate book often (`rex_htf_pullback_resume` book appearances: train 53,984; test 9,760; eval 3,296).
- But unconstrained v6 verifier selected unstable REX families (`rex_extreme_breakout_follow`, `rex_multiscale_extreme_fade`) and failed eval:
  - threshold 0.115 selected by 2025 test: train 22.74% / 24.48% / 309 trades; test 8.08% / 13.26% / 98 trades; eval -18.61% / 11.54% / 33 trades.

Structural fix:
- Add `--allowed-families` to `training/event_action_verifier_token_baseline.py` so a prior train/val family-selection layer can constrain the verifier's action universe before per-signal best-action selection.
- Tested whitelist `rex_htf_pullback_resume`, the only REX family that repeated positive in train/2025/2026 standalone probes.

Whitelist verifier result: `results/event_action_verifier_token_baseline_v7_rex_htf_whitelist_2026-07-01.json`

| threshold selected by 2025 test | train | 2025 test | 2026 eval | verdict |
| ---: | --- | --- | --- | --- |
| 0.07 | 17.16% CAGR / 32.04% MDD / 470 trades / p=0.061 | 19.16% / 8.34% / 62 trades / p=0.147 | 11.32% / 4.88% / 28 trades / p=0.565 | Best structural lead so far, but eval trade count still too low. |
| 0.08 | 16.32% / 25.52% / 296 trades / p=0.032 | -1.29% / 7.18% / 27 trades / p=0.883 | 55.36% / 4.55% / 12 trades / p=0.041 | Not selectable because 2025 test is negative. |

Readout:
- The important structure is now: **train/val stable family selection → conservative verifier within the selected family → untouched eval**.
- This avoids letting the verifier chase unstable REX subfamilies that looked good in 2025 but failed 2026.
- Still not production-ready: eval has only 28 trades and p=0.565, so the next goal is to widen `rex_htf_pullback_resume` opportunities without losing the low-MDD profile.
## 2026-07-02 REX whitelist low-threshold trade-count sweep

Purpose: check whether the `rex_htf_pullback_resume` verifier lead can be widened by lowering the score threshold, while keeping the train/2025-selected family whitelist and leaving 2026 eval untouched.

Run: `results/event_action_verifier_token_baseline_v8_rex_htf_whitelist_lowthr_2026-07-02.json` with `--allowed-families rex_htf_pullback_resume`, `--score-mode mean`, and thresholds `0.00..0.08`.

| threshold | train | 2025 test | 2026 eval | readout |
| ---: | --- | --- | --- | --- |
| 0.070 | 17.16% CAGR / 32.04% MDD / 470 trades / p=0.061 | 19.16% / 8.34% / 62 / p=0.147 | 11.32% / 4.88% / 28 / p=0.565 | Selected by 2025 test score; clean MDD but eval too thin. |
| 0.065 | 14.18% / 31.81% / 524 / p=0.112 | 21.22% / 9.91% / 81 / p=0.108 | 13.84% / 7.21% / 36 / p=0.529 | Better trade count, still weak significance. |
| 0.060 | 11.55% / 35.23% / 543 / p=0.177 | 17.79% / 11.21% / 98 / p=0.181 | 23.77% / 7.21% / 38 / p=0.304 | Best eval CAGR/MDD among wider settings, but not statistically enough. |
| 0.000-0.020 | 12.15% / 35.93% / 554 / p=0.161 | 16.40% / 11.48% / 99 / p=0.211 | 27.80% / 7.21% / 38 / p=0.242 | Score adds little once the family whitelist is loose. |

Readout:
- Lowering the threshold widens test/eval trades modestly and keeps all three splits positive, which supports `rex_htf_pullback_resume` as a real lead rather than a single threshold artifact.
- The score threshold is not the main edge below about `0.06`; the family-level rule is carrying most of the signal.
- Still not production-ready: train strict MDD is 31-36%, 2026 eval has only 28-38 trades, and all eval p-values remain weak.
- Next work should target risk/exit/hold variants for the selected family, then rolling-fold validation. Adding more verifier capacity is lower priority because prior sparse/linear variants overfit.

## 2026-07-02 REX whitelist risk-overlay mini sweep

Purpose: test whether strict MDD can be reduced after the REX family/threshold lead, without using eval for selection. The first broad grid was too slow because ATR recomputation dominated, so this pass intentionally isolates fast live-usable overlays: per-trade take-profit, stop-loss, monthly loss stop, and cooldown.

Run: `results/verifier_risk_overlay_sweep_v1_rex_htf_min_2026-07-02.json` from `training/verifier_risk_overlay_sweep.py`. Selection score uses train + 2025 test only; eval remains report-only.

Best selected overlay: threshold `0.065`, take-profit `8%`, monthly-loss stop `6%`, no stop-loss, no cooldown.

| split | result | delta vs no-overlay threshold 0.065 |
| --- | --- | --- |
| train | 13.33% CAGR / 23.52% MDD / 483 trades | MDD improved from 31.81%, CAGR slightly lower. |
| 2025 test | 21.22% / 9.91% / 81 trades | Essentially unchanged. |
| 2026 eval | 13.84% / 7.21% / 36 trades | Essentially unchanged. |

Readout:
- The overlay reduces some historical train tail risk, so it is useful as a risk-control layer.
- It does not create new alpha and does not fix the statistical weakness: eval remains only 36 trades and p-values are still weak.
- Stop-loss settings were not selected in the mini grid; hard stops likely cut winners/losers symmetrically at this 5m/hold horizon.
- Next structural work should improve entry/feature quality around `rex_htf_pullback_resume` rather than relying on gates. Candidate directions: HTF location buckets, multi-timeframe pullback depth, trend-strength/volatility context, and explicit short-side variants.

## 2026-07-02 REX pullback feature-family variants

Purpose: improve the actual alpha/entry structure instead of relying on gates. Added focused `rex_htf_*` variants to `training/event_candidate_pool_probe.py`:
- `rex_htf_pullback_reclaim`: higher-timeframe pullback plus local trend reclaim.
- `rex_htf_deep_pullback_resume`: emphasizes deeper range-location pullbacks.
- `rex_htf_context_pullback_resume`: requires long-range location/trend context.
- `rex_htf_long_pullback_resume` / `rex_htf_short_pullback_resume`: side-split diagnostics.

Also added `--family-include` so focused probes do not waste time on unrelated families.

Protocol: train 2020-2024, validation 2025, eval 2026-01-01..2026-06-01, hold 288, stride 24. Family/threshold selection uses train+validation only; eval is untouched.

Key focused probe results:

| q | family | train | 2025 val | 2026 eval | readout |
| ---: | --- | --- | --- | --- | --- |
| 0.80 | `rex_htf_pullback_resume` | 11.95% / 33.53% / 730 / p=0.174 | 2.69% / 14.30% / 111 / p=0.805 | 23.37% / 7.46% / 46 / p=0.486 | Original wider setting; positive eval but weak val. |
| 0.80 | `rex_htf_context_pullback_resume` | 8.59% / 38.52% / 833 / p=0.286 | 9.46% / 10.24% / 146 / p=0.506 | -13.90% / 10.74% / 57 / p=0.622 | Validation trap; reject despite best val. |
| 0.85 | `rex_htf_pullback_resume` | 20.90% / 20.11% / 573 / p=0.026 | 7.33% / 6.53% / 63 / p=0.581 | 14.71% / 6.72% / 33 / p=0.629 | Original stable narrow lead. |
| 0.85 | `rex_htf_pullback_reclaim` | 8.21% / 40.30% / 759 / p=0.303 | 14.63% / 12.34% / 103 / p=0.267 | 13.79% / 7.37% / 39 / p=0.611 | More validation trades/return; train MDD too high. |
| 0.85 | `rex_htf_deep_pullback_resume` | 11.88% / 29.93% / 628 / p=0.167 | 10.63% / 11.47% / 88 / p=0.405 | 8.24% / 7.60% / 38 / p=0.746 | Robustly positive but weak. |
| 0.85 | `rex_htf_short_pullback_resume` | -6.37% / 44.88% / 472 / p=0.550 | 12.29% / 12.59% / 123 / p=0.392 | 10.74% / 7.46% / 34 / p=0.823 | Short-only 2025/2026 positive but train negative; not selectable. |

Readout:
- `reclaim` is a useful new hypothesis because it improves 2025 trade count and return while keeping eval positive, but it is not yet stable enough: train MDD is ~40% and train p-value is weak.
- `context` demonstrates why eval must remain untouched: it looked best in 2025 but failed 2026.
- Side split shows shorts can work in recent regimes but are not stable over 2020-2024, so short specialization needs regime conditioning rather than unconditional side filters.
- Next: combine original `pullback_resume` and `pullback_reclaim` as candidate-book alternatives, then let the conservative verifier choose within that restricted pair. Do not promote `context`.

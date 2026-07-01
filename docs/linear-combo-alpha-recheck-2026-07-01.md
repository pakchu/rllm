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

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

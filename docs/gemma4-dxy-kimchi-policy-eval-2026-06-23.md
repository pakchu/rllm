# Gemma4 DXY/Kimchi Policy Evaluation — 2026-06-23

## Setup

- Model: `gemma4-e4b` alias -> `google/gemma-4-E4B-it`
- Adapter: `checkpoints/dxy_kimchi_policy_gemma4_e4b_sft24_2026-06-23`
- Policy shape: single compact JSON `{activate, action, exit_profile, confidence, reason_code}`
- Prompt inputs: causal DXY/Kimchi/Binance aux/regime text state plus train-fitted prior.
- Splits:
  - test: `2024-07-01 03:00:00` .. `2025-08-31 15:00:00`, 854 rows
  - eval: `2025-09-01 03:00:00` .. `2026-05-30 15:00:00`, 544 rows
- Leakage guard: model input excludes target; targets use future path reward only for supervised labels/metrics.

## Evaluator changes

`training/eval_dxy_kimchi_policy.py` now supports:

- `prediction_mode=model`: real adapter generation with left-padded batch decoding for decoder-only models.
- `prediction_mode=candidate_logprob`: candidate scoring smoke path. This is not production-valid for this adapter because it remains calibrated toward `NO_TRADE` even when generation works.
- `prediction_mode=target_echo`: oracle/upper-bound diagnostic only.
- prediction export compatible with `training.online_risk_overlay_backtest`.

## Results

### Generation classifier metrics

| Split | Rows | Target trades | Model trades | Activate/action acc. | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| test | 854 | 52 | 129 | 90.98% | False positives: 77 extra trades |
| eval | 544 | 27 | 75 | 91.18% | False positives: 48 extra trades |

### Strict backtest on real generated predictions

| Split | Period | Trades | CAGR | Strict MDD | CAGR/MDD | p approx |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| test | 2024-07-01 .. 2025-08-31 | 109 | 2.58% | 8.26% | 0.31 | 0.739 |
| eval | 2025-09-01 .. 2026-05-30 | 64 | -6.41% | 7.40% | -0.87 | 0.369 |

### Target-oracle diagnostic only

| Split | Trades | CAGR | Strict MDD | CAGR/MDD | Why not usable |
| --- | ---: | ---: | ---: | ---: | --- |
| test | 49 | 46.24% | 2.81% | 16.43 | Target uses future path reward |
| eval | 24 | 24.72% | 2.71% | 9.11 | Target uses future path reward |

## Interpretation

The adapter learned the label format and can reproduce target actions on balanced samples. The failure is deployment calibration: it over-activates the profitable prior and turns many target `NO_TRADE` rows into trades. The prior/label upper bound still has edge, but the current RLLM policy does not learn the abstention boundary tightly enough for live use.

## Next direction

Do not continue by gate-threshold tuning alone. The next useful change is to train the LLM on abstention/risk contrast:

1. Add hard-negative rows: prior present but future path rejected, plus near-threshold regime rows.
2. Make the output include a causal abstention rationale bucket, not just `activate`.
3. Add validation objective that penalizes false-positive trades more than missed oracle trades.
4. Evaluate with chronological full test/eval only; balanced samples are smoke tests, not performance evidence.

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

## Follow-up: hard-negative 10:1 SFT128

After the first evaluation showed deployment false positives, a train-only hard-negative export was generated with `NO_TRADE:ACTIVE = 10:1`:

- Train rows: 550
- Train targets: NO_TRADE 500, LONG 31, SHORT 19
- Adapter: `checkpoints/dxy_kimchi_policy_gemma4_e4b_hardneg10_sft128_2026-06-23`
- Training: 128 steps, effective batch 4, final train loss 0.2098

### Chronological full test result

| Model | Test trades | Pred mix | CAGR | Strict MDD | CAGR/MDD | p approx |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| SFT24 balanced 3:1 | 109 | LONG 59 / SHORT 70 / NO_TRADE 725 | 2.58% | 8.26% | 0.31 | 0.739 |
| SFT128 hard-negative 10:1 | 26 | LONG 27 / SHORT 0 / NO_TRADE 827 | 3.78% | 4.83% | 0.78 | 0.385 |

### Interpretation

Hard-negative weighting reduced false positives and drawdown, but overcorrected into a LONG-only conservative policy and lost all SHORT recall. This confirms the next iteration needs side-specific abstention contrast, not just more NO_TRADE rows. The likely useful dataset shape is:

- preserve `prior_signal_path_reward_rejected` rows separately for LONG and SHORT;
- balance active LONG, active SHORT, rejected LONG-prior, rejected SHORT-prior, and no-prior rows;
- add explicit `prior_side`/`abstain_reason` target fields or reason buckets so the model learns *why* a prior is rejected rather than treating abstention as one majority class.

## Follow-up: side-specific contrast and oversampling

Two side-aware train-only selectors were tested after hard-negative 10:1 collapsed SHORT recall.

| Adapter | Train rows | Train mix | Test pred mix | Test trades | CAGR | Strict MDD | CAGR/MDD | Outcome |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| sidecontrast05 SFT80 | 209 | active 50 / rejected-prior 89 / no-prior 70 | smoke only: LONG 9 / SHORT 0 / NO_TRADE 51 | n/a | n/a | n/a | n/a | Still no SHORT recall |
| sidecontrast_os1 SFT80 | 225 | LONG 45 / SHORT 45 / rejected 90 / no-prior 45 | full test: LONG 32 / SHORT 59 / NO_TRADE 763 | 83 | -8.31% | 11.63% | -0.71 | SHORT false positives dominate |

### Updated diagnosis

The policy is not simply under/over-weighting the inactive class. The LLM can be pushed between three bad regimes:

1. balanced 3:1: both sides active but too many false positives;
2. hard-negative 10:1: fewer false positives but LONG-only and low trade count;
3. side oversampling: SHORT recall returns but false-positive SHORTs destroy expectancy.

The next useful fix is not more class balancing. The model needs a causal feature that distinguishes rejected SHORT contexts from valid SHORT contexts, or the target schema must expose that distinction explicitly. Candidate next experiment: add compact numeric/rank tokens for side-specific margin from Kimchi thresholds, DXY depth, and recent BTC trend/volatility interaction, then train a cost-sensitive abstention objective.

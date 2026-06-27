# Short setup-quality audit (2026-06-27)

## Purpose

The prior sequence-macro short features failed under strict execution. This audit re-tests short-side alpha using non-sequence price-action events plus setup-quality buckets, with a longer robustness split from the start:

- Train: 2020-01-01 through 2023-12-31
- Test: 2024-01-01 through 2025-12-31
- Eval: 2026-01-01 through 2026-06-01

## Search surface

- Windows: 288, 576, 2016, 4032
- Events: `high_sweep_reject`, `failed_breakout_short`, `reject_mid_from_above`, `lower_high_mid_reject`, `lower_low_mid_fail`, `downtrend_pullback_reject`, `failed_mid_reclaim_short`, `break_below`
- Horizons: 72, 144, 288, 432
- Setup-quality buckets: train-quantile low/mid/high for `risk_bps`, `range_bps`, `body_frac`, `favorable_wick_frac`, `close_quality`
- Minimum train trades: 40

Command output:

- `trigger_rows`: 1,344,056
- `candidates`: 1,869
- Output: `results/setup_quality_filters_short_expanded_train2020_2023_test2024_2025_2026-06-27/report.json`

## Findings

Most top test-ranked candidates were negative in 2020-2023 train, confirming that the attractive short performance is concentrated in recent regimes.

A few top-100 candidates were train/test positive, but they are weak and mostly fail or become sparse in 2026 eval:

| Rule | Train CAGR/MDD/Trades | Test CAGR/MDD/Trades | Eval CAGR/MDD/Trades | Decision |
| --- | --- | --- | --- | --- |
| `pae_w4032_failed_mid_reclaim_short@144:risk_bps=mid` | 1.23 / 8.58 / 74 | 2.86 / 2.89 / 31 | -0.76 / 3.21 / 8 | Reject: eval negative, sparse |
| `pae_w288_downtrend_pullback_reject@288:range_bps=mid` | 4.17 / 17.28 / 174 | 3.02 / 21.30 / 98 | -6.18 / 9.88 / 21 | Reject: MDD too high, eval negative |
| `pae_w4032_lower_high_mid_reject@432:favorable_wick_frac=high` | 0.57 / 13.52 / 61 | 3.50 / 7.29 / 31 | -1.82 / 4.68 / 5 | Reject: sparse eval, weak train |
| `pae_w4032_lower_high_mid_reject@432:risk_bps=high` | 3.51 / 16.19 / 64 | 3.06 / 7.49 / 27 | -6.10 / 5.18 / 5 | Reject: sparse eval, negative |
| `pae_w288_downtrend_pullback_reject@432:range_bps=mid` | 2.18 / 29.27 / 163 | 2.81 / 23.77 / 90 | -11.88 / 12.68 / 21 | Reject: MDD too high, eval negative |
| `pae_w4032_failed_mid_reclaim_short@288:risk_bps=mid` | 2.71 / 9.50 / 58 | 2.49 / 4.81 / 28 | 9.20 / 3.22 / 7 | Reject for now: eval positive but only 7 trades |

## Decision

1. Short-side non-sequence setup-quality features are not yet production candidates.
2. The best short candidate is `pae_w4032_failed_mid_reclaim_short@288:risk_bps=mid`, but eval has only 7 trades, so it is not statistically useful.
3. Do not add these short rules to the portfolio yet.
4. Next short work should focus on generating less sparse short labels or using short rules as LLM context features rather than direct execution triggers.
5. Current best practical direction remains setup-quality feature engineering for RLLM text labels, not direct symbolic trading rules.

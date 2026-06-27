# Expanded setup-quality portfolio audit (2026-06-27)

## Purpose

After raw episode triggers and structural exits failed, setup-quality buckets became the only promising direction. This audit expands the long-side setup-quality surface and then tests a fixed predeclared setup-quality portfolio without using eval for rule application.

## Expanded long setup-quality audit

Spec surface:

- Windows: 288, 576, 2016, 4032
- Events: `failed_mid_loss_long`, `failed_breakdown_long`, `low_sweep_reclaim`, `reclaim_mid_from_below`, `higher_low_mid_reclaim`, `uptrend_pullback_reclaim`, `break_above`
- Horizons: 72, 144, 288, 432
- Quality buckets: train-quantile low/mid/high for `risk_bps`, `range_bps`, `body_frac`, `favorable_wick_frac`, `close_quality`

Command produced:

- `trigger_rows`: 350,260
- `candidates`: 1,439
- Output: `results/setup_quality_filters_long_expanded_2026-06-27/report.json`

Notable train/test/eval-positive rows from the top set:

| Rule | Train CAGR/MDD/Trades | Test CAGR/MDD/Trades | Eval CAGR/MDD/Trades | Test p |
| --- | --- | --- | --- | ---: |
| `pae_w576_low_sweep_reclaim@72:favorable_wick_frac=high` | 4.51 / 7.77 / 99 | 16.28 / 7.34 / 119 | 6.88 / 4.93 / 47 | 0.059 |
| `pae_w576_low_sweep_reclaim@144:range_bps=high` | 17.51 / 9.37 / 76 | 14.98 / 6.04 / 49 | 9.88 / 8.13 / 27 | 0.119 |
| `pae_w2016_reclaim_mid_from_below@288:body_frac=low` | 1.01 / 12.93 / 66 | 10.54 / 4.97 / 71 | 14.61 / 5.95 / 33 | 0.200 |
| `pae_w288_failed_breakdown_long@432:body_frac=low` | 17.05 / 22.79 / 66 | 14.36 / 9.54 / 74 | 9.20 / 10.85 / 29 | 0.203 |
| `pae_w2016_failed_mid_loss_long@288:favorable_wick_frac=low` | 3.15 / 6.57 / 30 | 6.74 / 3.57 / 29 | 34.48 / 1.69 / 12 | 0.187 |

These are still weak statistically, but they are better than raw event triggers and sequence macro shorts.

## Fixed setup-quality portfolio: 2024 train / 2025 test / 2026 eval

Rules:

```text
pae_w576_low_sweep_reclaim@72:favorable_wick_frac=high,
pae_w576_low_sweep_reclaim@144:range_bps=high,
pae_w2016_reclaim_mid_from_below@288:body_frac=low,
pae_w288_failed_breakdown_long@432:body_frac=low,
pae_w2016_failed_mid_loss_long@288:favorable_wick_frac=low
```

Output: `results/setup_quality_fixed_portfolio_long_train2024_test2025_eval2026jm_2026-06-27/report.json`

| Split | CAGR | Strict MDD | Ratio | Trades | p-value |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train 2024 | 2.83% | 21.33% | 0.13 | 153 | 0.806 |
| Test 2025 | 12.35% | 10.91% | 1.13 | 174 | 0.341 |
| Eval 2026-06 | 19.03% | 8.97% | 2.12 | 78 | 0.501 |

This is the best recent structure direction: more trades, lower eval MDD, and positive 2026 eval. However, it still fails the target and lacks statistical strength.

## Longer validation: 2020-2023 train / 2024-2025 test / 2026 eval

Same rules, but bucket thresholds fit on 2020-2023 train.

Output: `results/setup_quality_fixed_portfolio_long_train2020_2023_test2024_2025_eval2026jm_2026-06-27/report.json`

| Split | CAGR | Strict MDD | Ratio | Trades | p-value |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train 2020-2023 | -2.06% | 41.08% | -0.05 | 617 | 0.990 |
| Test 2024-2025 | 14.68% | 14.70% | 1.00 | 304 | 0.175 |
| Eval 2026-06 | -1.60% | 13.21% | -0.12 | 68 | 0.989 |

The same setup-quality rules do not survive older train history. This suggests the current long quality surface is regime-specific to 2024-2025 and not a durable all-history alpha.

## Decision

1. Setup-quality features are directionally useful, but current long-only rules are not a production edge.
2. The next RLLM dataset should not train on raw event names; it should encode setup quality and invalidation distance.
3. Selection must include a long-history robustness constraint. A rule that fails 2020-2023 train should not be promoted even if 2024-2026 looks attractive.
4. Need independent weak alphas outside long liquidity-reclaim / failed-breakdown families, especially short-side features that are not dense sequence triggers.
5. For the next search, optimize for stability first:
   - train 2020-2023 must not be negative;
   - test 2024-2025 must be positive;
   - eval 2026 is only a final report;
   - trade count should remain >200 in train+test combined.

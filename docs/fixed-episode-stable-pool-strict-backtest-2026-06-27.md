# Fixed stable episode pool strict backtest (2026-06-27)

## Why

The feature stability audit showed many episode features flip sign quickly, but identified a few train/test/eval-positive diagnostic candidates. This follow-up checks whether those candidates still work under the real strict simulator: next-open entry, scheduled hold, transaction costs, cooldown-by-position occupancy, and strict MDD with intrabar adverse excursions.

The key point: diagnostic open-to-open feature means are not enough. A deployable policy must survive strict bar simulation.

## Fixed portfolio command

```bash
.venv/bin/python -m training.fixed_episode_template_backtest \
  --input-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz \
  --output results/fixed_episode_stable_pool_train2024_test2025_eval2026jm_2026-06-27/report.json \
  --specs 'pae_w2016_seq_bear_failed_bounce@432,pae_w2016_seq_bear_breakdown_macro@432,pae_w4032_failed_breakout_short@288,pae_w2016_failed_mid_loss_long@288' \
  --train-start 2024-01-01 --train-end '2024-12-31 23:59:59' \
  --test-start 2025-01-01 --test-end '2025-12-31 23:59:59' \
  --eval-start 2026-01-01 --eval-end 2026-06-01 \
  --windows 36,72,144,288,576,2016,4032,8640 \
  --max-trigger-overlap 0.80
```

`pae_w2016_seq_bear_breakdown_macro@432` was rejected because it overlapped `pae_w2016_seq_bear_failed_bounce@432` with Jaccard 0.895.

## Fixed portfolio result

| Split | CAGR | Strict MDD | Ratio | Trades | Side counts | p-value |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Train 2024 | -15.91% | 21.17% | -0.75 | 131 | LONG 31 / SHORT 100 | 0.371 |
| Test 2025 | -3.16% | 13.66% | -0.23 | 117 | LONG 35 / SHORT 82 | 0.880 |
| Eval 2026-06 | -1.45% | 17.19% | -0.08 | 54 | LONG 15 / SHORT 39 | 0.988 |

## Single-template strict decomposition

| Spec | Train CAGR/MDD/Trades | Test CAGR/MDD/Trades | Eval CAGR/MDD/Trades | Interpretation |
| --- | --- | --- | --- | --- |
| `pae_w2016_seq_bear_failed_bounce@432` | -10.57 / 17.52 / 56 | -5.40 / 14.14 / 57 | -11.69 / 16.44 / 29 | Fails all splits under strict simulation. |
| `pae_w2016_seq_bear_failed_bounce@288` | -13.18 / 19.79 / 70 | -1.68 / 9.00 / 71 | -12.15 / 15.76 / 42 | Fails all splits. |
| `pae_w2016_seq_bear_failed_bounce@144` | -10.91 / 18.92 / 107 | -7.92 / 16.50 / 113 | -19.38 / 16.15 / 57 | Fails all splits. |
| `pae_w2016_seq_bear_breakdown_macro@432` | -8.67 / 15.29 / 44 | -9.70 / 17.89 / 42 | -3.79 / 14.09 / 23 | Fails all splits. |
| `pae_w2016_seq_bear_breakdown_macro@288` | -10.14 / 16.76 / 56 | -3.55 / 11.25 / 56 | -5.64 / 14.40 / 30 | Fails all splits. |
| `pae_w2016_seq_bear_breakdown_macro@144` | -3.14 / 12.30 / 89 | -10.41 / 17.87 / 97 | -17.79 / 16.50 / 48 | Fails all splits. |
| `pae_w4032_failed_breakout_short@288` | -4.61 / 9.03 / 48 | +2.60 / 6.81 / 27 | +1.49 / 3.69 / 13 | Too weak and train-negative. |
| `pae_w4032_failed_breakout_short@144` | -4.12 / 8.18 / 60 | -1.86 / 5.97 / 32 | +6.05 / 2.51 / 17 | Eval-positive only; not selectable. |
| `pae_w2016_failed_mid_loss_long@288` | +1.30 / 8.63 / 39 | +3.65 / 6.36 / 37 | +28.03 / 2.46 / 16 | Only train/test-positive single template, but p-values are weak and eval trade count is low. |
| `pae_w2016_failed_mid_loss_long@144` | -5.71 / 11.04 / 49 | -1.91 / 5.83 / 44 | +13.49 / 3.45 / 20 | Eval-positive only; not selectable. |

## Decision

1. The broad sequence bearish short features are not true deployable alpha under strict execution. Their diagnostic mean looked positive because overlapping trigger returns were counted independently, but actual scheduled holds lose money.
2. `pae_w2016_failed_mid_loss_long@288` is the only train/test-positive fixed template, but it is weak: train CAGR 1.30%, test CAGR 3.65%, p-values 0.846/0.553, and eval has only 16 trades. It is not enough for the target.
3. Future feature checks must use strict-simulator outcomes, not vectorized overlapping horizon returns, when deciding whether a feature is strategy-usable.
4. Next structure pass should focus on shorter occupancy / exit logic and event-conditioned exits, because long fixed holds convert weak positive event drift into drawdown and opportunity-cost drag.

# All alpha + wave feature combo scan (2026-07-08)

All known alpha sleeves combined with wave-feature train-only hard gates and portfolio BLOCK_RISK selector. Splits: train<2024, test=2024, eval=2025, ytd2026 through local data. Fees+slippage 5bp/side via existing event simulation; strict in-position MDD. Selector/gate thresholds fit on train only; selection ranking is diagnostic and uses test/eval/ytd report columns, not promoted live config.

## Event counts
```json
{
  "train": {
    "pb30_base": 189,
    "pb30_addon": 347,
    "nonpb30_taker": 242,
    "oi_raw": 1012,
    "rex_rule": 356,
    "short_kimchi3d": 175,
    "short_premium_panic": 135
  },
  "test2024": {
    "pb30_base": 20,
    "pb30_addon": 38,
    "nonpb30_taker": 34,
    "oi_raw": 197,
    "rex_rule": 62,
    "short_kimchi3d": 37,
    "short_premium_panic": 27
  },
  "eval2025": {
    "pb30_base": 20,
    "pb30_addon": 50,
    "nonpb30_taker": 35,
    "oi_raw": 124,
    "rex_rule": 33,
    "short_kimchi3d": 29,
    "short_premium_panic": 23
  },
  "ytd2026": {
    "pb30_base": 17,
    "pb30_addon": 46,
    "nonpb30_taker": 14,
    "oi_raw": 45,
    "rex_rule": 24,
    "short_kimchi3d": 14,
    "short_premium_panic": 14
  }
}
```

## Baseline best weight combo
weights: `{'pb30_base': 0, 'pb30_addon': 0, 'nonpb30_taker': 2.0, 'oi_raw': 1.0, 'rex_rule': 2.5, 'short_kimchi3d': 0, 'short_premium_panic': 0}` gross=5.5
| split | return | CAGR | strict MDD | CAGR/MDD | trades | win | sharpe-like |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | -24.26% | -8.00% | 88.99% | -0.09 | 1610 | 52.0% | 0.83 |
| test2024 | 429.38% | 427.57% | 23.60% | 18.12 | 293 | 56.3% | 3.12 |
| eval2025 | 352.23% | 352.70% | 10.85% | 32.50 | 192 | 62.0% | 4.34 |
| ytd2026 | 43.73% | 137.72% | 23.33% | 5.90 | 83 | 55.4% | 1.29 |

## Best wave selector
context_keys: `('side', 'wave_slope', 'wr_vwap_dev_z')` params: `{'min_train_context_events': 24, 'bad_mean_ret_bps': -20, 'bad_win_rate': 0.38}` blocked=4
## Best selector combo
weights: `{'pb30_base': 0, 'pb30_addon': 0, 'nonpb30_taker': 2.0, 'oi_raw': 1.0, 'rex_rule': 2.5, 'short_kimchi3d': 0, 'short_premium_panic': 0}` gross=5.5
| split | return | CAGR | strict MDD | CAGR/MDD | trades | win | sharpe-like |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 102.33% | 23.55% | 80.67% | 0.29 | 1450 | 52.5% | 1.31 |
| test2024 | 489.97% | 487.83% | 23.60% | 20.67 | 261 | 57.1% | 3.52 |
| eval2025 | 301.92% | 302.30% | 10.85% | 27.86 | 174 | 61.5% | 4.06 |
| ytd2026 | 44.81% | 142.01% | 23.33% | 6.09 | 78 | 57.7% | 1.32 |

## Top single wave gates
| sleeve | gate | 2024 ratio/trades | 2025 ratio/trades | 2026 ratio/trades |
|---|---|---:|---:|---:|
| oi_raw | `w_pos_144 <= 0.270659 (q0.2)` | 2.79/77 | 5.78/71 | -0.85/18 |
| oi_raw | `w_retr_144 >= 0.729341 (q0.8)` | 2.79/77 | 5.78/71 | -0.85/18 |
| rex_rule | `w_price_z_288 <= -1.18879 (q0.2)` | 4.33/24 | 2.75/5 | 0.88/1 |
| rex_rule | `w_price_z_144 <= -1.17447 (q0.2)` | 2.62/25 | 2.75/5 | 0.00/0 |
| rex_rule | `w_vol_z_144 >= 0.382363 (q0.8)` | 2.71/13 | 2.49/6 | 0.14/4 |
| rex_rule | `wr_vol_spike >= 0.382363 (q0.8)` | 2.71/13 | 2.49/6 | 0.14/4 |
| oi_raw | `w_pos_288 <= 0.265478 (q0.2)` | 2.46/73 | 5.71/70 | -0.69/17 |
| oi_raw | `w_retr_288 >= 0.734522 (q0.8)` | 2.46/73 | 5.71/70 | -0.69/17 |
| oi_raw | `w_price_z_144 <= -1.17447 (q0.2)` | 2.29/72 | 3.26/61 | -1.27/19 |
| oi_raw | `wr_vwap_dev_z <= -0.803766 (q0.2)` | 3.66/131 | 2.16/70 | -0.33/26 |
| oi_raw | `wr_flow_mom <= -0.0310765 (q0.2)` | 2.14/67 | 2.62/51 | 0.01/17 |
| short_premium_panic | `w_vol_z_144 <= -0.613949 (q0.2)` | 2.14/6 | 2.54/6 | 6.62/3 |
| short_premium_panic | `wr_vol_spike <= -0.613949 (q0.2)` | 2.14/6 | 2.54/6 | 6.62/3 |
| oi_raw | `w_slope_atr_288 <= -3.86843 (q0.2)` | 2.14/58 | 7.91/65 | -0.82/17 |
| rex_rule | `w_eff_288 >= 0.0974982 (q0.8)` | 2.12/11 | 3.17/7 | -0.91/4 |

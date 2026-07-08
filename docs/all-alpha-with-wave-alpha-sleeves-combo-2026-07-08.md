# All alpha + wave-derived alpha sleeves combo scan (2026-07-08)

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
    "short_premium_panic": 135,
    "oi_wave_lowpos144": 475,
    "oi_wave_slope288_low": 335,
    "rex_wave_pricez288_low": 90,
    "rex_wave_vol144_high": 146,
    "short_premium_wave_lowvol": 61
  },
  "test2024": {
    "pb30_base": 20,
    "pb30_addon": 38,
    "nonpb30_taker": 34,
    "oi_raw": 197,
    "rex_rule": 62,
    "short_kimchi3d": 37,
    "short_premium_panic": 27,
    "oi_wave_lowpos144": 84,
    "oi_wave_slope288_low": 65,
    "rex_wave_pricez288_low": 32,
    "rex_wave_vol144_high": 24,
    "short_premium_wave_lowvol": 11
  },
  "eval2025": {
    "pb30_base": 20,
    "pb30_addon": 50,
    "nonpb30_taker": 35,
    "oi_raw": 124,
    "rex_rule": 33,
    "short_kimchi3d": 29,
    "short_premium_panic": 23,
    "oi_wave_lowpos144": 74,
    "oi_wave_slope288_low": 69,
    "rex_wave_pricez288_low": 6,
    "rex_wave_vol144_high": 6,
    "short_premium_wave_lowvol": 11
  },
  "ytd2026": {
    "pb30_base": 17,
    "pb30_addon": 46,
    "nonpb30_taker": 14,
    "oi_raw": 45,
    "rex_rule": 24,
    "short_kimchi3d": 14,
    "short_premium_panic": 14,
    "oi_wave_lowpos144": 19,
    "oi_wave_slope288_low": 19,
    "rex_wave_pricez288_low": 1,
    "rex_wave_vol144_high": 6,
    "short_premium_wave_lowvol": 5
  }
}
```

## Baseline best weight combo
weights: `{'pb30_base': 0, 'pb30_addon': 0, 'nonpb30_taker': 2.0, 'oi_raw': 1.0, 'rex_rule': 2.5, 'short_kimchi3d': 0, 'short_premium_panic': 0, 'oi_wave_lowpos144': 0.5, 'oi_wave_slope288_low': 0, 'rex_wave_pricez288_low': 0, 'rex_wave_vol144_high': 0, 'short_premium_wave_lowvol': 0}` gross=6.0
| split | return | CAGR | strict MDD | CAGR/MDD | trades | win | sharpe-like |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | -37.21% | -13.03% | 90.98% | -0.14 | 2085 | 52.8% | 0.79 |
| test2024 | 500.03% | 497.83% | 23.60% | 21.10 | 377 | 57.3% | 3.18 |
| eval2025 | 422.52% | 423.11% | 11.22% | 37.71 | 266 | 62.8% | 4.47 |
| ytd2026 | 42.56% | 133.15% | 24.83% | 5.36 | 102 | 54.9% | 1.23 |

## Best wave selector
context_keys: `('side', 'wave_zone', 'wave_flow')` params: `{'min_train_context_events': 24, 'bad_mean_ret_bps': -20, 'bad_win_rate': 0.38}` blocked=6
## Best selector combo
weights: `{'pb30_base': 0, 'pb30_addon': 0, 'nonpb30_taker': 2.0, 'oi_raw': 1.0, 'rex_rule': 2.5, 'short_kimchi3d': 0, 'short_premium_panic': 0, 'oi_wave_lowpos144': 0.5, 'oi_wave_slope288_low': 0, 'rex_wave_pricez288_low': 0, 'rex_wave_vol144_high': 0, 'short_premium_wave_lowvol': 0}` gross=6.0
| split | return | CAGR | strict MDD | CAGR/MDD | trades | win | sharpe-like |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 54.47% | 13.94% | 85.42% | 0.16 | 1955 | 53.3% | 1.22 |
| test2024 | 474.07% | 472.01% | 23.60% | 20.00 | 356 | 57.6% | 3.15 |
| eval2025 | 416.48% | 417.06% | 11.22% | 37.17 | 249 | 62.7% | 4.48 |
| ytd2026 | 49.24% | 160.08% | 21.86% | 7.32 | 99 | 56.6% | 1.38 |

## Top single wave gates
| sleeve | gate | 2024 ratio/trades | 2025 ratio/trades | 2026 ratio/trades |
|---|---|---:|---:|---:|
| oi_wave_lowpos144 | `w_pos_144 <= 0.270659 (q0.2)` | 3.28/84 | 6.54/74 | -0.36/19 |
| oi_wave_lowpos144 | `w_retr_144 >= 0.729341 (q0.8)` | 3.28/84 | 6.54/74 | -0.36/19 |
| oi_wave_lowpos144 | `w_slope_atr_144 <= -2.55882 (q0.2)` | 2.80/65 | 4.99/63 | -0.30/17 |
| oi_raw | `w_pos_144 <= 0.270659 (q0.2)` | 2.79/77 | 5.78/71 | -0.85/18 |
| oi_raw | `w_retr_144 >= 0.729341 (q0.8)` | 2.79/77 | 5.78/71 | -0.85/18 |
| oi_wave_lowpos144 | `w_price_z_288 <= -1.18879 (q0.2)` | 2.75/72 | 5.81/68 | -0.16/18 |
| oi_wave_lowpos144 | `w_price_z_144 <= -1.17447 (q0.2)` | 2.75/71 | 5.49/62 | -0.36/19 |
| rex_rule | `w_price_z_288 <= -1.18879 (q0.2)` | 4.33/24 | 2.75/5 | 0.88/1 |
| rex_wave_pricez288_low | `w_slope_atr_144 <= -2.55882 (q0.2)` | 2.74/22 | 2.75/5 | 0.88/1 |
| rex_wave_pricez288_low | `w_price_z_288 <= -1.18879 (q0.2)` | 2.68/32 | 2.90/6 | 0.88/1 |
| oi_wave_lowpos144 | `w_pos_288 <= 0.265478 (q0.2)` | 2.68/76 | 5.00/68 | -0.16/18 |
| oi_wave_lowpos144 | `w_retr_288 >= 0.734522 (q0.8)` | 2.68/76 | 5.00/68 | -0.16/18 |
| rex_rule | `w_price_z_144 <= -1.17447 (q0.2)` | 2.62/25 | 2.75/5 | 0.00/0 |
| rex_rule | `w_vol_z_144 >= 0.382363 (q0.8)` | 2.71/13 | 2.49/6 | 0.14/4 |
| rex_rule | `wr_vol_spike >= 0.382363 (q0.8)` | 2.71/13 | 2.49/6 | 0.14/4 |

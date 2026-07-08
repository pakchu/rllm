# All alpha + volume/wave portfolio combo scan (2026-07-08)

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
    "short_premium_wave_lowvol": 61,
    "oi_vol_alt_ratio72": 958,
    "oi_vol_volmom288": 447,
    "oi_vol_alt_ratio288": 39,
    "oi_upbit_ratio288_low": 344
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
    "short_premium_wave_lowvol": 11,
    "oi_vol_alt_ratio72": 92,
    "oi_vol_volmom288": 116,
    "oi_vol_alt_ratio288": 69,
    "oi_upbit_ratio288_low": 79
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
    "short_premium_wave_lowvol": 11,
    "oi_vol_alt_ratio72": 53,
    "oi_vol_volmom288": 66,
    "oi_vol_alt_ratio288": 41,
    "oi_upbit_ratio288_low": 38
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
    "short_premium_wave_lowvol": 5,
    "oi_vol_alt_ratio72": 21,
    "oi_vol_volmom288": 20,
    "oi_vol_alt_ratio288": 14,
    "oi_upbit_ratio288_low": 14
  }
}
```

## Baseline best weight combo
weights: `{'pb30_base': 0, 'pb30_addon': 0, 'nonpb30_taker': 2.0, 'oi_raw': 1.25, 'rex_rule': 2.5, 'short_kimchi3d': 0, 'short_premium_panic': 0, 'oi_wave_lowpos144': 0, 'oi_wave_slope288_low': 0, 'rex_wave_pricez288_low': 0, 'rex_wave_vol144_high': 0, 'short_premium_wave_lowvol': 0, 'oi_vol_alt_ratio72': 0, 'oi_vol_volmom288': 0, 'oi_vol_alt_ratio288': 0, 'oi_upbit_ratio288_low': 0.25}` gross=6.0
| split | return | CAGR | strict MDD | CAGR/MDD | trades | win | sharpe-like |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | -41.76% | -14.98% | 91.24% | -0.16 | 1954 | 52.6% | 0.75 |
| test2024 | 529.10% | 526.74% | 24.03% | 21.92 | 372 | 55.9% | 3.25 |
| eval2025 | 412.34% | 412.91% | 12.66% | 32.61 | 230 | 62.6% | 4.41 |
| ytd2026 | 42.24% | 131.92% | 24.82% | 5.32 | 97 | 54.6% | 1.23 |

## Best wave selector
context_keys: `('sleeve', 'wr_vwap_dev_z', 'wr_vol_spike')` params: `{'min_train_context_events': 24, 'bad_mean_ret_bps': -20, 'bad_win_rate': 0.38}` blocked=3
## Best selector combo
weights: `{'pb30_base': 0, 'pb30_addon': 0, 'nonpb30_taker': 2.0, 'oi_raw': 0.75, 'rex_rule': 2.5, 'short_kimchi3d': 0, 'short_premium_panic': 0, 'oi_wave_lowpos144': 0, 'oi_wave_slope288_low': 0, 'rex_wave_pricez288_low': 0, 'rex_wave_vol144_high': 0, 'short_premium_wave_lowvol': 0, 'oi_vol_alt_ratio72': 0, 'oi_vol_volmom288': 0, 'oi_vol_alt_ratio288': 0, 'oi_upbit_ratio288_low': 0.75}` gross=6.0
| split | return | CAGR | strict MDD | CAGR/MDD | trades | win | sharpe-like |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 46.49% | 12.14% | 87.22% | 0.14 | 1888 | 52.7% | 1.13 |
| test2024 | 575.01% | 572.37% | 23.16% | 24.71 | 358 | 56.7% | 3.73 |
| eval2025 | 394.36% | 394.90% | 11.63% | 33.94 | 221 | 63.3% | 4.79 |
| ytd2026 | 36.45% | 109.99% | 21.39% | 5.14 | 96 | 54.2% | 1.14 |

## Top single wave gates
| sleeve | gate | 2024 ratio/trades | 2025 ratio/trades | 2026 ratio/trades |
|---|---|---:|---:|---:|
| oi_upbit_ratio288_low | `vg_upbit_binance_vol_ratio_z_288 <= -0.60446 (q0.2)` | 5.94/79 | 4.77/38 | 0.37/14 |
| oi_raw | `vg_upbit_binance_vol_ratio_z_288 <= -0.60446 (q0.2)` | 6.54/63 | 3.97/33 | -0.14/12 |
| rex_rule | `vg_alt_btc_qv_ratio_z_72 <= 0 (q0.2)` | 3.71/42 | 5.13/17 | 6.80/15 |
| oi_upbit_ratio288_low | `w_pos_144 <= 0.270659 (q0.2)` | 3.63/34 | 4.04/19 | -0.62/8 |
| oi_upbit_ratio288_low | `w_retr_144 >= 0.729341 (q0.8)` | 3.63/34 | 4.04/19 | -0.62/8 |
| oi_vol_volmom288 | `vg_alt_btc_qv_ratio_z_72 <= 0 (q0.2)` | 4.97/88 | 3.55/55 | 1.07/14 |
| oi_wave_lowpos144 | `vg_alt_btc_qv_ratio_z_72 <= 0 (q0.2)` | 3.45/61 | 3.70/53 | 0.75/16 |
| oi_upbit_ratio288_low | `vg_alt_btc_qv_ratio_z_72 <= 0 (q0.2)` | 6.37/64 | 3.34/30 | 0.17/13 |
| oi_raw | `vg_alt_btc_qv_ratio_z_288 >= 0 (q0.8)` | 4.40/64 | 3.32/32 | -1.22/12 |
| oi_wave_lowpos144 | `w_pos_144 <= 0.270659 (q0.2)` | 3.28/84 | 6.54/74 | -0.36/19 |
| oi_wave_lowpos144 | `w_retr_144 >= 0.729341 (q0.8)` | 3.28/84 | 6.54/74 | -0.36/19 |
| oi_vol_volmom288 | `vg_alt_btc_qv_ratio_z_288 <= 0 (q0.2)` | 3.06/92 | 3.24/55 | 1.04/15 |
| oi_raw | `vg_vol_mom_288 >= 1.29522 (q0.8)` | 3.06/97 | 3.21/54 | -0.33/19 |
| oi_wave_lowpos144 | `vg_upbit_binance_vol_ratio_z_288 <= -0.60446 (q0.2)` | 3.37/28 | 3.03/16 | -0.37/7 |
| oi_vol_volmom288 | `vg_vol_mom_288 >= 1.29522 (q0.8)` | 3.02/116 | 3.78/66 | 0.39/20 |

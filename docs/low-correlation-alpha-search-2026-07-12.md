# Low-correlation alpha search (2026-07-12)

## Objective and protocol

- Find a BTCUSDT standalone alpha whose activation is weakly related to existing alpha-feature/component masks.
- Cost: 6bp/side; next-open entry; strict intrabar MDD; full-window CAGR.
- Thresholds originate from train `<2024`; exit variant selected on test2024; eval2025 and YTD2026 reported afterward.
- Correlation measure: phi correlation of binary feature activation, measured against the existing alpha component frame.

## Search result

The new macro-relief long scan did not qualify. The useful low-correlation result is a tighter-exit refinement of the independent FX-stress short family:

`htf_3d_return_1 <= -0.0325294973 AND usdkrw_zscore >= 1.3870063775`

- side: short
- hold cap: 288 bars (24h)
- take profit: 2.5%
- stop loss: 1.5%
- leverage in validation: 0.5x
- entry scan stride: 12 bars

| split | abs return | CAGR | strict MDD | CAGR/MDD | trades | win | Sharpe-like* |
|---|---:|---:|---:|---:|---:|---:|---:|
| test2024 | 11.94% | 11.91% | 4.49% | **2.66** | 50 | 54.0% | ~1.74 |
| eval2025 | 9.93% | 9.94% | 3.27% | **3.04** | 41 | 53.7% | ~1.66 |
| ytd2026 | 2.27% | 5.55% | 4.14% | 1.34 | 33 | 42.4% | ~0.45 |

`*` Sharpe-like is the signed normal/t-like statistic implied by the stored approximate two-sided p-value; it is not annualized daily Sharpe.

This clears the project alpha-pool rule (test and eval ratio >=2.5), but not live-grade >=3 on both and not the 2026 ratio>=5 target.

## Correlation

Activation phi against existing components (2024~2026H1 correlation frame):

- short kimchi unwind: `+0.2615` (largest)
- promoted short premium+kimchi union: `+0.1938`
- momentum component: `-0.0299`
- premium panic short: `+0.0243`
- USDKRW macro relief long: `-0.0200`
- long funding/range/compression components: approximately `-0.011 .. +0.008`

Thus it is highly independent of the long-alpha complex and only moderately related to the existing short union. The alpha source is macro FX stress plus BTC weakness, not funding/range squeeze.

## Selection caution

- TP/SL choice was selected from the 2024 test grid; 2025 is the cleaner confirmation window.
- 2026 is weak despite positive return. Do not promote to live or optimize its thresholds against 2026.
- The looser TP4%/SL2.5% version has much stronger 2026 ratio (6.42) but test ratio 2.48, narrowly below the predefined alpha-pool cutoff; choosing it now would be 2026-influenced.

## Artifacts

- `results/nonrex_short_bear_tp_refine_cost6bp_2026-07-12.json`
- `training/nonrex_short_bear_tp_refine.py`
- `training/search_lowcorr_macro_alpha.py`
- `results/lowcorr_macro_alpha_scan_2026-07-12.json`
- `results/alpha_feature_correlation_oi_2026-07-10/component_phi.csv`

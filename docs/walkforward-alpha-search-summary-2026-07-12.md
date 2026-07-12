# Walk-forward alpha search summary (2026-07-12)

Goal: try a different alpha discovery method after fixed low-correlation quantile rules failed to reach the target.

## Methods tried

### 1. Monthly expanding-window ridge alpha surface

Script: `training/search_walkforward_linear_alpha.py`

Protocol:

- Every trade month fits a ridge model using only rows before that month.
- Prediction thresholds are also computed from prior rows only.
- Feature set includes market features, Alpha101-style primitives, VPIN/orderflow, OI divergence, funding/premium context.
- Strict non-overlapping OHLC path MDD and full-window CAGR are used.

Best non-inverted fast result still failed:

| candidate | 2024 abs/CAGR/MDD/R/trades | 2025 | 2026YTD |
|---|---:|---:|---:|
| `wflin_h72_r10_q0.95_hold72_s24_short` | -10.97/-10.95/15.90/-0.69/55 | 12.12/12.13/11.53/1.05/81 | 4.48/11.22/4.52/2.48/32 |

Interpretation: the short surface adapts to 2025/2026 but fails 2024 badly. It is not a robust alpha.

### 2. Inverted monthly ridge surface

Result: also failed. Some 2026-only short variants became positive, but 2024/2025 did not hold.

Example:

| candidate | 2024 | 2025 | 2026YTD |
|---|---:|---:|---:|
| `wflinINV_h72_r10_q0.95_hold72_s24_short` | -4.72/-4.71/12.56/-0.37/57 | -14.81/-14.82/24.18/-0.61/123 | 6.71/17.05/5.63/3.03/40 |

### 3. Monthly prior-only nonlinear stump ensemble

Protocol:

- Each month selects top univariate quantile conditions from prior data only.
- Active stumps vote long/short.
- Tested a fast version over low-correlation features.

Result: mostly no-trade or unstable. No candidate survived 2024/2025/2026 simultaneously.

## Conclusion

No new standalone alpha found from these alternative methods.

Important negative evidence:

1. Dynamic monthly retraining does not automatically solve the alpha problem.
2. Linear ridge surfaces learned from current features are either directionally wrong or regime-specific.
3. Nonlinear stump voting over the same low-correlation primitives is too sparse/unstable.
4. The current feature set may still be useful as **context/gating** for known sleeves, but not as independent entries.

## Next direction

The next different approach should not be “more thresholds over the same bars.” Better candidates:

1. Event-first alpha mining: define structurally meaningful events first, then learn only exit/skip/size.
2. Cross-sectional / market-wide crypto pool features: BTC-only microstructure appears too weak alone.
3. Live-orderbook or liquidation/positioning features if available; current bar-level OHLCV/taker/OI/funding data is not enough for robust standalone edge.
4. Use RLLM as a meta-controller over a small set of weak, diverse sleeves rather than asking it to discover a raw numeric alpha from scratch.

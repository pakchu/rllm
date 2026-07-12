# Low-correlation cross-asset breadth alpha scan (2026-07-12)

## Protocol
- Trading target remains BTCUSDT.
- ETH/SOL/BNB/XRP/ADA/DOGE 5m closes are causal predictor inputs only.
- Features: median alt return, positive breadth fraction, BTC-minus-alt residual, cross-sectional dispersion over 1h/4h/12h.
- Thresholds fit on 2023 train; rank on test2024; eval2025/YTD2026 attached after selection.
- 6bp/side, 0.5x, strict MDD; max activation phi <=0.20 versus existing alpha components.

## Result
- 3,840 eligible low-correlation variants.
- test/eval CAGR/strict-MDD >=2.5: **0**.
- live-grade >=3: **0**.
- Best stable family was 4h cross-sectional dispersion reversion long, max phi around 0.10, but best test/eval minimum ratio was only 1.04/1.22.

## Interpretation
Alt breadth and BTC relative-value residuals are genuinely orthogonal but not standalone alpha under the current fixed-hold/TP-SL formulation. Keep as beta features for portfolio gates or event selectors; do not promote or repeat the same quantile grid.

## Artifacts
- `training/search_lowcorr_crossasset_breadth_alpha.py`
- `results/lowcorr_crossasset_breadth_alpha_scan_2026-07-12.json`

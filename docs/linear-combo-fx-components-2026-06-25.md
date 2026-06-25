# Linear composite alpha scan with FX components — 2026-06-25

## Purpose
After single-feature gates failed, this tested whether a ridge-style composite predictor can combine weak price action, flow, Kimchi/DXY/USDKRW, and individual FX component features into a stable no-leak alpha.

## Protocol
- Train: 2023-01-01 through 2024-12-31.
- Test/selection: 2025-01-01 through 2025-12-31.
- Eval/untouched: 2026-01-01 through 2026-06-01.
- Horizons: 12, 24, 36, 72 bars.
- Quantiles: 0.05, 0.10, 0.20.
- Leverage: 1.0.
- Feature groups included market, external, FX components, and combined groups.
- A second run added `inverted` variants, allowing 2025 test to choose original vs flipped rule direction before evaluating untouched 2026.

## Results
No deployable candidate emerged.

### Original composite run
Best ranked rows were already negative on 2025 test:
- `kimchi_only`, h=72, q=0.05: test CAGR -12.79 / strict MDD 43.02 / ratio -0.30 / 661 trades; eval CAGR -63.77 / strict MDD 36.54 / ratio -1.75 / 288 trades.
- `fx_components`, h=36, q=0.05: test CAGR -40.91 / strict MDD 54.69 / ratio -0.75 / 781 trades; eval CAGR -75.40 / strict MDD 44.81 / ratio -1.68 / 281 trades.

### Inverted variant run
Allowing test-selected sign inversion did not rescue the setup:
- Best by test ratio: `trend` inverted, h=72, q=0.05: test CAGR -9.15 / strict MDD 44.62 / ratio -0.21 / 544 trades; eval CAGR -26.45 / strict MDD 29.10 / ratio -0.91 / 249 trades.
- No top candidate was positive on both 2025 test and 2026 eval.

## Interpretation
The ridge composite setup is not learning a tradable edge from these feature groups. It mostly creates dense, high-drawdown trading, and flipping direction only reduces loss in one group instead of creating positive risk-adjusted returns.

The next step should avoid dense continuous prediction with simple ridge thresholds. More promising directions:
1. sparse event construction first, then LLM/RL ranking on event contexts;
2. path/price-action labels that reward asymmetric excursions, not simple forward return;
3. explicit abstention as a primary action, with positive-trade opportunity mining rather than every-tail quantile trading.

## Leakage guard
Selection uses only 2025 test. 2026 eval is reported after selection. External features use backward-asof joins, and strict replay uses actual OHLC bar-by-bar execution.

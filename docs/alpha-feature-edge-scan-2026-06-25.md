# Alpha feature edge scan (2026-06-25)

## Context
After the path-outcome gate failed to create enough alpha, we checked whether current market features contain standalone predictive edge before doing more LLM/RL work.

## Diagnostic scan
Command output: `results/alpha_feature_scan_market_recent_2026-06-25/report.json`

The scan found that most high-ranked volatility/drawdown features invert sign in the recent eval split. Stable-sign candidates were mostly kimchi-premium and weekly-momentum features:

- `kimchi_premium_zscore`, horizon 288
- `kimchi_premium_change`, horizons 72/144/288
- `weekly_return_1w` / `htf_1w_return_1`, horizon 288

These looked promising on forward-return quantile spread diagnostics.

## Strict OHLC extended backtests
Fit: 2024-01-01 through 2025-08-31.  
Eval: 2025-09-01 through 2026-05-31.  
Leverage: 1.0. Entry delay: 1 bar. Quantile: 0.20.

| Feature | Horizon | CAGR | Strict MDD | Trades | Verdict |
|---|---:|---:|---:|---:|---|
| kimchi_premium_zscore | 288 | -22.08% | 44.40% | 172 | fail |
| kimchi_premium_zscore | 144 | -21.06% | 40.76% | 342 | fail |
| kimchi_premium_zscore | 72 | -43.88% | 38.65% | 642 | fail |
| kimchi_premium_change | 288 | -27.53% | 40.66% | 172 | fail |
| kimchi_premium_change | 144 | -42.76% | 39.75% | 331 | fail |
| kimchi_premium_change | 72 | -35.10% | 33.80% | 616 | fail |
| weekly_return_1w | 288 | -26.15% | 35.99% | 91 | fail |
| weekly_return_1w | 144 | -35.87% | 37.90% | 182 | fail |
| weekly_return_1w | 72 | -46.82% | 43.37% | 361 | fail |

## Interpretation
Forward-return spread diagnostics are not sufficient for this project's objective. The apparent edge collapses under strict bar-by-bar replay with costs and intrabar adverse excursion. This explains why gate/ranker optimization has repeatedly produced plausible validation results but unacceptable recent drawdown.

## Next direction
Do not keep optimizing single gates. Search for compound context rules where the feature is only active under compatible regimes, then validate with strict OHLC and rolling no-leak folds.

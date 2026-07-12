# State-transition standalone alpha scan (2026-07-12)

## Protocol

- Target: BTCUSDT only; OI/taker/price state-transition features are predictors, not alternate trading assets.
- Splits: train `<2024`, test `2024`, eval `2025`, YTD `2026-01-01..2026-06-01`.
- Selection: quantile thresholds fit on train only; candidates ranked on test2024 only; eval2025/YTD2026 attached after the top-400 test selection.
- Execution: next-open entry, 6bp/side, full-calendar CAGR, strict intrabar MDD, period-contained exits.
- Search: 6 predeclared families, 4 fixed past-only regime variants, 6 holds, 2 strides; 6,950 eligible test variants.

## Result

No new alpha-pool or live-grade standalone alpha was found.

- `CAGR/strict MDD >= 2.5` on both test2024 and eval2025: **0**
- `CAGR/strict MDD >= 3.0` on both test2024 and eval2025: **0**
- Among the top-400 test-selected rows, only 8 reached eval ratio >=1 and none reached 2.5.
- The best test/eval minimum was `oi_flow_confirmation + sma30_rising`, hold 216, stride 12:
  - train: return 64.53%, CAGR 13.26%, strict MDD 49.22%, ratio 0.27, 198 trades
  - test2024: return 51.17%, CAGR 51.04%, strict MDD 18.83%, ratio 2.71, 76 trades, win 53.95%, p≈0.063
  - eval2025: return 11.81%, CAGR 11.82%, strict MDD 6.66%, ratio 1.78, 42 trades, win 54.76%, p≈0.304
  - ytd2026: return -10.79%, CAGR -23.99%, strict MDD 12.46%, ratio -1.93, 22 trades

## Interpretation

- OI/taker state transitions contain some conditional information, but this formulation is not a standalone alpha.
- The strongest 2024 OI-squeeze rows (test ratio roughly 8-11) collapsed in 2025; these are explicit examples of test-regime overfit and must not enter alpha_pool.
- `oi_flow_confirmation` remains beta-feature evidence because it stayed positive in 2025 for some fixed bullish regimes, but weak train risk-adjusted performance and negative 2026 prevent promotion.
- Do not repeat the same raw OI-impulse + taker-confirmation + fixed-hold grid. A future pass must change the economic mechanism (for example event-time normalization or an independent exit), not just thresholds.

## Artifacts

- `training/search_state_transition_alpha.py`
- `results/state_transition_regime_alpha_scan_top400_2026-07-12.json`

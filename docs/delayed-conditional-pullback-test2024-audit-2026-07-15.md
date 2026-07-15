# Delayed conditional-pullback 2024 test audit — 2026-07-15

**2024 test passed; 2025+ remains sealed.**

Metric: absolute return / CAGR / strict MDD / CAGR-to-strict-MDD / trades.

- All predictive features except current funding/premium event identity are delayed by 12×5m (one hour).
- The delay was fixed as a pre-2024 timing control before 2024 was opened; no delay grid was searched on test.
- 1,000-tree individual joint passes: **3/5**.
- 2,000-tree individual joint passes: **4/5**.
- Three/five-model means pass at both sizes: **True**.

## Selected 5×2,000-tree model

| Window | Result |
|---|---:|
| train | 105.65% / 33.39% / 8.02% / 4.16 / 126 |
| select_2023 | 11.99% / 12.00% / 3.12% / 3.85 / 26 |
| select_2023_h1 | 10.63% / 22.62% / 3.12% / 7.26 / 17 |
| select_2023_h2 | 1.23% / 2.46% / 2.88% / 0.85 / 9 |
| pre_2024 | 130.32% / 26.90% / 8.02% / 3.35 / 152 |
| test_2024 | 16.31% / 16.28% / 4.62% / 3.52 / 27 |

The next step is a write-once eval manifest pinning this exact delay, forest ensemble, thresholds, feature graph, source-owned exits, and 2025/2026 windows before either is evaluated.

Audit hash: `742b993373a46dcd5c94681132fa764228ccd3ab7c75c811f48bbaac70a8adb5`

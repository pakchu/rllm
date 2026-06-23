# Multi-feature alpha checkpoint (2026-06-23)

## Purpose

After broad univariate feature scans failed fold-consistency gates, the next check tested whether a simple leakage-safe multi-feature learner finds a stronger BTC or cross-sectional alpha.

## BTC linear-combo scan

Protocol:

- Train fit window: `2020-01-01` through `2024-06-30`.
- Test/selection window: `2024-07-01` through `2025-12-31`.
- Untouched eval window: `2026-01-01` through `2026-06-01`.
- Inputs: BTCUSDT 5m futures, wave-trading DXY/Kimchi/USDKRW features, Binance funding and premium aux.
- Model: ridge linear predictor over causal feature groups; thresholds and direction fit on train; ranking by test only.
- Output: `results/alpha_linear_combo_wave_aux_2026h1_2026-06-23.json`.

Top test-ranked rows:

| Group | Horizon | Quantile | Test CAGR | Test strict MDD | Test ratio | Test trades | Eval CAGR | Eval strict MDD | Eval ratio | Eval trades |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 288 | 0.05 | 17.96% | 14.28% | 1.26 | 182 | -6.83% | 16.78% | -0.41 | 58 |
| all | 144 | 0.10 | 23.85% | 21.92% | 1.09 | 514 | -13.83% | 16.73% | -0.83 | 162 |
| all | 144 | 0.15 | 25.00% | 23.29% | 1.07 | 733 | -16.15% | 12.24% | -1.32 | 220 |
| external | 288 | 0.05 | 17.18% | 14.58% | 1.18 | 269 | 28.93% | 9.55% | 3.03 | 89 |
| all | 144 | 0.05 | 14.98% | 13.08% | 1.15 | 212 | 9.57% | 9.13% | 1.05 | 69 |

Decision: **NO_GO**. The tempting `external/h288/q0.05` row clears the eval target, but its test ratio is only 1.18, so using it would be eval cherry-picking.

## Multiasset rolling cross-sectional check

Protocol:

- Universe: `XRPUSDT`, `SOLUSDT`, `ADAUSDT`, `BNBUSDT`, `DOGEUSDT`, `ETHUSDT` 5m futures, with funding/premium aux.
- Eval: `2025-01-01` through `2026-06-01`.
- Each month fits only on the preceding 365 days.
- Fixed params source in script: prior train/validation selection, not eval tuning.
- Output: `results/multiasset_feature_rolling_eval_2025_2026_2026-06-23.json`.

Best diagnostic row:

| Min score spread | Eval CAGR | Strict MDD | Ratio | Rebalances | Mean-return p approx |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.004 | 0.98% | 7.85% | 0.12 | 1548 | 0.0777 |
| 0.002 | 0.71% | 7.49% | 0.10 | 1548 | 0.0530 |
| 0.001 | 0.05% | 7.49% | 0.01 | 1548 | 0.0649 |

Decision: **NO_GO**. Trade count is statistically useful, and p-values are closer than the BTC sparse policies, but the economic magnitude is far below the target.

## Working conclusion

The current evidence says the missing piece is not “more gates” or “more prior thresholds.” The promising direction is to make the LLM/RLLM component act as a **causal state abstraction and regularizer**, not a direct numeric threshold picker:

1. Mine stable causal contexts from train/test only.
2. Convert them into compact textual state/rationale tokens.
3. Train a single Gemma 4 policy to choose among a small action set with abstention.
4. Validate by rolling monthly retraining; eval remains untouched until a candidate is fixed.

The immediate next implementation should therefore create a candidate-context miner that produces LLM-sized state/action examples from stable multi-feature regimes rather than raw numeric windows.

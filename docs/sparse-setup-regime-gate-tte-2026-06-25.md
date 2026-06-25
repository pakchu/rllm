# Sparse setup regime gate TTE (2026-06-25)

## Purpose

Stop using sparse setup candidates as direct trade triggers. Treat them as weak event proposals and learn a regime-aware `trade/abstain` score from market/context features.

Protocol:

1. Candidate pool is discovered on train only: `2020H1` through `2024H2`.
2. Ridge gate is fit on train event rows only.
3. Score quantile threshold is selected on test only: `2025H1`, `2025H2`.
4. Final eval refits on train+test and applies the selected quantile to untouched `2026H1`.

Implementation:

- `training/sparse_setup_regime_gate_tte.py`
- Uses sparse setup candidate events from the train-only sparse report.
- Uses regime/context features from market, wave, and price-action extreme feature frames.
- Learns utility or net-return target with ridge.
- Selects one best candidate per signal bar if score is above threshold.
- Low-trade test candidates are penalized with capped CAGR to avoid 1-trade annualization explosions.

## Main run

Artifact:

- `results/sparse_setup_tte_2020train_combined_pa_macro_2026-06-25/regime_gate_tte_min20_v3.json`

Config summary:

- candidate limit: `80`
- ridge alpha: `300`
- target: `utility`
- score quantiles: `0.90,0.925,0.95,0.975,0.99`
- min test trades: `20`
- features: DXY/kimchi/USDKRW/HTF/range/trend/volume, wave momentum/CVD/flow/vol, PA extreme features

Selected by test:

```json
{"q": 0.95, "threshold": 0.9864900705706608, "score": 1.2895445535887369}
```

| Period | CAGR | Strict MDD | CAGR/MDD | Trades |
| --- | ---: | ---: | ---: | ---: |
| Train 2020-2024 | 34.89% | 18.30% | 1.91 | 191 |
| Test 2025 | 6.63% | 6.14% | 1.08 | 21 |
| Eval 2026H1 | -4.34% | 3.73% | -1.16 | 10 |

## Net-return target run

Artifact:

- `results/sparse_setup_tte_2020train_combined_pa_macro_2026-06-25/regime_gate_tte_net_min20_v3.json`

The net-return target selected a very sparse q=0.99 candidate because every candidate was weak under the min-trade rule. It produced only 1 trade in test and 1 trade in eval, so it is not meaningful.

## Interpretation

The regime gate improved the shape versus direct sparse setup triggers in one narrow sense: it can reduce 2025 damage and find a small positive 2025 test slice. But the edge does not generalize to 2026H1 and the trade count is too low.

Current conclusion:

- Sparse setup events contain some weak conditional signal.
- The current regime feature surface and utility label are still insufficient for robust live deployment.
- Selector/gate tuning alone should not continue on this pool.

Next required direction:

1. Build explicit failure-regime labels rather than only per-event return labels.
2. Add regime classes for 2025-like conditions: chop, failed breakout, DXY impulse, kimchi premium shock, volatility transition, and distance-to-extreme compression.
3. Train the model to abstain by regime first, then rank candidate trades inside safe regimes.
4. Keep the same TTE protocol: train discovery, test threshold selection, untouched eval.

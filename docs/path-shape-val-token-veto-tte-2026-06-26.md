# Path-shape val-selected token veto TTE (2026-06-26)

## Purpose

The PA+micro token policy remained loss-making, but token return diagnostics showed that certain token regions were strongly harmful or favorable. This pass promotes that idea into a leakage-safer protocol:

1. Fit token model on train only.
2. Generate val predictions.
3. Backtest val predictions and compute bad tokens from **val executed returns only**.
4. Sweep token-veto size and prediction thresholds on val.
5. Apply selected fixed veto/thresholds to untouched OOS.

Implementation:

- `training/path_shape_val_token_veto_tte.py`
- `tests/test_path_shape_val_token_veto_tte.py`

This is not an OOS token-mining script. OOS is used only once for final evaluation.

## Main run

Artifact:

- `results/path_shape_val_token_veto_tte_h144_t1p0_s0p6_pa_micro/report.json`

Selected by val:

```json
{
  "side_mode": "normal",
  "veto_size": 20,
  "prob_threshold": 0.34,
  "margin_threshold": 0.30
}
```

Val:

- CAGR: `28.54%`
- strict MDD: `6.54%`
- CAGR/MDD: `4.37`
- trades: `85`
- mean trade: `0.151%`
- p approx: `0.076`

OOS:

- CAGR: `4.56%`
- strict MDD: `8.93%`
- CAGR/MDD: `0.51`
- trades: `86`
- mean trade: `0.028%`
- p approx: `0.730`

Interpretation: turns OOS from strongly negative to slightly positive, but not statistically meaningful.

## Robustness sweep

Additional settings:

| Setting | Selected veto | Val CAGR/MDD | Val trades | OOS CAGR | OOS MDD | OOS CAGR/MDD | OOS trades | OOS p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `min_token_trades=12`, mean <= -0.05 | 20 | 4.37 | 85 | 4.56% | 8.93% | 0.51 | 86 | 0.730 |
| `min_token_trades=16`, mean <= -0.05 | 20 | 8.13 | 72 | 19.50% | 7.82% | 2.49 | 79 | 0.184 |
| `min_token_trades=16`, mean <= -0.10 | 20 | 8.13 | 72 | 19.50% | 7.82% | 2.49 | 79 | 0.184 |
| `min_token_trades=24`, mean <= -0.05 | 12 | 8.28 | 84 | 10.87% | 9.39% | 1.16 | 94 | 0.466 |
| `min_token_trades=24`, mean <= -0.10 | 12 | 8.28 | 84 | 10.87% | 9.39% | 1.16 | 94 | 0.466 |

Best OOS among this sweep:

- `min_token_trades=16`
- `max_veto_mean_ret_pct=-0.05` or `-0.10`
- OOS CAGR `19.50%`, strict MDD `7.82%`, ratio `2.49`, trades `79`

## Conclusion

This is the first leakage-controlled path in the current branch that moved OOS from large negative CAGR to positive CAGR under strict OHLC execution. It still does **not** satisfy the target:

- CAGR is below `50%`.
- CAGR/strict-MDD is below `3`.
- OOS trade count is underpowered and p-value is not significant.

But it gives a concrete next direction:

1. Treat token-veto/abstention as the useful layer, not raw action prediction.
2. Increase OOS/eval horizon and rolling splits to test whether the veto layer is stable.
3. Replace naive token Naive-Bayes with an LLM/RL-compatible abstention model only after the veto effect survives longer rolling evaluation.

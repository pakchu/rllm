# Economic action-space sweep (2026-06-07)

## Purpose
Previous pressure-label systems failed because direction classification was not the same as positive expected trade value. This sweep changed the economic action template itself before doing more LLM fine-tuning.

## Method
- Fit train-only value calibration tables for every target/stop/horizon combo.
- Select config on validation only, requiring at least 50 trades.
- Evaluate the single selected combo/config once on OOS.
- Strict costs/MDD: 0.5 leverage, 4 bps fee, 1 bp slippage, next-bar entry, intrabar strict adverse excursion, stop-first ambiguous bars.

## Sweep space
- Horizons: 36, 72, 144 bars.
- Targets: 0.8%, 1.2%, 1.8%, 2.5%.
- Stops: 0.6%, 1.0%, 1.5%.
- Trader configs per combo: 4 context levels × 4 min bucket sizes × 5 score thresholds × 2 score modes × 3 side gates.

## Best validation-selected combo
```json
{
  "economics": {"horizon_bars": 144, "target_pct": 1.8, "stop_pct": 1.5},
  "config": {"level": "teacher_only", "min_n": 50, "min_score": 0.0005, "score_mode": "mean", "side_gate": "free"}
}
```

## Result
| Split | Trades | Return | CAGR | Strict MDD | CAGR/MDD | Mean trade | p-value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Val-selected | 68 | +6.41% | +13.20% | 4.19% | 3.15 | +0.0932% | 0.202 |
| OOS-fixed | 87 | -1.60% | -3.26% | 8.11% | -0.40 | -0.0164% | 0.819 |

## Interpretation
This is the first sweep where validation crossed the user ratio target (`CAGR/strict MDD >= 3`), but it failed OOS and the validation p-value was weak. Treat it as a useful direction signal, not a monetizable candidate.

Key implication: longer horizon / wider target-stop economics reduce fee drag and can create validation edge, but the simple train bucket trader overfits regime-specific validation structure.

## Next step
Do not fine-tune on the selected val winner as if it were proven. The next repair should target generalization:
1. Use rolling train→test→eval folds across more than one validation period.
2. Penalize config instability and require multiple folds with positive CAGR/MDD.
3. Add a trader objective that predicts net trade value and uncertainty, not just bucket mean.

# Monthly symbolic selector result (2026-06-25)

## Goal
Avoid final-eval threshold selection by choosing symbolic-policy target/threshold candidates from the prior validation window for each eval month.

## Implementation
- `training/regime_symbolic_monthly_selector.py`
- Default compact mode removes temporary split/candidate artifacts and keeps only aggregate replay files.
- Validation gates use only prior data.

## Run 1: aggregate validation gates
Path: `results/regime_symbolic_monthly_selector_compact_2026-06-25/report.json`

- Eval period: 2026-01-01 through 2026-06-01
- CAGR: -12.78%
- Strict MDD: 15.67%
- Trades: 122
- p-value: 0.638

Monthly eval:
- 2026-01: abstain
- 2026-02: abstain
- 2026-03: +3.68 summed trade-return pct, 45 trades
- 2026-04: -7.38 summed trade-return pct, 41 trades
- 2026-05: -1.31 summed trade-return pct, 36 trades

## Run 2: validation month-consistency gates
Path: `results/regime_symbolic_monthly_selector_consistent_2026-06-25/report.json`

Added prior-validation constraints:
- at least 2 positive validation months
- worst validation month return >= -3%

Result:
- CAGR: -21.53%
- Strict MDD: 12.40%
- Trades: 72
- p-value: 0.188

This blocked the profitable March eval month but still allowed April and May losses.

## Interpretation
The symbolic monthly selector is structurally safer than eval threshold sweeps, but it does not solve the alpha problem. The April break is especially important: prior validation months were all positive for the selected April candidate, yet April eval was strongly negative. This suggests the current symbolic feature surface/targets do not identify the regime transition fast enough.

## Decision
Keep the selector as a no-leak guardrail/research tool, but do not treat it as a live candidate. Future work should shift away from static symbolic thresholds toward either:
1. faster regime-break detection from actual market path features, or
2. a different alpha source before LLM/RL policy selection.

## Exit overlay diagnostic
After the monthly selector failed, the combined eval predictions were replayed with a small exit overlay sweep:

- stop-loss: 0%, 1%, 2%
- take-profit: 0%, 2%, 4%
- ATR trailing stop: off / 2x

Best observed variant among this small sweep was still negative:

- baseline: CAGR -12.78%, strict MDD 15.67%, 122 trades
- take-profit 2%, no stop/ATR: CAGR -11.98%, strict MDD 15.41%, 127 trades
- stop-loss and ATR variants generally worsened results

Conclusion: the monthly symbolic failure is not mainly an exit-overlay problem. The entry/ranking alpha is insufficient.

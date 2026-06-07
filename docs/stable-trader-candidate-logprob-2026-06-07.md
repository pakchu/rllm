# Stable trader candidate log-prob check (2026-06-07)

## Purpose
Free generation was slow and over-traded. This check added candidate log-prob evaluation over 9 fixed JSON candidates:
`NO_TRADE/LONG/SHORT × HIGH/MEDIUM/LOW`.

## Result summary
Candidate log-prob fixes output format, but it worsened the no-trade problem for the step16 Gemma checkpoint.

### Val128 metrics
- Action accuracy: 9.38%.
- Exact accuracy: 9.38%.
- Target trades: 27/128.
- Predicted trades: 128/128.
- Side accuracy when target trade: 44.44%.

Val128 strict backtest:
- Trades: 90.
- CAGR: +56.56%.
- Strict MDD: 3.98%.
- CAGR/MDD: 14.20.
- Mean trade: +0.0599%.
- p-value: 0.438.

### Eval128 metrics
- Action accuracy: 6.25%.
- Exact accuracy: 6.25%.
- Target trades: 15/128.
- Predicted trades: 128/128.
- Side accuracy when target trade: 53.33%.

Eval128 strict backtest:
- Trades: 72.
- CAGR: -61.94%.
- Strict MDD: 11.47%.
- CAGR/MDD: -5.40.
- Mean trade: -0.1529%.
- p-value: 0.0108.

## Interpretation
The checkpoint has learned directional-looking completions, but it has not learned abstention. Candidate scoring amplifies that by forcing every row into a candidate and consistently under-ranking `NO_TRADE`.

## Decision
Reject candidate log-prob for the current step16 checkpoint. Keep the evaluator mode because it is useful once the model is trained with no-trade-preserving objectives.

## Next step
The next SFT iteration should preserve the original action prior or overweight `NO_TRADE`, rather than balanced sampling that makes the model think trades are common. Checkpoint selection should include predicted trade rate as a hard constraint before backtesting.

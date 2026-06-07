# Stable trader candidate log-prob check (2026-06-07)

## Purpose
Free generation was slow and over-traded. This check evaluated the stable trader over 9 fixed JSON candidates:
`NO_TRADE/LONG/SHORT × HIGH/MEDIUM/LOW`.

## Important correction: old candidate-logprob results invalidated
A later audit found that candidate-logprob evaluators inherited tokenizer padding defaults. With left padding, candidate spans were scored at the wrong token positions for batched/padded sequences. The affected code paths are now fixed by forcing `tokenizer.padding_side = "right"` and by recording candidate scoring metadata.

Do **not** compare against the older unbatched reports unless they are regenerated with the right-padding fix.

Fix commits:
- `5aa59e0` — split JSON-key candidate scoring right-padding/batching.
- `3f98629` — stable trader action/risk candidate scoring right-padding/batching.
- `e54062b` — text trader candidate scoring right-padding/batching.

## Corrected step16 stable-trader candidate-logprob results
Adapter: `checkpoints/stable_trader_gemma4_e4b_h144_t1p8_s1p5_step16`
Prediction mode: `candidate_logprob`, `score_normalization=mean`, `batch_size=4`.

### Corrected Val128 metrics
Report: `results/stable_trader_gemma4_e4b_h144_t1p8_s1p5_step16_val128_logprob_batched.json`

- Action accuracy: 72.66%.
- Risk accuracy: 4.69%.
- Exact accuracy: 0.78%.
- Target trades: 27/128.
- Predicted trades: 10/128.
- Side accuracy when target trade: 3.70%.

Corrected Val128 strict backtest:
`results/stable_trader_gemma4_e4b_h144_t1p8_s1p5_step16_val128_logprob_batched_backtest.json`

- Trades: 10.
- CAGR: +3.67%.
- Strict MDD: 2.80%.
- CAGR/MDD: 1.31.
- Mean trade: +0.0442%.
- p-value: 0.859.
- Required n for 80% power rule: 2492; gap: 2482.

### Corrected Eval128 metrics
Report: `results/stable_trader_gemma4_e4b_h144_t1p8_s1p5_step16_eval128_logprob_batched.json`

- Action accuracy: 85.16%.
- Risk accuracy: 0.00%.
- Exact accuracy: 0.00%.
- Target trades: 15/128.
- Predicted trades: 4/128.
- Side accuracy when target trade: 0.00%.

Corrected Eval128 strict backtest:
`results/stable_trader_gemma4_e4b_h144_t1p8_s1p5_step16_eval128_logprob_batched_backtest.json`

- Trades: 4.
- CAGR: +3.24%.
- Strict MDD: 1.01%.
- CAGR/MDD: 3.22.
- Mean trade: +0.0933%.
- p-value: 0.770.
- Required n for 80% power rule: 368; gap: 364.

## Interpretation
The old Val128 candidate-logprob backtest looked attractive because it produced 90 trades and CAGR/MDD 14.20. After the scoring fix, the same checkpoint produces only 10 Val trades and 4 Eval trades. The apparent edge does not survive correction and is statistically meaningless.

The corrected model mostly abstains and fails directional trade selection when a trade is actually required. Eval CAGR/MDD above 3 is not acceptable because it comes from only 4 trades and p=0.770.

## Decision
Reject this checkpoint and reject all older candidate-logprob-derived high-return claims unless regenerated after the right-padding fix.

## Next step
Use corrected candidate-logprob only as a rejection/diagnostic tool. The next profitable-search stage should focus on improving representation/teacher signal and then run leak-free train/test/eval with minimum trade-count and power constraints before backtesting claims are considered.

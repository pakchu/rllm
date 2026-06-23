# Event side rationale score-spread abstention (2026-06-23)

## Purpose

After prior subtraction, `event_side_rationale_gemma4_e4b_dpo16` lost the previously apparent profitable behavior. This pass tests whether the residual prior-adjusted score spread can be used as an abstention confidence filter before applying NORMAL/INVERSE side decisions to the rolling event-context trade proposals.

## Inputs

- Model eval: `results/event_side_rationale_gemma4_dpo16_eval2026_prior_adjusted_mean_2026-06-23.json`
- Event eval rows: `data/event_side_pair_h288_start2022_eval2026_2026-06-23.jsonl`
- Base proposals: `results/rolling_event_context_preference_h288_start2022_predictions_2026-06-23.jsonl`
- Market data: `data/2020-01-01_2026-06-01_btcusdt_futures_5m.csv.gz`
- Backtest window: 2026-01-02 09:00:00 to 2026-05-29 21:00:00

## Method

`training/threshold_side_rationale_eval.py` computes:

```text
score_spread = abs(score_normal - score_inverse)
```

Rows below `min_spread` are changed to `UNRELIABLE`. The existing application layer treats labels other than `NORMAL` / `INVERSE` as `NO_TRADE`, so this is an abstention-only filter and does not modify side choice for kept rows.

Thresholds tested here are diagnostic quantile-style cutoffs from the eval score-spread distribution. They are not selected by a separate validation split, so they must not be treated as production-tuned parameters.

## Results

| threshold | kept eval decisions | side accuracy | executed trades | CAGR | strict MDD | CAGR/strict MDD | mean-ret p approx | conclusion |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.0062401 (q75) | 48 | 58.33% | 37 | 3.26% | 9.26% | 0.35 | 0.846 | Not enough edge |
| 0.0067253 (q80) | 39 | 56.41% | 32 | 2.45% | 6.74% | 0.36 | 0.867 | Not enough edge |
| 0.0091300 (q90) | 20 | 60.00% | 19 | -3.44% | 7.22% | -0.48 | Fails |
| 0.0102859 (q95) | 9 | 77.78% | 8 | 12.30% | 1.88% | 6.55 | 0.023 | Too few trades; statistically fragile |

## Interpretation

- The high-confidence tail exists, but it is too sparse in the 2026 Jan-May eval window.
- The only ratio-over-target result is q95 with 8 executed trades. This is not a statistically meaningful trading system and is vulnerable to selection noise.
- q75/q80 provide enough trades for a first sanity check but do not produce meaningful profitability.
- Therefore, score-spread abstention alone does not solve the current RLLM alpha problem.

## Next direction

Do not optimize the gate further on this eval window. The useful signal is that LLM score confidence may identify a small subset; the next valid test should derive the threshold from pre-2026 train/validation score distribution and then replay 2026 once, or expand to a rolling out-of-sample protocol with enough trades.

# Event-action value Gemma4 SFT64 full 2026 eval — 2026-06-24

## Purpose

Test whether a single Gemma4 E4B text LLM can act as a candidate-level value selector:

1. each signal expands into fixed candidate actions (`LONG/SHORT × hold_bars` plus variants),
2. the model scores `TAKE` vs `SKIP` for each candidate,
3. the highest TAKE-minus-SKIP candidate is selected only if its margin exceeds a train-calibrated threshold,
4. 2026 eval is backtested with strict bar-by-bar execution, fees, slippage, entry delay, and intrabar adverse excursion MDD.

This is intended to keep the LLM in a role it can plausibly handle: discrete candidate evaluation over textual market context, not raw numeric time-series regression.

## Data and model

- Train value dataset: `data/event_action_value_train_pre2026_2026-06-24.jsonl`
  - rows: 116,880 candidate rows
  - labels: SKIP 107,399 / TAKE 9,481
- Eval value dataset: `data/event_action_value_eval2026_2026-06-24.jsonl`
  - rows: 11,940 candidate rows
  - labels: SKIP 11,113 / TAKE 827
- Model: `google/gemma-4-E4B-it` via alias `gemma4-e4b-it`
- Adapter: `checkpoints/event_action_value_gemma4_e4b_sft64_2026-06-24`
- SFT command shape:
  - balanced 8,192 samples
  - max steps 64
  - LoRA r=16, alpha=32, dropout=0.05
  - max sequence length 1600
- Training output:
  - runtime: 432.4s
  - train loss: 0.3973
  - epoch: 0.0625

## Scoring implementation

Added `training/fast_score_action_value_candidates.py`.

The first implementation used row-wise prompt KV-cache scoring. That was correct but too slow for full eval:

- observed speed: about 435s / 1,000 candidate rows
- expected full 11,940 rows: about 85+ minutes

The scorer now supports batched full-sequence TAKE/SKIP scoring:

- stable batch size: 8
- batch size 16 was rejected: too close to 32GB VRAM limit and slower under this prompt length distribution
- full eval runtime: 2,844s for 11,940 rows
- progress logging now uses `done >= next_progress`, not modulo, so it works with arbitrary batch sizes

## Calibration

Thresholds below were selected from pre-2026 train score distribution, not from 2026 eval:

| name | margin threshold |
| --- | ---: |
| zero | 0.0 |
| trainq50 | 1.471116542816162 |
| trainq75 | 1.5456857681274414 |
| trainq90 | 1.6311779022216797 |
| trainq95 | 1.649836540222168 |

The original 100-signal smoke looked promising at trainq75, but full 2026 rejected it.

## Full 2026 strict backtest

Period: `2026-01-01 02:55:00` to `2026-05-30 02:55:00` (`0.40794` years).  
Market: `data/2020-01-01_2026-06-01_btcusdt_futures_5m.csv.gz`.  
Execution assumptions: leverage 0.5, fee 0.0004, slippage 0.0001, entry delay 1 bar, max hold 432 bars.

| threshold | trades | ret % | CAGR % | strict MDD % | CAGR/MDD | p approx | mean trade % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero | 147 | -15.02 | -32.90 | 20.98 | -1.57 | 0.195 | -0.106 |
| trainq50 | 141 | -10.08 | -22.93 | 20.96 | -1.09 | 0.363 | -0.071 |
| trainq75 | 128 | -11.30 | -25.48 | 20.38 | -1.25 | 0.306 | -0.089 |
| trainq90 | 93 | -8.10 | -18.71 | 11.80 | -1.59 | 0.369 | -0.087 |
| trainq95 | 79 | -12.01 | -26.92 | 14.49 | -1.86 | 0.196 | -0.156 |

Main result: train-calibrated thresholds all lose on full 2026. The 100-signal result was not robust.

## Direction diagnostics

Using trainq75 selected candidates:

| variant | trades | ret % | CAGR % | strict MDD % | CAGR/MDD | p approx | mean trade % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| selected side | 128 | -11.30 | -25.48 | 20.38 | -1.25 | 0.306 | -0.089 |
| side inverted | 128 | -2.43 | -5.84 | 18.37 | -0.32 | 0.869 | -0.014 |
| long only | 107 | -6.31 | -14.76 | 19.70 | -0.75 | 0.592 | -0.055 |
| short only | 82 | -4.50 | -10.68 | 13.06 | -0.82 | 0.658 | -0.051 |

Interpretation: the failure is not a simple long/short sign inversion. The candidate value model is selecting weak/noisy timing overall.

## Leakage and validity notes

- Thresholds in the table are train-calibrated from pre-2026 score distribution.
- 2026 full threshold sweep is diagnostic; it must not be used to choose a live threshold without a new held-out eval.
- Strict MDD includes intrabar adverse excursion while a position is open.
- Entry is delayed by one bar after the signal.
- Backtest does not use future prices for gate decisions.
- `skipped_cooldown` is implicit non-overlap behavior from the backtester: while a trade is open, later signals before exit are skipped.

## Conclusion

This SFT64 candidate-value approach is useful infrastructure but not a profitable model. It fails the target by a large margin:

- target: CAGR / strict MDD >= 3 with statistically meaningful trades
- observed best train-calibrated full 2026 ratio: negative for all tested thresholds
- trade count is statistically more meaningful than the smoke test, and the mean-trade p-values do not support a positive edge

The next productive direction is not more threshold/gate tuning on this adapter. The model needs a different learning target or stronger candidate generation:

1. train on realized forward utility/rank among candidates rather than binary TAKE/SKIP labels dominated by SKIP,
2. add walk-forward calibration windows where threshold selection is frozen before each eval segment,
3. use price-action candidate features that describe stop proximity, range location, breakout/fade context, and multi-timeframe extrema,
4. keep a single LLM scorer, but make it rank a small, higher-quality candidate set rather than score many weak actions.

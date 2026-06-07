# Pressure analyzer direct-trade rejection (2026-06-07)

## Question
Can the compact Gemma4 pressure analyzer be traded directly by mapping `direction_pressure` to a stop/target futures position?

## Setup
- Model: `gemma4-e4b` adapter `checkpoints/teacher_compact_pressure_analyzer_gemma4_e4b_h36_t0p5_s0p6_step16`.
- Labels: teacher-compact pressure, horizon 36 bars, target 0.5%, stop 0.6%.
- Backtest: enter at next-bar open, leverage 0.5, fee 4 bps, slippage 1 bp, target/stop intrabar, ambiguous same-bar target+stop resolved as stop-first.
- Strict MDD: includes entry/exit costs and intrabar adverse excursion while position is open.
- Leakage guard: backtest consumes only prediction JSONL + market OHLC; no forward-return or eval label is used for entry filtering.

## Analyzer accuracy
| Split | Samples | Pressure accuracy | Notes |
| --- | ---: | ---: | --- |
| Val | 552 | 41.12% | Better than majority baseline but below softmax teacher baseline. |
| OOS | 535 | 47.85% | Slightly above the softmax full-OOS baseline observed in the run. |

Important confusion pattern: the model over-predicts `SHORT_FAVORED` and misses many `LONG_FAVORED` rows. Direction classification is learnable, but not enough to infer positive expected trade value after costs.

## Direct stop/target backtest
| Split | Trades | Return | CAGR | Strict MDD | CAGR/MDD | Mean trade | p-value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Val | 354 | -18.86% | -34.11% | 19.40% | -1.76 | -0.0588% | 2.74e-6 |
| OOS | 353 | -11.71% | -22.55% | 13.65% | -1.65 | -0.0350% | 0.00697 |

Exit mix:
- Val: target 133, stop 108, time 113.
- OOS: target 157, stop 109, time 87, ambiguous same-bar 1.

## Live-safe filter sweep on validation
Filters used only model prediction, train-derived teacher confidence, agreement, and allowed side. They did not use eval labels or future returns.

Best validation setting was still negative:
- `min_conf=0.46`, long-only, teacher agreement optional.
- 21 trades, return -0.34%, CAGR -0.68%, strict MDD 2.13%, CAGR/MDD -0.32, mean trade -0.016%, p=0.736.

## Decision
Reject direct `direction_pressure -> LONG/SHORT` trading and do not scale this mapping. The analyzer signal has classification lift, but the economic mapping is net negative after realistic costs and strict drawdown accounting.

## Next implication
The next architecture should make the trader estimate cost-aware expected value / abstention directly, using analyzer output as context rather than as the action. Candidate next unit:
1. Build a value-calibrated trader target from realized stop/target trade returns.
2. Train or baseline a thresholded trader using only train/val selection.
3. Evaluate once on OOS with no parameter changes.

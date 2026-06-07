# Pressure value calibration trader (2026-06-07)

## Question
Can we keep the Gemma4 analyzer as context but avoid direct pressure trading by fitting a train-only, cost-aware value calibration trader?

## Method
- Fit calibration buckets on train split only (`2023-01-01` based SFT rows).
- Each train row is scored for hypothetical LONG and SHORT realized stop/target return after fee+slippage.
- Bucket keys use compact analyzer context (`teacher_pressure`, regime, trend alignment, risk, macro/kimchi, confidence/margin buckets).
- Validation selects one configuration with at least 50 trades.
- OOS is evaluated once using the selected validation config.
- No OOS labels or returns are used for selection.

## Selected validation config
```json
{"level":"teacher_only","min_n":35,"min_score":0.0,"score_mode":"mean","side_gate":"free"}
```

## Result
| Split | Trades | Return | CAGR | Strict MDD | CAGR/MDD | Mean trade | p-value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Val-selected | 57 | -0.95% | -1.90% | 1.64% | -1.16 | -0.0166% | 0.584 |
| OOS-fixed | 59 | -2.40% | -4.87% | 3.81% | -1.28 | -0.0410% | 0.171 |

The trader reduced loss and drawdown versus direct pressure mapping, but it still did not produce positive expected value.

## Contrarian diagnostic
A full inversion of Gemma pressure was also negative:
- Val inverse: 354 trades, CAGR -24.13%, strict MDD 15.01%, ratio -1.61, mean trade -0.0388%, p=0.00187.
- OOS inverse: 353 trades, CAGR -41.65%, strict MDD 23.28%, ratio -1.79, mean trade -0.0741%, p=1.00e-8.

This means the failure is not just direction sign. The current 0.5% target / 0.6% stop / 36-bar setup plus costs leaves little economic edge even when filtered.

## Decision
Reject the simple table-calibrated pressure trader as monetizable. Keep the tool because it gives a leakage-safe train/val/OOS calibration scaffold, but the next attempt must change the economic target/action space rather than only gating pressure labels.

## Next implication
Move from pressure labels to a trader label that optimizes expected net return directly:
1. Sweep target/stop/horizon spaces with strict MDD and cost accounting.
2. Prefer asymmetric payoffs or longer horizons where fee drag is smaller relative to target.
3. Train analyzer/trader text on the selected economic action template only after a non-LLM leakage-safe baseline shows positive validation edge.

# Pressure-only Gemma4 analyzer POC (2026-06-07)

## Hypothesis

The full path-shape analyzer target may be too large for a short POC. Train a smaller analyzer that predicts only:

```json
{"direction_pressure": "LONG_FAVORED|SHORT_FAVORED|NO_TRADE_FAVORED|BOTH_SIDES_VOLATILE"}
```

## Data

- source: `economic_path_shape_h144_t1p0_s0p6_{train,val,oos}.jsonl`
- output: `economic_pressure_analyzer_sft_h144_t1p0_s0p6_{train,val,oos}.jsonl`
- train rows: 2370
- val rows: 552
- OOS rows: 535

Majority baselines:

| split | majority baseline |
| --- | ---: |
| train | 36.79% |
| val | 41.12% |
| OOS | 38.88% |

## Training

- model: `google/gemma-4-E4B-it` via `gemma4-e4b`
- checkpoint: `checkpoints/pressure_analyzer_gemma4_e4b_h144_t1p0_s0p6_step16`
- rows: 512 balanced
- max steps: 16
- runtime: 134.1s
- train loss: 0.3678
- late token accuracy: ~95-97%

## Validation/OOS generation

| split | samples | direction_pressure accuracy | majority baseline | result |
| --- | ---: | ---: | ---: | --- |
| val | 128 | 36.72% | 41.12% | fail |
| OOS | 128 | 34.38% | 38.88% | fail |

## Interpretation

Reducing the analyzer target to one field fixed training loss but did not produce out-of-sample signal. This suggests the current `h144 target=1.0 stop=0.6 direction_pressure` label is not learnable enough from the current past-only summary, at least with the current sampling and short SFT.

The trader POC remains useful because it can map a supplied pressure/path output into actions. The bottleneck is now clearly analyzer label/features, not trader action formatting.

## Next move

Change the label definition before spending more GPU:

1. Sweep shorter horizons and easier targets/stops for pressure learnability.
2. Prefer labels tied to nearer path events, e.g. first 12h/24h impulse, drawdown warning, or no-trade safety.
3. Run a cheap non-LLM learnability baseline before another Gemma SFT.

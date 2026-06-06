# Pressure label learnability sweep (2026-06-07)

## Purpose

Before spending more GPU on Gemma, test which path-pressure label definitions are learnable from the existing past-only analyzer summary features.

## Method

- Fit a cheap train-only softmax classifier on `signal_feature_row` features.
- Evaluate val for model-selection signal.
- Report OOS without tuning on it.
- Swept:
  - horizons: 36, 72, 144 bars
  - targets: 0.5%, 0.8%, 1.0%
  - stops: 0.4%, 0.6%

## Top result

Best val edge over majority:

| horizon | target | stop | val acc | val majority | val edge | OOS acc | OOS majority | OOS edge |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 36 | 0.5% | 0.6% | 42.39% | 36.05% | +6.34pp | 46.54% | 35.51% | +11.03pp |
| 36 | 0.5% | 0.4% | 43.48% | 38.59% | +4.89pp | 45.42% | 36.82% | +8.60pp |
| 72 | 0.5% | 0.4% | 40.04% | 38.95% | +1.09pp | 45.79% | 41.12% | +4.67pp |

## Interpretation

The failed h144/t1.0/s0.6 pressure-only Gemma run used a weak label definition. The shorter 36-bar / 0.5% target pressure label is materially more learnable and keeps its edge on OOS in a train-only softmax baseline.

This is not yet a trading result. It is a label-definition go/no-go signal saying the next Gemma analyzer POC should use h36/t0.5/s0.6 rather than h144/t1.0/s0.6.

## Next step

Generate h36/t0.5/s0.6 path-pressure SFT splits, run the same short Gemma4 pressure analyzer POC, and compare val/OOS against the softmax baseline and majority baseline.

# Causal Kalman state-gate alpha search

Date: 2026-07-13

## Summary

A local-linear Kalman filter was tested as a causal context gate on the fixed
`long_minimal_funding_premium` setup. The strict Train+Test-only winner did not
generalize to Eval 2025 or 2026 YTD, so this experiment does **not** promote a
new alpha.

The useful result is negative but important: a previously attractive Kalman
row was discovered only after looking at later windows. Re-ranking the complete
grid with a frozen robustness rule exposed that apparent success as a
selection artifact rather than a defensible winner.

## Model and leakage controls

- Model: causal local-linear state-space model with latent level and slope.
- Input: completed hourly BTC log price.
- Signal state: 3 slope buckets x 3 standardized-innovation buckets.
- Train (`2020-2023`) fits return variance, state quantiles and state-level
  trade quality.
- Variants are ranked by `min(Train CAGR/MDD, Test-2024 CAGR/MDD)`.
- Entry is on the next 5-minute bar.
- Hold is 576 five-minute bars; cost is 6 bp/side.
- Strict MDD includes intraposition adverse excursion.
- Prefix-causality is regression-tested.

Reference: R. E. Kalman, *A New Approach to Linear Filtering and Prediction
Problems* (1960), <https://doi.org/10.1115/1.3662552>.

### Covariance audit

The first prototype included Kalman predictive covariance as an “uncertainty”
bucket. With fixed process and observation noise, that covariance converged to
a constant after roughly 100 hourly updates and contained no market
information. It was removed before the final scan. The final state therefore
uses only filtered slope and standardized innovation.

## Search

- 3,166 eligible parameter/state-quality variants.
- 1,347 distinct entry masks after exact signal deduplication.
- Parameters varied level noise, slope noise, observation noise, state
  quantiles, minimum Train state count and minimum Train trade edge.
- Eval 2025 and 2026 are report-only in the final script. However, these later
  windows had already been viewed during an earlier exploratory Kalman run, so
  the experiment is not presented as pristine untouched OOS.

## Frozen Train+Test winner

Parameters:

- `q_level = 0.1`
- `q_slope = 0.01`
- `r_obs = 1.0`
- state quantiles: `0.25 / 0.75`
- minimum Train state trades: `5`
- minimum Train mean trade return: `0.50%`
- allowed states:
  - low slope / middle innovation
  - low slope / high innovation
  - middle slope / low innovation
  - middle slope / high innovation

| split | absolute return | CAGR | strict MDD | CAGR/MDD | trades | win rate |
|---|---:|---:|---:|---:|---:|---:|
| Train | +168.37% | 27.99% | 11.13% | 2.51 | 155 | 58.71% |
| Test 2024 | +24.85% | 24.79% | 2.54% | 9.75 | 17 | 82.35% |
| Eval 2025 | +9.55% | 9.56% | 4.27% | 2.24 | 16 | 50.00% |
| 2026 YTD | +1.73% | 4.21% | 4.20% | 1.00 | 18 | 61.11% |

The frozen winner misses the required ratio of 3 in Eval and degrades further
in 2026.

## Ungated baseline

| split | absolute return | CAGR | strict MDD | CAGR/MDD | trades |
|---|---:|---:|---:|---:|---:|
| Train | +128.26% | 22.92% | 14.99% | 1.53 | 206 |
| Test 2024 | +30.36% | 30.29% | 5.05% | 6.00 | 29 |
| Eval 2025 | +18.03% | 18.04% | 4.26% | 4.23 | 26 |
| 2026 YTD | +11.57% | 30.09% | 3.80% | 7.93 | 29 |

The Kalman gate improves Train and Test risk efficiency but removes too much
of the base setup's later-period return.

## Calendar stability

| year | absolute return | CAGR | strict MDD | CAGR/MDD | trades |
|---|---:|---:|---:|---:|---:|
| 2020 | +34.28% | 34.20% | 10.83% | 3.16 | 44 |
| 2021 | +63.61% | 63.67% | 11.13% | 5.72 | 46 |
| 2022 | -0.21% | -0.21% | 10.35% | -0.02 | 41 |
| 2023 | +21.70% | 21.72% | 2.86% | 7.58 | 23 |
| 2024 | +24.85% | 24.79% | 2.54% | 9.75 | 17 |
| 2025 | +9.55% | 9.56% | 4.27% | 2.24 | 16 |
| 2026 YTD | +1.73% | 4.21% | 4.20% | 1.00 | 18 |

The weak 2022 block and the 2025-2026 decay reject a stable cross-regime alpha
interpretation.

## Cost and component stress

At 10 bp/side, the winner's CAGR/MDD is `2.00` in Eval 2025 and `0.57` in 2026
YTD. At 15 bp/side, 2026 YTD is approximately flat (`+0.10%` absolute return,
ratio `0.05`).

Leave-one-state-out analysis also shows that the middle-slope states carry most
of the later-period performance. This is not a broad, redundant state effect.

## Selection-bias verdict

Some lower-ranked variants have attractive Eval/2026 numbers. Promoting one of
them now would explicitly select on the report-only windows and is therefore
rejected. The correct conclusion from the frozen ranking is:

> Kalman slope/innovation state is a useful diagnostic context, but this grid
> did not produce a defensible standalone alpha gate.

## Artifacts

- Script: `training/search_kalman_state_gated_alpha.py`
- Result: `results/kalman_state_gated_alpha_scan_2026-07-13.json`
- Causality test: `tests/test_markov_alpha_models.py`

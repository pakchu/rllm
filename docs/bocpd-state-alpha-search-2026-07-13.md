# Bayesian online change-point alpha search

Date: 2026-07-13

## Summary

Bayesian Online Change Point Detection (BOCPD) was tested as a causal regime
gate for the fixed `long_minimal_funding_premium` setup. The frozen
Train+Test-only winner failed in Eval 2025, so no BOCPD alpha is promoted.

This result also rejects selecting one of the attractive lower-ranked rows
after inspecting Eval/2026. Those rows remain diagnostics, not valid winners.

## Model

The implementation follows the Adams-MacKay online run-length recursion with a
Normal-Gamma/Student-t predictive model:

- source: <https://arxiv.org/abs/0710.3742>
- inference is forward-only and uses completed hourly observations;
- standardization parameters are frozen from Train;
- the run-length posterior is capped for bounded live cost;
- entry occurs on the next 5-minute bar.

Three observation families were tested:

1. hourly return;
2. hourly return plus 24-hour taker-flow mean;
3. 24-hour trend plus short/long volatility ratio.

The state combines:

- posterior segment mean, three buckets;
- posterior mass in run lengths of 0-6 hours, two buckets;
- posterior segment context or predictive surprise, two buckets.

This makes 12 causal states per model.

### Constant-hazard detail

With a constant BOCPD hazard, `P(r_t = 0)` is nearly the hazard itself and is
not a useful market feature. The strategy instead uses posterior mass over
short run lengths, which varies with the filtered run-length distribution.

## Leakage protocol

- State standardization, state thresholds and state trade quality use
  `2020-2023` only.
- Variants are ranked by
  `min(Train CAGR/MDD, Test-2024 CAGR/MDD)`.
- Eval 2025 and 2026 YTD are report-only for the BOCPD overlay.
- Hold is 576 five-minute bars; transaction cost is 6 bp/side.
- Strict MDD includes intraposition adverse excursion.
- Prefix-causality is regression-tested.
- The underlying funding/premium setup has prior research exposure, so the
  composite would still be a research-forward candidate even if the overlay
  passed.

## Search

- 1,521 eligible model/state-quality variants.
- 1,009 distinct entry masks after exact signal deduplication.
- Model horizon, state quantiles, minimum state count and minimum Train edge
  were varied.

## Frozen Train+Test winner

Model:

- observations: 24-hour trend and volatility-term ratio;
- expected regime horizon: 168 hours;
- primary state quantiles: `0.20 / 0.80`;
- short-run-mass threshold: Train median;
- volatility-term threshold: Train 67th percentile;
- minimum Train state trades: `5`;
- minimum Train mean trade return: `0.50%`.

| split | absolute return | CAGR | strict MDD | CAGR/MDD | trades | win rate |
|---|---:|---:|---:|---:|---:|---:|
| Train | +201.28% | 31.75% | 10.29% | 3.09 | 140 | 61.43% |
| Test 2024 | +22.68% | 22.63% | 3.94% | 5.74 | 22 | 77.27% |
| Eval 2025 | -3.68% | -3.68% | 7.72% | -0.48 | 14 | 42.86% |
| 2026 YTD | +8.52% | 21.70% | 4.84% | 4.48 | 19 | 78.95% |

The Eval-2025 sign reversal is a hard rejection despite good Train/Test and a
partial 2026 recovery.

## Calendar stability

| year | absolute return | CAGR | strict MDD | CAGR/MDD | trades |
|---|---:|---:|---:|---:|---:|
| 2020 | +42.05% | 41.95% | 8.44% | 4.97 | 37 |
| 2021 | +52.77% | 52.81% | 10.29% | 5.13 | 39 |
| 2022 | +10.11% | 10.11% | 10.25% | 0.99 | 44 |
| 2023 | +24.44% | 24.46% | 3.39% | 7.21 | 19 |
| 2024 | +22.68% | 22.63% | 3.94% | 5.74 | 22 |
| 2025 | -3.68% | -3.68% | 7.72% | -0.48 | 14 |
| 2026 YTD | +8.52% | 21.70% | 4.84% | 4.48 | 19 |

## Cost and component stress

- At 10 bp/side, Eval 2025 remains negative (`-4.22%`) and 2026 YTD falls to
  CAGR/MDD `3.92`.
- Removing individual states does not repair Eval 2025 without using Eval to
  choose the removal. Such a repair would be post-hoc leakage.

## Verdict

BOCPD extracts interpretable, causal regime-age information, but this search
did not produce a defensible alpha. Lower-ranked rows that happen to pass later
windows are not promoted because choosing them now would select directly on
those windows.

The next experiment should model regime dwell time explicitly and use an
additional pre-Eval internal holdout, rather than adding more BOCPD gates.

## Artifacts

- Script: `training/search_bocpd_state_gated_alpha.py`
- Result: `results/bocpd_state_gated_alpha_scan_2026-07-13.json`
- Causality test: `tests/test_bocpd_alpha_model.py`

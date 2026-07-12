# Explicit-duration semi-Markov alpha search

Date: 2026-07-13

## Summary

An observable explicit-duration semi-Markov entry gate was tested on the fixed
`long_minimal_funding_premium` setup. It produced a strong first untouched test
in 2024 but failed immediately in Eval 2025 and remained weak in 2026 YTD. No
new alpha is promoted.

## Why this model

The previous observable Markov alpha benefited from state persistence. This
experiment tests persistence directly by augmenting each completed-hour market
state with the causal age of that state.

The implementation is a deliberately small observable semi-Markov-style model,
not a full hidden-state HSMM. Duration affects **entry gating only**; exits remain
the fixed 48-hour policy.

Reference: Kevin Murphy, *Hidden semi-Markov models* (2002),
<https://www.cs.ubc.ca/~murphyk/papers/segment.pdf>.

## Predeclared model

Hourly observable state:

- 24-hour BTC trend: low / middle / high;
- 24-hour hourly-return volatility: low / high;
- 24-hour taker flow: low / high.

State age buckets were fixed before the run:

- 1 hour;
- 2-6 hours;
- 7-24 hours;
- 25-72 hours;
- more than 72 hours.

Only one trend binning (`0.33 / 0.67`) and one duration scheme were used. This
avoids searching many duration partitions after seeing later results.

For every state-age key, 2020-2022 fits:

- mean setup-trade return;
- number of setup trades;
- Laplace-smoothed empirical state-exit hazard.

The hazard is an entry-context feature only. It is not used as a trade-exit
forecast.

## Leakage and selection protocol

- All state thresholds, duration-key quality and hazards use `2020-2022` only.
- Hyperparameter candidates vary only minimum key count, minimum Train edge and
  maximum empirical exit hazard: 21 eligible variants, 14 distinct signals.
- Selection ranks the worst CAGR/strict-MDD across 2020, 2021, 2022 and the
  internal 2023 holdout.
- 2024 is the first untouched overlay test.
- 2025 is the second evaluation window; 2026 YTD is forward diagnostic.
- Run age uses completed hourly bars and resets across missing-hour gaps.
- Entry is the next 5-minute bar; repeated entries are blocked during the
  48-hour hold.
- Cost is 6 bp/side at 0.5x; strict MDD includes intraposition adverse excursion.
- The base funding/premium setup still has broader research-history exposure.

## Frozen winner

- minimum Fit key trades: `3`
- minimum Fit mean trade return: `0.50%`
- maximum smoothed state-exit hazard: `0.50`
- allowed state-age keys: `12, 15, 16, 17, 20, 31, 35, 47, 52, 55`

| split | absolute return | CAGR | strict MDD | CAGR/MDD | trades | win rate |
|---|---:|---:|---:|---:|---:|---:|
| Fit 2020-2022 | +110.25% | 28.10% | 16.99% | 1.65 | 134 | 56.72% |
| 2020 | +36.08% | 36.00% | 11.60% | 3.10 | 47 | 63.83% |
| 2021 | +66.17% | 66.23% | 11.04% | 6.00 | 43 | 65.12% |
| 2022 | +1.19% | 1.19% | 16.99% | 0.07 | 43 | 41.86% |
| Holdout 2023 | +17.82% | 17.83% | 6.09% | 2.93 | 21 | 42.86% |
| Untouched Test 2024 | +27.84% | 27.78% | 1.78% | 15.64 | 20 | 80.00% |
| Eval 2025 | +1.63% | 1.63% | 7.33% | 0.22 | 11 | 54.55% |
| 2026 YTD | +3.61% | 8.90% | 4.98% | 1.79 | 18 | 66.67% |

The model passes 2024 impressively but fails the next independent annual block.
It also never solved the known weak 2022 long regime.

## Trade-return bootstrap

Non-overlapping trade returns were resampled 20,000 times.

| split | trades | mean/trade | 95% bootstrap CI | P(mean > 0) |
|---|---:|---:|---:|---:|
| Holdout 2023 | 21 | +0.82% | -0.18% to +2.04% | 93.64% |
| Test 2024 | 20 | +1.24% | +0.73% to +1.77% | 100.00% |
| Eval 2025 | 11 | +0.16% | -0.99% to +1.20% | 62.75% |
| 2026 YTD | 18 | +0.21% | -0.38% to +0.76% | 75.44% |

Only Test 2024 has a clearly positive interval. Eval 2025 does not meet the
predeclared statistical screen.

## Cost stress

At 10 bp/side:

- Test 2024 remains strong: CAGR/MDD `14.58`.
- Eval 2025 remains negligible: CAGR/MDD `0.16`.
- 2026 YTD falls to CAGR/MDD `1.39`.

The failure is not caused merely by optimistic transaction cost.

## Component stress

Leave-one-key-out results show several keys have no observations in later
windows, while removing key `55` would improve Eval 2025. Removing it now would
use Eval to repair the model and is therefore rejected as post-hoc selection.

## Verdict

The duration model captured a real 2024 effect but not a stable multi-year
alpha. It is rejected for promotion. The experiment supports two broader
conclusions:

1. regime persistence is useful context, but finer duration conditioning
   rapidly fragments the trade sample;
2. a good single untouched year is insufficient when the next independent
   year loses the effect.

## Artifacts

- Script: `training/search_semimarkov_duration_alpha.py`
- Result: `results/semimarkov_duration_alpha_scan_2026-07-13.json`
- Causality tests: `tests/test_semimarkov_alpha_model.py`

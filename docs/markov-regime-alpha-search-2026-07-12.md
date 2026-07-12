# Hidden/observable Markov regime alpha search

Date: 2026-07-12

## Summary

A Gaussian HMM traded as a standalone hourly direction model failed. An observable Markov-chain gate applied only when the fixed funding/premium setup fires produced a materially stronger and sparser alpha candidate.

The successful mechanism is **regime persistence**, not a generic state forecast: trade only when the previous hourly state and current hourly state are the same and that transition had positive setup-level expectancy in Train.

## Leakage protocol

- State thresholds and transition trade quality are fit on `2020-2023` only.
- Current state uses the completed hourly observation at or before the signal.
- Entry occurs on the following 5-minute bar.
- Test 2024 ranks variants.
- Eval 2025 and 2026 YTD are report-only for this Markov overlay experiment.
- Cost is 6 bp/side; strict MDD includes intraposition adverse excursion.
- The underlying funding/premium setup predates this experiment and has broader research-history contamination, so the composite is not labelled pristine final OOS.

## Failed Gaussian HMM standalone

The diagonal Gaussian HMM used causal filtered probabilities, 3-5 states and three feature sets. Direct hourly long/short trading generated too much low-edge turnover.

Best Test-2024-ranked HMM variant:

| split | absolute return | CAGR | strict MDD | CAGR/MDD | trades |
|---|---:|---:|---:|---:|---:|
| Train | -95.46% | -53.83% | 95.94% | -0.56 | 4,093 |
| Test 2024 | +4.30% | 4.29% | 22.10% | 0.19 | 741 |
| Eval 2025 | -21.71% | -21.72% | 25.67% | -0.85 | 630 |
| 2026 YTD | -15.31% | -32.93% | 18.68% | -1.76 | 262 |

Conclusion: HMM state is context, not a standalone action policy.

## Successful observable Markov alpha

### Base setup

`long_minimal_funding_premium`, hold 576 five-minute bars. The base trigger is the union of:

- low funding with positive medium trend; and
- depressed premium change with strong completed daily momentum.

### Observable state

Hourly state combines:

- 24-hour log trend: low / neutral / high;
- 24-hour hourly-return volatility: low / high;
- 24-hour taker flow: low / high.

Frozen Train thresholds:

- trend low: `-0.01342395`
- trend high: `+0.01660670`
- volatility median: `0.00523541`
- taker-flow median: `-0.00351340`

State encoding is `trend_bucket * 4 + volatility_bucket * 2 + flow_bucket`. The selected transition keys are `52, 143, 26, 39, 65`; all are self-transitions. Therefore the gate requires a qualifying state to persist for at least two consecutive completed hourly observations.

Selection requirements:

- at least 8 Train setup trades in the transition;
- Train mean trade return at least `0.20%`;
- selected transition keys are fixed before Eval 2025/2026.

## Main statistics

| split | absolute return | CAGR | strict MDD | CAGR/MDD | trades | win rate |
|---|---:|---:|---:|---:|---:|---:|
| Train | +194.76% | 31.03% | 12.31% | 2.52 | 180 | 58.33% |
| Test 2024 | +37.88% | 37.79% | 3.27% | 11.56 | 22 | 86.36% |
| Eval 2025 | +18.33% | 18.34% | 2.83% | 6.48 | 19 | 73.68% |
| 2026 YTD | +8.82% | 22.51% | 3.74% | 6.03 | 23 | 69.57% |

### Ungated baseline comparison

| split | baseline CAGR/MDD | Markov CAGR/MDD | baseline trades | Markov trades |
|---|---:|---:|---:|---:|
| Train | 1.53 | 2.52 | 206 | 180 |
| Test 2024 | 6.00 | 11.56 | 29 | 22 |
| Eval 2025 | 4.23 | 6.48 | 26 | 19 |
| 2026 YTD | 7.93 | 6.03 | 29 | 23 |

The gate improves Train, 2024 and 2025 risk efficiency. It reduces 2026 absolute return and ratio, but 2026 remains above the target ratio of 5.

## Calendar-year stability

| year | absolute return | CAGR | strict MDD | CAGR/MDD | trades |
|---|---:|---:|---:|---:|---:|
| 2020 | +29.20% | 29.14% | 10.70% | 2.72 | 53 |
| 2021 | +74.05% | 74.12% | 12.31% | 6.02 | 48 |
| 2022 | +7.43% | 7.44% | 11.83% | 0.63 | 50 |
| 2023 | +22.48% | 22.50% | 5.04% | 4.46 | 28 |
| 2024 | +37.88% | 37.79% | 3.27% | 11.56 | 22 |
| 2025 | +18.33% | 18.34% | 2.83% | 6.48 | 19 |
| 2026 YTD | +8.82% | 22.51% | 3.74% | 6.03 | 23 |

Every calendar block is profitable, but 2022 is weak. This prevents calling the alpha universally regime-independent.

## Cost stress

At 10 bp/side, CAGR/MDD remains:

- Test 2024: `10.87`
- Eval 2025: `5.95`
- 2026 YTD: `5.18`

At 15 bp/side, 2026 falls to `4.19`; live deployment therefore requires normal futures execution costs and should not assume unlimited taker slippage.

## Leave-one-transition-out

Removing one transition at a time generally preserves positive performance. Key `26` is important for 2026 and key `143` is important for 2025, so performance is diversified across several transition regimes but not completely insensitive to each component.

## Verdict

Promote as a **strong research alpha / forward-shadow candidate**. It is a genuine Markov transition overlay with improved multi-period risk efficiency, but it is not yet a pristine final OOS strategy because the underlying setup was discovered during prior iterative research.

## Artifacts

- Gaussian HMM script: `training/search_gaussian_hmm_regime_alpha.py`
- Gaussian HMM result: `results/gaussian_hmm_regime_alpha_scan_2026-07-12.json`
- HMM setup-gate script: `training/search_hmm_gated_funding_premium_alpha.py`
- HMM setup-gate result: `results/hmm_gated_funding_premium_alpha_scan_2026-07-12.json`
- Observable Markov script: `training/search_markov_transition_gated_alpha.py`
- Observable Markov result: `results/markov_transition_gated_alpha_scan_2026-07-12.json`


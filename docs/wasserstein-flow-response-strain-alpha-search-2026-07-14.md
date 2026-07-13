# Wasserstein Flow-Response Strain Alpha Search — 2026-07-14

## Thesis

The same amount of aggressive buying and selling need not move price equally.
Instead of fitting one average impact coefficient, this experiment compares the
**entire side-aligned price-response distributions**:

- buy response: volatility-normalized return during strong aggressive buying;
- sell response: negated volatility-normalized return during strong aggressive
  selling, so positive values mean movement in the aggressive side's direction.

At each hourly decision, the exact one-dimensional monotone transport map pairs
the nine deciles of the two distributions. Its displacement has:

- a central location term measuring which side transports into more price
  movement;
- a tail-shape term measuring whether the asymmetry grows in extreme responses;
- a Wasserstein-1 magnitude used as a diagnostic.

Positive signed strain means buy aggression has the thinner opposing liquidity
surface and maps to long; negative strain maps to short. This is a local
distribution-geometry state, not historical analogue retrieval, linear impact
regression, transfer entropy or a path-order statistic.

## Causal protocol

- Source rows are physically cut before `2024-01-01`.
- Bar response is completed-bar `log(close/open)` divided by a seven-day
  volatility estimate ending at the previous bar, then clipped to `[-5,5]`.
- Strong flow is `q70(abs(taker imbalance))`, fitted on 2020-2022 only.
- Lookback distributions end at the completed decision bar.
- Open-time-labelled row `:55` is completed at the next hour boundary; its
  signal enters the following row `:00` open.
- Fit is 2020-10-15 through 2022. 2023 is inspected internal selection. 2024+
  remains sealed.
- Position size is `0.5x`, cost is `6bp/side`, and strict MDD uses
  favorable-first/adverse-second OHLC high-water accounting.

## Search disclosure

The final architect-reviewed grid contains eight policies:

- response distribution lookback: 24 hours, seven days;
- absolute strain tail: fit-only q80, q90;
- hold: six hours, 12 hours;
- one fixed positive-long/negative-short mapping.

Before that grid, two researcher probes were inspected on pre-2024 data:

1. within-window flow terciles, raw Wasserstein magnitude and directional
   coherence — eight policies;
2. the same state centered on a trailing seven-day median — eight policies.

Both were weak. Those 16 precursor policies and the final eight policies are
fully contaminated and frozen. This disclosure prevents the final grid from
being misrepresented as the only attempted interpretation.

The final fit-only absolute-flow cutoff was `0.206398`. The 24-hour state had
20,519 valid hourly decisions with median 55 buy and 57 sell observations; the
seven-day state had 34,906 decisions with median 335 and 354 observations.

## Best ranked policy

24-hour distributions, q90 absolute strain and 12-hour hold:

| Period | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |
|---|---:|---:|---:|---:|---:|---:|
| fit | +0.53% | +0.24% | 28.47% | 0.01 | 164 | 89/75 |
| 2023 | +12.54% | +12.55% | 7.06% | 1.78 | 114 | 88/26 |
| 2023H1 | +4.08% | +8.40% | 5.38% | 1.56 | 46 | 37/9 |
| 2023H2 | +7.95% | +16.41% | 5.79% | 2.83 | 68 | 51/17 |

The 2023 breadth is notable, but fit is economically flat and strict MDD is far
above target. The policy loses in 2021H2, 2022H1 and 2022H2. Zero of eight final
policies passed the required fit-and-2023 CAGR/MDD `>=3` gate.

The q80/12-hour sibling had stronger fit (`+33.95%`, ratio `0.85`) and positive
2023 (`+8.83%`, ratio `1.03`) but lost `-1.06%` in 2023H2. This confirms a
threshold/regime trade-off rather than a robust static policy.

## Structural controls

| Control | Fit return / ratio / trades | 2023 return / ratio / trades |
|---|---:|---:|
| Exact direction flip | -20.93% / -0.31 / 164 | -23.23% / -0.93 / 114 |
| First state onset only | -4.81% / -0.07 / 149 | +18.27% / 3.16 / 100 |
| Mean response gap only | +17.09% / 0.34 / 164 | +4.19% / 0.40 / 110 |
| Mean/std/IQR moment state | -0.45% / -0.01 / 174 | -5.89% / -0.37 / 123 |
| Signed W1 magnitude; no transport shape | +20.41% / 0.44 / 169 | +1.71% / 0.18 / 100 |
| Signal delayed one hour | -12.40% / -0.24 / 164 | +11.76% / 2.05 / 114 |
| Signal delayed six hours | +27.65% / 0.55 / 164 | +0.02% / 0.00 / 114 |
| Signal delayed seven days | -4.95% / -0.12 / 167 | -5.75% / -0.51 / 113 |

The exact direction is strongly identified, and the quantile transport shape
adds substantial 2023 information over mean, moment and unsigned-distance
controls. Onset-only reaches a 2023 ratio above three, but its negative fit and
three losing fit half-years reject it as another regime-local expression.

## Cost stress

| Cost per side | Fit return / ratio | 2023 return / ratio |
|---|---:|---:|
| 0bp | +10.93% / 0.19 | +20.51% / 3.54 |
| 1bp | +9.12% / 0.16 | +19.14% / 3.31 |
| 3bp | +5.60% / 0.09 | +16.46% / 2.72 |
| 6bp | +0.53% / 0.01 | +12.54% / 1.78 |
| 10bp | -5.85% / -0.09 | +7.52% / 0.90 |
| 15bp | -13.27% / -0.18 | +1.56% / 0.15 |

Turnover matters, but even zero-cost fit risk efficiency is only `0.19`; cost
does not explain the structural rejection.

## Decision

**Do not promote or open 2024+.** Keep the signed transport location, shape,
W1 magnitude, side sample counts and flow cutoff as a weak beta representation.
The static lookbacks, tails, mapping, onset interpretation and holds are gamma
failure provenance and must not be retuned on the same sample.

The feature is worth preserving because direction, distribution shape and
timing survive meaningful falsification controls. It is not an executable alpha
because its fit risk efficiency and regime stability are inadequate.

## Research context

Optimal-transport regime research motivates comparing distributions rather
than only moments. The side-conditioned flow-response transport and execution
mapping are this repository's own falsifiable construction:

- [Market regime detection with Wasserstein distance](https://arxiv.org/abs/2110.11848)
- [Journal of Computational Finance version](https://doi.org/10.21314/JCF.2024.005)

These sources motivate distributional geometry; they do not validate WFRS.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python -m training.search_wasserstein_flow_response_strain_alpha
PYTHONPATH=. .venv/bin/pytest -q tests/test_search_wasserstein_flow_response_strain_alpha.py
```

Artifacts:

- `training/search_wasserstein_flow_response_strain_alpha.py`
- `tests/test_search_wasserstein_flow_response_strain_alpha.py`
- `results/wasserstein_flow_response_strain_alpha_scan_2026-07-14.json`

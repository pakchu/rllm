# Nonequilibrium probability-current alpha search (2026-07-14)

## Decision

**Reject the complete static family; no beta promotion; keep 2024+ sealed.**

This label-free experiment encoded each completed hour as one of eight
observable microstates:

`price-return sign × aggressive-taker-flow sign × open-interest-change sign`.

Using only the previous 720 hourly transitions, a Dirichlet-smoothed joint
transition flux was split into symmetric traffic and antisymmetric probability
current. From the current state, the policy followed the expected price sign of
destination states reached by positive irreversible current. Its score was the
local entropy-production contribution times directional certainty.

This is not the repository's outcome-fitted Markov gate: no trade return or
future label entered transition estimation, signal direction, or score.

## Protocol

- Source physically truncated before `2024-01-01`; 2024+ was never opened.
- The rolling graph at hour `t` contains transitions ending no later than
  `t-1`; the current transition is excluded.
- Current microstate uses the completed minute-55 bar and enters next
  minute-00 open.
- One fixed 30-day graph, q80 fit-only current-score tail, and two holds
  (6h/12h): two policies total.
- 0.5x exposure, 6 bp per side, non-overlapping holds, strict conservative MDD.
- 2023 is inspected internal selection; all pre-2024 work is exploratory.

## Results

| Hold / split | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades (L/S) |
|---|---:|---:|---:|---:|---:|
| 12h / fit | -52.32% | -28.45% | 54.49% | -0.52 | 900 (461/439) |
| 12h / 2023 | -8.09% | -8.10% | 27.53% | -0.29 | 455 (167/288) |
| 6h / fit | -44.05% | -23.09% | 47.53% | -0.49 | 1,399 (725/674) |
| 6h / 2023 | -42.28% | -42.30% | 48.74% | -0.87 | 714 (286/428) |

Only 2023 H1 of the 12-hour policy was positive (+9.62%, ratio 1.40); all five
fit half-year blocks and 2023 H2 failed. Zero of two policies passed admission.

## Falsification controls on the 12-hour policy

| Control | Fit return / ratio | 2023 return / ratio |
|---|---:|---:|
| Exact direction flip | -38.86% / -0.43 | -38.75% / -0.95 |
| Time-reversed current | -49.81% / -0.47 | -32.70% / -0.96 |
| Ordinary Markov expectation | -46.02% / -0.34 | -17.78% / -0.66 |
| Same timestamps, current price side | -24.83% / -0.34 | -53.26% / -1.00 |
| Remove OI state | -26.01% / -0.29 | -22.65% / -0.66 |
| Delay signal 1 hour | -47.41% / -0.47 | -11.67% / -0.41 |
| Delay signal 7 days | +9.44% / 0.12 | -0.81% / -0.04 |

Neither current orientation, exact inversion, detailed-balance baseline, nor
state ablation creates stable edge. The seven-day delay improving fit while
remaining flat/negative in 2023 further contradicts a local irreversible-cycle
interpretation.

## Cost stress

| Cost/side | Fit return / ratio | 2023 return / ratio |
|---|---:|---:|
| 0 bp | -18.17% / -0.24 | +20.76% / 1.47 |
| 1 bp | -25.21% / -0.34 | +15.39% / 0.95 |
| 3 bp | -37.53% / -0.45 | +5.35% / 0.26 |
| 6 bp | -52.32% / -0.52 | -8.09% / -0.29 |
| 10 bp | -66.74% / -0.58 | -23.39% / -0.64 |
| 15 bp | -78.80% / -0.64 | -38.99% / -0.83 |

Fit loses even at zero cost. Turnover is severe, but costs are not the sole
cause. The exact state encoding, graph window, q80 tail, current projection and
holds are frozen as gamma failure provenance.

## Artifacts

- `training/search_nonequilibrium_probability_current_alpha.py`
- `tests/test_search_nonequilibrium_probability_current_alpha.py`
- `results/nonequilibrium_probability_current_alpha_scan_2026-07-14.json`

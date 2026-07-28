# Gross9 + fixed OI-pullback pre-2025 marginal audit

Metric: `absolute return / full-calendar CAGR / strict MDD / CAGR-MDD / trades`.

- evaluated weights: 4
- passers: 0
- decision: **reject_marginal**

| weight | pass | train | 2024 | train ratio Δ / control Δ | 2024 ratio Δ / control Δ |
|---:|:---:|---:|---:|---:|---:|
| 0.25 | N | 2650.66/170.40/37.28/4.57/1174 | 237.85/237.01/16.48/14.38/267 | -0.008 / +0.066 | +1.306 / +0.234 |
| 0.50 | N | 2735.63/172.88/37.99/4.55/1174 | 253.75/252.83/16.09/15.71/267 | -0.028 / +0.133 | +2.636 / +0.472 |
| 0.75 | N | 2806.73/174.92/38.70/4.52/1174 | 270.18/269.19/15.71/17.14/267 | -0.059 / +0.200 | +4.057 / +0.715 |
| 1.00 | N | 2862.79/176.50/39.87/4.43/1174 | 287.15/286.08/15.34/18.65/267 | -0.152 / +0.267 | +5.573 / +0.962 |

## Fixed sleeve

- standalone train: `19.00/5.36/26.25/0.20/313`
- standalone 2024: `20.74/20.69/3.84/5.39/64`
- maximum entry Jaccard versus a Gross9 sleeve: `0.0320`

## Boundary

- The signal config, four thresholds, long side, hold, stride, and costs were not changed.
- Only train and 2024 shared-clock arrays were passed to weight selection.
- Same-gross control scales frozen Gross9 pro rata instead of adding the candidate.
- 2025, 2026, and July results are absent from this artifact and cannot rerank it.

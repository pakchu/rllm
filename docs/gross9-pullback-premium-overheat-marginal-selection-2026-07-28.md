# Gross9 + frozen PPOSM same-gross marginal selection

Metric: `absolute return / full-calendar CAGR / strict MDD / CAGR-MDD / trades`.

- decision: **freeze_top1_for_future_veto**
- tested/pass: `4/2`
- research boundary: the standalone candidate future was already exposed; only the exact Gross9 portfolio interaction was previously unmeasured.

| weight | pass | train combined | 2024 combined | train Δ ratio | 2024 Δ ratio |
|---:|:---:|---:|---:|---:|---:|
| 0.75 | Y | 4672.53% / 219.03% / 36.58% / 5.99 / 949 | 268.27% / 267.29% / 15.61% / 17.12 / 221 | +1.209 | +3.331 |
| 0.25 | Y | 3137.99% / 183.97% / 36.58% / 5.03 / 949 | 237.23% / 236.39% / 16.26% / 14.54 / 221 | +0.384 | +1.222 |
| 1.00 | N | 5663.89% / 237.62% / 37.56% / 6.33 / 949 | 284.58% / 283.52% / 15.81% / 17.93 / 221 | +1.480 | +3.894 |
| 0.50 | N | 3837.94% / 201.15% / 36.58% / 5.50 / 949 | 252.49% / 251.58% / 15.57% / 16.15 / 221 | +0.788 | +2.604 |

## Integrity

- candidate freeze: `1360cf620b8afcead476e2aa0c1394e1419b70c42cac7e03dcc91912ab60c81f`
- max exact-entry Jaccard: `0.0809`
- Weight selection saw only train and 2024 arrays.
- 2025/2026 cannot rerank or repair the frozen top1.

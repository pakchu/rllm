# Gross9 AFGI-12 pre-2025 marginal battery

Metric: `absolute return / full-calendar CAGR / strict MDD / CAGR-MDD / trades`.

- tested cells: `12`
- passing cells: `0`
- decision: **reject_and_close_six_alt_flow_model_family**
- future opened: `False`

## Common-coverage frozen Gross9

- 2023 H2: `1.45/2.91/19.98/0.15/93`
- 2024: `222.48/221.71/16.95/13.08/203`

## Ranked cells

| # | candidate | w | pass | standalone 2023H2 | standalone 2024 | portfolio 2023H2 | portfolio 2024 | same-gross Δ 2023H2/2024 | MDD Δ 2023H2/2024 |
|---:|---|---:|:---:|---:|---:|---:|---:|---:|---:|
| 1 | `alt_flow_geometry_h144_unrestricted` | 0.25 | N | 0.00/0.00/0.00/0.00/0 | 0.00/0.00/0.00/0.00/0 | 1.45/2.91/19.98/0.15/93 | 222.48/221.71/16.95/13.08/203 | 0.005/-0.234 | 0.00/0.00 |
| 2 | `alt_flow_geometry_h144_gross9_flat_at_signal` | 0.25 | N | 0.00/0.00/0.00/0.00/0 | 0.00/0.00/0.00/0.00/0 | 1.45/2.91/19.98/0.15/93 | 222.48/221.71/16.95/13.08/203 | 0.005/-0.234 | 0.00/0.00 |
| 3 | `alt_flow_geometry_h144_gross9_drawdown_ge_5pct` | 0.25 | N | 0.00/0.00/0.00/0.00/0 | 0.00/0.00/0.00/0.00/0 | 1.45/2.91/19.98/0.15/93 | 222.48/221.71/16.95/13.08/203 | 0.005/-0.234 | 0.00/0.00 |
| 4 | `alt_flow_geometry_h144_unrestricted` | 0.50 | N | 0.00/0.00/0.00/0.00/0 | 0.00/0.00/0.00/0.00/0 | 1.45/2.91/19.98/0.15/93 | 222.48/221.71/16.95/13.08/203 | 0.010/-0.472 | 0.00/0.00 |
| 5 | `alt_flow_geometry_h144_gross9_flat_at_signal` | 0.50 | N | 0.00/0.00/0.00/0.00/0 | 0.00/0.00/0.00/0.00/0 | 1.45/2.91/19.98/0.15/93 | 222.48/221.71/16.95/13.08/203 | 0.010/-0.472 | 0.00/0.00 |
| 6 | `alt_flow_geometry_h144_gross9_drawdown_ge_5pct` | 0.50 | N | 0.00/0.00/0.00/0.00/0 | 0.00/0.00/0.00/0.00/0 | 1.45/2.91/19.98/0.15/93 | 222.48/221.71/16.95/13.08/203 | 0.010/-0.472 | 0.00/0.00 |
| 7 | `alt_flow_geometry_h144_unrestricted` | 0.75 | N | 0.00/0.00/0.00/0.00/0 | 0.00/0.00/0.00/0.00/0 | 1.45/2.91/19.98/0.15/93 | 222.48/221.71/16.95/13.08/203 | 0.014/-0.715 | 0.00/0.00 |
| 8 | `alt_flow_geometry_h144_gross9_flat_at_signal` | 0.75 | N | 0.00/0.00/0.00/0.00/0 | 0.00/0.00/0.00/0.00/0 | 1.45/2.91/19.98/0.15/93 | 222.48/221.71/16.95/13.08/203 | 0.014/-0.715 | 0.00/0.00 |
| 9 | `alt_flow_geometry_h144_gross9_drawdown_ge_5pct` | 0.75 | N | 0.00/0.00/0.00/0.00/0 | 0.00/0.00/0.00/0.00/0 | 1.45/2.91/19.98/0.15/93 | 222.48/221.71/16.95/13.08/203 | 0.014/-0.715 | 0.00/0.00 |
| 10 | `alt_flow_geometry_h144_unrestricted` | 1.00 | N | 0.00/0.00/0.00/0.00/0 | 0.00/0.00/0.00/0.00/0 | 1.45/2.91/19.98/0.15/93 | 222.48/221.71/16.95/13.08/203 | 0.019/-0.962 | 0.00/0.00 |
| 11 | `alt_flow_geometry_h144_gross9_flat_at_signal` | 1.00 | N | 0.00/0.00/0.00/0.00/0 | 0.00/0.00/0.00/0.00/0 | 1.45/2.91/19.98/0.15/93 | 222.48/221.71/16.95/13.08/203 | 0.019/-0.962 | 0.00/0.00 |
| 12 | `alt_flow_geometry_h144_gross9_drawdown_ge_5pct` | 1.00 | N | 0.00/0.00/0.00/0.00/0 | 0.00/0.00/0.00/0.00/0 | 1.45/2.91/19.98/0.15/93 | 222.48/221.71/16.95/13.08/203 | 0.019/-0.962 | 0.00/0.00 |

## Fold thresholds

| fold | fit rows | prediction rows | raw prior q95 | threshold |
|---|---:|---:|---:|---:|
| `calibration_2023q2` | 1475 | 2172 | calibration only | — |
| `selection_2023q3` | 3659 | 2026 | -0.005335 | 0.000000 |
| `selection_2023q4` | 5697 | 2196 | -0.005691 | 0.000000 |
| `selection_2024` | 7905 | 8597 | -0.005331 | 0.000000 |

## Executable schedules

| candidate | 2023H2 L/S | 2024 L/S |
|---|---:|---:|
| `alt_flow_geometry_h144_unrestricted` | 0/0 | 0/0 |
| `alt_flow_geometry_h144_gross9_flat_at_signal` | 0/0 | 0/0 |
| `alt_flow_geometry_h144_gross9_drawdown_ge_5pct` | 0/0 | 0/0 |

## Boundary

- The source prefix, BTC market, exact funding, and Gross9 context are physically/logically truncated before 2025.
- Q2 2023 predictions calibrate only; every later threshold comes from the immediately prior OOS score fold.
- Exact 10bp stress replays unchanged features, models, scores, sides, gates, and schedules.
- 2025/2026 were not opened and cannot rerank or repair this family.

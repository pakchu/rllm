# RITT-1 pre-2025 annual selection

Metric: `absolute return / full-calendar CAGR / strict MDD / CAGR-MDD / trades (long/short)`.

- policy cells: 4
- standalone passers: 0
- decision: **reject_pre2025**
- table: top failed cells (diagnostic only)

| # | policy | pass | 2023 | 2023 H1 | 2023 H2 | 2024 | 2024 H1 | 2024 H2 |
|---:|---|:---:|---:|---:|---:|---:|---:|---:|
| 1 | `ritt_h72_lcb00` | N | 0.00/0.00/0.00/0.00/0 (0/0) | 0.00/0.00/0.00/0.00/0 (0/0) | 0.00/0.00/0.00/0.00/0 (0/0) | 0.00/0.00/0.00/0.00/0 (0/0) | 0.00/0.00/0.00/0.00/0 (0/0) | 0.00/0.00/0.00/0.00/0 (0/0) |
| 2 | `ritt_h72_lcb05` | N | 0.00/0.00/0.00/0.00/0 (0/0) | 0.00/0.00/0.00/0.00/0 (0/0) | 0.00/0.00/0.00/0.00/0 (0/0) | 0.00/0.00/0.00/0.00/0 (0/0) | 0.00/0.00/0.00/0.00/0 (0/0) | 0.00/0.00/0.00/0.00/0 (0/0) |
| 3 | `ritt_h144_lcb00` | N | 0.00/0.00/0.00/0.00/0 (0/0) | 0.00/0.00/0.00/0.00/0 (0/0) | 0.00/0.00/0.00/0.00/0 (0/0) | 0.00/0.00/0.00/0.00/0 (0/0) | 0.00/0.00/0.00/0.00/0 (0/0) | 0.00/0.00/0.00/0.00/0 (0/0) |
| 4 | `ritt_h144_lcb05` | N | 0.00/0.00/0.00/0.00/0 (0/0) | 0.00/0.00/0.00/0.00/0 (0/0) | 0.00/0.00/0.00/0.00/0 (0/0) | 0.00/0.00/0.00/0.00/0 (0/0) | 0.00/0.00/0.00/0.00/0 (0/0) | 0.00/0.00/0.00/0.00/0 (0/0) |

## Rejection diagnosis

- hold `72` / 2023: global utility `-0.004177/-0.004192` (long/short), best supported posterior `-0.002586/-0.002639`, positive supported side cells `0`.
- hold `72` / 2024: global utility `-0.003595/-0.003831` (long/short), best supported posterior `-0.002008/-0.002465`, positive supported side cells `0`.
- hold `144` / 2023: global utility `-0.005701/-0.005781` (long/short), best supported posterior `-0.002964/-0.002930`, positive supported side cells `0`.
- hold `144` / 2024: global utility `-0.004808/-0.005365` (long/short), best supported posterior `-0.002875/-0.002931`, positive supported side cells `0`.

All adequately supported transition-side posteriors remained below zero after frozen costs and adverse-excursion penalty. The zero-trade result is therefore a fail-closed model decision, not missing data or an execution error.

## Boundary

- Market, spot/premium, and funding sources were physically truncated before 2025.
- OI is delayed one complete 5m bar; all rolling regressions and z-scores use statistics ending at t-1.
- 2023 and 2024 use separate expanding fits with exits purged before each cutoff.
- Entry is the next 5m open; costs, idle calendar time, non-overlap, and split-contained exits are included.
- Gross9 marginal and 2025/2026 outcomes remain unopened unless a standalone cell passes.

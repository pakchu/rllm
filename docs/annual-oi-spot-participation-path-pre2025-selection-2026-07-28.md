# AOSP-1 pre-2025 annual selection

Metric: `absolute return / full-calendar CAGR / strict MDD / CAGR-MDD / trades (long/short)`.

- features: 35
- policy cells: 24
- standalone passers: 0
- decision: **reject_pre2025**
- table: top failed cells (diagnostic only; no cell admitted)

| # | policy | pass | 2023 | 2023 H1 | 2023 H2 | 2024 | 2024 H1 | 2024 H2 |
|---:|---|:---:|---:|---:|---:|---:|---:|---:|
| 1 | `aosp_h144_mean_lam05_q90_both` | N | 5.21/5.21/4.28/1.22/27 (2/25) | 0.90/1.82/2.86/0.64/14 (2/12) | 4.27/8.66/4.28/2.02/13 (0/13) | 1.26/1.26/2.56/0.49/6 (1/5) | -0.08/-0.15/2.56/-0.06/3 (0/3) | 1.34/2.68/1.76/1.53/3 (1/2) |
| 2 | `aosp_h144_mean_lam05_q80_both` | N | 5.21/5.21/4.28/1.22/27 (2/25) | 0.90/1.82/2.86/0.64/14 (2/12) | 4.27/8.66/4.28/2.02/13 (0/13) | 1.26/1.26/2.56/0.49/6 (1/5) | -0.08/-0.15/2.56/-0.06/3 (0/3) | 1.34/2.68/1.76/1.53/3 (1/2) |
| 3 | `aosp_h144_mean_lam05_q90_long` | N | 0.32/0.32/0.76/0.42/2 (2/0) | 0.32/0.65/0.76/0.85/2 (2/0) | 0.00/0.00/0.00/0.00/0 (0/0) | 1.01/1.01/1.51/0.67/1 (1/0) | 0.00/0.00/0.00/0.00/0 (0/0) | 1.01/2.01/1.51/1.34/1 (1/0) |
| 4 | `aosp_h144_mean_lam05_q80_long` | N | 0.32/0.32/0.76/0.42/2 (2/0) | 0.32/0.65/0.76/0.85/2 (2/0) | 0.00/0.00/0.00/0.00/0 (0/0) | 1.01/1.01/1.51/0.67/1 (1/0) | 0.00/0.00/0.00/0.00/0 (0/0) | 1.01/2.01/1.51/1.34/1 (1/0) |
| 5 | `aosp_h144_mean_lam05_q90_short` | N | 4.87/4.88/4.28/1.14/25 (0/25) | 0.58/1.17/2.86/0.41/12 (0/12) | 4.27/8.66/4.28/2.02/13 (0/13) | 0.25/0.25/2.56/0.10/5 (0/5) | -0.08/-0.15/2.56/-0.06/3 (0/3) | 0.33/0.65/1.76/0.37/2 (0/2) |
| 6 | `aosp_h144_mean_lam05_q80_short` | N | 4.87/4.88/4.28/1.14/25 (0/25) | 0.58/1.17/2.86/0.41/12 (0/12) | 4.27/8.66/4.28/2.02/13 (0/13) | 0.25/0.25/2.56/0.10/5 (0/5) | -0.08/-0.15/2.56/-0.06/3 (0/3) | 0.33/0.65/1.76/0.37/2 (0/2) |
| 7 | `aosp_h144_robust_lam10_q90_long` | N | 0.55/0.55/0.73/0.75/5 (5/0) | 0.26/0.53/0.57/0.93/1 (1/0) | 0.28/0.56/0.73/0.77/4 (4/0) | 0.00/0.00/0.00/0.00/0 (0/0) | 0.00/0.00/0.00/0.00/0 (0/0) | 0.00/0.00/0.00/0.00/0 (0/0) |
| 8 | `aosp_h144_robust_lam10_q80_long` | N | 0.55/0.55/0.73/0.75/5 (5/0) | 0.26/0.53/0.57/0.93/1 (1/0) | 0.28/0.56/0.73/0.77/4 (4/0) | 0.00/0.00/0.00/0.00/0 (0/0) | 0.00/0.00/0.00/0.00/0 (0/0) | 0.00/0.00/0.00/0.00/0 (0/0) |

## Leakage and execution

- Both market and spot frames are physically truncated before 2025.
- 2023 and 2024 use separate annual expanding fits.
- Every fit target is purged unless its exit is strictly before the fold cutoff.
- OI is delayed one complete 5m bar; incomplete spot or missing OI fails closed.
- Entry is the next 5m open; costs, idle calendar time, non-overlap, and split-contained exits are included.
- 2025/2026 and Gross9 portfolio marginal outcomes are not opened here.

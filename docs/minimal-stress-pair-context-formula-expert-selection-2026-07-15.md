# Minimal-stress pair-context formula expert selection

Metric: absolute return / CAGR / strict MDD / CAGR-MDD / trades.

## Verdict

**Frozen for OOS.** Minimal funding/premium stress owns entry. Prior-bar Alpha101/VPIN formulas are routed by independent USD/KRW and dollar-flow context responsibilities; the critic chooses ABSTAIN or LONG/SHORT TP4/time.

Multiplicity: 640 cells; 1 clear the absolute gate and beat the long time-only base.

| Policy | Train | 2023 selection | Pre-2024 | Score |
|---|---:|---:|---:|---:|
| Long time-only base | 51.86% / 18.17% / 14.88% / 1.22 / 125 | 20.25% / 20.27% / 8.77% / 2.31 / 31 | 82.62% / 18.77% / 15.29% / 1.23 / 156 | `[1.221301942119578, 1.2272029135936964, 156.0]` |
| Pair-context expert | 233.46% / 61.81% / 14.16% / 4.37 / 129 | 26.50% / 26.52% / 8.64% / 3.07 / 29 | 321.83% / 50.84% / 14.16% / 3.59 / 158 | `[3.070707908188333, 3.591631017983954, 158.0]` |

## Leakage controls

- Selection market, funding, and premium sources are physically truncated before 2024.
- Formula features are shifted one complete 5-minute bar; context fields use the audited live feature contract.
- Context thresholds, scaling, and model fitting stop before 2023; every 48-hour utility label exits before 2023.
- Trades enter next-open, pay 6bp/notional/side plus realized funding, remain split-contained, and use strict favorable-before-adverse MDD.

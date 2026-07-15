# Minimal-stress weak action expert selection

Metric: absolute return / CAGR / strict MDD / CAGR-MDD / trades.

## Verdict

**Frozen for OOS.** Funding-relief and premium-discount events own entry. A year-balanced source-aware weak tensor chooses ABSTAIN or LONG/SHORT with TP4/TP8/TP12/time; it is not a scalar gate sweep.

Multiplicity: 972 cells; 23 clear the absolute gate and beat the long time-only base.

| Policy | Train | 2023 selection | Pre-2024 | Score |
|---|---:|---:|---:|---:|
| Long time-only base | 58.26% / 20.14% / 16.47% / 1.22 / 125 | 22.66% / 22.67% / 9.65% / 2.35 / 31 | 94.11% / 20.85% / 16.92% / 1.23 / 156 | `[1.2225698256482864, 1.2324431970697747, 156.0]` |
| Weak action expert | 222.88% / 59.74% / 11.98% / 4.99 / 129 | 31.53% / 31.56% / 5.89% / 5.36 / 30 | 324.69% / 51.13% / 11.98% / 4.27 / 159 | `[4.267038245124276, 4.985418342113164, 159.0]` |

## Leakage controls

- Selection market, funding, and premium sources are physically truncated before 2024.
- Market features are prior completed 5-minute bars; BOCPD uses completed hours and exact boundary mapping.
- Scaling and model fitting stop before 2023; every 48-hour utility label exits before 2023.
- Selection schedules enter next-open, pay 6bp/notional/side plus realized funding, force split-contained exits, and use strict favorable-before-adverse MDD.

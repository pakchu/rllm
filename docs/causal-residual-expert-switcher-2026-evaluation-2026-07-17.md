# CRES-1 2026 one-shot evaluation

## Decision

- Strategy gate: **FAIL**.
- Disposition: **retire_cres1_no_2026_repair**.
- This file is the first and only CRES-1 opening of 2026 post-entry outcomes.
- All decisions were materialized before each event's outcome was computed.

## Primary metrics

| Window | Absolute return | Full-calendar CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 2026H1 | -0.50% | -1.00% | 4.26% | -0.23 | 11 |
| 2026 Q1 | -2.71% | -10.54% | 4.26% | -2.48 | 5 |
| 2026 Q2 | 2.27% | 9.44% | 2.83% | 3.33 | 6 |

## Controls (2026H1)

| Control | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 10 bp/side | -1.31% | -2.62% | 4.77% | -0.55 | 11 |
| Entry/exit +5m | -0.20% | -0.41% | 4.27% | -0.10 | 11 |
| Direction flip | -2.02% | -4.03% | 5.37% | -0.75 | 11 |

## Evidence boundary

- base events: 68; executed: 11; continuation: 7; reversion: 4; flat: 57;
- weekly-cluster sign-flip p-value: 0.56537;
- 2023-2025 were development; no 2026 threshold/sign/model repair is permitted.

Portfolio orthogonality is evaluated only if the strategy gate passes. Live use additionally requires an atomic two-leg alt executor and partial-fill neutralization.

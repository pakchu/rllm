# Conformal pressure ignition: pre-outcome rejection

Date: 2026-07-14

## Decision

The proposed `conformal-SR pressure -> price ignition + continuing OI build`
transition was rejected **before any return, CAGR, MDD, or trade outcome was
computed**. No 2024+ segment was opened.

The parent representation had already been inspected, so the child gate would
at best have been contaminated same-family exploration rather than independent
validation. More importantly, its support could not meet the frozen admission
floor.

## Frozen candidate

- Parent: the previously evaluated conformal-SR pressure event.
- Child condition: `pressure_side * current hourly price_z > 0` and
  one-source-row-delayed `oi_z > 0`.
- Direction: trade `pressure_side`.
- Intended entry: next 5-minute open after the completed minute-55 decision.
- Intended hold: 12 hours.

## Support-only preflight

These counts were computed without simulating returns.

| Mask | Fit raw / executable | 2023 raw / executable |
|---|---:|---:|
| Parent release, no child condition | 150 / 142 | 49 / 48 |
| Price ignition only | 87 / 87 | 26 / 26 |
| Continuing OI build only | 105 / 101 | 29 / 28 |
| Full price ignition + OI build | **75 / 75** | **22 / 22** |
| Opposed price + OI build | 30 / 28 | 7 / 7 |

The frozen admission rule requires at least 80 fit trades and 24 trades in
2023. The full conjunction has only 75 and 22, so no possible realized returns
could make it admissible.

## Why the branch was stopped

1. The full mechanism fails the predeclared minimum support before outcomes.
2. `price_z` and `oi_z` are components of the already inspected parent state;
   adding a gate is not a new independent representation.
3. A minute-55 decision followed by a minute-00 entry also needs an explicit
   one-bar execution-delay control before any live claim.
4. Replacing the failed conjunction with a single condition only because it has
   more support would silently change the original hypothesis. That is reserved
   for a separately predeclared experiment, not this one.

## Outcome disclosure

No profitability statistics exist for this rejected design. In particular,
there is no absolute return, CAGR, strict MDD, CAGR/MDD, or trade-outcome table
to report. This is intentional: stopping an underpowered, contaminated branch
before looking at returns prevents another degree of researcher freedom.

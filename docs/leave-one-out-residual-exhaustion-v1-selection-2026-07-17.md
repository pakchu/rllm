# LORE v1 2023–2024 selection — 2026-07-17

> Only 2023–2024 outcomes were opened. Calendar 2025 and 2026 remain sealed for LORE v1.

| Rank | Policy | Residual/hold | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | Decision |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | L01 | 6h/12h | -44.153% | -25.254% | 53.002% | -0.476 | 284 | REJECT |
| 2 | L04 | 12h/24h | -79.283% | -54.460% | 80.628% | -0.675 | 163 | REJECT |
| 3 | L03 | 12h/12h | -82.370% | -57.987% | 82.915% | -0.699 | 224 | REJECT |
| 4 | L02 | 6h/24h | -65.964% | -41.638% | 67.221% | -0.619 | 214 | REJECT |

## Decision

- status: **rejected_before_2025_holdout**
- selected policy: `None`
- passing policies: `[]`
- CAGR includes the full calendar, including idle periods.
- strict MDD includes global/pre-entry HWM, favorable-before-adverse two-leg OHLC, entry and hypothetical liquidation costs, exact funding event ordering, and scheduled exit cost.
- Diagnostics cannot rescue a rejected policy; no sign/threshold/hold/pair repair is allowed.

## Failure diagnosis

- The hypothesized **mean reversion** is wrong at both horizons: all four policies lose money after 6 bp/notional/side costs in both the 2023 fit and 2024 test years.
- Best L01 annual results were 2023 absolute `-35.405%`, CAGR `-35.425%`, strict MDD `39.000%`, 127 trades; and 2024 absolute `-13.542%`, CAGR `-13.516%`, strict MDD `29.728%`, 157 trades.
- The weekly-cluster sign-flip result for L01 is Bonferroni p=`1.000`; this is not sampling noise around a positive mean.
- Removing flow disconfirmation, replacing residuals with raw returns, delaying entry, shifting the clock, and monthly pair permutation all remained negative. LORE v1 is therefore retired before any 2025 access.

## Research-only continuation clue

The preregistered direction-flip diagnostic was profitable for the 12-hour residual policies:

| Frozen clock | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| L03 direction flip | +191.325% | +70.620% | 22.934% | 3.079 | 224 |
| L04 direction flip | +179.605% | +67.155% | 23.157% | 2.900 | 163 |

These rows **do not rescue LORE v1** and are contaminated exploratory evidence on 2023–2024. They may only motivate a separately named, single-policy residual-continuation family whose first confirmatory test is untouched calendar 2025.

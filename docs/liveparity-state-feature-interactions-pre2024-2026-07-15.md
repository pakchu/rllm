# Live-parity state/feature interaction rejection — 2026-07-15

This formalizes the completed broad weak interaction tree search on physically truncated pre-2024 data.

## Protocol

- Decisions occur only on `:00` rows.
- `live_decision_features` excludes the current market bar; completed hourly state uses `[HH-1:00, HH:00)` at `HH:00`.
- Tree targets and thresholds are fit only on 2020-07-01 through 2022-12-31.
- Market/funding/premium inputs are truncated before 2024; 2024+ remains sealed.
- Entry is the next 5-minute open; costs are 6 bp/notional/side; realized funding and strict MDD are included.
- Schedules are non-overlapping and exits must stay inside each split.

## Result

**REJECTION.** The grid evaluated `762` cells and found `0` qualifiers.

## Top ranked cells

| Rank | Group | Cooldown | Hold | Depth | Leaf | Threshold | Train CAGR/MDD | 2023 CAGR/MDD | Pre-2024 CAGR/MDD | Pre-2024 trades | Pass |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | state_pa | 144 | 288 | 2 | 32 | 0.00332147 | 2.66 | 2.40 | 2.17 | 155 | False |
| 2 | state_pa_macro | 144 | 288 | 2 | 32 | 0.00327060 | 2.66 | 2.40 | 2.17 | 155 | False |
| 3 | state_all | 144 | 288 | 2 | 32 | 0.00327060 | 2.66 | 2.40 | 2.17 | 155 | False |
| 4 | state_pa | 144 | 288 | 2 | 20 | 0.00532858 | 2.61 | 2.12 | 2.09 | 142 | False |
| 5 | state_pa_macro | 144 | 288 | 4 | 32 | 0.00000000 | 2.73 | 2.20 | 2.04 | 153 | False |

## Interpretation

The best-ranked cells are economically interesting but fail the frozen selection contract, primarily because the minimum train/2023/pre-2024 CAGR-to-strict-MDD target is not met across all required windows. No 2024+ evaluation is justified.

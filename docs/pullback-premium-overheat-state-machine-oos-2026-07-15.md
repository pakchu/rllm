# Pullback premium-overheat state machine OOS

- Status: **rejected by the frozen OOS gate**.
- Entry: causal confirmed pullback squeeze; next 5-minute open.
- Capitulation: weak week AND (wide 48h range OR dry quote activity) -> 4% take.
- Premium overheat: high premium-index change AND high 48h range position -> skip.
- Orderly remainder: 12% take. All routes use a 48h cap and no stop.
- Leverage 0.50x; cost 6bp/notional/side plus realized funding.
- Strict MDD uses global/pre-entry HWM and favorable-then-adverse position envelopes.
- Eight state-machine cells were selected on physically truncated pre-2024 data.
- Family-level 2024+ is not pristine because related pullback variants were previously inspected; the exact policy was not selected from future outcomes.

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| test_2024 | 21.20% | 21.16% | 5.29% | 4.00 | 18 |
| eval_2025 | 6.78% | 6.78% | 5.39% | 1.26 | 11 |
| holdout_2026 | 13.69% | 36.12% | 4.09% | 8.83 | 19 |
| oos_2024_2026 | 47.13% | 17.32% | 5.72% | 3.03 | 48 |

`holdout_2026` is a short diagnostic and is not part of the frozen OOS gate.

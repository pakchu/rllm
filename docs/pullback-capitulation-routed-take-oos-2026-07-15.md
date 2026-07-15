# Pullback capitulation-routed take OOS

- Status: **rejected by the frozen OOS gate**.
- Entry signal: confirmed pullback squeeze at a completed hourly boundary; enter next 5-minute open.
- Route: weak completed weekly return AND (wide completed 48h range OR dry 1d quote activity).
- Stress route: 4% take; normal route: 12% take; both use 48h cap and no stop.
- Leverage: 0.60x; cost: 6bp/notional/side plus realized funding.
- Strict MDD: global/pre-entry HWM plus position favorable envelope before adverse envelope.
- Selection sources were physically truncated before `2024-01-01`; route thresholds use active fit events only.
- Multiplicity disclosed in the frozen specification; this is a conditional interaction, not a score blend.

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| test_2024 | 26.16% | 26.10% | 6.33% | 4.12 | 18 |
| eval_2025 | 8.16% | 8.16% | 6.44% | 1.27 | 11 |
| holdout_2026 | 16.36% | 43.92% | 4.90% | 8.97 | 20 |
| oos_2024_2026 | 58.77% | 21.07% | 6.84% | 3.08 | 49 |

`holdout_2026` is reported as a shorter diagnostic window and is not part of the frozen OOS gate.

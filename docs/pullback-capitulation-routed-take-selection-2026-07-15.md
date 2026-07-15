# Pullback capitulation-routed take selection

- Status: **frozen pre-2024 candidate; 2024+ not opened**.
- Entry signal: confirmed pullback squeeze at a completed hourly boundary; enter next 5-minute open.
- Route: weak completed weekly return AND (wide completed 48h range OR dry 1d quote activity).
- Stress route: 4% take; normal route: 12% take; both use 48h cap and no stop.
- Leverage: 0.60x; cost: 6bp/notional/side plus realized funding.
- Strict MDD: global/pre-entry HWM plus position favorable envelope before adverse envelope.
- Selection sources were physically truncated before `2024-01-01`; route thresholds use active fit events only.
- Multiplicity disclosed in the frozen specification; this is a conditional interaction, not a score blend.

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| train | 147.06% | 43.54% | 11.80% | 3.69 | 81 |
| train_2020h2 | 27.73% | 62.55% | 8.25% | 7.58 | 14 |
| train_2021 | 48.23% | 48.27% | 11.80% | 4.09 | 27 |
| train_2022 | 27.19% | 27.21% | 11.60% | 2.35 | 39 |
| select_2023 | 17.42% | 17.43% | 4.68% | 3.72 | 17 |
| select_2023_h1 | 11.00% | 23.44% | 4.68% | 5.00 | 9 |
| select_2023_h2 | 5.78% | 11.80% | 3.68% | 3.21 | 8 |
| pre_2024 | 190.09% | 35.55% | 11.80% | 3.01 | 98 |

## Frozen thresholds

```json
{
  "week_low": 0.004204836782434551,
  "range_wide": 0.08667875748951356,
  "quote_activity_dry": -1.162656660957789,
  "week_quantile": 0.5,
  "range_quantile": 0.5,
  "quote_activity_quantile": 0.2,
  "fit_active_events": 408
}
```

## Search accounting

The route emerged after 23 trade-feature diagnostics, 12 conditional-route cells, 12 bounded exit refinements, five leverage points, and eight earlier rejected mechanism families. This multiplicity is retrospective and is not presented as a pristine single-hypothesis test.

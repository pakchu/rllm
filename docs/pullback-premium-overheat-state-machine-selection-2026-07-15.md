# Pullback premium-overheat state machine selection

- Status: **frozen pre-2024 candidate; 2024+ not opened for this exact policy**.
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
| train | 116.08% | 36.06% | 7.96% | 4.53 | 75 |
| train_2020h2 | 16.67% | 35.80% | 4.88% | 7.33 | 13 |
| train_2021 | 45.62% | 45.66% | 7.96% | 5.73 | 26 |
| train_2022 | 24.49% | 24.51% | 7.96% | 3.08 | 35 |
| select_2023 | 15.04% | 15.05% | 3.92% | 3.84 | 15 |
| select_2023_h1 | 9.11% | 19.23% | 3.92% | 4.91 | 9 |
| select_2023_h2 | 5.44% | 11.09% | 3.08% | 3.60 | 6 |
| pre_2024 | 148.58% | 29.70% | 7.96% | 3.73 | 90 |

## Selection grid

Only `premium_range_overheat + skip` passed. COUA, CERF and PWPM transition families were also tested and rejected before this state-machine refinement.

```json
{
  "fit_active_events": 408,
  "htf_1w_return_1_q50": 0.004204836782434551,
  "htf_1w_return_1_q67": 0.028898560185342426,
  "rex_576_range_width_pct_q50": 0.08667875748951356,
  "quote_vol_z_1d_q20": -1.162656660957789,
  "quote_vol_z_1d_q67": -0.1409544241230363,
  "rex_576_range_pos_q67": 0.15534622924628988,
  "bb_z_q67": 0.538528750322368,
  "premium_index_change_q67": 2.924929999999999e-05,
  "htf_3d_return_1_q67": 0.013121525434418135
}
```

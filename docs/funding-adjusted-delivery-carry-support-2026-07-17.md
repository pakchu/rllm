# FADC-21 outcome-blind support — 2026-07-17

## Boundary

This unit loaded only current/past completed closes, settled funding rates and
delivery clocks. It did not load quarterly/perpetual future entry or exit
prices, held high/low paths, funding settlement marks, returns, PnL, CAGR or
MDD. No row at or after 2024-01-01 was opened.

## Support result

| Window | Entries | Direction split | Max month share | Active days | Median hold |
|---|---:|---|---:|---:|---:|
| 2021–2022 | 30 | {'perp_long_quarter_short': 11, 'perp_short_quarter_long': 19} | 10.00% | 352.3 | 10.17d |
| 2023 diagnostic | 9 | {'perp_long_quarter_short': 3, 'perp_short_quarter_long': 6} | 22.22% | 153.7 | 15.00d |

Year/half counts:

- pre-2023 years: `{'2021': 16, '2022': 14}`
- pre-2023 halves: `{'2021H1': 8, '2021H2': 8, '2022H1': 9, '2022H2': 5}`
- 2023 halves: `{'2023H1': 3, '2023H2': 6}`

Disposition: **PASS_SUPPORT_OPEN_2021_2022_PNL**.

Passing this unit permits only the frozen 2021–2022 stage-1 evaluator. It does
not establish alpha, profitability or orthogonality. 2023 PnL stays sealed
unless stage 1 passes; 2024 remains source-and-outcome sealed.

Content hash: `dab72f4495f54b91ccccf052e1a6c9d0dccadd8a90130336516d7dd3b8276785`

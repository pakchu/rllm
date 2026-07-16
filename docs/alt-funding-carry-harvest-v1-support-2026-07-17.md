# AFCH v1 outcome-blind support — 2026-07-17

> Exact funding features and sleeve clocks only. No post-entry price return, PnL, CAGR, or MDD was calculated.

## Support result

- events: `127`; years `{'2023': 45, '2024': 46, '2025': 36}`
- halves: `{'2023H1': 22, '2023H2': 23, '2024H1': 26, '2024H2': 20, '2025H1': 24, '2025H2': 12}`
- unique ordered high>low pairs: `13`
- maximum pair share: `23.6220%`
- long symbols: `['ADAUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']`
- short symbols: `['ADAUSDT', 'BNBUSDT', 'DOGEUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']`
- maximum active sleeves: `4`
- projected 28d carry: minimum `18.374 bp`, median `44.187 bp`
- maximum monthly source quarantine: `0.4032%`
- decision: **PASS**
- clock SHA-256: `d0cc451e74a45d5ab9f08a8a7c3d8c3df756d35d2e020d1f70dc0198ebaf0ef9`

| Short high-funding > long low-funding | Sleeves |
|---|---:|
| ADAUSDT>BNBUSDT | 13 |
| ADAUSDT>SOLUSDT | 3 |
| BNBUSDT>ADAUSDT | 1 |
| DOGEUSDT>ADAUSDT | 2 |
| DOGEUSDT>BNBUSDT | 30 |
| DOGEUSDT>SOLUSDT | 6 |
| DOGEUSDT>XRPUSDT | 1 |
| ETHUSDT>ADAUSDT | 1 |
| ETHUSDT>BNBUSDT | 25 |
| ETHUSDT>SOLUSDT | 4 |
| SOLUSDT>BNBUSDT | 10 |
| XRPUSDT>BNBUSDT | 26 |
| XRPUSDT>SOLUSDT | 5 |

Each row is fixed before outcomes, enters at Monday 00:10 UTC, holds 28 days,
uses a 0.25-gross sleeve, and has projected carry of at least 18 bp at normalized
gross one. At most four vintages overlap.

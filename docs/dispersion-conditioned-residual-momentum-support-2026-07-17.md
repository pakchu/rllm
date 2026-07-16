# DCRM-1 outcome-blind support — 2026-07-17

- Post-entry returns/PnL calculated: **no**
- Accepted weekly states: `92`; years `{'2023': 39, '2024': 53}`; halves `{'2023H1': 13, '2023H2': 26, '2024H1': 26, '2024H2': 27}`
- Gross buckets: `{'0.25': 29, '1.0': 63}`
- Unique ordered pairs: `25`
- Maximum pair share: `13.04%`
- Maximum month share: `5.43%`
- Long symbols: `['ADAUSDT', 'BNBUSDT', 'DOGEUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']`
- Short symbols: `['ADAUSDT', 'BNBUSDT', 'DOGEUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']`
- LORE exact-entry Jaccard: `0.0015`
- LORE 5m position-time Jaccard: `0.3557`
- Support gate: **PASS**
- Clock SHA-256: `d7cb7b5066692b8dccc6dbc2051d01c9522acc1e0769e63b7a6135bbffeae992`

The builder read only timestamps and closes through 2024. The weekly clock,
leave-one-out factors, shifted betas, pair identities, and gross scales contain
no entry/exit price, post-entry return, PnL, equity, 2025, or 2026 outcome.
The 0.25 and 1.0 buckets are reported separately so nominal event count does
not hide reduced exposure. LORE overlap also uses clocks only, not returns.
